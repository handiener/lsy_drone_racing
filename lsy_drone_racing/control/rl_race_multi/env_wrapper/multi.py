"""Two physical drones, one learner, frozen opponents; no official environment changes.

Training ends on the first crash/finish or world timeout. Only a genuine first
completion is a win; an opponent crash alone is NOT a learner win. Official
full-race evaluation uses MultiDroneRaceEnv separately (see evaluation.py).
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from crazyflow.sim.sim import sync_sim2mjx

from lsy_drone_racing.envs import race_core
from lsy_drone_racing.envs.multi_drone_race import VecMultiDroneRaceEnv
from lsy_drone_racing.utils import load_config

from ..league import probabilities
from ..network import reset_carry, zeros
from . import rewards
from .actions import ActionConfig, to_command
from .env import METRICS, EnvConfig, RaceEnv
from .observations import encode

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "config/multi_new_level2.toml"
MIN_START_DISTANCE_M = 0.22  # Horizontal center separation; cf21B box diagonal is about 0.20 m.
MULTI_METRICS = (
    "win",
    "loss",
    "draw",
    "opponent_finished",
    "opponent_crashed",
    "collision",
    "ego_out_of_bounds",
    "opponent_out_of_bounds",
    "opponent_solo",
    "outcome_done",
)


def outcome(finished: Any, crashed: Any, timeout: Any) -> tuple:
    """Completion determines wins; both crash, an interrupted race, or timeout is a draw."""
    done = jnp.any(finished | crashed, axis=1) | timeout
    win = finished[:, 0] & ~finished[:, 1] & ~crashed[:, 0]
    loss = finished[:, 1] & ~finished[:, 0] & ~crashed[:, 1]
    draw = done & ~win & ~loss
    return done, win, loss, draw


class MultiRaceEnv(RaceEnv):
    def __init__(
        self,
        n_envs: int,
        cfg: EnvConfig,
        league: Any,
        network: Any,
        config_path: str | Path | None = None,
        device: str = "gpu",
        seed: int = 1,
        swap_starts: bool = True,
        snapshot_interval: int = 100,
        transitions: int = 1,
        full_race: bool = False,
        fixed_opponent: int | None = None,
        deterministic_opponent: bool = True,
    ):
        self.cfg, self.n_envs, self.league, self.network = cfg, int(n_envs), league, network
        self.seed = seed
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        config = load_config(self.config_path)
        if len(config.env.track.drones) != 2:
            raise ValueError("League environment requires exactly two drones")
        config.sim.render = False
        self.freq = 50
        self.full_race, self.fixed_opponent = full_race, fixed_opponent
        self.deterministic_opponent = deterministic_opponent
        self.swap_starts = swap_starts
        self.snapshot_interval, self.transitions = snapshot_interval, transitions
        self.last_snapshot = 0
        self._core = VecMultiDroneRaceEnv(
            num_envs=n_envs,
            freq=50,
            sim_config=config.sim,
            track=config.env.track,
            sensor_range=config.env.kwargs[0]["sensor_range"],
            control_mode="attitude",
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=seed,
            max_episode_steps=cfg.max_episode_steps,
            device=device,
        )
        self._core.settings = self._core.settings.replace(autoreset=False)
        self._template = self._core.data
        low, high = self._core.single_action_space.low[0], self._core.single_action_space.high[0]
        a = cfg.action
        self.action_cfg = ActionConfig(
            a.roll_limit_deg,
            a.pitch_limit_deg,
            a.yaw_limit_deg,
            a.thrust_min_fraction,
            a.thrust_max_fraction,
            float(low[3] + (high[3] - low[3]) * a.thrust_min_fraction),
            float(low[3] + (high[3] - low[3]) * a.thrust_max_fraction),
        )
        if asdict(self.action_cfg) != league.action:
            raise ValueError("Frozen opponents must use the same action scaling as learner")
        self.n_seq = int(self._template.gate_sequence.shape[0])
        self._physics = self._build_physics()

    def _build_physics(self) -> Any:
        """Reuse the official core helpers, observing contacts before disabled drones warp.

        This mirrors RaceCore's orchestration, not Crazyflow integration or gate logic.
        The extra pair mask identifies inter-agent contact rather than guessing from distance.
        """
        core = self._core
        apply = core.build_apply_action_fn()
        sim_step = core.sim.build_step_fn()
        template = core.sim.mjx_data
        model = core.sim.mjx_model
        masks = race_core._load_contact_masks(core.sim)
        pair_mask = jnp.asarray(masks[:, 0] & masks[:, 1])
        gate_ids, obstacle_ids = core.mocap_ids

        def step(data: Any, action: Any) -> tuple:
            data = apply(action, data)
            data = data.replace(sim_data=sim_step(data.sim_data, data.sim_data.core.freq // 50))
            physical_pos = data.sim_data.states.pos
            mocap_pos = template.mocap_pos.at[..., gate_ids, :].set(data.gates_pos)
            mocap_pos = mocap_pos.at[..., obstacle_ids, :].set(data.obstacles_pos)
            mocap_quat = template.mocap_quat.at[..., gate_ids, :].set(
                jnp.roll(data.gates_quat, 1, axis=-1)
            )
            _, mjx = sync_sim2mjx(
                data.sim_data, template.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat), model
            )
            contacts = mjx._impl.contact.dist < 0
            per_drone = jnp.any(contacts[:, None, :] & jnp.asarray(masks), axis=-1)
            pair = jnp.any(contacts & pair_mask, axis=-1)
            data = race_core._update_disabled_drones(data, per_drone)
            data = race_core._warp_disabled_drones(data)
            data = race_core._update_visited_objects(data)
            data = race_core._update_target_gates(data)
            data = race_core._mark_drones_for_reset(data)
            return data.replace(steps=data.steps + 1), pair, physical_pos

        return jax.jit(step)

    def _reset_worlds(self, data: Any, mask: Any, key: Any) -> Any:
        data, _ = self._core._reset(data, None, mask)
        states = data.sim_data.states

        def too_close(pos: Any) -> Any:
            distance_sq = jnp.sum((pos[:, 0, :2] - pos[:, 1, :2]) ** 2, axis=-1)
            return mask & (distance_sq < MIN_START_DISTANCE_M**2)

        def resample(carry: tuple) -> tuple:
            pos, rng = carry
            rng, sample_key = jax.random.split(rng)
            proposed = self._core.sim.default_data.states.pos + self._core.settings.randomizations[
                "drone_pos"
            ](sample_key, shape=pos.shape)
            return jnp.where(too_close(pos)[:, None, None], proposed, pos), rng

        pos, key = jax.lax.while_loop(
            lambda carry: jnp.any(too_close(carry[0])), resample, (states.pos, key)
        )
        data = data.replace(
            sim_data=data.sim_data.replace(
                states=states.replace(pos=pos), core=data.sim_data.core.replace(mjx_synced=False)
            )
        )
        if self.swap_starts:
            swap = mask & jax.random.bernoulli(key, 0.5, (self.n_envs,))
            states = data.sim_data.states
            changed = {}
            for name in ("pos", "vel", "quat", "ang_vel"):
                value = getattr(states, name)
                changed[name] = jnp.where(swap[:, None, None], value[:, ::-1], value)
            states = states.replace(**changed)
            takeoff = jnp.where(swap[:, None, None], data.takeoff_pos[:, ::-1], data.takeoff_pos)
            data = data.replace(
                sim_data=data.sim_data.replace(
                    states=states, core=data.sim_data.core.replace(mjx_synced=False)
                ),
                takeoff_pos=takeoff,
            )
        return race_core._reset_env_data(data, mask)

    def _features(self, data: Any, previous: Any, solo_opponent: Any) -> tuple:
        raw = race_core.obs(data)
        a = {k: v[:, 0] for k, v in raw.items()}
        b = {k: v[:, 1] for k, v in raw.items()}
        alive = ~data.disabled_drones
        ego = encode(a, previous[:, 0], self.cfg.obs, b, alive[:, 1])
        opponent = encode(b, previous[:, 1], self.cfg.obs, a, alive[:, 0] & ~solo_opponent)
        return ego, opponent

    def _sample(self, key: Any, pool: Any) -> Any:
        if self.fixed_opponent is not None:
            return jnp.full((self.n_envs,), self.fixed_opponent, jnp.int32)
        p = probabilities(
            pool["count"],
            self.league.capacity,
            self.league.solo_fraction,
            self.league.exponent,
            pool["ranks"],
        )
        return jax.random.choice(key, self.league.capacity, (self.n_envs,), p=p)

    def reset(self, rng: Any) -> State:
        key, sim_key, swap_key, opp_key = jax.random.split(rng[0], 4)
        data = self._template
        data = data.replace(
            sim_data=data.sim_data.replace(
                core=data.sim_data.core.replace(rng_key=jax.random.wrap_key_data(sim_key))
            )
        )
        data = self._reset_worlds(data, jnp.ones(self.n_envs, bool), swap_key)
        pool = self.league.packed()
        index = self._sample(opp_key, pool)
        prev = jnp.zeros((self.n_envs, 2, 4))
        obs, opp = self._features(data, prev, pool["solo"][index])
        z = jnp.zeros(self.n_envs)
        metrics = {k: z for k in (*METRICS, *MULTI_METRICS)}
        info = dict(
            rng=key,
            previous_action=prev,
            official=jnp.ones(self.n_envs),
            truncation=z,
            steps=z,
            episode_done=z,
            terminal_obs=obs,
            learner_reset=z,
            finish_times=jnp.full((self.n_envs, 2), jnp.inf),
            recurrent=(
                zeros((self.n_envs,), self.cfg.hidden_size),
                zeros((self.n_envs,), self.cfg.hidden_size),
            ),
            opponent_carry=zeros((self.n_envs,), self.cfg.hidden_size),
            opponent_obs=opp,
            opponent_index=index,
            league=pool,
            episode_metrics=dict(sum_reward=z, length=z, **metrics),
        )
        return State(pipeline_state=data, obs=obs, reward=z, done=z, metrics=metrics, info=info)

    def step(self, state: State, action: Any) -> State:
        info = state.info
        pool, index = info["league"], info["opponent_index"]
        key, opp_key, swap_key, sample_key = jax.random.split(info["rng"], 4)
        actor = jax.tree.map(lambda x: x[index], pool["actor"])
        stats = jax.tree.map(lambda x: x[index], pool["stats"])

        def forward(params, normalizer, obs, carry, key):
            carry, logits = self.network.policy_network.apply(normalizer, params, obs, carry)
            raw = (
                logits[..., :4]
                if self.deterministic_opponent
                else self.network.parametric_action_distribution.sample_no_postprocessing(
                    logits, key
                )
            )
            return carry, self.network.parametric_action_distribution.postprocess(raw)

        opponent_carry, opponent_action = jax.vmap(forward)(
            actor,
            stats,
            info["opponent_obs"],
            info["opponent_carry"],
            jax.random.split(opp_key, self.n_envs),
        )
        opponent_carry, opponent_action = jax.tree.map(
            jax.lax.stop_gradient, (opponent_carry, opponent_action)
        )
        joint = jnp.stack([action, opponent_action], axis=1)
        prev = state.pipeline_state
        data, pair, positions = self._physics(prev, to_command(joint, self.action_cfg))
        passed = data.n_gates_passed
        before = prev.n_gates_passed
        finished = (passed >= self.n_seq) & (before < self.n_seq)
        crashed = data.disabled_drones & ~(passed >= self.n_seq)
        timeout = data.steps >= self.cfg.max_episode_steps
        done, win, loss, draw = outcome(finished, crashed, timeout)
        finish_times = jnp.where(finished, data.steps[:, None] / 50, info["finish_times"])
        if self.full_race:
            ended = data.disabled_drones | (passed >= self.n_seq)
            done = jnp.all(ended, axis=1) | timeout
            win = (
                done & jnp.isfinite(finish_times[:, 0]) & (finish_times[:, 0] < finish_times[:, 1])
            )
            loss = (
                done & jnp.isfinite(finish_times[:, 1]) & (finish_times[:, 1] < finish_times[:, 0])
            )
            draw = done & ~win & ~loss
        crash_event = crashed & ~prev.disabled_drones
        # An opponent event ends the learner trajectory too, with a real terminal
        # outcome; only pure time limits bootstrap the physical terminal state.
        trunc = timeout & ~jnp.any(finished | crashed, axis=1)
        event = rewards.events(before[:, 0], passed[:, 0], crash_event[:, 0], self.n_seq)
        seconds = data.steps.astype(jnp.float32) / 50
        reward, parts = rewards.compute(
            self.cfg.reward,
            event,
            data.sim_data.states.ang_vel[:, 0],
            jnp.zeros(self.n_envs),
            seconds,
        )
        delta = (passed - before).astype(jnp.float32)
        reward, parts = rewards.competitive(
            self.cfg.reward,
            parts,
            delta[:, 0],
            (delta[:, 0] - delta[:, 1]) / self.n_seq,
            jnp.linalg.norm(positions[:, 1] - positions[:, 0], axis=-1),
            pair.astype(jnp.float32),
            win.astype(float),
            loss.astype(float),
            draw.astype(float),
        )
        terminal, _ = self._features(data, joint, pool["solo"][index])
        outside = (
            jnp.any((positions < data.pos_limit_low) | (positions > data.pos_limit_high), axis=-1)
            & crash_event
        )
        metrics = dict(
            gates=event["gates"].astype(float),
            finished=finished[:, 0].astype(float),
            crashed=crash_event[:, 0].astype(float),
            timeout=trunc.astype(float),
            official_done=done.astype(float),
            success=finished[:, 0].astype(float),
            lap_time=finished[:, 0] * seconds,
            **{f"r_{k}": v for k, v in parts.items()},
            win=win.astype(float),
            loss=loss.astype(float),
            draw=draw.astype(float),
            opponent_finished=finished[:, 1].astype(float),
            opponent_crashed=crash_event[:, 1].astype(float),
            collision=pair.astype(float),
            ego_out_of_bounds=outside[:, 0].astype(float),
            opponent_out_of_bounds=outside[:, 1].astype(float),
            opponent_solo=done.astype(float) * pool["solo"][index],
            outcome_done=done.astype(float),
        )
        keep = 1 - info["episode_done"]
        episode = {k: v * keep for k, v in info["episode_metrics"].items()}
        episode["sum_reward"] += reward
        episode["length"] += 1
        for k, v in metrics.items():
            episode[k] += v
        reset = self._reset_worlds(data, done, swap_key)
        next_index = jnp.where(done, self._sample(sample_key, pool), index)
        previous = jnp.where(done[:, None, None], 0.0, jnp.clip(joint, -1.0, 1.0))
        obs, opponent_obs = self._features(reset, previous, pool["solo"][next_index])
        next_info = dict(
            info,
            rng=key,
            previous_action=previous,
            opponent_index=next_index,
            opponent_carry=reset_carry(opponent_carry, done | finished[:, 1] | crashed[:, 1]),
            opponent_obs=opponent_obs,
            finish_times=jnp.where(done[:, None], jnp.inf, finish_times),
            learner_reset=(done | finished[:, 0] | crashed[:, 0]).astype(float),
            terminal_obs=terminal,
            truncation=trunc.astype(float),
            steps=data.steps.astype(float),
            episode_done=done.astype(float),
            episode_metrics=episode,
        )
        return State(
            pipeline_state=reset,
            obs=obs,
            reward=reward,
            done=done.astype(float),
            metrics=metrics,
            info=next_info,
        )

    def update_league_state(self, state: State, params: Any, steps: int) -> State:
        iteration = steps // self.transitions
        if iteration - self.last_snapshot < self.snapshot_interval:
            return state
        self.last_snapshot = iteration
        from dataclasses import asdict

        from ..checkpoint import save

        file = self.league.directory / f"iteration_{iteration:08d}.pkl"
        config = asdict(self.cfg)
        config["action"] = asdict(self.action_cfg)
        save(
            params,
            file,
            network=self.league.network,
            env=config,
            stage="multi",
            steps=getattr(self, "step_offset", 0) + steps,
            league=self.league.metadata(),
            metadata=dict(seed=self.seed, run_id=self.league.directory.parent.name),
        )
        active = set(np.asarray(state.info["opponent_index"]).ravel().tolist())
        installed = self.league.add(file, solo=False, active_slots=active)
        import json

        with (self.league.directory / "snapshots.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(iteration=iteration, checkpoint=str(file), installed=installed))
                + "\n"
            )
        if not installed:
            return state
        pool = self.league.packed()
        devices = state.obs.shape[0]
        packed = jax.tree.map(lambda x: jnp.broadcast_to(x, (devices, *x.shape)), pool)
        return state.replace(info=dict(state.info, league=packed))
