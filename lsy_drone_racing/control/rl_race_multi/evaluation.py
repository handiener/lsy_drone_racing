"""Reproducible recurrent evaluation; official full-race rules are the multi default."""

import hashlib
import json
from pathlib import Path
from typing import Any

import fire
import jax
import jax.numpy as jnp
import numpy as np

from . import checkpoint
from .controller import network_from_metadata
from .env_wrapper.actions import ActionConfig
from .env_wrapper.env import EnvConfig, RaceEnv
from .env_wrapper.multi import MultiRaceEnv
from .env_wrapper.observations import ObsConfig
from .env_wrapper.rewards import RewardConfig
from .env_wrapper.starts import StartConfig
from .league import League
from .network import make_inference_fn, reset_carry


def ranking_score(
    success_rate: float, lap_time: float | None, minimum_success: float = 0.5
) -> float:
    return (
        lap_time
        if success_rate >= minimum_success and lap_time is not None
        else 100 + 100 * max(minimum_success - success_rate, 0.0)
    )


def config_from_record(record: dict) -> EnvConfig:
    return EnvConfig(
        max_episode_steps=record["max_episode_steps"],
        hidden_size=record["hidden_size"],
        action=ActionConfig(**record["action"]),
        obs=ObsConfig(**record["obs"]),
        reward=RewardConfig(**record["reward"]),
        start=StartConfig(official_prob=1.0),
    )


def episode_batch(env: Any, policy: Any, params: Any, key: Any, count: int, length: int) -> dict:
    state = env.reset(jax.random.split(key, count))
    totals = {k: jnp.zeros(count) for k in state.metrics}
    totals["reward"] = jnp.zeros(count)
    totals["duration"] = jnp.zeros(count)

    def step(carry, _):
        state, seen, totals, key = carry
        key, subkey = jax.random.split(key)
        hidden, action, _ = policy(state.obs, state.info["recurrent"], subkey)
        next_state = env.step(state, action)
        next_state = next_state.replace(
            info=dict(
                next_state.info,
                recurrent=reset_carry(
                    hidden, next_state.info.get("learner_reset", next_state.done) > 0
                ),
            )
        )
        metrics = dict(next_state.metrics, reward=next_state.reward, duration=jnp.ones(count) / 50)
        totals = {k: totals[k] + jnp.where(~seen, metrics[k], 0.0) for k in totals}
        return (next_state, seen | (next_state.done > 0), totals, key), None

    (_, seen, totals, _), _ = jax.lax.scan(
        step, (state, jnp.zeros(count, bool), totals, key), None, length=length
    )
    return dict(totals, seen=seen)


def evaluate(
    run_dir: str,
    stage: str = "multi",
    episodes: int = 128,
    seeds: tuple[int, ...] = (410000, 420000, 430000),
    batch_size: int = 32,
    device: str = "gpu",
    report_name: str = "final",
    official: bool = True,
) -> dict:
    if min(episodes, batch_size) < 1 or not seeds:
        raise ValueError("Positive episodes/batch and seeds required")
    if Path(report_name).name != report_name or report_name in {".", "..", ""}:
        raise ValueError("Invalid report name")
    run = Path(run_dir).expanduser().resolve()
    path = run / "checkpoints" / f"{stage}_final.pkl"
    model = checkpoint.load(path)
    record = json.loads((run / f"{stage}_config.json").read_text())
    cfg = config_from_record(model["env"])
    nets = network_from_metadata(model["network"]).factory()(57, 4)
    params = jax.tree.map(jnp.asarray, model["params"])
    policy = make_inference_fn(nets)(params, deterministic=True)
    local_track = run / ("multi_track.toml" if stage == "multi" else "solo_track.toml")
    track_path = str(local_track) if local_track.exists() else record["config_path"]
    count = min(batch_size, episodes)
    rows = []
    pool = None
    if stage == "multi":
        entries = model["league"]["entries"]
        # Read existing metadata only; evaluation does not mutate training pool files.
        directory = run / f"{report_name}_opponents"
        manifest = directory / "source.json"
        checkpoint.write_json(manifest, model["league"])
        pool = League(
            entries[0]["checkpoint"],
            model["network"],
            model["env"]["action"],
            directory,
            model["league"]["capacity"],
            model["league"]["solo_fraction"],
            model["league"]["exponent"],
            str(manifest),
        )
    opponents = list(range(len(pool.entries))) if pool else [None]
    for opponent in opponents:
        if pool:
            env = MultiRaceEnv(
                count,
                cfg,
                pool,
                nets,
                track_path,
                device,
                1,
                True,
                full_race=official,
                fixed_opponent=opponent,
            )
        else:
            env = RaceEnv(count, cfg, track_path, device, 1)
        try:
            batch = jax.jit(
                lambda key: episode_batch(env, policy, params, key, count, cfg.max_episode_steps)
            )
            for seed in seeds:
                for offset in range(0, episodes, count):
                    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), offset // count)
                    result = jax.tree.map(np.asarray, batch(key))
                    for i in range(min(count, episodes - offset)):
                        if not result["seen"][i]:
                            raise RuntimeError("Episode failed to reach a terminal/time limit")
                        rows.append(
                            dict(
                                seed=int(seed),
                                episode=offset + i,
                                opponent=opponent,
                                opponent_solo=bool(pool.entries[opponent]["solo"])
                                if pool
                                else True,
                                **{
                                    k: float(v[i])
                                    for k, v in result.items()
                                    if k not in {"seen", "opponent_solo"}
                                },
                            )
                        )
        finally:
            env.close()
    rate = float(np.mean([r["success"] for r in rows]))
    times = [r["lap_time"] for r in rows if r["success"] > 0]
    report = dict(
        stage=stage,
        report_name=report_name,
        episodes=len(rows),
        seeds=list(seeds),
        rules="official_full_race"
        if stage == "multi" and official
        else "first_event_training"
        if pool
        else "solo",
        success_rate=rate,
        completion_rate=rate,
        mean_success_time=float(np.mean(times)) if times else None,
        mean_gates=float(np.mean([r["gates"] for r in rows])),
        crash_rate=float(np.mean([r["crashed"] for r in rows])),
        checkpoint=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    report["leaderboard_score"] = ranking_score(rate, report["mean_success_time"])
    report["balanced_score"] = ranking_score(rate, report["mean_success_time"], 0.8)
    if pool:
        for key in ["win", "loss", "draw", "collision"]:
            report[f"{key}_rate"] = float(np.mean([r[key] for r in rows]))
        for solo, label in [(True, "frozen_solo"), (False, "historical")]:
            group = [r for r in rows if (r["opponent"] == 0) == solo]
            report[f"win_rate_vs_{label}"] = (
                float(np.mean([r["win"] for r in group])) if group else None
            )
    checkpoint.write_json(run / f"{report_name}_evaluation.json", report)
    checkpoint.write_json(run / f"{report_name}_episodes.json", rows)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    fire.Fire(evaluate)
