"""Small real-Crazyflow recurrent train/export/official-controller smoke tests.

Thresholds are zero solely to exercise stage transitions, not demonstrate skill.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.rl_race_multi import checkpoint
from lsy_drone_racing.control.rl_race_multi.controller import network_from_metadata
from lsy_drone_racing.control.rl_race_multi.env_wrapper.multi import MultiRaceEnv
from lsy_drone_racing.control.rl_race_multi.env_wrapper.observations import ObsConfig, encode
from lsy_drone_racing.control.rl_race_multi.evaluation import config_from_record
from lsy_drone_racing.control.rl_race_multi.export import export
from lsy_drone_racing.control.rl_race_multi.league import League
from lsy_drone_racing.control.rl_race_multi.network import zeros
from lsy_drone_racing.control.rl_race_multi.train import RL_DIR, train
from lsy_drone_racing.utils import load_config, load_controller


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("recurrent_e2e")
    run = train(config_file=str(RL_DIR / "configs/smoke.yaml"), output_root=str(root), device="cpu")
    return Path(run)


@pytest.mark.integration
def test_real_three_stage_smoke_and_league_snapshots(trained):
    stages = json.loads((trained / "stages.json").read_text())
    assert [s["stage"] for s in stages] == ["pretrain", "race", "multi"]
    assert stages[0]["switched_on_success"] and stages[1]["switched_on_success"]
    solo = checkpoint.load(trained / "checkpoints/race_final.pkl")
    multi = checkpoint.load(trained / "checkpoints/multi_final.pkl")
    assert solo["network"] == multi["network"]
    assert len(multi["league"]["entries"]) >= 2
    assert multi["resume"].startswith("Weights/normalizer restored")
    assert "optimizer" not in multi
    report = json.loads((trained / "final_evaluation.json").read_text())
    assert report["stage"] == "multi" and report["rules"] == "official_full_race"
    assert report["episodes"] >= 2
    assert "win_rate_vs_frozen_solo" in report and "win_rate_vs_historical" in report
    # Frozen solo opponent remains byte-identical after league PPO and snapshots.
    assert multi["league"]["entries"][0]["checkpoint"] == str(
        trained / "checkpoints/race_final.pkl"
    )


@pytest.mark.integration
def test_export_recurrent_controller_matches_actor_and_100hz_holds(trained):
    destination = Path(export(str(trained), stage="multi"))
    model = checkpoint.load(trained / "checkpoints/multi_final.pkl")
    nets = network_from_metadata(model["network"]).factory()(57, 4)
    cfg = load_config(trained / "multi_track.toml")
    cfg.controller = [dict(file="rl_race_multi_controller.py", model_dir=str(destination))] * 2
    cfg.env.freq = 100
    env_cfg = config_from_record(model["env"])
    pool = League(
        str(trained / "checkpoints/race_final.pkl"),
        model["network"],
        model["env"]["action"],
        trained / "test_pool",
        capacity=3,
    )
    env = MultiRaceEnv(
        2, env_cfg, pool, nets, str(trained / "multi_track.toml"), device="cpu", swap_starts=False
    )
    try:
        state = env.reset(jax.random.split(jax.random.PRNGKey(1), 2))
        from lsy_drone_racing.envs.race_core import obs as raw_obs

        raw = {k: np.asarray(v[0]) for k, v in raw_obs(state.pipeline_state).items()}
        ctrl_class = load_controller(
            Path(__file__).parents[2] / "lsy_drone_racing/control/rl_race_multi_controller.py"
        )
        controller = ctrl_class(raw, dict(rank=1), cfg)
        command = controller.compute_control(raw, dict(rank=1))
        hidden = jax.tree.map(np.asarray, controller._carry)
        held = controller.compute_control(raw, dict(rank=1))
        np.testing.assert_allclose(command, held)
        for a, b in zip(jax.tree.leaves(hidden), jax.tree.leaves(controller._carry)):
            np.testing.assert_array_equal(a, b)
        ego = {k: jnp.asarray(v[1])[None] for k, v in raw.items()}
        opp = {k: jnp.asarray(v[0])[None] for k, v in raw.items()}
        features = encode(ego, jnp.zeros((1, 4)), ObsConfig(), opp, jnp.array([True]))
        expected_carry, logits = nets.policy_network.apply(
            model["params"][0], model["params"][1], features, zeros((1,), 8)
        )
        from lsy_drone_racing.control.rl_race_multi.env_wrapper.actions import (
            ActionConfig,
            to_command,
        )

        expected = to_command(jnp.tanh(logits[..., :4]), ActionConfig(**model["env"]["action"]))
        np.testing.assert_allclose(command, expected[0], atol=1e-6)
        for a, b in zip(jax.tree.leaves(expected_carry), jax.tree.leaves(hidden)):
            np.testing.assert_allclose(a, b, atol=1e-6)
        controller.episode_reset()
        for leaf in jax.tree.leaves(controller._carry):
            np.testing.assert_array_equal(leaf, 0)
        # Joint action reaches two drone states in ONE simulator world.
        previous = np.asarray(state.pipeline_state.sim_data.states.pos)
        next_state = env.step(state, jnp.zeros((2, 4)))
        assert next_state.pipeline_state.sim_data.states.pos.shape == (2, 2, 3)
        assert np.isfinite(next_state.obs).all()
        assert np.any(previous != np.asarray(next_state.pipeline_state.sim_data.states.pos))
    finally:
        env.close()


@pytest.mark.integration
def test_real_collision_and_training_vs_official_world_termination(trained):
    from lsy_drone_racing.control.rl_race_multi.network import make_inference_fn
    from lsy_drone_racing.control.rl_race_multi.recurrent import generate_unroll

    model = checkpoint.load(trained / "checkpoints/multi_final.pkl")
    cfg = config_from_record(model["env"])
    nets = network_from_metadata(model["network"]).factory()(57, 4)
    pool = League(
        str(trained / "checkpoints/race_final.pkl"),
        model["network"],
        model["env"]["action"],
        trained / "collision_pool",
        capacity=3,
    )
    frozen_bytes = (trained / "checkpoints/race_final.pkl").read_bytes()

    def relocate(state, positions):
        data = state.pipeline_state
        states = data.sim_data.states.replace(
            pos=jnp.broadcast_to(jnp.array(positions), (2, 2, 3)), vel=jnp.zeros((2, 2, 3))
        )
        data = data.replace(
            sim_data=data.sim_data.replace(
                states=states, core=data.sim_data.core.replace(mjx_synced=False)
            )
        )
        return state.replace(pipeline_state=data)

    for official in (False, True):
        env = MultiRaceEnv(
            2,
            cfg,
            pool,
            nets,
            str(trained / "multi_track.toml"),
            device="cpu",
            swap_starts=False,
            full_race=official,
            fixed_opponent=0,
        )
        try:
            state = env.reset(jax.random.split(jax.random.PRNGKey(4), 2))
            # An opponent alone exceeds upper bounds. This is not a learner victory.
            state = relocate(state, [[0.0, 0.0, 0.8], [4.0, 0.0, 0.8]])
            result = env.step(state, jnp.zeros((2, 4)))
            np.testing.assert_array_equal(result.metrics["opponent_crashed"], 1)
            np.testing.assert_array_equal(result.metrics["win"], 0)
            np.testing.assert_array_equal(result.done, 0 if official else 1)
            for leaf in jax.tree.leaves(result.info["opponent_carry"]):
                np.testing.assert_array_equal(leaf, 0)
            if official:
                np.testing.assert_array_equal(result.obs[:, -1], 0)
                terminal = env.step(result, jnp.zeros((2, 4)))
                np.testing.assert_array_equal(terminal.done, 1)
                np.testing.assert_array_equal(terminal.metrics["draw"], 1)
            else:
                # Real inter-agent contact: both drones at the same airborne position.
                state = env.reset(jax.random.split(jax.random.PRNGKey(6), 2))
                state = relocate(state, [[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]])
                nonzero = jax.tree.map(lambda x: jnp.ones_like(x), state.info["recurrent"])
                state = state.replace(info=dict(state.info, recurrent=nonzero))
                policy = make_inference_fn(nets)(model["params"])
                after, data = generate_unroll(
                    env, state, policy, jax.random.PRNGKey(7), 1, ("truncation",)
                )
                np.testing.assert_array_equal(after.metrics["collision"], 1)
                np.testing.assert_array_equal(after.metrics["crashed"], 1)
                np.testing.assert_array_equal(after.metrics["opponent_crashed"], 1)
                np.testing.assert_array_equal(after.metrics["draw"], 1)
                np.testing.assert_array_equal(data.discount, 0)
                for leaf in jax.tree.leaves(after.info["recurrent"]):
                    np.testing.assert_array_equal(leaf, 0)
                np.testing.assert_allclose(
                    after.metrics["r_crash"] + after.metrics["r_inter_collision"],
                    -cfg.reward.inter_collision,
                )
                assert np.isfinite(after.obs).all()
        finally:
            env.close()
    assert (trained / "checkpoints/race_final.pkl").read_bytes() == frozen_bytes
