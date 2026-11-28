"""CPU smoke: independent dynamic-reference LSTM solo followed by a mixed frozen league."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from lsy_drone_racing.control.rl_race.network import NetworkConfig as SingleNetwork
from lsy_drone_racing.control.rl_race_multi import checkpoint
from lsy_drone_racing.control.rl_race_multi.network import NetworkConfig
from lsy_drone_racing.control.rl_race_multi.train import RL_DIR, train


@pytest.mark.integration
def test_dynamic_solo_then_multi_with_mlp_and_larger_lstm(tmp_path):
    config = str(RL_DIR / "configs/smoke.yaml")
    dense = Path(
        train(
            config_file=config,
            output_root=str(tmp_path),
            run_name="dense",
            device="cpu",
            reference_path="dynamic",
            end_stage="pretrain",
            thrust_min_fraction=0.1,
            thrust_max_fraction=0.9,
        )
    )
    assert [s["stage"] for s in json.loads((dense / "stages.json").read_text())] == ["pretrain"]
    dense_checkpoint = dense / "checkpoints/pretrain_final.pkl"
    dense_bytes = dense_checkpoint.read_bytes()
    base = Path(
        train(
            config_file=config,
            output_root=str(tmp_path),
            run_name="sparse",
            device="cpu",
            reference_path="dynamic",
            start_stage="race",
            thrust_min_fraction=0.1,
            thrust_max_fraction=0.9,
            start_params=str(dense_checkpoint),
            end_stage="race",
            stop_on_success=False,
            race_steps=8,
        )
    )
    stages = json.loads((base / "stages.json").read_text())
    assert [s["stage"] for s in stages] == ["race"]
    assert stages[-1]["reason"] == "budget_complete" and stages[-1]["steps"] == 8
    assert not (base / "league").exists()
    assert dense_checkpoint.read_bytes() == dense_bytes
    solo_model = checkpoint.load(base / "checkpoints/race_final.pkl")
    assert solo_model["env"]["reference_path"] == "dynamic"
    solo_bytes = (base / "checkpoints/race_final.pkl").read_bytes()
    weights = tmp_path / "manual_weights"
    weights.mkdir()
    # A real single-racer MLP pickle plus its original metadata, with different tilt limits.
    from lsy_drone_racing.control.rl_race.env_wrapper.env import EnvConfig

    env = asdict(EnvConfig())
    env["action"].update(roll_limit_deg=85, pitch_limit_deg=85)
    for name, single in [
        ("mlp_small", SingleNetwork((8,), (8,))),
        ("mlp_deep", SingleNetwork((12, 8), (12,))),
    ]:
        nets = single.factory()(49, 4)
        params = (
            running_statistics.init_state(specs.Array((49,), jnp.float32)),
            nets.policy_network.init(jax.random.PRNGKey(3)),
            nets.value_network.init(jax.random.PRNGKey(4)),
        )
        (weights / f"{name}.pkl").write_bytes(pickle.dumps(jax.device_get(params)))
        (weights / f"{name}.json").write_text(
            json.dumps(
                dict(
                    env=env,
                    network=single.metadata(),
                    freq=50,
                    stage="race",
                    thrust_min=0.08545,
                    thrust_max=0.8,
                )
            )
        )
    larger = NetworkConfig((16,), (16,), hidden_size=12)
    nets = larger.factory()(57, 4)
    params = (
        solo_model["params"][0],
        nets.policy_network.init(jax.random.PRNGKey(5)),
        nets.value_network.init(jax.random.PRNGKey(6)),
    )
    frozen_env = dict(solo_model["env"], hidden_size=12)
    checkpoint.save(
        params,
        weights / "larger.pkl",
        network=larger.metadata(),
        env=frozen_env,
        stage="multi",
        steps=0,
    )
    frozen = {p.name: p.read_bytes() for p in weights.glob("*.pkl")}
    result = Path(
        train(
            config_file=config,
            output_root=str(tmp_path),
            run_name="multi",
            device="cpu",
            start_stage="multi",
            start_params=str(base / "checkpoints/race_final.pkl"),
            solo_checkpoint=str(base / "checkpoints/race_final.pkl"),
            league_weights=str(weights),
            league_capacity=5,
            multi_steps=4,
            reference_path="dynamic",
            thrust_min_fraction=0.1,
            thrust_max_fraction=0.9,
        )
    )
    report = json.loads((result / "final_evaluation.json").read_text())
    assert report["stage"] == "multi" and report["episodes"] == 10
    model = checkpoint.load(result / "checkpoints/multi_final.pkl")
    entries = model["league"]["entries"]
    assert len(entries) == 5 and sum(e["fixed"] for e in entries) == 4
    assert {p.name: p.read_bytes() for p in weights.glob("*.pkl")} == frozen
    assert (base / "checkpoints/race_final.pkl").read_bytes() == solo_bytes
    assert "win_rate_vs_fixed" in report and "win_rate_vs_historical" in report

    from lsy_drone_racing.control.rl_race_multi.env_wrapper.multi import MultiRaceEnv
    from lsy_drone_racing.control.rl_race_multi.evaluation import config_from_record
    from lsy_drone_racing.control.rl_race_multi.league import League
    from lsy_drone_racing.control.rl_race_multi.network import network_from_metadata

    # Check the two-agent policy's actual wrapper observations and episode-local memory.
    pool = League(
        str(base / "checkpoints/race_final.pkl"),
        model["network"],
        model["env"]["action"],
        tmp_path / "routing_pool",
        capacity=5,
        manifest=str(result / "league/manifest.json"),
    )
    cfg = config_from_record(model["env"])
    nets = network_from_metadata(model["network"]).factory()(57, 4)
    env = MultiRaceEnv(2, cfg, pool, nets, result / "multi_track.toml", device="cpu")
    try:
        state = env.reset(jax.random.split(jax.random.PRNGKey(6), 2))
        previous = state.info["previous_action"]
        _, obs = env._features(state.pipeline_state, previous, jnp.array([True, False]))
        np.testing.assert_array_equal(obs[0, 49:], 0)
        assert obs[1, -1] == 1
        np.testing.assert_allclose(obs[:, :3], state.pipeline_state.sim_data.states.pos[:, 1])
        disabled = state.pipeline_state.replace(
            disabled_drones=state.pipeline_state.disabled_drones.at[:, 0].set(True)
        )
        _, absent = env._features(disabled, previous, jnp.array([False, False]))
        np.testing.assert_array_equal(absent[:, 49:], 0)
        multi_slot = next(i for i, e in enumerate(pool.entries) if e["fixed"] and not e["solo"])
        index = jnp.full((2,), multi_slot, jnp.int32)
        data = state.pipeline_state.replace(steps=jnp.array([cfg.max_episode_steps - 1, 0]))
        _, opponent_obs = env._features(data, previous, jnp.array([False, False]))
        carry = jax.tree.map(jnp.ones_like, state.info["opponent_carry"])
        state = state.replace(
            pipeline_state=data,
            info=dict(
                state.info, opponent_index=index, opponent_obs=opponent_obs, opponent_carry=carry
            ),
        )
        expected, _, _ = pool.predict(
            state.info["league"], index, opponent_obs, carry, jax.random.PRNGKey(0)
        )
        after = jax.jit(env.step)(state, jnp.zeros((2, 4)))
        np.testing.assert_array_equal(after.done, [1, 0])
        assert after.info["opponent_index"][1] == multi_slot
        for actual, predicted in zip(after.info["opponent_carry"], expected):
            np.testing.assert_array_equal(actual[0], 0)
            np.testing.assert_allclose(actual[1], predicted[1], atol=1e-6)
    finally:
        env.close()
