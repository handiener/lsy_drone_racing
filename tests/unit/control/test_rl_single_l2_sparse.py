"""Sparse-reward (Pasumarti et al. 2026) and paper-observation contracts."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    ActionConfig,
    decode_action,
    rotation_increment,
)
from lsy_drone_racing.control.rl_single_l2.reward import (
    SparseRewardConfig,
    compute_sparse_reward,
    reward_config_from_dict,
)

DT = 0.02


def observation() -> dict:
    return {
        "pos": np.zeros(3),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "n_gates_passed": np.array(0),
        "gate_sequence": np.array([0, 1, 0]),
        "gate_sequence_direction": np.array([1, 1, -1]),
        "gates_pos": np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        "gates_quat": np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        "gates_visited": np.array([False, False]),
        "obstacles_pos": np.array([[3.0, 0.0, 1.0]]),
        "obstacles_visited": np.array([False]),
    }


def batch(count: int) -> dict:
    return {key: np.repeat(value[None], count, axis=0) for key, value in observation().items()}


def sparse(
    before: dict,
    after: dict,
    terminated: Any,
    truncated: Any = None,
    delta: Any = None,
    leading: Any = None,
    **config: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    count = np.asarray(before["pos"]).reshape(-1, 3).shape[0]
    truncated = np.zeros(count, bool) if truncated is None else truncated
    delta = np.zeros((count, 3)) if delta is None else delta
    return compute_sparse_reward(
        before,
        after,
        np.asarray(terminated),
        np.asarray(truncated),
        delta,
        DT,
        leading,
        SparseRewardConfig(**config),
    )


def test_gate_pass_pays_leading_or_trailing_value():
    before = batch(2)
    after = {key: value.copy() for key, value in before.items()}
    after["n_gates_passed"] = np.array([1, 1])
    reward, terms = sparse(before, after, [False, False], leading=np.array([True, False]))
    np.testing.assert_array_equal(terms["gate_pass"], [10, 5])
    np.testing.assert_array_equal(terms["lap"], 0)
    np.testing.assert_array_equal(reward, [10, 5])
    _, terms = sparse(before, after, [False, False], single_agent_leading=False)
    np.testing.assert_array_equal(terms["gate_pass"], [5, 5])
    _, terms = sparse(before, after, [False, False])
    np.testing.assert_array_equal(terms["gate_pass"], [10, 10])


def test_final_pass_pays_gate_and_lap_and_crash_pays_nothing_but_penalty():
    before = batch(4)
    before["n_gates_passed"] = np.array([2, 2, 1, 1])
    after = {key: value.copy() for key, value in before.items()}
    after["n_gates_passed"] = np.array([3, 3, 2, 1])
    terminated = np.array([True, True, True, False])
    truncated = np.array([False, False, False, True])
    leading = np.array([True, False, True, True])
    reward, terms = sparse(before, after, terminated, truncated, leading=leading)
    np.testing.assert_array_equal(terms["gate_pass"], [10, 5, 0, 0])
    np.testing.assert_array_equal(terms["lap"], [50, 0, 0, 0])
    # Crash (row 2) gets -2 and no gate reward even if the counter advanced;
    # a timeout (row 3) and finishes get no crash penalty.
    np.testing.assert_array_equal(terms["crash"], [0, 0, -2, 0])
    np.testing.assert_array_equal(reward, [60, 5, -2, 0])
    assert set(terms) == {"gate_pass", "lap", "crash", "cmd"}


def test_command_penalty_uses_implied_rate_and_dt_flag():
    obs = observation()
    delta = np.array([0.02, -0.04, 0.06])  # rad per step -> (1, -2, 3) rad/s
    rate_cost = 0.15 * (1 + 4) + 0.05 * 9
    _, terms = sparse(obs, obs, False, False, delta=delta)
    np.testing.assert_allclose(terms["cmd"], [-rate_cost * DT], rtol=1e-6)
    reward, terms = sparse(obs, obs, False, False, delta=delta, cmd_scale_by_dt=False)
    np.testing.assert_allclose(terms["cmd"], [-rate_cost], rtol=1e-6)
    np.testing.assert_allclose(reward, [-rate_cost], rtol=1e-6)
    _, terms = sparse(obs, obs, False, False, delta=delta, cmd_roll_pitch=0.0, cmd_yaw=0.0)
    assert terms["cmd"][0] == 0


def test_single_and_batch_shapes_and_jit_compatibility():
    obs = observation()
    reward, terms = sparse(obs, obs, False, False)
    assert reward.shape == (1,) and all(value.shape == (1,) for value in terms.values())
    before = batch(7)
    reward, terms = sparse(before, before, np.zeros(7, bool))
    assert reward.shape == (7,) and all(value.shape == (7,) for value in terms.values())
    assert reward.dtype == jnp.float32


def test_rotation_increment_matches_decode_action():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(16, 4)).astype(np.float32) * 3
    config = ActionConfig(alpha_max_rad=0.16)
    delta = np.asarray(rotation_increment(raw, config))
    assert np.all(np.linalg.norm(delta, axis=1) <= 0.16 + 1e-6)
    np.testing.assert_allclose(rotation_increment(raw[0], config), delta[0])
    from scipy.spatial.transform import Rotation

    commands = decode_action(raw, np.tile([0, 0, 0, 1.0], (16, 1)), 0.0, 0.6, config)
    expected = Rotation.from_rotvec(delta).as_euler("xyz")
    np.testing.assert_allclose(commands[:, :3], expected, atol=2e-6)


def test_reward_config_round_trip():
    config = SparseRewardConfig(crash=3.0, single_agent_leading=False)
    from dataclasses import asdict

    assert reward_config_from_dict("sparse", asdict(config)) == config
    with pytest.raises(ValueError):
        reward_config_from_dict("other", {})
