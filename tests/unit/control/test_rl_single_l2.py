"""Contracts shared by training and competition inference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    ObservationConfig,
    RunningNormalizer,
    compute_reward,
    decode_action,
    encode_obs,
)
from lsy_drone_racing.control.rl_single_l2.policy import LOG_STD_MAX, LOG_STD_MIN


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


def test_sensor_features_repeated_reverse_gate_and_padding():
    obs = observation()
    obs["n_gates_passed"] = np.array(2)
    single = encode_obs(obs)
    batch = encode_obs({key: value[None] for key, value in obs.items()})
    assert single.shape == (ObservationConfig().size,)
    np.testing.assert_array_equal(single, batch[0])
    np.testing.assert_allclose(single[23:29], [1, 0, 1, -1, 0, 0])
    np.testing.assert_array_equal(single[31:39], 0)  # no next target after repeated gate
    np.testing.assert_array_equal(single[-15:], 0)  # padded obstacle slots
    obs["n_gates_passed"] = np.array(3)
    assert np.all(encode_obs(obs)[23:39] == 0)


def test_body_frame_rotation_and_bounded_action_mapping():
    obs = observation()
    obs["quat"] = Rotation.from_euler("z", np.pi / 2).as_quat()
    obs["vel"] = np.array([0.0, 1.0, 0.0])
    features = encode_obs(obs)
    np.testing.assert_allclose(features[12:15], [1, 0, 0], atol=1e-6)
    commands = decode_action(
        np.array([[1e3, 1e3, -1e3, 1e3], [-1e3, 0, 0, 0]]), np.tile([0, 0, 0, 1], (2, 1)), 0.0, 0.6
    )
    assert np.isfinite(commands).all()
    assert np.all(np.abs(commands[:, :3]) <= np.pi / 2 + 1e-6)
    np.testing.assert_allclose(commands[:, 3], [0.6, 0], atol=1e-6)
    # With an identity starting attitude, a saturated raw rotation reaches the
    # reference increment limit before the official Euler-angle bounds are active.
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", commands[0, :3]).magnitude(), 0.16, atol=1e-6
    )


def test_reward_target_changes_reveals_and_termination():
    before = observation()
    after = {key: value.copy() for key, value in before.items()}
    after["pos"][0] = 0.2
    after["gates_pos"][0, 0] = 10  # sensor update must not create distance progress
    after["n_gates_passed"] = np.array(1)
    _, terms = compute_reward(before, after, np.array([False]), np.array([False]))
    expected = 2 * (np.sqrt(2) - np.linalg.norm([0.8, 0, 1]))
    np.testing.assert_allclose(terms["progress"], [expected], atol=1e-6)
    _, terms = compute_reward(before, after, np.array([True]), np.array([False]))
    assert terms["crash"][0] == -10 and terms["progress"][0] == 0
    _, terms = compute_reward(before, after, np.array([False]), np.array([True]))
    assert terms["crash"][0] == 0
    after["n_gates_passed"] = np.array(3)
    _, terms = compute_reward(before, after, np.array([True]), np.array([False]))
    assert terms["completion"][0] == 20 and terms["crash"][0] == 0


def test_action_mapping_preserves_half_turn_signs_and_gimbal_lock():
    # Equivalent +/-pi endpoints become different commands after +/-pi/2 clipping.
    angles = np.array(
        [
            [0, 0, np.pi],
            [np.pi, 0, 0],
            [0, 0, -np.pi],
            [-np.pi, 0, 0],
            [0.4, np.pi / 2, 0.7],
            [0.4, -np.pi / 2, 0.7],
        ]
    )
    quats = Rotation.from_euler("xyz", angles).as_quat()
    quats = np.concatenate([quats, -quats])
    with pytest.warns(UserWarning, match="Gimbal lock"):
        expected = Rotation.from_quat(quats).as_euler("xyz")
    commands = decode_action(np.zeros((len(quats), 4)), quats, 0.0, 0.6)
    np.testing.assert_allclose(commands[:, :3], np.clip(expected, -np.pi / 2, np.pi / 2), atol=1e-6)


def test_normalizer_merges_raw_moments_and_rejects_corruption():
    values = np.random.default_rng(4).normal(size=(30, 3))
    normalizer = RunningNormalizer(3)
    normalizer.update(values[:11])
    normalizer.update(values[11:])
    # Device-local moments use float32, including when the input is NumPy float64.
    np.testing.assert_allclose(normalizer.mean, values.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(normalizer.var, values.var(axis=0), atol=1e-6)
    restored = RunningNormalizer.from_state_dict(normalizer.state_dict())
    np.testing.assert_array_equal(normalizer.normalize(values), restored.normalize(values))
    state = normalizer.state_dict() | {"var": [-1, 1, 1]}
    with pytest.raises(ValueError):
        normalizer.load_state_dict(state)


def test_jax_batch_geometry_matches_scipy_and_compiles_together():
    rng = np.random.default_rng(18)
    count = 64
    obs = {key: np.repeat(value[None], count, axis=0) for key, value in observation().items()}
    rotations = Rotation.random(count, random_state=rng)
    obs["quat"] = rotations.as_quat().astype(np.float32)
    obs["vel"] = rng.normal(size=(count, 3)).astype(np.float32)
    raw = rng.normal(size=(count, 4)).astype(np.float32)
    after = obs | {"pos": np.tile([0.2, 0.0, 0.0], (count, 1)).astype(np.float32)}
    done = jnp.zeros(count, dtype=bool)

    @jax.jit
    def process(
        before: dict[str, jax.Array], current: dict[str, jax.Array], actions: jax.Array
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, dict[str, jax.Array]]]:
        return (
            encode_obs(before),
            decode_action(actions, before["quat"], 0.0, 0.6),
            compute_reward(before, current, done, done),
        )

    features, commands, (reward, components) = process(
        jax.tree.map(jnp.asarray, obs), jax.tree.map(jnp.asarray, after), jnp.asarray(raw)
    )
    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree.leaves((features, commands, reward))
    )
    np.testing.assert_allclose(
        features[:, 3:12], rotations.as_matrix().reshape(count, 9), atol=1e-6
    )
    np.testing.assert_allclose(features[:, 12:15], rotations.inv().apply(obs["vel"]), atol=1e-6)
    centers = obs["gates_pos"][:, 0] - obs["pos"]
    np.testing.assert_allclose(features[:, 23:26], rotations.inv().apply(centers), atol=1e-6)
    np.testing.assert_allclose(
        features[:, 26:29], rotations.inv().apply(np.tile([1, 0, 0], (count, 1))), atol=1e-6
    )
    magnitude = np.linalg.norm(raw[:, 1:], axis=1, keepdims=True)
    delta = raw[:, 1:] * (np.tanh(magnitude) / np.maximum(magnitude, 1e-12)) * 0.16
    expected_angles = (rotations * Rotation.from_rotvec(delta)).as_euler("xyz")
    np.testing.assert_allclose(
        commands[:, :3], np.clip(expected_angles, -np.pi / 2, np.pi / 2), atol=2e-6
    )
    np.testing.assert_allclose(commands[:, 3], 0.6 / (1 + np.exp(-raw[:, 0])), atol=1e-6)
    np.testing.assert_allclose(reward, sum(components.values()), atol=1e-6)


def test_gaussian_heads_depend_on_state_and_have_finite_gradients():
    from lsy_drone_racing.control.rl_single_l2.policy import Actor

    actor = Actor()
    obs = jnp.stack([jnp.zeros(59), jnp.ones(59)])
    params = actor.init(jax.random.PRNGKey(0), obs)
    dist = actor.apply(params, obs)
    assert dist.mean().shape == (2, 4)
    assert np.all(np.abs(dist.mean()) <= 1)
    assert not np.allclose(dist.stddev()[0], dist.stddev()[1])
    assert np.all(dist.stddev() >= np.exp(LOG_STD_MIN))
    assert np.all(dist.stddev() <= np.exp(LOG_STD_MAX))
    grad = jax.grad(lambda p: -actor.apply(p, obs).log_prob(jnp.ones((2, 4))).mean())(params)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(grad))
    assert np.any(np.asarray(grad["params"]["log_std_head"]["kernel"]) != 0)
