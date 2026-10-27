"""Real simulator/PPO/checkpoint and competition-loader integration."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation
from sbx import PPO
from stable_baselines3.common.logger import configure

from lsy_drone_racing.control.rl_single_l2.callbacks import TrainingCallback, evaluate_policy
from lsy_drone_racing.control.rl_single_l2.checkpoint import export_actor, load_checkpoint
from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    Level2SBXEnv,
    decode_action,
    encode_obs,
)
from lsy_drone_racing.control.rl_single_l2.policy import Level2Policy
from lsy_drone_racing.control.rl_single_l2.reset_mid_start import ResetConfig
from lsy_drone_racing.control.rl_single_l2.reward import compute_reward
from lsy_drone_racing.envs.race_core import obs as race_obs
from lsy_drone_racing.utils import load_config, load_controller

CONFIG = Path(__file__).parents[2] / "config/level2.toml"


def assert_tree_equal(left: Any, right: Any):
    a, a_tree = jax.tree.flatten(left)
    b, b_tree = jax.tree.flatten(right)
    assert a_tree == b_tree
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.integration
def test_masked_reset_preserves_other_world_and_timeout_terminal_observation():
    env = Level2SBXEnv(
        CONFIG, n_envs=2, max_episode_steps=3, reset_config=ResetConfig(segment_init_prob=0)
    )
    try:
        env.seed(7)
        env.reset()
        previous = env._current_obs
        assert all(isinstance(value, jax.Array) for value in previous.values())
        env.jax_env.data = env.jax_env.data.replace(steps=jnp.array([2, 0]))
        raw_actions = np.zeros((2, 4), dtype=np.float32)
        commands = decode_action(
            raw_actions, env._current_obs["quat"], env.thrust_min, env.thrust_max
        )
        _, (expected, _, _, _, _) = env.jax_env._step(env.jax_env.data, commands)
        obs, rewards, done, infos = env.step(raw_actions)
        np.testing.assert_array_equal(done, [True, False])
        assert infos[0]["TimeLimit.truncated"] and "terminal_observation" in infos[0]
        expected = {key: value[:, 0] for key, value in expected.items()}
        expected_terminal = env.normalizer.normalize(encode_obs(expected, commands))
        np.testing.assert_allclose(
            infos[0]["terminal_observation"], expected_terminal[0], atol=1e-6
        )
        np.testing.assert_allclose(env._current_obs["pos"][1], np.asarray(expected["pos"])[1])
        np.testing.assert_array_equal(env._previous_action[0], 0)
        np.testing.assert_array_equal(env._previous_action[1], commands[1])
        assert obs.shape == (2, 59) and rewards.flags.writeable
        assert env.jax_env.data.steps.tolist() == [0, 1]
        assert infos[1] == {}  # ongoing worlds need no per-step metric dictionary
        assert isinstance(env._previous_action, jax.Array)
        assert isinstance(env._pending_normalizer.mean, jax.Array)
        assert all(isinstance(value, jax.Array) for value in env._current_obs.values())
        _, terms = compute_reward(previous, expected, jnp.zeros(2, bool), jnp.array([True, False]))
        means, episodes = env.drain_metrics()
        for key, values in terms.items():
            np.testing.assert_allclose(means[key], np.mean(values), atol=1e-6)
        assert len(episodes) == 1 and episodes[0]["timeout"]
        empty_means, empty_episodes = env.drain_metrics()
        assert not empty_episodes and all(value == 0 for value in empty_means.values())
    finally:
        env.close()


@pytest.mark.integration
def test_reward_uses_official_directional_gate_pass_counter():
    env = Level2SBXEnv(CONFIG, n_envs=2, reset_config=ResetConfig(segment_init_prob=0))
    try:
        env.seed(7)
        env.reset()
        data = env.jax_env.data
        centers = data.gates_pos[:, 0]
        normals = Rotation.from_quat(data.gates_quat[:, 0]).as_matrix()[:, :, 0]
        direction = jnp.array([1.0, -1.0])[:, None]
        start = centers - direction * normals * 0.005
        velocity = direction * normals
        states = data.sim_data.states.replace(pos=start[:, None], vel=velocity[:, None])
        data = data.replace(
            sim_data=data.sim_data.replace(states=states),
            last_drone_pos=start[:, None],
            gates_visited=jnp.ones_like(data.gates_visited),
        )
        env.jax_env.data = data
        env._current_obs = {key: value[:, 0] for key, value in race_obs(data).items()}
        _, _, done, _ = env.step(np.zeros((2, 4), dtype=np.float32))
        np.testing.assert_array_equal(done, False)
        np.testing.assert_array_equal(env._current_obs["n_gates_passed"].reshape(-1), [1, 0])
        means, _ = env.drain_metrics()
        assert means["gate_pass"] == 5.0  # +10 for the forward crossing, +0 for the wrong direction
    finally:
        env.close()


@pytest.mark.integration
def test_phase1_pose_noise_sensor_masks_reproducibility_and_masked_reset():
    cfg = ResetConfig(segment_init_prob=1.0)
    env = Level2SBXEnv(CONFIG, n_envs=16, reset_config=cfg)
    try:
        env.seed(7)
        env.reset()
        data = env.jax_env.data
        selected = env._episode_is_segment.copy()
        assert selected.any()
        targets = env._episode_start_gate.copy()
        rows = np.arange(env.num_envs)
        gate_ids = np.asarray(data.gate_sequence)[targets]
        centers = np.asarray(data.nominal_gates_pos)[rows, gate_ids]
        normals = np.asarray(
            Rotation.from_quat(data.nominal_gates_quat[rows, gate_ids]).as_matrix()
        )[:, :, 0]
        normals = normals * np.asarray(data.gate_sequence_direction)[targets, None]
        positions = np.asarray(data.sim_data.states.pos)[:, 0]
        noise = positions - (centers - cfg.segment_init_offset_m * normals)
        assert np.max(np.abs(noise[selected])) <= cfg.segment_init_perturb_m + 1e-6
        assert np.std(noise[selected]) > 0.01
        np.testing.assert_allclose(
            np.asarray(data.sim_data.states.vel)[selected, 0],
            cfg.segment_init_vel_mps * normals[selected],
            atol=1e-6,
        )
        np.testing.assert_array_equal(data.sim_data.states.ang_vel, 0)
        angles = np.asarray(Rotation.from_quat(data.sim_data.states.quat[:, 0]).as_euler("xyz"))
        np.testing.assert_allclose(angles[selected, :2], 0, atol=1e-6)
        assert np.max(np.abs(angles[selected, 2])) <= np.pi / 2 + 1e-6
        np.testing.assert_array_equal(data.last_drone_pos, data.sim_data.states.pos)
        np.testing.assert_array_equal(data.takeoff_pos, data.sim_data.states.pos)
        distance = np.linalg.norm(
            positions[:, None, :2] - np.asarray(data.gates_pos)[:, :, :2], axis=-1
        )
        np.testing.assert_array_equal(np.asarray(data.gates_visited)[:, 0], distance < 0.7)
        distance = np.linalg.norm(
            positions[:, None, :2] - np.asarray(data.obstacles_pos)[:, :, :2], axis=-1
        )
        np.testing.assert_array_equal(np.asarray(data.obstacles_visited)[:, 0], distance < 0.7)
        assert not np.asarray(env.jax_env.build_contact_check_fn()(data))[selected].any()
        before = data.sim_data.states
        env.seed(7)
        env.reset()
        assert_tree_equal(before, env.jax_env.data.sim_data.states)
        np.testing.assert_array_equal(selected, env._episode_is_segment)
        # Only one world resets; all other physical states stay exactly unchanged.
        before = env.jax_env.data.sim_data.states
        mask = np.zeros(16, dtype=bool)
        mask[0] = True
        env._reset_worlds(mask=mask)
        for old, new in zip(
            jax.tree.leaves(before), jax.tree.leaves(env.jax_env.data.sim_data.states), strict=True
        ):
            np.testing.assert_array_equal(old[1:], new[1:])
    finally:
        env.close()


@pytest.mark.integration
def test_phase1_respects_reverse_sequence_and_rejects_unsafe_candidates():
    env = Level2SBXEnv(CONFIG, n_envs=16, reset_config=ResetConfig(segment_init_prob=1.0))
    try:
        env.seed(13)
        env.reset()
        data = env.jax_env.data.replace(
            gate_sequence=jnp.array([0, 0]), gate_sequence_direction=jnp.array([1, -1])
        )
        spawned, accepted, targets = env._segment_reset(data, jnp.ones(16, dtype=bool))
        accepted, targets = np.asarray(accepted), np.asarray(targets)
        assert set(targets[accepted]) == {0, 1}
        local = Rotation.from_quat(spawned.nominal_gates_quat[:, 0]).apply(
            spawned.sim_data.states.vel[:, 0], inverse=True
        )
        expected = np.where(targets == 0, 2.5, -2.5)
        np.testing.assert_allclose(np.asarray(local)[accepted, 0], expected[accepted], atol=1e-6)
        np.testing.assert_allclose(np.asarray(local)[accepted, 1:], 0, atol=1e-6)
        # A nominal proposal outside the world must leave the official reset intact.
        invalid = data.replace(
            nominal_gates_pos=data.nominal_gates_pos + jnp.array([100.0, 0.0, 0.0])
        )
        result, accepted, _ = env._segment_reset(invalid, jnp.ones(16, dtype=bool))
        assert not np.asarray(accepted).any()
        assert_tree_equal(result.sim_data.states, invalid.sim_data.states)
    finally:
        env.close()


@pytest.mark.integration
def test_evaluation_rejects_phase1_starts():
    env = Level2SBXEnv(CONFIG, n_envs=1, reset_config=ResetConfig(segment_init_prob=0.4))
    try:
        with pytest.raises(ValueError, match="official starts"):
            evaluate_policy(None, env, episodes=1, seed=0)
    finally:
        env.close()


@pytest.mark.integration
def test_phase1_suffix_completion_is_not_a_full_race_success():
    env = Level2SBXEnv(CONFIG, n_envs=2, reset_config=ResetConfig(segment_init_prob=1.0))
    try:
        env.seed(7)
        env.reset()
        data = env.jax_env.data
        last = data.gate_sequence.shape[0] - 1
        gate_id = data.gate_sequence[last]
        centers = data.gates_pos[:, gate_id]
        normals = Rotation.from_quat(data.gates_quat[:, gate_id]).as_matrix()[:, :, 0]
        normals *= data.gate_sequence_direction[last]
        positions = centers - 0.005 * normals
        states = data.sim_data.states.replace(pos=positions[:, None], vel=normals[:, None])
        env.jax_env.data = data.replace(
            sim_data=data.sim_data.replace(states=states),
            last_drone_pos=positions[:, None],
            n_gates_passed=jnp.full_like(data.n_gates_passed, last),
            gates_visited=jnp.ones_like(data.gates_visited),
        )
        env._current_obs = {key: value[:, 0] for key, value in race_obs(env.jax_env.data).items()}
        env._episode_start_gate.fill(last)
        env._episode_is_segment.fill(True)
        _, _, done, infos = env.step(np.zeros((2, 4), dtype=np.float32))
        assert done.all()
        for info in infos:
            assert info["finished"] and not info["is_success"]
            assert info["segment_start"] and info["gates_passed"] == 1
            assert info["gate_progress"] == 1.0
        means, episodes = env.drain_metrics()
        assert means["gate_pass"] == 10 and means["completion"] == 100
        assert all(row["segment_start"] for row in episodes)
    finally:
        env.close()


@pytest.mark.integration
def test_ppo_resume_warm_start_and_actor_export(tmp_path: Path):
    env = Level2SBXEnv(CONFIG, n_envs=2, max_episode_steps=3)
    eval_env = Level2SBXEnv(
        CONFIG, n_envs=1, max_episode_steps=3, reset_config=ResetConfig(segment_init_prob=0)
    )
    logger = configure(str(tmp_path / "logs"), ["csv"])
    try:
        kwargs = dict(n_steps=4, batch_size=4, n_epochs=1, seed=3, device="cpu")
        model = PPO(Level2Policy, env, **kwargs)
        model.set_logger(logger)
        original = model.policy.actor_state.params
        callback = TrainingCallback(
            env,
            tmp_path / "checkpoints",
            {"ppo": kwargs},
            save_freq=8,
            eval_env=eval_env,
            eval_freq=8,
            eval_episodes=2,
        )
        model.learn(16, callback=callback)
        assert list((tmp_path / "checkpoints/best").glob("step_*/model.zip"))
        assert "reward/progress" in (tmp_path / "logs/progress.csv").read_text()
        assert "reward/gate_pass" in (tmp_path / "logs/progress.csv").read_text()
        assert int(model.policy.actor_state.step) > 0
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(original),
                jax.tree.leaves(model.policy.actor_state.params),
                strict=True,
            )
        )
        restored = load_checkpoint(callback.last_checkpoint, env)
        assert restored.num_timesteps == 16
        assert restored._n_updates == model._n_updates
        assert_tree_equal(restored.policy.actor_state.params, model.policy.actor_state.params)
        assert_tree_equal(restored.policy.actor_state.opt_state, model.policy.actor_state.opt_state)
        assert_tree_equal(restored.policy.vf_state.opt_state, model.policy.vf_state.opt_state)
        np.testing.assert_array_equal(restored.key, model.key)
        np.testing.assert_array_equal(restored.policy.key, model.policy.key)
        assert restored.l2_callback_state == model.l2_callback_state
        restored.set_logger(logger)
        resume_callback = TrainingCallback(env, tmp_path / "resumed", {"ppo": kwargs})
        restored.learn(8, callback=resume_callback, reset_num_timesteps=False)
        assert restored.num_timesteps == 24
        assert int(restored.policy.actor_state.step) > int(model.policy.actor_state.step)
        fresh = load_checkpoint(callback.last_checkpoint, env, mode="warm_start", ppo_kwargs=kwargs)
        assert fresh.num_timesteps == 0 and int(fresh.policy.actor_state.step) == 0
        assert int(fresh.policy.vf_state.step) == 0
        assert_tree_equal(fresh.policy.actor_state.params, model.policy.actor_state.params)
        export = Path(export_actor(str(callback.last_checkpoint), str(tmp_path / "deploy_model")))
        assert sorted(p.name for p in export.iterdir()) == ["actor.msgpack", "config.json"]
        config = load_config(CONFIG)
        config.controller.model_dir = str(export)
        ctrl_cls = load_controller(
            CONFIG.parents[1] / "lsy_drone_racing/control/rl_single_level2_controller.py"
        )
        env.reset()
        raw = {key: value[0] for key, value in env._current_obs.items()}
        ctrl = ctrl_cls(raw, {}, config)
        features = encode_obs(raw, np.zeros(4))
        expected_raw, _ = model.predict(env.normalizer.normalize(features), deterministic=True)
        expected_command = decode_action(expected_raw, raw["quat"], env.thrust_min, env.thrust_max)
        np.testing.assert_allclose(ctrl.compute_control(raw), expected_command, atol=1e-6)
        assert ctrl.step_callback(expected_command, raw, 0, False, False, {}) is False
        ctrl.episode_reset()
        np.testing.assert_array_equal(ctrl.previous_action, 0)
    finally:
        eval_env.close()
        env.close()
        logger.close()
