"""Real simulator/PPO/checkpoint and competition-loader integration."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sbx import PPO
from stable_baselines3.common.logger import configure

from lsy_drone_racing.control.rl_single_l2.callbacks import TrainingCallback
from lsy_drone_racing.control.rl_single_l2.checkpoint import export_actor, load_checkpoint
from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    Level2SBXEnv,
    compute_reward,
    decode_action,
    encode_obs,
)
from lsy_drone_racing.control.rl_single_l2.policy import Level2Policy
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
    env = Level2SBXEnv(CONFIG, n_envs=2, max_episode_steps=3)
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
def test_ppo_resume_warm_start_and_actor_export(tmp_path: Path):
    env = Level2SBXEnv(CONFIG, n_envs=2, max_episode_steps=3)
    eval_env = Level2SBXEnv(CONFIG, n_envs=1, max_episode_steps=3)
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
