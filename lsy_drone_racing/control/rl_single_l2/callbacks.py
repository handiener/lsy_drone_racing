"""Rollout statistics, frozen-stat evaluation and post-update checkpoint timing."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from lsy_drone_racing.control.rl_single_l2.checkpoint import save_checkpoint

if TYPE_CHECKING:
    from lsy_drone_racing.control.rl_single_l2.env_wrapper import Level2SBXEnv


def evaluate_policy(model: Any, env: Level2SBXEnv, episodes: int, seed: int) -> dict[str, float]:
    """Evaluate on the same independent seeds each time, with deterministic actions."""
    if env.num_envs != 1 or episodes < 1:
        raise ValueError("Evaluation needs one world and a positive episode count")
    results = []
    for episode in range(episodes):
        env.seed(seed + episode)
        obs = env.reset()
        for _ in range(env.max_episode_steps):
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(actions)
            if done[0]:
                results.append(infos[0])
                break
    success_times = [row["success_time"] for row in results if row["finished"]]
    return {
        "success_rate": float(np.mean([row["finished"] for row in results])),
        "gate_progress": float(np.mean([row["gate_progress"] for row in results])),
        "crash_rate": float(np.mean([row["crash"] for row in results])),
        "timeout_rate": float(np.mean([row["timeout"] for row in results])),
        "episode_return": float(np.mean([row["episode"]["r"] for row in results])),
        "success_time": float(np.mean(success_times)) if success_times else 0.0,
    }


class TrainingCallback(BaseCallback):
    """Keep sampling/update statistics consistent and save only completed PPO updates.

    SBX calls rollout_end *before* optimizing. Pending raw moments are therefore
    merged at the next rollout_start, where the previous optimizer update has
    finished. The model's cached last observation is re-encoded at that boundary.
    """

    def __init__(
        self,
        env: Level2SBXEnv,
        checkpoint_dir: Path,
        config: dict[str, Any],
        save_freq: int = 20_000_000,
        keep: int = 3,
        eval_env: Level2SBXEnv | None = None,
        eval_freq: int = 0,
        eval_episodes: int = 5,
        eval_seed: int = 10_000,
    ) -> None:
        """Receive paths from train.py; frequencies count total environment transitions."""
        super().__init__()
        self.env = env
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.save_freq, self.keep = save_freq, keep
        self.eval_env, self.eval_freq = eval_env, eval_freq
        self.eval_episodes, self.eval_seed = eval_episodes, eval_seed
        self.episodes: deque = deque(maxlen=1000)
        self.last_save = self.last_eval = 0
        self.best_score = (-1.0, -1.0, -float("inf"))
        self.last_checkpoint: Path | None = None

    def _on_training_start(self) -> None:
        state = getattr(self.model, "l2_callback_state", {})
        self.last_save = state.get("last_save", self.model.num_timesteps)
        self.last_eval = state.get("last_eval", self.model.num_timesteps)
        if state.get("best_score") is not None:
            self.best_score = tuple(state["best_score"])

    def _state(self) -> None:
        self.model.l2_callback_state = {
            "last_save": self.last_save,
            "last_eval": self.last_eval,
            "best_score": list(self.best_score) if self.best_score[0] >= 0 else None,
        }

    def _save(self) -> None:
        self.last_save = self.model.num_timesteps
        self._state()
        self.last_checkpoint = save_checkpoint(
            self.checkpoint_dir, self.model, self.env, self.config, keep=self.keep
        )

    def _evaluate(self) -> None:
        if self.eval_env is None:
            return
        self.eval_env.set_normalizer(self.env.normalizer)
        metrics = evaluate_policy(self.model, self.eval_env, self.eval_episodes, self.eval_seed)
        self.last_eval = self.model.num_timesteps
        for key, value in metrics.items():
            self.logger.record(f"eval/{key}", value)
        score = (metrics["success_rate"], metrics["gate_progress"], -metrics["success_time"])
        if score > self.best_score:
            self.best_score = score
            self._state()
            best = save_checkpoint(
                self.checkpoint_dir / "best", self.model, self.env, self.config, keep=1
            )
            self.logger.record("eval/best_checkpoint", str(best), exclude="tensorboard")
        self._state()

    def _on_rollout_start(self) -> None:
        self.env.update_normalizer()
        self.model._last_obs = self.env.get_observation()
        # At this point the previous rollout has been fully optimized.
        if self.eval_freq and self.model.num_timesteps - self.last_eval >= self.eval_freq:
            self._evaluate()
        if self.save_freq and self.model.num_timesteps - self.last_save >= self.save_freq:
            self._save()

    def _on_step(self) -> bool:
        # The environment aggregates metrics on-device; no per-world Python scan.
        return True

    def _on_rollout_end(self) -> None:
        reward_means, episodes = self.env.drain_metrics()
        self.episodes.extend(episodes)
        for key, value in reward_means.items():
            self.logger.record(f"reward/{key}", value)
        if self.episodes:
            for metric in ("finished", "crash", "timeout", "gate_progress"):
                self.logger.record(
                    f"episodes/{metric}", np.mean([row[metric] for row in self.episodes])
                )
            times = [row["success_time"] for row in self.episodes if row["finished"]]
            if times:
                self.logger.record("episodes/success_time", np.mean(times))
        buffer = self.model.rollout_buffer.observations
        sample = buffer.reshape(-1, self.env.observation_config.size)
        sample = sample[:: max(1, len(sample) // 1024)][:1024]
        dist = self.model.policy.actor.apply(
            self.model.policy.actor_state.params, jnp.asarray(sample)
        )
        mean, std = np.asarray(dist.mean()), np.asarray(dist.stddev())
        for index, label in enumerate(("thrust", "rotation_x", "rotation_y", "rotation_z")):
            self.logger.record(f"actions/{label}_mean", float(mean[:, index].mean()))
            self.logger.record(f"actions/{label}_std", float(std[:, index].mean()))
        self.logger.record("normalizer/count", self.env.normalizer.count)

    def _on_training_end(self) -> None:
        # Final update just finished; save its original frozen moments and pending
        # moments together. Resume commits the pending batch at the next start.
        if self.eval_env is not None and self.last_eval != self.model.num_timesteps:
            self._evaluate()
        if self.last_checkpoint is None or self.last_save != self.model.num_timesteps:
            self._save()
        self.logger.dump(step=self.model.num_timesteps)
