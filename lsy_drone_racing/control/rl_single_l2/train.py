"""Train Level 2 PPO with SBX; all output paths are owned by this entry point."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import fire
from sbx import PPO
from stable_baselines3.common.logger import configure

from lsy_drone_racing.control.rl_single_l2.callbacks import TrainingCallback
from lsy_drone_racing.control.rl_single_l2.checkpoint import load_checkpoint, read_metadata
from lsy_drone_racing.control.rl_single_l2.env_wrapper import (
    ActionConfig,
    Level2SBXEnv,
    ObservationConfig,
)
from lsy_drone_racing.control.rl_single_l2.policy import Level2Policy
from lsy_drone_racing.control.rl_single_l2.reset_mid_start import ResetConfig
from lsy_drone_racing.control.rl_single_l2.reward import RewardConfig

RL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = RL_DIR.parents[2] / "config" / "level2.toml"


def train(
    run_name: str | None = None,
    total_timesteps: int = 300_000_000,
    n_envs: int = 18432,
    n_steps: int = 256,
    batch_size: int = 18432,
    learning_rate: float = 3e-4,
    n_epochs: int = 3,
    clip_range: float = 0.2,
    ent_coef: float = 0.005,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    seed: int = 42,
    device: str = "cpu",
    config: str | None = None,
    max_episode_steps: int = 1500,
    save_freq: int = 20_000_000,
    keep: int = 3,
    checkpoint: str | None = None,
    mode: str = "new",
    output_root: str | None = None,
    eval_freq: int = 0,
    eval_episodes: int = 5,
    eval_seed: int = 10_000,
    gate_lookahead: int = 2,
    obstacle_count: int = 4,
    alpha_max_rad: float = 0.16,
    segment_init_prob: float = 0.40,
    segment_init_offset_m: float = 0.50,
    segment_init_vel_mps: float = 2.50,
    segment_init_perturb_m: float = 0.10,
) -> str:
    """Run PPO and return the final complete checkpoint directory.

    total_timesteps is an absolute target (including restored steps for resume).
    PPO rounds it up to a whole rollout. device selects the JAX simulator/policy
    backend; SBX's rollout-buffer tensors stay on CPU. Resume reuses the saved
    PPO/environment settings; warm_start uses the supplied new settings.
    """
    if mode not in {"new", "resume", "warm_start"}:
        raise ValueError("mode must be new, resume or warm_start")
    if (checkpoint is None) != (mode == "new"):
        raise ValueError("Provide a checkpoint exactly when mode is resume or warm_start")
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    if (
        total_timesteps < 1
        or min(n_envs, n_steps, n_epochs, max_episode_steps, keep, eval_episodes) < 1
    ):
        raise ValueError("Training counts and episode limits must be positive")
    if save_freq < 0 or eval_freq < 0 or learning_rate <= 0 or not 0 < clip_range < 1:
        raise ValueError("Invalid save/eval interval, learning rate or PPO clipping range")
    config_path = Path(config).expanduser().resolve() if config else DEFAULT_CONFIG
    observation = ObservationConfig(gate_lookahead=gate_lookahead, obstacle_count=obstacle_count)
    action = ActionConfig(alpha_max_rad=alpha_max_rad)
    reward = RewardConfig()
    reset = ResetConfig(
        segment_init_prob=segment_init_prob,
        segment_init_offset_m=segment_init_offset_m,
        segment_init_vel_mps=segment_init_vel_mps,
        segment_init_perturb_m=segment_init_perturb_m,
    )
    ppo_kwargs = dict(
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device="cpu",
        verbose=1,
    )
    if mode == "resume":
        saved = read_metadata(checkpoint)
        ppo_kwargs = saved["config"]["ppo"]
        old_env = saved["environment"]
        observation = ObservationConfig(**old_env["observation"])
        action, reward = ActionConfig(**old_env["action"]), RewardConfig(**old_env["reward"])
        reset = (
            ResetConfig(**old_env["reset"])
            if "reset" in old_env
            else ResetConfig(segment_init_prob=0.0)
        )
        max_episode_steps = old_env["max_episode_steps"]
        n_envs = saved["config"]["n_envs"]
        seed = ppo_kwargs["seed"]
        if config is None:
            config_path = Path(old_env["config_path"])
        if total_timesteps <= saved["num_timesteps"]:
            raise ValueError("Resume total_timesteps must exceed the restored cumulative steps")
    rollout_size = n_envs * ppo_kwargs["n_steps"]
    if ppo_kwargs["batch_size"] < 2 or rollout_size % ppo_kwargs["batch_size"]:
        raise ValueError("batch_size must be >= 2 and divide n_envs * n_steps exactly")
    run_name = run_name or datetime.now().strftime("l2_%Y%m%d_%H%M%S")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    root = Path(output_root).expanduser().resolve() if output_root else RL_DIR / "outputs"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir, checkpoint_dir = run_dir / "logs", run_dir / "checkpoints"
    log_dir.mkdir()
    checkpoint_dir.mkdir()
    env = eval_env = logger = None
    # JAX policy operations and simulator allocations use the selected backend.
    import jax

    try:
        with jax.default_device(jax.devices(device)[0]):
            env_kwargs = dict(
                config_path=config_path,
                device=device,
                max_episode_steps=max_episode_steps,
                observation_config=observation,
                action_config=action,
                reward_config=reward,
            )
            env = Level2SBXEnv(n_envs=n_envs, seed=seed, reset_config=reset, **env_kwargs)
            model = (
                PPO(Level2Policy, env, **ppo_kwargs)
                if mode == "new"
                else load_checkpoint(checkpoint, env, mode=mode, ppo_kwargs=ppo_kwargs)
            )
            effective = {
                "run_name": run_name,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "device": device,
                "mode": mode,
                "source_checkpoint": checkpoint,
                "ppo": ppo_kwargs,
                "environment": env.get_config(),
                "save_freq": save_freq,
                "keep": keep,
                "eval_freq": eval_freq,
                "eval_episodes": eval_episodes,
                "eval_seed": eval_seed,
            }
            (run_dir / "config.json").write_text(json.dumps(effective, indent=2) + "\n")
            logger = configure(str(log_dir), ["stdout", "log", "csv", "json"])
            model.set_logger(logger)
            if eval_freq:
                eval_env = Level2SBXEnv(
                    n_envs=1,
                    seed=eval_seed,
                    reset_config=ResetConfig(segment_init_prob=0.0),
                    **env_kwargs,
                )
            callback = TrainingCallback(
                env,
                checkpoint_dir,
                effective,
                save_freq,
                keep,
                eval_env,
                eval_freq,
                eval_episodes,
                eval_seed,
            )
            remaining = total_timesteps - model.num_timesteps
            model.learn(remaining, callback=callback, reset_num_timesteps=(mode != "resume"))
            return str(callback.last_checkpoint)
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    fire.Fire(train)
