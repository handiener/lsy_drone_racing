r"""Train the racing policy with Brax PPO in two stages (see DESIGN.md).

Stage ``pretrain``: reward ``dense_path`` (race reward + arc-length progress along the nominal
path), until the deterministic evaluation from the official start reaches ``switch_success``.
Stage ``race``: reward ``race``; the critic is re-initialized and trained alone for
``critic_only_iterations`` PPO iterations, then the policy resumes with ``policy_lr_after_switch``.

Usage::

    python -m lsy_drone_racing.control.rl_race.train --run_name=race_s1 --seed=1 \
        --wandb_project=lsy-drone-racing --wandb_group=rl-race
"""

from __future__ import annotations

import functools
import json
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import jax
import numpy as np

from lsy_drone_racing.control.rl_race import brax_ppo_train as ppo
from lsy_drone_racing.control.rl_race import rewards
from lsy_drone_racing.control.rl_race.actions import ActionConfig
from lsy_drone_racing.control.rl_race.env import EnvConfig, RaceEnv
from lsy_drone_racing.control.rl_race.observations import ObsConfig
from lsy_drone_racing.control.rl_race.optim import iteration_updates, separate_lr
from lsy_drone_racing.control.rl_race.starts import StartConfig

RL_DIR = Path(__file__).resolve().parent
ACTIVATIONS = ("swish", "elu", "relu", "tanh")


class _SwitchReward(Exception):
    """Raised from progress_fn to end pre-training once the success threshold is met."""


def env_config(
    stage: str,
    theta_max_deg: float,
    official_prob: float,
    max_steps: int,
    pretrain_reward: dict[str, float] | None = None,
) -> EnvConfig:
    """Environment configuration of a stage (``pretrain`` or ``race``).

    ``pretrain_reward`` overrides coefficients of the ``dense_path`` shaping reward; the race
    reward always uses the ``race`` preset.
    """
    if stage == "pretrain":
        reward = rewards.preset("dense_path", **(pretrain_reward or {}))
    else:
        reward = rewards.preset("race")
    return EnvConfig(
        max_episode_steps=max_steps,
        action=ActionConfig(theta_max_deg=theta_max_deg),
        obs=ObsConfig(),
        reward=reward,
        start=StartConfig(official_prob=official_prob),
    )


def eval_summary(metrics: dict[str, Any]) -> dict[str, float]:
    """Success rate and mean lap time from Brax evaluation metrics (official starts only)."""
    episodes = float(metrics.get("eval/episode_official_done", 0.0))
    success = float(metrics.get("eval/episode_success", 0.0))
    lap_time = float(metrics.get("eval/episode_lap_time", 0.0))
    return {
        "eval/success_rate": success / episodes if episodes > 0 else 0.0,
        "eval/lap_time": lap_time / success if success > 0 else float("nan"),
        "eval/gates_passed": float(metrics.get("eval/episode_gates", 0.0)),
        "eval/crash_rate": float(metrics.get("eval/episode_crashed", 0.0)),
    }


def save_params(params: Any, target: Path) -> None:
    """Atomically pickle (normalizer, policy, value) parameters on the host."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(jax.device_get(params)))
    temporary.replace(target)


def load_params(source: str | Path) -> Any:
    """Load parameters written by ``save_params`` (trusted local files only)."""
    return pickle.loads(Path(source).read_bytes())


def train(
    run_name: str | None = None,
    output_root: str | None = None,
    seed: int = 0,
    device: str = "gpu",
    # Problem.
    theta_max_deg: float = 70.0,
    official_start_prob: float = 0.5,
    max_episode_steps: int = 750,
    # Schedule.
    pretrain_max_steps: int = 500_000_000,
    pretrain_crash: float | None = None,
    pretrain_time: float | None = None,
    pretrain_progress: float | None = None,
    pretrain_path_lateral: float | None = None,
    race_steps: int = 1_000_000_000,
    switch_success: float = 0.3,
    critic_only_iterations: int = 10,
    start_params: str | None = None,
    start_stage: str = "pretrain",
    # PPO (Brax): one iteration = num_envs * unroll_length * (batch * minibatches / num_envs).
    num_envs: int = 8192,
    unroll_length: int = 32,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_updates_per_batch: int = 4,
    learning_rate: float = 3e-4,
    policy_lr_after_switch: float = 1e-4,
    entropy_cost: float = 5e-3,
    discounting: float = 0.99,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.2,
    max_grad_norm: float = 1.0,
    policy_layers: tuple[int, ...] = (256, 256, 256),
    value_layers: tuple[int, ...] = (256, 256, 256, 256),
    activation: str = "swish",
    # Evaluation and logging.
    eval_every_steps: int = 20_000_000,
    num_eval_envs: int = 512,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
) -> str:
    """Run the staged training and return the run directory."""
    from brax.training.agents.ppo import networks as ppo_networks

    if start_stage not in {"pretrain", "race"}:
        raise ValueError("start_stage must be pretrain or race")
    if start_stage == "race" and start_params is None:
        raise ValueError("start_stage=race needs start_params from a pre-training run")
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}")
    if (batch_size * num_minibatches) % num_envs:
        raise ValueError("batch_size * num_minibatches must be a multiple of num_envs")

    run_name = run_name or datetime.now().strftime("race_%Y%m%d_%H%M%S")
    root = Path(output_root).expanduser().resolve() if output_root else RL_DIR / "outputs"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {key: value for key, value in locals().items() if key not in {"ppo_networks"}}
    settings["policy_layers"], settings["value_layers"] = list(policy_layers), list(value_layers)
    network = {
        "policy_layers": list(policy_layers),
        "value_layers": list(value_layers),
        "activation": activation,
        "distribution": "tanh_normal",
    }
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(policy_layers),
        value_hidden_layer_sizes=tuple(value_layers),
        activation=getattr(jax.nn, activation),
        distribution_type="tanh_normal",
    )
    wandb_run = None
    if wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=run_name,
            dir=str(run_dir),
            config={
                key: str(value) if isinstance(value, Path) else value
                for key, value in settings.items()
            },
        )

    overrides = {
        "crash": pretrain_crash,
        "time": pretrain_time,
        "progress": pretrain_progress,
        "path_lateral": pretrain_path_lateral,
    }
    pretrain_reward = {key: value for key, value in overrides.items() if value is not None}
    step_offset = 0
    params = load_params(start_params) if start_params else None
    stage_log: list[dict[str, Any]] = []
    stages = ["pretrain", "race"] if start_stage == "pretrain" else ["race"]
    try:
        for stage in stages:
            cfg = env_config(
                stage, theta_max_deg, official_start_prob, max_episode_steps, pretrain_reward
            )
            eval_cfg = env_config(stage, theta_max_deg, 1.0, max_episode_steps, pretrain_reward)
            env = RaceEnv(num_envs, cfg, device=device, seed=seed)
            eval_env = RaceEnv(num_eval_envs, eval_cfg, device=device, seed=seed + 10_000)
            steps = pretrain_max_steps if stage == "pretrain" else race_steps
            (run_dir / f"{stage}_config.json").write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "env": asdict(cfg),
                        "network": network,
                        "freq": env.freq,
                        "thrust_min": env.action_cfg.thrust_min,
                        "thrust_max": env.action_cfg.thrust_max,
                        "settings": {
                            k: str(v) if isinstance(v, Path) else v for k, v in settings.items()
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            latest: dict[str, Any] = {}
            started = time.time()

            def keep_params(step: int, _make_policy: Any, current: Any, stage: str = stage) -> None:
                latest["step"], latest["params"] = step, current
                save_params(current, run_dir / "checkpoints" / f"{stage}_latest.pkl")

            def progress(step: int, metrics: dict[str, Any], stage: str = stage) -> None:
                values = {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}
                is_eval = "eval/episode_reward" in values
                if is_eval:
                    values.update(eval_summary(values))
                values["stage"] = 0.0 if stage == "pretrain" else 1.0
                if wandb_run is not None:
                    wandb_run.log(values, step=step_offset + step)
                if not is_eval:
                    return
                rate, lap = values["eval/success_rate"], values["eval/lap_time"]
                elapsed = time.time() - started
                print(
                    f"[{stage}] step {step_offset + step:>13,d} | success {rate:5.1%} | "
                    f"lap {lap:5.2f}s | gates {values['eval/gates_passed']:4.2f} | "
                    f"{elapsed / 60:6.1f} min",
                    flush=True,
                )
                if stage == "pretrain" and step > 0 and rate >= switch_success:
                    raise _SwitchReward

            kwargs: dict[str, Any] = {}
            if params is not None:
                kwargs["restore_params"] = params
            if stage == "race" and params is not None:
                kwargs["restore_value_fn"] = False  # critic re-initialized for the new reward
                kwargs["optimizer_override"] = separate_lr(
                    policy_lr=policy_lr_after_switch,
                    value_lr=learning_rate,
                    max_grad_norm=max_grad_norm,
                    freeze_policy_steps=critic_only_iterations
                    * iteration_updates(num_updates_per_batch, num_minibatches),
                )
            switched = False
            try:
                _, params, _ = ppo.train(
                    environment=env,
                    eval_env=eval_env,
                    wrap_env=False,
                    num_timesteps=steps,
                    episode_length=max_episode_steps,
                    num_envs=num_envs,
                    num_eval_envs=num_eval_envs,
                    unroll_length=unroll_length,
                    batch_size=batch_size,
                    num_minibatches=num_minibatches,
                    num_updates_per_batch=num_updates_per_batch,
                    learning_rate=learning_rate,
                    entropy_cost=entropy_cost,
                    discounting=discounting,
                    gae_lambda=gae_lambda,
                    clipping_epsilon=clipping_epsilon,
                    max_grad_norm=max_grad_norm,
                    normalize_observations=True,
                    network_factory=network_factory,
                    num_evals=max(steps // eval_every_steps, 1) + 1,
                    deterministic_eval=True,
                    log_training_metrics=True,
                    seed=seed,
                    progress_fn=progress,
                    policy_params_fn=keep_params,
                    **kwargs,
                )
                end_step = latest["step"]  # Brax rounds up to whole iterations
            except _SwitchReward:
                switched = True
                params, end_step = latest["params"], latest["step"]
            save_params(params, run_dir / "checkpoints" / f"{stage}_final.pkl")
            stage_log.append({"stage": stage, "steps": end_step, "switched_on_success": switched})
            step_offset += end_step
            if stage == "pretrain" and not switched:
                print(
                    f"Pre-training did not reach {switch_success:.0%} success within "
                    f"{steps:,d} steps; stopping before the race stage.",
                    flush=True,
                )
                break
    finally:
        (run_dir / "stages.json").write_text(json.dumps(stage_log, indent=2) + "\n")
        if wandb_run is not None:
            wandb_run.finish()
    return str(run_dir)


if __name__ == "__main__":
    fire.Fire(train)
