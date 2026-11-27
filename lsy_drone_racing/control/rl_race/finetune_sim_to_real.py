"""Fine-tune a chosen race/anneal checkpoint with the original sim-to-real profiles."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import fire

from .evaluation import evaluate
from .export import export
from .train import train

RL_DIR = Path(__file__).resolve().parent


def run(
    source: str,
    source_stage: str | None = None,
    sim_to_real: str = "estimator_mix",
    sim_to_real_nominal_prob: float = 0.35,
    output_root: str | None = None,
    run_name: str | None = None,
    steps: int = 2_147_483_648,
    seed: int = 22,
    device: str = "gpu",
    checkpoint_every_iterations: int = 128,
    num_envs: int = 16384,
    batch_size: int = 4096,
    num_minibatches: int = 32,
    unroll_length: int = 32,
    num_updates_per_batch: int = 4,
    num_eval_envs: int = 512,
    max_episode_steps: int | None = None,
    intermediate_eval: bool = False,
    final_eval: bool = True,
    final_eval_episodes: int = 1024,
    final_eval_seeds: tuple[int, ...] = (610000, 620000, 630000),
    final_eval_batch_size: int = 256,
) -> str:
    """Continue one selected model, keeping its problem and architecture settings.

    ``source`` is a run directory or one of its trusted ``checkpoints/*.pkl`` files.
    A directory uses its final evaluation's stage, or the latest available final stage.
    Optimizer/rollout state starts fresh; the actor, critic and normalizer are restored.
    The default 512 PPO iterations double the old-track 256-iteration trial budget.
    Final evaluation measures both the training mixture and ordinary simulation.
    The exported actor is stored inside the new run, leaving deployed weights untouched.
    """
    from .env_wrapper.sim_to_real import SimToRealConfig

    SimToRealConfig(sim_to_real, sim_to_real_nominal_prob)
    if sim_to_real == "none":
        raise ValueError("Fine-tuning needs sim_to_real=reference or estimator_mix")
    if min(num_envs, batch_size, num_minibatches, unroll_length, num_updates_per_batch) <= 0:
        raise ValueError("PPO sizes must be positive")
    transitions = batch_size * num_minibatches * unroll_length
    if steps <= 0 or steps % transitions:
        raise ValueError(f"steps must be a positive multiple of {transitions:,}")
    if checkpoint_every_iterations <= 0:
        raise ValueError("checkpoint_every_iterations must be positive")
    if max_episode_steps is not None and max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if (batch_size * num_minibatches) % num_envs:
        raise ValueError("batch_size * num_minibatches must be a multiple of num_envs")

    selected = Path(source).expanduser().resolve()
    if selected.is_dir():
        source_run = selected
        if source_stage is None:
            report = source_run / "final_evaluation.json"
            if report.is_file():
                source_stage = json.loads(report.read_text())["stage"]
            else:
                source_stage = (
                    "anneal" if (source_run / "checkpoints/anneal_final.pkl").is_file() else "race"
                )
        checkpoint = source_run / "checkpoints" / f"{source_stage}_final.pkl"
    else:
        checkpoint = selected
        if checkpoint.parent.name != "checkpoints" or checkpoint.suffix != ".pkl":
            raise ValueError("source must be a run directory or its checkpoints/*.pkl file")
        source_run = checkpoint.parent.parent
        file_stage = checkpoint.name.split("_", 1)[0]
        if source_stage is not None and source_stage != file_stage:
            raise ValueError("source_stage does not match the chosen checkpoint")
        source_stage = file_stage
    if source_stage not in {"race", "anneal"}:
        raise ValueError("Choose a completed race or anneal policy")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = json.loads((source_run / f"{source_stage}_config.json").read_text())
    env, network = config["env"], config["network"]
    if config["freq"] != 50:
        raise ValueError("The original sim-to-real profiles require 50 Hz")
    if env["obs"].get("n_obstacles", 4) != 4:
        raise ValueError("This training entry requires the four-obstacle observation layout")
    allowed = inspect.signature(train).parameters
    options = {k: v for k, v in config["settings"].items() if k in allowed}
    # Read the actual saved problem, including different references, tilt and rewards.
    action = env["action"]
    options.update(
        {
            k: action[k]
            for k in (
                "roll_limit_deg",
                "pitch_limit_deg",
                "yaw_limit_deg",
                "thrust_min_fraction",
                "thrust_max_fraction",
            )
        }
    )
    options.update(
        {
            f"race_{k}": env["reward"][k]
            for k in ("time", "crash", "gate", "lap", "lap_bonus", "lap_ref")
        }
    )
    interval = min(checkpoint_every_iterations, steps // transitions) * transitions
    options.update(
        run_name=run_name or f"{source_run.name}_sim2real_{sim_to_real}_s{seed}",
        output_root=output_root or str(RL_DIR / "outputs/sim2real_training"),
        seed=seed,
        device=device,
        reference_path=env.get("reference_path", "hermite"),
        n_gates=env["obs"]["n_gates"],
        official_start_prob=env["start"]["official_prob"],
        max_episode_steps=(
            env["max_episode_steps"] if max_episode_steps is None else max_episode_steps
        ),
        policy_layers=tuple(network["policy_layers"]),
        value_layers=tuple(network["value_layers"]),
        activation=network["activation"],
        start_stage="race",
        continue_race=True,
        start_params=str(checkpoint),
        start_step=0,
        race_steps=steps,
        anneal_steps=0,
        sim_to_real=sim_to_real,
        sim_to_real_nominal_prob=sim_to_real_nominal_prob,
        num_envs=num_envs,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        num_eval_envs=num_eval_envs,
        unroll_length=unroll_length,
        learning_rate=1e-4,
        policy_lr_after_switch=2e-5,
        lr_decay_to=0.5,
        warmup_iterations=5,
        entropy_cost=0.001,
        intermediate_eval=intermediate_eval,
        save_step_checkpoints=True,
        eval_every_steps=interval,
        race_eval_every_steps=interval,
        final_eval=final_eval,
        final_eval_episodes=final_eval_episodes,
        final_eval_seeds=final_eval_seeds,
        final_eval_batch_size=final_eval_batch_size,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
    )
    print(
        f"Fine-tune {checkpoint}\n"
        f"reference={options['reference_path']}, profile={sim_to_real}, "
        f"nominal_prob={sim_to_real_nominal_prob}; "
        f"{steps // transitions} requested PPO iterations ({steps:,} transitions)",
        flush=True,
    )
    run_dir = train(**options)
    if final_eval:
        evaluate(
            run_dir,
            stage="race",
            episodes=final_eval_episodes,
            seeds=final_eval_seeds,
            batch_size=final_eval_batch_size,
            device=device,
            sim_to_real="none",
            report_name="nominal",
        )
    export(run_dir, stage="race", destination=str(Path(run_dir) / "exported"))
    return run_dir


if __name__ == "__main__":
    fire.Fire(run)
