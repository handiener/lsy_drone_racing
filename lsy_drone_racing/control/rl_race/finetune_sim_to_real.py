"""Fine-tune the fastest local sweep actor with the two retained stress profiles."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import fire

from .export import export
from .train import train

RL_DIR = Path(__file__).resolve().parent
TRIALS = {"reference": "sim2real_fixed", "estimator_mix": "sim2real_mixed"}


def select_base() -> Path:
    """Choose the fastest completed sweep actor with at least 50% nominal success."""
    candidates = []
    for path in (RL_DIR / "outputs/sweeps").rglob("final_evaluation.json"):
        report = json.loads(path.read_text())
        if (
            report.get("stage", "race") == "race"
            and report.get("success_rate", 0) >= 0.5
            and report.get("mean_success_time") is not None
            and (path.parent / "checkpoints/race_final.pkl").is_file()
        ):
            candidates.append((report["mean_success_time"], path.parent))
    if not candidates:
        raise FileNotFoundError("No eligible completed race sweep checkpoint found")
    return min(candidates, key=lambda item: item[0])[1]


def run(
    output_root: str = str(RL_DIR / "outputs/sim2real_training"),
    steps: int = 1_073_741_824,
    profiles: tuple[str, ...] = ("reference", "estimator_mix"),
) -> str:
    """Train independent continuations at original PPO sizes; assess separately.

    Intermediate rollout evaluations are disabled. Four curve checkpoints and training
    metrics are saved. Existing output directories are never overwritten.
    """
    if steps <= 0 or steps % 16_777_216:
        raise ValueError("steps must be a positive multiple of 16,777,216")
    if (
        not profiles
        or len(set(profiles)) != len(profiles)
        or any(p not in TRIALS for p in profiles)
    ):
        raise ValueError("profiles accepts reference and estimator_mix, without duplicates")
    root = Path(output_root).expanduser().resolve()
    for profile in profiles:
        if (root / TRIALS[profile]).exists():
            raise FileExistsError(f"Use a fresh output directory: {root / TRIALS[profile]}")
    source = select_base()
    source_config = json.loads((source / "race_config.json").read_text())
    allowed = inspect.signature(train).parameters
    settings = {k: v for k, v in source_config["settings"].items() if k in allowed}
    for profile in profiles:
        options = dict(settings)
        options.update(
            run_name=TRIALS[profile],
            output_root=str(root),
            seed=22,
            device="gpu",
            start_stage="race",
            continue_race=True,
            start_params=str(source / "checkpoints/race_final.pkl"),
            start_step=0,
            race_steps=steps,
            anneal_steps=0,
            sim_to_real=profile,
            sim_to_real_nominal_prob=0.35,
            num_envs=16384,
            batch_size=4096,
            num_minibatches=32,
            num_updates_per_batch=4,
            unroll_length=32,
            learning_rate=1e-4,
            policy_lr_after_switch=2e-5,
            lr_decay_to=0.5,
            warmup_iterations=5,
            entropy_cost=0.001,
            intermediate_eval=False,
            save_step_checkpoints=True,
            eval_every_steps=steps // 4,
            final_eval=False,
            wandb_project=None,
            wandb_entity=None,
            wandb_group=None,
        )
        train(**options)
        run_dir = root / TRIALS[profile]
        export(str(run_dir), destination=str(run_dir / "exported"))
    return str(root)


if __name__ == "__main__":
    fire.Fire(run)
