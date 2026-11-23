"""Evaluate one retained trial checkpoint in nominal and both stress environments."""

import fire

from .evaluation import evaluate


def assess(
    run_dir: str,
    episodes: int = 512,
    seeds: tuple[int, ...] = (1710001, 1720001),
    checkpoint: str = "final",
    device: str = "gpu",
) -> dict:
    """Use paired seeds and write three reports beside the chosen checkpoint."""
    return {
        profile: evaluate(
            run_dir,
            episodes=episodes,
            seeds=seeds,
            checkpoint=checkpoint,
            sim_to_real=profile,
            report_name=f"eval_{profile}",
            device=device,
        )
        for profile in ("none", "reference", "estimator_mix")
    }


if __name__ == "__main__":
    fire.Fire(assess)
