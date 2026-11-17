"""Run the frozen policy; the default is the current three-factor actuator_delay reference."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from harness import Runner, OUT, jp
from initial_state import add_start_perturbations
from presets import PRESETS, REFERENCE_PRESET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default=REFERENCE_PRESET)
    parser.add_argument("--scene", choices=["113735", "115247"], default="113735")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="0 is the exact start; positive seeds add the archived small perturbations.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--real-bounds", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    runner = Runner(args.scene)
    add_start_perturbations(runner)
    if args.real_bounds:
        original_reset = runner.reset

        def real_bounds(cfg, seed):
            data, raw = original_reset(cfg, seed)
            return data.replace(
                pos_limit_low=jp.array([-2.5, -1.5, -0.001], jp.float32),
                pos_limit_high=jp.array([2.5, 1.5, 2.0], jp.float32),
            ), raw

        runner.reset = real_bounds
    seed_suffix = f"_seed{args.seed}" if args.seed else ""
    bounds_suffix = "_realbounds" if args.real_bounds else ""
    output = (
        Path(args.output)
        if args.output
        else OUT / (f"reproduced_{args.scene}_{args.preset}{seed_suffix}{bounds_suffix}.json")
    )
    try:
        metrics, _ = runner.run(PRESETS[args.preset], args.seed, save=output)
        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "seed": args.seed,
                    "real_bounds": args.real_bounds,
                    "output": str(output),
                    "metrics": metrics,
                },
                indent=2,
            )
        )
    finally:
        runner.env.close()


if __name__ == "__main__":
    main()
