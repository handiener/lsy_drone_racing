"""Plot bundled current-reference trajectories against nominal controls and recorded policy inputs."""

from __future__ import annotations
import argparse
import json
import numpy as np
from paths import REFERENCE, OUT
from rank import REALS, describe
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["113735", "115247"], default="113735")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="0: exact start. Seed 16 is the explicitly post-hoc selected illustration.",
    )
    args = parser.parse_args()
    rows = json.loads((REFERENCE / "metrics.json").read_text())

    def trajectory(role):
        row = next(
            (
                r
                for r in rows
                if r["scene"] == args.scene and r["role"] == role and r["seed"] == args.seed
            ),
            None,
        )
        if row is None:
            raise ValueError(
                f"No bundled {role} trajectory for scene {args.scene}, seed {args.seed}"
            )
        return dict(np.load(REFERENCE / row["trajectory"]))

    curves = [
        (trajectory("baseline"), "#347fb6", "Original sim"),
        (trajectory("jitter"), "#df8b27", "Response 0.8 + rate age 40 ms + cap 0.70 N"),
        (REALS[args.scene], "#bd3039", "Recorded policy observation"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for arrays, color, label in curves:
        info = describe(arrays)
        if info is None:
            print(f"{label}: does not reach gate 2; omitted from event-aligned plot")
            continue
        ids = np.flatnonzero((arrays["passed"] >= 1) & (arrays["passed"] < 3))
        time = arrays["t"] - info["t2"]
        series = [
            (axes[0, 0], arrays["pos"][:, 2], "Reference height (m)"),
            (axes[0, 1], info["tilt"], "Tilt (deg)"),
            (axes[0, 2], arrays["vel"][:, 2], "Vertical velocity (m/s)"),
            (axes[1, 0], arrays["requested"][:, 3], "Requested thrust (N)"),
            (axes[1, 1], arrays["seen_ang_vel"][:, 1], "Policy pitch-axis rate (rad/s)"),
        ]
        for axis, values, ylabel in series:
            valid = ids[np.isfinite(values[ids])]
            axis.plot(time[valid], values[valid], color=color, label=label)
            axis.set(xlabel="Seconds after first gate 2 passage", ylabel=ylabel, xlim=(-0.15, 1.6))
        axes[1, 2].plot(arrays["pos"][ids, 0], arrays["pos"][ids, 1], color=color, label=label)
    axes[0, 0].axhspan(0, 0.15, color="#d3a354", alpha=0.12)
    axes[0, 0].legend(fontsize=7)
    axes[1, 2].set(xlabel="World x (m)", ylabel="World y (m)")
    axes[1, 2].set_aspect("equal", adjustable="datalim")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    selection = "exact recorded start" if args.seed == 0 else f"perturbed start, seed {args.seed}"
    if args.seed == 16:
        selection += "; selected after inspection, not independent validation"
    fig.suptitle(f"Current diagnostic reference: scene {args.scene}\n{selection}", fontsize=11)
    fig.tight_layout()
    target = OUT / f"reference_{args.scene}_seed{args.seed}"
    fig.savefig(target.with_suffix(".png"), dpi=160)
    fig.savefig(target.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {target.with_suffix('.png')} and PDF")


if __name__ == "__main__":
    main()
