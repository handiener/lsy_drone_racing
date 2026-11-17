"""Compare freshly reproduced simulation records with the branch's existing hardware log."""

from __future__ import annotations

import argparse
import json

from paths import OUT
from rank import REALS, describe
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_replay(scene, preset):
    path = OUT / f"reproduced_{scene}_{preset}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. First run reproduce.py --preset {preset} --scene {scene} "
            "using the same DRONE_DEBUG_OUTPUT_DIR."
        )
    records = json.loads(path.read_text())["records"]
    arrays = {
        "t": np.array([row["t"] for row in records]),
        "passed": np.array([row["passed"] for row in records]),
    }
    for field in ["pos", "quat", "vel", "ang_vel", "seen_ang_vel", "requested"]:
        shape = np.asarray(records[0][field]).shape
        arrays[field] = np.asarray([row.get(field, np.full(shape, np.nan)) for row in records])
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["113735", "115247"], default="113735")
    parser.add_argument("--preset", choices=["paired", "input_match", "minimal"], default="paired")
    args = parser.parse_args()
    baseline = read_replay(args.scene, "baseline")
    disturbed = read_replay(args.scene, args.preset)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for arrays, color, label in [
        (baseline, "#347fb6", "Original sim"),
        (disturbed, "#df8b27", f"Disturbed sim ({args.preset})"),
        (REALS[args.scene], "#bd3039", "Recorded hardware"),
    ]:
        info = describe(arrays)
        if info is None:
            raise ValueError(f"{label} never reaches gate 2; no gate-2-aligned comparison exists.")
        ids = np.flatnonzero((arrays["passed"] >= 1) & (arrays["passed"] < 3))
        time = arrays["t"] - info["t2"]
        series = [
            (axes[0, 0], arrays["pos"][:, 2], "Reference height (m)"),
            (axes[0, 1], info["tilt"], "Physical tilt (deg)"),
            (axes[0, 2], arrays["vel"][:, 2], "Vertical velocity (m/s)"),
            (axes[1, 0], arrays["requested"][:, 3], "Requested thrust (N)"),
            (axes[1, 1], arrays["seen_ang_vel"][:, 1], "Policy pitch-axis rate (rad/s)"),
        ]
        for axis, values, ylabel in series:
            valid = ids[np.isfinite(values[ids])]
            axis.plot(time[valid], values[valid], color=color, label=label)
            axis.set(xlabel="Seconds after gate 2 crossing", ylabel=ylabel, xlim=(-0.1, 1.6))
        axes[1, 2].plot(arrays["pos"][ids, 0], arrays["pos"][ids, 1], color=color, label=label)
    axes[0, 0].axhspan(0, 0.15, color="#d3a354", alpha=0.12)
    axes[0, 0].legend(fontsize=8)
    axes[1, 2].set(xlabel="World x (m)", ylabel="World y (m)")
    axes[1, 2].set_aspect("equal", adjustable="datalim")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(f"Scene {args.scene}: original sim, {args.preset} and recorded hardware")
    fig.tight_layout()
    destination = OUT / f"replay_comparison_{args.scene}_{args.preset}"
    fig.savefig(destination.with_suffix(".png"), dpi=160)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {destination.with_suffix('.png')} and PDF")


if __name__ == "__main__":
    main()
