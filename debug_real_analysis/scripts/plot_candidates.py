from rank import *
from paths import read_metrics, resolve_trajectory
import os

os.environ["MPLCONFIGDIR"] = str(OUT / "mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for phase in ["coarse", "extended", "fine", "focused", "validation"]:
    rows.extend(read_metrics(phase))


def find(scene, key):
    return next(r for r in rows if r["scene"] == scene and r["id"] == key)


def plot(scene, key, name):
    row = find(scene, key)
    a = dict(np.load(resolve_trajectory(row["trajectory"])))
    real = REALS[scene]
    sd = describe(a)
    rd = DESCS[scene]
    baseline = next(
        r
        for r in rows
        if r["scene"] == scene
        and (r.get("preset") == "baseline" or r["family"] == "omega_lowpass_capacity")
        and r["noise"]["omega_tau"] == 0
        and r["noise"]["thrust_cap"] == 0.8
    )
    b = dict(np.load(resolve_trajectory(baseline["trajectory"])))
    bd = describe(b)
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    for d, desc, color, label in [
        (b, bd, "#347fb6", "Unmodified sim"),
        (a, sd, "#df8b27", "Disturbed sim"),
        (real, rd, "#bd3039", "Recorded real"),
    ]:
        n = d["passed"]
        ids = np.flatnonzero((n >= 1) & (n < 3))
        x = d["t"][ids] - desc["t2"]
        ax[0, 0].plot(x, d["pos"][ids, 2], color=color, label=label, lw=1.8)
        ax[0, 1].plot(x, desc["tilt"][ids], color=color, lw=1.6)
        finite = ids[np.isfinite(d["requested"][ids, 3])]
        ax[1, 0].plot(
            d["t"][finite] - desc["t2"], d["requested"][finite, 3], color=color, label=label, lw=1.5
        )
        ax[1, 1].plot(x, d["vel"][ids, 2], color=color, lw=1.5)
        ax[2, 0].plot(d["pos"][ids, 0], d["pos"][ids, 1], color=color, lw=1.8)
        good = ids[np.all(np.isfinite(d["seen_ang_vel"][ids]), axis=1)]
        ax[2, 1].plot(d["t"][good] - desc["t2"], d["seen_ang_vel"][good, 1], color=color, lw=1.5)
    if "motor_thrust" in a:
        good = np.flatnonzero((a["passed"] == 2) & np.isfinite(a["motor_thrust"]))
        ax[1, 0].plot(
            a["t"][good] - sd["t2"],
            a["motor_thrust"][good],
            color="#765b36",
            ls="--",
            label="Disturbed sim actual rotor thrust",
            alpha=0.8,
        )
    ax[0, 0].set(ylabel="Height (m)", title="Height before gate 3")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].set(ylabel="Tilt (deg)", title="Physical attitude")
    ax[1, 0].set(ylabel="Thrust (N)", title="Policy demand and simulated motor output")
    ax[1, 0].legend(fontsize=7)
    ax[1, 1].set(ylabel="Vertical velocity (m/s)", title="Descent and braking")
    ax[2, 0].set(
        xlabel="World x (m)", ylabel="World y (m)", title="Horizontal trajectory, gate 2 to gate 3"
    )
    ax[2, 0].scatter([-1], [-0.25], marker="x", c="k", label="Gate 3 center")
    ax[2, 0].axis("equal")
    ax[2, 1].set(
        ylabel="Policy pitch-axis angular velocity (rad/s)",
        title="Angular velocity actually seen by policy",
    )
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]:
        ax[r, c].set_xlabel("Seconds since gate 2 crossing")
        ax[r, c].set_xlim(-0.1, 1.5)
    for aa in ax.flat:
        aa.grid(alpha=0.2)
    changed = {
        k: v for k, v in row["noise"].items() if v not in [0, 0.0, 1, 1.0, 0.8, 7000.0, "global"]
    }
    fig.suptitle(f"Scene {scene}, candidate {key}\n{changed}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("PLOT", name, "noise", row["noise"], "metrics", row)


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "113735"
    key = sys.argv[2] if len(sys.argv) > 2 else "bc60e02f023f"
    name = sys.argv[3] if len(sys.argv) > 3 else "candidate_progress"
    plot(scene, key, name)
