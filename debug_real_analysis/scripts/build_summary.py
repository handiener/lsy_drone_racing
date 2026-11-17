from rank import *
from paths import read_metrics, resolve_trajectory
import os

os.environ["MPLCONFIGDIR"] = str(OUT / "mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
rows = []
for phase in ["coarse", "extended", "focused", "validation"]:
    rows.extend(read_metrics(phase))
for aa, scene, title in zip(
    ax,
    ["113735", "115247"],
    ["12:37 Berlin hardware initial state", "12:52 Berlin hardware initial state"],
):
    real = REALS[scene]
    rd = describe(real)
    b = dict(np.load(resolve_trajectory(f"validation/trajectories/{scene}_baseline_00.npz")))
    bd = describe(b)
    a = dict(np.load(resolve_trajectory(f"validation/trajectories/{scene}_B_full_00.npz")))
    ad = describe(a)
    for d, desc, color, label in [
        (b, bd, "#347fb6", "Original sim"),
        (a, ad, "#df8b27", "Sim with combined disturbances"),
        (real, rd, "#bd3039", "Recorded hardware"),
    ]:
        ids = np.flatnonzero(d["passed"] == 2)
        aa.plot(d["t"][ids] - desc["t2"], d["pos"][ids, 2], color=color, label=label, lw=2.1)
    aa.axhspan(0, 0.15, color="#d3a354", alpha=0.12)
    aa.set(
        title=title, xlabel="Seconds after gate 2 crossing", xlim=(-0.02, 2.15), ylim=(-0.025, 1.4)
    )
    aa.grid(alpha=0.2)
    aa.set_ylabel("Recorded / simulated reference height (m)")
ax[0].legend(fontsize=8)
fig.suptitle(
    "Same frozen policy: third-gate low pass reproduced from both recorded starts", fontsize=12
)
fig.tight_layout()
fig.savefig(OUT / "summary.png", dpi=170)
fig.savefig(OUT / "summary.pdf")
plt.close(fig)
# Reproducibility manifest and totals.
import importlib.metadata as md, hashlib

manifest = {
    "date": "2026-11-17",
    "repo_commit": "d27923b",
    "policy_files": {},
    "versions": {},
    "episode_counts": {},
    "notes": [
        "All baseline and disturbed trajectories start at takeoff; no gate-triggered interference in the scans.",
        "Physical terminal state is captured before the official crash teleport.",
        "Parameter fitting and outcome matching do not identify the actual onboard configuration.",
    ],
}
for name in ["config.json", "policy.msgpack"]:
    p = ROOT / "lsy_drone_racing/control/rl_race/deploy_model" / name
    manifest["policy_files"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
for pkg in ["crazyflow", "jax", "jaxlib", "numpy", "scipy", "mujoco", "flax"]:
    try:
        manifest["versions"][pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        pass
for phase in ["coarse", "extended", "focused", "validation"]:
    rs = read_metrics(phase)
    manifest["episode_counts"][phase] = len(rs)
(OUT / "reproducibility.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
