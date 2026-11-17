from pathlib import Path
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from paths import BASE, ROOT, OUT, EVIDENCE, read_metrics, resolve_trajectory


def real_arrays(scene):
    p = next((ROOT / "runs").glob(f"deploy_*T{scene}.*Z"))
    es = [json.loads(l) for l in (p / "events.jsonl").read_text().splitlines()]
    oo = [e for e in es if e["kind"] == "observation"]
    aa = {e["step"]: e["action"] for e in es if e["kind"] == "action"}
    t0 = oo[0]["monotonic_ns"]
    arr = {
        "t": np.array([(e["monotonic_ns"] - t0) / 1e9 for e in oo]),
        "passed": np.array([e["obs"]["n_gates_passed"] for e in oo]),
        "requested": np.array([aa[e["step"]] for e in oo]),
    }
    for f in ["pos", "quat", "vel", "ang_vel"]:
        arr[f] = np.array([e["obs"][f] for e in oo])
        arr["seen_" + f] = arr[f].copy()
    return arr


REALS = {s: real_arrays(s) for s in ["113735", "115247"]}


def describe(a):
    ids = np.flatnonzero(a["passed"] == 2)
    if not len(ids):
        return None
    t2 = float(a["t"][ids[0]])
    idx = ids[a["pos"][ids, 2].argmin()]
    tilt = np.degrees(np.arccos(np.clip(R.from_quat(a["quat"]).as_matrix()[:, 2, 2], -1, 1)))
    return {
        "t2": t2,
        "minz": float(a["pos"][idx, 2]),
        "minxy": a["pos"][idx, :2],
        "minrel": float(a["t"][idx] - t2),
        "tilt": tilt,
    }


DESCS = {s: describe(a) for s, a in REALS.items()}


def score(row, a):
    rd = DESCS[row["scene"]]
    sd = describe(a)
    if sd is None:
        return 1e6, {}
    real = REALS[row["scene"]]
    t = a["t"] - sd["t2"]
    tr = real["t"] - rd["t2"]
    grid = np.linspace(0, 1.15, 59)

    # Event-aligned comparison; no time warping or fitted offsets.
    def samples(ar, tt, key, col):
        return np.interp(grid, tt, ar[key][:, col])

    zerr = np.sqrt(np.mean((samples(a, t, "pos", 2) - samples(real, tr, "pos", 2)) ** 2))
    vzerr = np.sqrt(np.mean((samples(a, t, "vel", 2) - samples(real, tr, "vel", 2)) ** 2))
    tilt = np.interp(grid, t, sd["tilt"])
    rtilt = np.interp(grid, tr, rd["tilt"])
    tilterr = np.sqrt(np.mean((tilt - rtilt) ** 2))
    # Ignore terminal NaN command in interpolation.
    amask = np.isfinite(a["requested"][:, 3])
    thrust = np.interp(grid, t[amask], a["requested"][amask, 3])
    rthrust = np.interp(grid, tr, real["requested"][:, 3])
    terr = np.sqrt(np.mean((thrust - rthrust) ** 2))
    xyerr = float(np.linalg.norm(sd["minxy"] - rd["minxy"]))
    g1 = row["gate_times"].get("1", 20)
    rg1 = float(real["t"][np.flatnonzero(real["passed"] >= 1)[0]])
    early = abs(g1 - rg1) / 0.25 + abs(sd["t2"] - rd["t2"]) / 0.35
    total = (
        2 * abs(sd["minz"] - rd["minz"]) / 0.15
        + xyerr / 0.35
        + abs(sd["minrel"] - rd["minrel"]) / 0.5
        + zerr / 0.2
        + vzerr / 1
        + tilterr / 30
        + terr / 0.25
        + early
    )
    return total, {
        "minxy_error": xyerr,
        "height_curve_rmse": float(zerr),
        "vz_curve_rmse": float(vzerr),
        "tilt_curve_rmse_deg": float(tilterr),
        "thrust_curve_rmse_N": float(terr),
        "early_timing_penalty": float(early),
    }


def main():
    rows = []
    for phase in sys.argv[1:] or ["coarse", "extended"]:
        rows.extend(read_metrics(phase))
    scored = []
    for row in rows:
        if row["g2g3_min_z"] is None:
            continue
        trajectory = resolve_trajectory(row["trajectory"])
        if trajectory is None:
            continue
        a = dict(np.load(trajectory))
        s, details = score(row, a)
        scored.append({**row, "similarity_score": s, **details})
    scored.sort(key=lambda r: r["similarity_score"])
    (OUT / "ranked.json").write_text(json.dumps(scored, indent=2))
    print(
        "TOTAL_METRICS",
        len(rows),
        "BUNDLED_OR_GENERATED_TRAJECTORIES_SCORED",
        len(scored),
        "near_gate3_under15cm",
        sum(r["near_gate3_min_z"] is not None and r["near_gate3_min_z"] < 0.15 for r in scored),
    )
    print("Bulk unbundled trajectories are skipped; regenerate full sweeps to score all rows.")
    for row in scored[:20]:
        print(
            row["scene"],
            row["id"],
            row["family"],
            "score",
            round(row["similarity_score"], 2),
            "minz",
            round(row["g2g3_min_z"], 3),
            "xyerr",
            round(row["minxy_error"], 2),
            "gates",
            row["passed"],
            "gate times",
            row["gate_times"],
        )


if __name__ == "__main__":
    main()
