from rank import *
from paths import read_metrics, resolve_trajectory


def trend(a):
    valid = np.all(np.isfinite(a["seen_quat"]), axis=1) & np.all(
        np.isfinite(a["seen_ang_vel"]), axis=1
    )
    t = a["t"][valid]
    q = R.from_quat(a["seen_quat"][valid])
    w = a["seen_ang_vel"][valid]
    tm = (t[:-1] + t[1:]) / 2
    derived = (q[:-1].inv() * q[1:]).as_rotvec() / np.diff(t)[:, None]
    mask = (tm > 0.25) & (tm < min(3.2, t[-1] - 0.15))
    errs = []
    grid = np.linspace(-0.1, 0.2, 61)
    for lag in grid:
        pred = np.column_stack([np.interp(tm + lag, t, w[:, i]) for i in range(3)])
        errs.append(np.mean((pred[mask] - derived[mask]) ** 2))
    d = describe(a)
    tg = a["t"] - d["t2"]
    ids = np.flatnonzero((tg > 0.15) & (tg < 0.75) & np.isfinite(a["seen_ang_vel"][:, 1]))
    idx = ids[np.argmax(a["seen_ang_vel"][ids, 1])]
    return {
        "input_rate_relative_lag_s": float(grid[np.argmin(errs)]),
        "pitch_rate_peak": float(a["seen_ang_vel"][idx, 1]),
        "pitch_rate_peak_after_gate2_s": float(tg[idx]),
    }


rows = []
for phase in ["coarse", "extended", "focused", "validation"]:
    rows.extend(read_metrics(phase))
for scene in ["113735", "115247"]:
    real = REALS[scene]
    print("REAL", scene, trend(real))
    for key in [
        "bc60e02f023f",
        "ceab5bdf368b",
        "1fb719373e8d",
        "789a0031821f",
        "e0c56b11e59a",
        "1be1c4be8f86",
        "325bba60f881",
    ]:
        r = next((r for r in rows if r["scene"] == scene and r["id"] == key), None)
        if not r or r["g2g3_min_z"] is None:
            continue
        a = dict(np.load(resolve_trajectory(r["trajectory"])))
        d = describe(a)
        rd = DESCS[scene]
        grid = np.linspace(0, 1.0, 51)
        t = a["t"] - d["t2"]
        tr = real["t"] - rd["t2"]
        inp = {}
        for f in ["seen_ang_vel", "seen_vel", "seen_pos"]:
            valid = np.all(np.isfinite(a[f]), axis=1)
            v = np.column_stack([np.interp(grid, t[valid], a[f][valid, i]) for i in range(3)])
            vr = np.column_stack([np.interp(grid, tr, real[f][:, i]) for i in range(3)])
            inp[f + "_rmse"] = float(np.sqrt(np.mean((v - vr) ** 2)))
        print(scene, key, "z", r["g2g3_min_z"], trend(a), inp)
