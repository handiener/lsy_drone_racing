"""Check two actuator-only candidates near both recorded starts; no controller-gain changes."""

from __future__ import annotations
import sys
from harness import *
from rank import DESCS
import time

DEST = OUT / "jitter"
DEST.mkdir(exist_ok=True)
(DEST / "trajectories").mkdir(exist_ok=True)
presets = {
    "response55_cap070": Noise(rotor_rate=0.55, thrust_cap=0.70),
    "response50_cap075": Noise(rotor_rate=0.50, thrust_cap=0.75),
}
metrics_path = DEST / "metrics.jsonl"
rows = (
    [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if metrics_path.exists()
    else []
)
known = {(r["scene"], r["preset"], r["seed"]) for r in rows}
fields = [
    "pos",
    "quat",
    "vel",
    "ang_vel",
    "seen_pos",
    "seen_quat",
    "seen_vel",
    "seen_ang_vel",
    "requested",
    "applied",
    "motor_thrust",
    "rotor_rpm",
]
started = time.monotonic()
for scene in ["113735", "115247"]:
    runner = Runner(scene)
    original_reset = runner.reset

    def jitter_reset(cfg, seed):
        data, raw = original_reset(cfg, seed)
        if seed > 0:
            rng = np.random.default_rng(900000 + seed)
            delta_pos = rng.normal(size=3) * np.array([0.005, 0.005, 0.001])
            delta_rotation = rng.normal(size=3) * np.deg2rad(0.3)
            states = data.sim_data.states.replace(
                pos=jp.asarray(raw["pos"] + delta_pos, dtype=jp.float32)[None, None],
                quat=jp.asarray(
                    (R.from_quat(raw["quat"]) * R.from_rotvec(delta_rotation)).as_quat(),
                    dtype=jp.float32,
                )[None, None],
            )
            data = data.replace(sim_data=data.sim_data.replace(states=states))
            data = rc._reset_env_data(data)
            data = data.replace(
                sim_data=data.sim_data.replace(
                    controls=data.sim_data.controls.replace(
                        body_rate=data.sim_data.controls.body_rate.replace(cmd=states.quat)
                    )
                )
            )
            raw = {k: np.asarray(v[0, 0]) for k, v in rc.obs(data).items()}
        return data, raw

    runner.reset = jitter_reset
    for name, cfg in presets.items():
        for seed in range(21):
            if (scene, name, seed) in known:
                continue
            metric, records = runner.run(cfg, seed)
            metric.update(
                preset=name,
                initial_position_sd_m=[0.005, 0.005, 0.001] if seed else [0, 0, 0],
                initial_rotation_sd_deg=0.3 if seed else 0,
            )
            for field in ["t", "pos", "quat", "vel", "ang_vel"]:
                assert np.isfinite(np.asarray([r[field] for r in records])).all(), (
                    scene,
                    name,
                    seed,
                    field,
                )
            ids = [r for r in records if r["passed"] == 2]
            if ids:
                minimum = min(ids, key=lambda r: r["pos"][2])
                t2 = ids[0]["t"]
                ref = DESCS[scene]
                metric.update(
                    min_xy_error_to_real_m=float(
                        np.linalg.norm(np.asarray(minimum["pos"][:2]) - ref["minxy"])
                    ),
                    min_t_after_gate2_s=minimum["t"] - t2,
                    min_t_error_to_real_s=abs(minimum["t"] - t2 - ref["minrel"]),
                )
            path = DEST / "trajectories" / f"{scene}_{name}_{seed:02d}.npz"
            arrays = {
                "t": np.asarray([r["t"] for r in records], np.float32),
                "passed": np.asarray([r["passed"] for r in records], np.int16),
            }
            for field in fields:
                shape = np.asarray(records[0][field]).shape
                arrays[field] = np.asarray(
                    [r.get(field, np.full(shape, np.nan)) for r in records], np.float32
                )
            np.savez_compressed(path, **arrays)
            metric["trajectory"] = str(path.relative_to(DEST))
            rows.append(metric)
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(metric) + "\n")
            if seed == 0 or len(rows) % 10 == 0:
                print(
                    "PROGRESS",
                    len(rows),
                    "scene",
                    scene,
                    name,
                    "seed",
                    seed,
                    "g2g3_min_z",
                    metric["g2g3_min_z"],
                    "passed",
                    metric["passed"],
                    "seconds",
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
    runner.env.close()
(DEST / "metrics.json").write_text(json.dumps(rows, indent=2))
summary = []
for scene in ["113735", "115247"]:
    for name in presets:
        group = [r for r in rows if r["scene"] == scene and r["preset"] == name and r["seed"] > 0]
        g2 = [r for r in group if r["passed"] >= 2]
        low = [r for r in g2 if r["g2g3_min_z"] < 0.15]
        close = [
            r for r in low if r["min_xy_error_to_real_m"] < 0.3 and r["min_t_error_to_real_s"] < 0.3
        ]
        entry = {
            "scene": scene,
            "preset": name,
            "episodes": len(group),
            "reached_gate2": len(g2),
            "passed_gate3": sum(r["passed"] >= 3 for r in group),
            "finished_all5": sum(r["finished"] for r in group),
            "g2g3_below15cm": len(low),
            "g2g3_below1cm": sum(r["g2g3_min_z"] < 0.01 for r in g2),
            "near_gate3_below15cm": sum(
                r["near_gate3_min_z"] is not None and r["near_gate3_min_z"] < 0.15 for r in g2
            ),
            "low_with_xy_error_under30cm_and_t_error_under300ms": len(close),
            "g2g3_height_range_m": [
                min((r["g2g3_min_z"] for r in g2), default=None),
                max((r["g2g3_min_z"] for r in g2), default=None),
            ],
            "median_low_xy_error_m": float(np.median([r["min_xy_error_to_real_m"] for r in low]))
            if low
            else None,
        }
        summary.append(entry)
(DEST / "summary.json").write_text(json.dumps(summary, indent=2))
print("SUMMARY", json.dumps(summary), flush=True)
print("DONE", len(rows), flush=True)
