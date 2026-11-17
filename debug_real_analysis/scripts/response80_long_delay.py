"""Only rotor response, policy input filtering and motor capacity vary; all gains stay nominal."""

from __future__ import annotations
import sys
from harness import *
from rank import DESCS
import time

DEST = OUT / "response80_long_delay"
DEST.mkdir(exist_ok=True)
(DEST / "trajectories").mkdir(exist_ok=True)
caps = [0.68, 0.70, 0.72, 0.75, 0.76, 0.78, 0.80]
profiles = [
    (scope, delay) for scope in ["all_kinematics", "omega_only"] for delay in [0.040, 0.050, 0.100]
]
metrics_path = DEST / "metrics.jsonl"
rows = (
    [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if metrics_path.exists()
    else []
)
known = {(r["scene"], r["profile"], r["input_delay_s"], r["noise"]["thrust_cap"]) for r in rows}
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
start = time.monotonic()
for scene in ["113735", "115247"]:
    runner = Runner(scene)
    for profile, tau in profiles:
        for cap in caps:
            if (scene, profile, tau, cap) in known:
                continue
            steps = 2 if tau in [0.040, 0.050] else 5
            window = 2 if tau == 0.050 else 1
            cfg = Noise(
                rotor_rate=0.8,
                thrust_cap=cap,
                **(
                    {"state_delay": steps, "state_average": window}
                    if profile == "all_kinematics"
                    else {"omega_delay": steps, "omega_average": window}
                ),
            )
            nominal = asdict(Noise())
            assert all(
                value == nominal[key]
                for key, value in asdict(cfg).items()
                if key
                not in {
                    "rotor_rate",
                    "thrust_cap",
                    "state_delay",
                    "state_average",
                    "omega_delay",
                    "omega_average",
                }
            )
            metric, records = runner.run(cfg)
            metric.update(profile=profile, input_delay_s=tau)
            for field in ["t", "pos", "quat", "vel", "ang_vel"]:
                assert np.isfinite(np.asarray([r[field] for r in records])).all(), (
                    scene,
                    profile,
                    tau,
                    cap,
                    field,
                )
            passed2 = [r for r in records if r["passed"] == 2]
            if passed2:
                minimum = min(passed2, key=lambda r: r["pos"][2])
                ref = DESCS[scene]
                metric.update(
                    min_xy_error_to_real_m=float(
                        np.linalg.norm(np.asarray(minimum["pos"][:2]) - ref["minxy"])
                    ),
                    min_t_after_gate2_s=minimum["t"] - passed2[0]["t"],
                    min_t_error_to_real_s=abs(minimum["t"] - passed2[0]["t"] - ref["minrel"]),
                )
            key = f"{scene}_{profile}_delay{round(tau * 1000):03d}_cap{round(cap * 100):02d}"
            path = DEST / "trajectories" / f"{key}.npz"
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
            if metric["near_gate3_min_z"] is not None and metric["near_gate3_min_z"] < 0.15:
                print(
                    "LOW_GATE3",
                    key,
                    "z",
                    metric["g2g3_min_z"],
                    "passed",
                    metric["passed"],
                    "xyerror",
                    metric["min_xy_error_to_real_m"],
                    flush=True,
                )
            if len(rows) % 10 == 0:
                print(
                    "PROGRESS",
                    len(rows),
                    "/",
                    len(profiles) * len(caps) * 2,
                    "seconds",
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
    runner.env.close()
(DEST / "metrics.json").write_text(json.dumps(rows, indent=2))
summary = []
for scene in ["113735", "115247"]:
    for profile, tau in profiles:
        group = [
            r
            for r in rows
            if r["scene"] == scene and r["profile"] == profile and r["input_delay_s"] == tau
        ]
        summary.append(
            {
                "scene": scene,
                "profile": profile,
                "input_delay_s": tau,
                "rows": [
                    {
                        "cap": r["noise"]["thrust_cap"],
                        "passed": r["passed"],
                        "min_g2g3": r["g2g3_min_z"],
                        "min_near3": r.get("near_gate3_min_z"),
                        "finished": r["finished"],
                    }
                    for r in group
                ],
            }
        )
(DEST / "summary.json").write_text(json.dumps(summary, indent=2))
print("SUMMARY", json.dumps(summary), flush=True)
print("DONE", len(rows), flush=True)
