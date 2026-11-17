from harness import *
from itertools import product
import time

DEST = OUT / "focused"
DEST.mkdir(exist_ok=True)
(DEST / "trajectories").mkdir(exist_ok=True)
configs = {}


def add(family, **kw):
    c = Noise(**kw)
    key = hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest()[:12]
    if key not in configs:
        configs[key] = (family, c)


base = {
    "omega_tau": 0.1,
    "pose_tau": 0.02,
    "thrust_cap": 0.72,
    "damping": 0.7,
    "command_substeps": 3,
    "drag_scale": 1.4,
}
for bits in product([False, True], repeat=len(base)):
    kw = {k: v for (k, v), on in zip(base.items(), bits) if on}
    add("all_subset_ablation", **kw)
for cap, omega, pose, sub, damp, drag in product(
    [0.68, 0.7, 0.72, 0.74, 0.76],
    [0.05, 0.07, 0.1],
    [0.01, 0.02, 0.03],
    [0, 2, 3, 4],
    [0.7, 1.0],
    [1.0, 1.4],
):
    # A bounded local grid around the first spatially close ground case, including default gains.
    add(
        "local_grid",
        thrust_cap=cap,
        omega_tau=omega,
        pose_tau=pose,
        command_substeps=sub,
        damping=damp,
        drag_scale=drag,
    )
(DEST / "manifest.json").write_text(
    json.dumps(
        [{"id": k, "family": f, "noise": asdict(c)} for k, (f, c) in configs.items()], indent=2
    )
)
print("CONFIGS", len(configs), "EPISODES", len(configs) * 2, flush=True)
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
    "obstacles_pos",
]
rows = (
    [json.loads(l) for l in (DEST / "metrics.jsonl").read_text().splitlines()]
    if (DEST / "metrics.jsonl").exists()
    else []
)
known = {(r["scene"], r["id"]) for r in rows}
start = time.monotonic()
for scene in ["113735", "115247"]:
    runner = Runner(scene)
    for key, (family, cfg) in configs.items():
        if (scene, key) in known:
            continue
        dest = DEST / "trajectories" / f"{scene}_{key}.npz"
        m, rs = runner.run(cfg)
        m.update(id=key, family=family, trajectory=str(dest))
        rows.append(m)
        arr = {
            "t": np.array([r["t"] for r in rs], np.float32),
            "passed": np.array([r["passed"] for r in rs], np.int16),
        }
        for f in fields:
            shape = np.asarray(rs[0][f]).shape
            arr[f] = np.asarray([r.get(f, np.full(shape, np.nan)) for r in rs], np.float32)
        np.savez_compressed(dest, **arr)
        with (DEST / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(m) + "\n")
        if m["near_gate3_min_z"] is not None and m["near_gate3_min_z"] < 0.15:
            print(
                "GATE3_LOW",
                scene,
                key,
                "z",
                m["near_gate3_min_z"],
                "gate times",
                m["gate_times"],
                flush=True,
            )
        if len(rows) % 100 == 0:
            print(
                "PROGRESS",
                len(rows),
                "/",
                len(configs) * 2,
                "seconds",
                round(time.monotonic() - start, 1),
                flush=True,
            )
    runner.env.close()
(DEST / "metrics.json").write_text(json.dumps(rows, indent=2))
print("DONE", len(rows), flush=True)
