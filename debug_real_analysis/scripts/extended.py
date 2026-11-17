from harness import *
from itertools import product
import time

DEST = OUT / "extended"
DEST.mkdir(exist_ok=True)
(DEST / "trajectories").mkdir(exist_ok=True)
configs = {}


def add(family, **kw):
    c = Noise(**kw)
    key = hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest()[:12]
    if key not in configs:
        configs[key] = (family, c)


for cap, g, w in product(
    [0.8, 0.74, 0.7, 0.68], [0, 0.002, 0.005, 0.008, 0.012, 0.02, 0.03], [0, 0.07, 0.15]
):
    add("onboard_gyro_filter", thrust_cap=cap, gyro_tau=g, omega_tau=w)
for cap, p, w in product([0.8, 0.7, 0.68], [0.002, 0.005, 0.01, 0.02], [0.07]):
    add("onboard_attitude_filter", thrust_cap=cap, onboard_pose_tau=p, omega_tau=w)
for cap, idle, w in product([0.8, 0.75, 0.72, 0.7, 0.68, 0.65], [0, 3000, 7000], [0, 0.07, 0.15]):
    add(
        "uniform_allocator_capacity",
        thrust_cap=cap,
        idle_pwm=idle,
        uniform_allocation=1.0,
        omega_tau=w,
    )
for cap, sub, w in product([0.8, 0.75, 0.7, 0.68], range(1, 10), [0, 0.07]):
    add("subcycle_command_delay", thrust_cap=cap, command_substeps=sub, omega_tau=w)
for cap, rate, w in product(
    [0.8, 0.75, 0.7, 0.68], [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], [0, 0.07, 0.15]
):
    add("rotor_boundary", thrust_cap=cap, rotor_rate=rate, omega_tau=w)
for cap, mass, inertia, drag in product(
    [0.8, 0.7], [0.8, 0.9, 1, 1.1, 1.2], [0.7, 1, 1.3], [0.5, 1, 2]
):
    add(
        "physical_params",
        thrust_cap=cap,
        mass_scale=mass,
        inertia_scale=inertia,
        drag_scale=drag,
        omega_tau=0.07,
    )
rng = np.random.default_rng(20261117)
choices = {
    "gyro_tau": [0.002, 0.005, 0.008, 0.012, 0.02],
    "onboard_pose_tau": [0.002, 0.005, 0.008],
    "command_substeps": [1, 2, 3, 4, 5, 6],
    "rotor_rate": [0.45, 0.55, 0.65, 0.75, 0.85, 1.15],
    "vel_tau": [0.007, 0.02, 0.04, 0.07],
    "state_average": [2, 3, 4],
    "pose_tau": [0.005, 0.01, 0.02],
    "motor_gain": [0.85, 0.9, 0.95],
    "mass_scale": [0.9, 1.05, 1.1],
    "inertia_scale": [0.7, 0.85, 1.15, 1.3],
    "drag_scale": [0.6, 1.4, 2.0],
    "uniform_allocation": [1.0],
    "damping": [0.7, 0.85, 1.15, 1.3],
    "pose_bias_pitch": [-3.0, 3.0],
    "vel_bias_z": [-0.2, 0.2],
    "omega_scale": [0.8, 1.2],
}
for i in range(1200):
    kw = {
        "omega_tau": float(rng.choice([0, 0.035, 0.07, 0.1, 0.15, 0.2])),
        "thrust_cap": float(rng.choice([0.66, 0.68, 0.7, 0.72, 0.75, 0.8])),
    }
    for key in rng.choice(list(choices), size=int(rng.integers(2, 5)), replace=False):
        kw[key] = rng.choice(choices[key]).item()
    add("combined_sparse", **kw)
manifest = [{"id": k, "family": f, "noise": asdict(c)} for k, (f, c) in configs.items()]
(DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
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
            print("GATE3_LOW", scene, key, "z", m["near_gate3_min_z"], "family", family, flush=True)
        if len(rows) % 50 == 0:
            print(
                "PROGRESS",
                len(rows),
                "/",
                len(configs) * 2,
                "seconds",
                round(time.monotonic() - start, 1),
                "near_floor",
                sum(x["near_floor"] for x in rows),
                "near_gate3_low",
                sum(
                    x["near_gate3_min_z"] is not None and x["near_gate3_min_z"] < 0.15 for x in rows
                ),
                flush=True,
            )
    runner.env.close()
(DEST / "metrics.json").write_text(json.dumps(rows, indent=2))
print("DONE", len(rows), flush=True)
