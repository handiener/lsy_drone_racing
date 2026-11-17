from harness import *
from dataclasses import replace
import time

DEST = OUT / "validation"
DEST.mkdir(exist_ok=True)
(DEST / "trajectories").mkdir(exist_ok=True)
B = Noise(
    omega_tau=0.05, pose_tau=0.01, thrust_cap=0.76, damping=0.7, command_substeps=2, drag_scale=1.4
)
A = Noise(
    omega_tau=0.1, pose_tau=0.02, thrust_cap=0.72, damping=0.7, command_substeps=3, drag_scale=1.4
)
M = Noise(omega_tau=0.1, thrust_cap=0.72, damping=0.7)
presets = {
    "baseline": Noise(),
    "observer_only": Noise(omega_tau=0.05, pose_tau=0.01, command_substeps=2),
    "actuation_only": Noise(thrust_cap=0.76, damping=0.7, drag_scale=1.4),
    "B_full": B,
    "A_full": A,
    "minimal_three": M,
}
for field, value in [
    ("omega_tau", 0.0),
    ("pose_tau", 0.0),
    ("thrust_cap", 0.8),
    ("damping", 1.0),
    ("command_substeps", 0),
    ("drag_scale", 1.0),
]:
    presets["B_without_" + field] = replace(B, **{field: value})
rows = (
    [json.loads(l) for l in (DEST / "metrics.jsonl").read_text().splitlines()]
    if (DEST / "metrics.jsonl").exists()
    else []
)
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
start = time.monotonic()
known = {(r["scene"], r["preset"], r["seed"]) for r in rows}
for scene in ["113735", "115247"]:
    runner = Runner(scene)
    original_reset = runner.reset

    def reset_with_jitter(cfg, seed):
        data, raw = original_reset(cfg, seed)
        if seed > 0:
            rng = np.random.default_rng(900000 + seed)
            dp = rng.normal(size=3) * np.array([0.005, 0.005, 0.001])
            dr = rng.normal(size=3) * np.deg2rad(0.3)
            pos = raw["pos"] + dp
            quat = (R.from_quat(raw["quat"]) * R.from_rotvec(dr)).as_quat()
            st = data.sim_data.states.replace(
                pos=jp.asarray(pos, dtype=jp.float32)[None, None],
                quat=jp.asarray(quat, dtype=jp.float32)[None, None],
            )
            data = data.replace(sim_data=data.sim_data.replace(states=st))
            data = rc._reset_env_data(data)
            data = data.replace(
                sim_data=data.sim_data.replace(
                    controls=data.sim_data.controls.replace(
                        body_rate=data.sim_data.controls.body_rate.replace(cmd=st.quat)
                    )
                )
            )
            raw = {k: np.asarray(v[0, 0]) for k, v in rc.obs(data).items()}
        return data, raw

    runner.reset = reset_with_jitter
    jobs = [(name, cfg, 0) for name, cfg in presets.items()]
    jobs += [
        (name, presets[name], seed)
        for name in ["baseline", "B_full", "A_full"]
        for seed in range(1, 21)
    ]
    for name, cfg, seed in jobs:
        if (scene, name, seed) in known:
            continue
        key = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]
        dest = DEST / "trajectories" / f"{scene}_{name}_{seed:02d}.npz"
        m, rs = runner.run(cfg, seed)
        m.update(
            id=key,
            preset=name,
            family="validation",
            trajectory=str(dest),
            initial_position_sd_m=[0.005, 0.005, 0.001] if seed else [0, 0, 0],
            initial_rotation_sd_deg=0.3 if seed else 0,
        )
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
        if seed == 0:
            print(
                "ABLATION",
                scene,
                name,
                "z",
                m["g2g3_min_z"],
                "passed",
                m["passed"],
                "times",
                m["gate_times"],
                flush=True,
            )
        if len(rows) % 20 == 0:
            print("PROGRESS", len(rows), "elapsed", round(time.monotonic() - start, 1), flush=True)
    runner.env.close()
(DEST / "metrics.json").write_text(json.dumps(rows, indent=2))
print("DONE", len(rows), flush=True)
