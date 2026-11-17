from harness_v1 import *
from dataclasses import replace
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--phase", default="coarse")
args = parser.parse_args()
OUT = OUT / args.phase
OUT.mkdir(exist_ok=True)
(OUT / "trajectories").mkdir(exist_ok=True)
configs = {}


def add(family, **kw):
    c = Noise(**kw)
    key = hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest()[:12]
    if key not in configs:
        configs[key] = (family, c)


for cap, tau in product(
    [0.8, 0.75, 0.72, 0.70, 0.68, 0.65, 0.62, 0.60, 0.56],
    [0, 0.035, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5],
):
    add("omega_lowpass_capacity", thrust_cap=cap, omega_tau=tau)
for cap, delay in product([0.8, 0.7, 0.68, 0.65], [1, 2, 3, 4, 5, 6, 8, 10, 15]):
    add("omega_delay_capacity", thrust_cap=cap, omega_delay=delay)
for cap, window in product([0.8, 0.7, 0.68], [3, 5, 7, 11, 16, 26]):
    add("omega_average_capacity", thrust_cap=cap, omega_average=window)
for cap, window, tau in product([0.8, 0.7, 0.68], [2, 3, 4, 6, 9, 13], [0, 0.07]):
    add("state_average_capacity", thrust_cap=cap, state_average=window, omega_tau=tau)
for cap, delay in product([0.8, 0.7, 0.68], [1, 2, 3, 4, 6, 8, 12]):
    add("state_delay_capacity", thrust_cap=cap, state_delay=delay)
for cap, ptau, wtau in product(
    [0.8, 0.7, 0.68], [0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15], [0.07, 0.15]
):
    add("pose_filter_capacity", thrust_cap=cap, pose_tau=ptau, omega_tau=wtau)
for cap, vtau, wtau in product(
    [0.8, 0.7, 0.68], [0.007, 0.02, 0.04, 0.07, 0.1, 0.15, 0.25, 0.4], [0.07, 0.15]
):
    add("velocity_filter_capacity", thrust_cap=cap, vel_tau=vtau, omega_tau=wtau)
for cap, ttau, wtau in product([0.8, 0.7, 0.68], [0.02, 0.04, 0.08, 0.12, 0.2, 0.3], [0.07, 0.15]):
    add("collective_filter_capacity", thrust_cap=cap, thrust_tau=ttau, omega_tau=wtau)
for cap, rate, wtau in product([0.8, 0.7, 0.68], [0.8, 0.6, 0.4, 0.25, 0.15], [0, 0.07, 0.15]):
    add("rotor_dynamics_capacity", thrust_cap=cap, rotor_rate=rate, omega_tau=wtau)
for cap, delay, wtau in product([0.8, 0.7, 0.68], [1, 2, 3, 4, 5], [0, 0.07, 0.15]):
    add("command_delay_capacity", thrust_cap=cap, command_delay=delay, omega_tau=wtau)
for cap, damp, stiff in product([0.8, 0.7], [0.5, 0.7, 1, 1.3, 1.6], [0.7, 1, 1.3]):
    add("inner_loop_gains", thrust_cap=cap, damping=damp, stiffness=stiff, omega_tau=0.07)
for cap, bias in product([0.8, 0.7, 0.68], [-6, -3, 0, 3, 6]):
    add("pose_frame_bias", thrust_cap=cap, pose_bias_pitch=bias, omega_tau=0.07)
manifest = [{"id": k, "family": f, "noise": asdict(c)} for k, (f, c) in configs.items()]
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
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
    [json.loads(l) for l in (OUT / "metrics.jsonl").read_text().splitlines()]
    if (OUT / "metrics.jsonl").exists()
    else []
)
start = __import__("time").monotonic()
known = {(r["scene"], r["id"]) for r in rows}
for scene in ["113735", "115247"]:
    runner = Runner(scene)
    for key, (family, cfg) in configs.items():
        if (scene, key) in known:
            continue
        dest = OUT / "trajectories" / f"{scene}_{key}.npz"
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
        with (OUT / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(m) + "\n")
        if m["near_floor"]:
            print("NEAR_FLOOR", json.dumps(m), flush=True)
        if len(rows) % 25 == 0:
            print(
                "PROGRESS",
                len(rows),
                "/",
                len(configs) * 2,
                "seconds",
                round(__import__("time").monotonic() - start, 1),
                "near_floor",
                sum(x["near_floor"] for x in rows),
                flush=True,
            )
    runner.env.close()
(OUT / "metrics.json").write_text(json.dumps(rows, indent=2))
print("DONE", len(rows), flush=True)
