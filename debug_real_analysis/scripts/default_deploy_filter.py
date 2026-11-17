"""Offline default DiffFilter from the lockfile wheel, fed pose samples from 500 Hz physics.

Only motor response/capacity and offboard observation processing differ. ROS transport ages
and mocap sampling rates are explicit sensitivity assumptions, not measured hardware values.
"""

from __future__ import annotations
import sys
from pathlib import Path
from harness import *
from rank import DESCS
import argparse, time
from estimator_dependency import load_diff_filter
import harness as h

DEFAULT_TAU = (0.0001, 0.007, 0.09, 0.005, 0.07)
DEST = OUT / "default_deploy_filter"
DEST.mkdir(exist_ok=True)
(DEST / "trajectories").mkdir(exist_ok=True)


class PoseObserver:
    def __init__(self, initial, frequency, age):
        self.filter = load_diff_filter()(DEFAULT_TAU, jit=False)
        self.filter.set_state(initial["pos"].astype(np.float64), initial["quat"].astype(np.float64))
        # Warm-start from the first recorded estimate; no artificial zero-velocity startup.
        self.filter.data = self.filter.data.replace(
            vel=initial["vel"].astype(np.float64),
            ang_vel=initial["ang_vel"].astype(np.float64),
            vel_prev=initial["vel"].astype(np.float64),
            vel_meas=initial["vel"].astype(np.float64),
        )
        self.filter._update_estimate()
        self.frequency = frequency
        self.age = age
        self.next_sample = 1
        self.last_sample = 0.0
        self.positions = [np.array(initial["pos"], np.float64)]
        self.quaternions = [np.array(initial["quat"], np.float64)]

    def add_segment(self, positions, quaternions):
        self.positions.extend(np.asarray(positions)[1:])
        self.quaternions.extend(np.asarray(quaternions)[1:])

    def state(self, now):
        while self.next_sample / self.frequency <= now - self.age + 1e-10:
            ts = self.next_sample / self.frequency
            index = ts / 0.002
            lo = min(int(np.floor(index + 1e-9)), len(self.positions) - 1)
            hi = min(lo + 1, len(self.positions) - 1)
            frac = max(0.0, min(1.0, index - lo))
            pos = (1 - frac) * self.positions[lo] + frac * self.positions[hi]
            r0 = R.from_quat(self.quaternions[lo])
            q = (
                r0
                * R.from_rotvec(frac * (r0.inv() * R.from_quat(self.quaternions[hi])).as_rotvec())
            ).as_quat()
            self.filter.predict(ts - self.last_sample)
            self.filter.correct(pos, q)
            self.last_sample = ts
            self.next_sample += 1
        estimate = self.filter._estimate
        return {
            k: np.asarray(getattr(estimate, k), np.float32)
            for k in ["pos", "quat", "vel", "ang_vel"]
        }


class PhysicsRecorder:
    def __init__(self, runner):
        self.observer = None
        simstep = runner.env.sim.build_step_fn()
        apply = runner.env.build_apply_action_fn()
        contacts = runner.env.build_contact_check_fn()

        @jax.jit
        def advance(data, action, previous, substeps):
            positions = (
                jp.zeros((11, 3), dtype=data.sim_data.states.pos.dtype)
                .at[0]
                .set(data.sim_data.states.pos[0, 0])
            )
            quaternions = (
                jp.zeros((11, 4), dtype=data.sim_data.states.quat.dtype)
                .at[0]
                .set(data.sim_data.states.quat[0, 0])
            )

            def body(i, carry):
                sd, pp, qq = carry
                sd = simstep(sd, 1)
                return (
                    sd,
                    pp.at[i + 1].set(sd.states.pos[0, 0]),
                    qq.at[i + 1].set(sd.states.quat[0, 0]),
                )

            data = jax.lax.cond(substeps > 0, lambda d: apply(previous, d), lambda d: d, data)
            sd, positions, quaternions = jax.lax.fori_loop(
                0, substeps, body, (data.sim_data, positions, quaternions)
            )
            data = data.replace(sim_data=sd)
            data = apply(action, data)
            sd, positions, quaternions = jax.lax.fori_loop(
                substeps, 10, body, (data.sim_data, positions, quaternions)
            )
            data = data.replace(sim_data=sd)
            hit = contacts(data)
            data = rc._update_disabled_drones(data, hit)
            data = rc._update_visited_objects(data)
            data = rc._update_target_gates(data)
            data = rc._mark_drones_for_reset(data)
            data = data.replace(steps=data.steps + 1)
            return data, rc.obs(data), hit, positions, quaternions

        self.compiled = advance

    def __call__(self, *args):
        data, obs, hit, positions, quaternions = self.compiled(*args)
        if self.observer is not None:
            self.observer.add_segment(positions, quaternions)
        return data, obs, hit


def save_trace(records, path):
    arrays = {
        "t": np.asarray([r["t"] for r in records], np.float32),
        "passed": np.asarray([r["passed"] for r in records], np.int16),
    }
    for field in [
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
    ]:
        shape = np.asarray(records[0][field]).shape
        arrays[field] = np.asarray(
            [r.get(field, np.full(shape, np.nan)) for r in records], np.float32
        )
    for field in ["t", "pos", "quat", "vel", "ang_vel"]:
        assert np.isfinite(arrays[field]).all(), (path, field)
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "grid"], default="grid")
    args = parser.parse_args()
    native_wrapper = h.DisturbanceWrapper
    if args.phase == "smoke":
        runner = Runner("113735")
        reference, refrecords = runner.run(Noise(rotor_rate=0.8, thrust_cap=0.7))
        recorder = PhysicsRecorder(runner)
        runner.advance = recorder
        check, records = runner.run(Noise(rotor_rate=0.8, thrust_cap=0.7))
        for key in ["g2g3_min_z", "duration", "passed", "finished", "gate_times"]:
            assert reference[key] == check[key], (key, reference[key], check[key])
        for field in ["pos", "quat", "vel", "ang_vel", "requested"]:
            assert all(
                np.array_equal(a.get(field), b.get(field)) for a, b in zip(refrecords, records)
            ), field
        assert len(refrecords) == len(records)
        print("PHYSICS_RECORDER_IDENTICAL", json.dumps(check), flush=True)
        jobs = [(300, 0.0, 0.8), (300, 0.0, 0.7)]
        scenes = ["113735"]
    else:
        scenes = ["113735", "115247"]
        jobs = [
            (freq, age, cap)
            for freq in [300, 120]
            for age in [0.0, 0.01, 0.02, 0.03]
            for cap in [0.68, 0.70, 0.72, 0.75, 0.76, 0.78, 0.80]
        ]
    path = DEST / ("smoke_metrics.jsonl" if args.phase == "smoke" else "metrics.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    known = {
        (r["scene"], r["mocap_hz"], r["measurement_transport_age_s"], r["noise"]["thrust_cap"])
        for r in rows
    }
    start = time.monotonic()
    for scene in scenes:
        if args.phase != "smoke":
            runner = Runner(scene)
            recorder = PhysicsRecorder(runner)
            runner.advance = recorder
        for frequency, age, cap in jobs:
            if (scene, frequency, age, cap) in known:
                continue

            class DefaultObserver(native_wrapper):
                def __init__(self, cfg, initial):
                    super().__init__(cfg, initial)
                    self.observer = PoseObserver(initial, frequency, age)
                    recorder.observer = self.observer

                def observe(self, raw, step):
                    seen = {k: np.array(v, copy=True) for k, v in raw.items()}
                    seen.update(self.observer.state(step * 0.02))
                    return seen

            h.DisturbanceWrapper = DefaultObserver
            cfg = Noise(rotor_rate=0.8, thrust_cap=cap)
            result, records = runner.run(cfg)
            result.update(
                mocap_hz=frequency,
                measurement_transport_age_s=age,
                default_tau_s=DEFAULT_TAU,
                filter_implementation="locked drone-estimators 0.1.0 DiffFilter, jit=False numpy64",
                policy_event_logic="physical counter/reveals unchanged",
            )
            ids = [r for r in records if r["passed"] == 2]
            if ids:
                minimum = min(ids, key=lambda r: r["pos"][2])
                ref = DESCS[scene]
                result.update(
                    min_xy_error_to_real_m=float(
                        np.linalg.norm(np.asarray(minimum["pos"][:2]) - ref["minxy"])
                    ),
                    min_t_after_gate2_s=minimum["t"] - ids[0]["t"],
                    min_t_error_to_real_s=abs(minimum["t"] - ids[0]["t"] - ref["minrel"]),
                )
            key = f"{scene}_hz{frequency}_age{round(age * 1000):02d}_cap{round(cap * 100):02d}"
            target = DEST / "trajectories" / f"{key}.npz"
            save_trace(records, target)
            result["trajectory"] = str(target.relative_to(DEST))
            rows.append(result)
            with path.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            if args.phase == "smoke" or (
                result["near_gate3_min_z"] is not None and result["near_gate3_min_z"] < 0.15
            ):
                print(
                    "RESULT",
                    key,
                    "min",
                    result["g2g3_min_z"],
                    "passed",
                    result["passed"],
                    "near3",
                    result["near_gate3_min_z"],
                    flush=True,
                )
            if len(rows) % 10 == 0:
                print(
                    "PROGRESS",
                    len(rows),
                    "/",
                    len(scenes) * len(jobs),
                    "seconds",
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
        runner.env.close()
    h.DisturbanceWrapper = native_wrapper
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
