"""Resolve the 50 ms delay at 500 Hz physics timestamps, avoiding policy-rate interpolation."""

from __future__ import annotations
import sys
from harness import *
from rank import DESCS
import harness as h

DEST = OUT / "response80_delay50_physics"
DEST.mkdir(exist_ok=True)
(DEST / "trajectories").mkdir(exist_ok=True)
Base = h.DisturbanceWrapper


class Recorder:
    def __init__(self, runner):
        self.history = []
        step = runner.env.sim.build_step_fn()
        apply = runner.env.build_apply_action_fn()
        contacts = runner.env.build_contact_check_fn()

        def pack(sd):
            return jp.concatenate(
                [
                    sd.states.pos[0, 0],
                    sd.states.quat[0, 0],
                    sd.states.vel[0, 0],
                    sd.states.ang_vel[0, 0],
                ]
            )

        @jax.jit
        def advance(data, action, previous, substeps):
            trace = (
                jp.zeros((11, 13), dtype=data.sim_data.states.pos.dtype)
                .at[0]
                .set(pack(data.sim_data))
            )

            def body(i, carry):
                sd, trace = carry
                sd = step(sd, 1)
                return sd, trace.at[i + 1].set(pack(sd))

            data = jax.lax.cond(substeps > 0, lambda d: apply(previous, d), lambda d: d, data)
            sd, trace = jax.lax.fori_loop(0, substeps, body, (data.sim_data, trace))
            data = data.replace(sim_data=sd)
            data = apply(action, data)
            sd, trace = jax.lax.fori_loop(substeps, 10, body, (data.sim_data, trace))
            data = data.replace(sim_data=sd)
            hit = contacts(data)
            data = rc._update_disabled_drones(data, hit)
            data = rc._update_visited_objects(data)
            data = rc._update_target_gates(data)
            data = rc._mark_drones_for_reset(data)
            data = data.replace(steps=data.steps + 1)
            return data, rc.obs(data), hit, trace

        self.advance = advance

    def __call__(self, *args):
        data, obs, hit, trace = self.advance(*args)
        self.history.extend(np.asarray(trace)[1:])
        return data, obs, hit


rows = []
for scene in ["113735", "115247"]:
    runner = Runner(scene)
    recorder = Recorder(runner)
    runner.advance = recorder
    for scope, delay in [("omega_only", 0.04), ("omega_only", 0.05), ("all_kinematics", 0.05)]:
        for cap in [0.68, 0.70, 0.72, 0.75, 0.76, 0.78, 0.80]:

            class Delayed(Base):
                def __init__(self, cfg, initial):
                    super().__init__(cfg, initial)
                    recorder.history = [
                        np.concatenate([initial[k] for k in ["pos", "quat", "vel", "ang_vel"]])
                    ]

                def observe(self, raw, policy_step):
                    seen = {k: np.array(v, copy=True) for k, v in raw.items()}
                    index = max(0, policy_step * 10 - round(delay / 0.002))
                    old = recorder.history[index]
                    if scope == "all_kinematics":
                        seen.update(pos=old[:3].copy(), quat=old[3:7].copy(), vel=old[7:10].copy())
                    seen["ang_vel"] = old[10:13].copy()
                    return seen

            h.DisturbanceWrapper = Delayed
            result, records = runner.run(Noise(rotor_rate=0.8, thrust_cap=cap))
            result.update(profile=scope, input_delay_s=delay, input_history_hz=500)
            ids = [r for r in records if r["passed"] == 2]
            if ids:
                low = min(ids, key=lambda r: r["pos"][2])
                ref = DESCS[scene]
                result.update(
                    min_xy_error_to_real_m=float(
                        np.linalg.norm(np.asarray(low["pos"][:2]) - ref["minxy"])
                    ),
                    min_t_after_gate2_s=low["t"] - ids[0]["t"],
                )
            key = f"{scene}_{scope}_delay{round(delay * 1000):03d}_cap{round(cap * 100):02d}"
            path = DEST / "trajectories" / f"{key}.npz"
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
                assert np.isfinite(arrays[field]).all()
            np.savez_compressed(path, **arrays)
            result["trajectory"] = str(path.relative_to(DEST))
            rows.append(result)
            if result["near_floor"]:
                print(
                    "LOW",
                    key,
                    "height",
                    result["g2g3_min_z"],
                    "pos",
                    result["g2g3_min_pos"],
                    flush=True,
                )
    runner.env.close()
h.DisturbanceWrapper = Base
(DEST / "metrics.json").write_text(json.dumps(rows, indent=2))
print("DONE", len(rows), flush=True)
