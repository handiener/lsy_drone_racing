# Return-to-home requests out-of-bounds braking targets and loses altitude after the final gate

## Problem

On two recorded hardware flights, the controller completed all five gate crossings and the deployment runner submitted return-to-home commands immediately afterward. No further policy attitude commands were submitted after the first return command. During the first return maneuver, the drone descended to approximately floor height and flew beyond the configured y boundary. It subsequently recovered, returned toward home, and performed the planned landing.

There is a concrete target-generation issue: **the first return/braking target itself is outside the configured flight bounds**. The policy also hands over a demanding state (about 3 m/s, 67–69° tilt, and downward velocity). Please investigate both the return trajectory constraints and the continuity of the low-level-to-high-level handoff.

“Return command submitted” here means the cflib2 send operation returned and the host recorder logged `go_to_sent`. It does **not** establish that the firmware received/executed the command correctly. The logs establish that the runner entered the return path, not that return navigation succeeded.

## Environment and reproduction context

- Repository: `handiener/lsy_drone_racing`.
- Recorded base commit: `dbcb5cb13d9d12257f1829df7a0054cbe4b7d95d`, **dirty working tree**. The deployment recorder was a local modification; the commit alone is not an exact source snapshot.
- Date: 2026-11-17. Run-directory timestamps are UTC.
- Config: `level2.toml`; controller: `rl_race_controller.py`; `env.control_mode = "attitude"`; policy/environment frequency: 50 Hz.
- Drone: `cf2`, radio channel 80, configured model `cf21B_500`, configured mass 0.0434 kg.
- `real_track_objects = true`; track/start checks enabled.
- Gate order: **[1, 2, 3, 4, 2]** (one-based gate numbers; the last crossing is gate 2 again).
- Nominal home/start: **[-1.5, 0.75, 0.01] m**.
- Bounds: x **[-2.5, 2.5] m**, y **[-1.5, 1.5] m**, z **[-0.001, 2.0] m**.
- The two flights used identical exported policy/config hashes.

The recorded invocation is equivalent to:

```bash
pixi shell -e deploy
python scripts/deploy.py --config level2.toml --record=True
```

Mocap and the offboard estimator must already be running. The precise configuration used for each flight is embedded in its `metadata.json`; use that snapshot rather than assuming the current `level2.toml` is unchanged.

## Expected behavior

After the last crossing, the return path should account for current velocity/attitude and the available flight volume, arrest motion without unintended near-ground descent, and remain inside the configured bounds before returning and landing. If the current state cannot be recovered within those constraints, this should be handled explicitly rather than silently requesting an out-of-bounds target.

This report does not assume that every aggressive terminal state is physically recoverable; assessing that is part of the investigation.

## Recorded outcomes

Times in this table are relative to the final gate / `race_end` event unless otherwise stated.

| Evidence | Flight A | Flight B |
| --- | --- | --- |
| Run | `deploy_20261117T100017.801841Z` | `deploy_20261117T100827.697140Z` |
| Race time | 5.1227 s | 5.3048 s |
| Gate counter at terminal transition | 5 / 5 | 5 / 5 |
| First return command after race_end | +5.470 ms | +5.414 ms |
| Policy attitude commands after first return command | **0** | **0** |
| First requested target [x, y, z], m | [0.54375, **1.62474**, 1.75000] | [0.51694, **1.63124**, 1.75000] |
| Target exceeds y bound by | **0.12474 m** | **0.13124 m** |
| First measured y > 1.5 m, relative to first return command | +1.185 s | +0.599 s |
| Near-floor minimum in first 3 s of return | z = **0.05598 m**, +0.658 s after race_end | z = **0.04955 m**, +0.804 s after race_end |
| Position at this minimum, m | [-0.00342, 1.26061, 0.05598] | [-0.08467, 1.60248, 0.04955] |
| Preflight resting mocap z | 0.04412 m | 0.04431 m |
| Maximum measured y in first 3 s of return | **2.18158 m** | **2.11385 m** |
| Next command: home at 1.75 m | +3.006 s | +3.006 s |
| Next command: planned landing at z=0.05 m | +8.052 s | +8.050 s |
| Emergency stop submitted | +11.054 s | +11.054 s |

The near-floor minima occurred well before the planned landing command. We observed ground contact during these deployments; the mocap minima are consistent with that observation, but the recorder does not independently measure contact. Both flights subsequently climbed back to roughly 1.7 m.

## Last recorded state before return

These are **offboard estimator observations from the final environment transition**, not onboard firmware state and not a separately recorded snapshot at the exact first go-to call. The return method reads the state again a few milliseconds later.

Position and velocity use the environment's world frame. Quaternions are **[x, y, z, w]**. Angular velocity below is the environment's `ang_vel` field in rad/s.

| Field | Flight A | Flight B |
| --- | --- | --- |
| Position [m] | [1.152827, 0.877749, 1.256350] | [1.127300, 0.860827, 1.282388] |
| Velocity [m/s] | [-1.836305, 2.252087, **-0.803504**] | [-1.874223, 2.365706, **-0.565585**] |
| Speed [m/s] | **3.0149** | **3.0707** |
| Quaternion xyzw | [0.287024, -0.489205, 0.001747, 0.823586] | [0.207594, -0.513896, -0.045633, 0.831104] |
| Tilt from vertical | **69.11°** | **67.32°** |
| `ang_vel` [rad/s] | [3.282894, 0.859463, -0.290692] | [2.801731, 0.749699, -0.141080] |
| Last policy command [roll, pitch, third attitude field, total thrust], [rad, rad, rad, N] | [1.109500, -1.018443, -0.210154, 0.787444] | [0.527281, -0.886906, 0.184985, 0.542948] |

The current wrapper sets `flightmode.stabModeYaw = 1`; interpret the third attitude field according to the configured firmware mode. The recorder's `yaw_rate_deg_s` label is not sufficient to establish that the firmware was in yaw-rate mode.

The terminal velocity and tilt likely make the transition difficult, but these recordings alone do not establish the exact onboard cause of the altitude loss. They also do not exclude an influence from the policy's terminal state.

## Track geometry recorded at reset

Measured gate centers, in meters:

| Gate | Flight A [x, y, z] | Flight B [x, y, z] |
| --- | --- | --- |
| 1 | [0.49613, 0.27518, 0.67006] | [0.49614, 0.27514, 0.66998] |
| 2, also final gate | [1.08245, 0.74989, 1.21682] | [1.08227, 0.74985, 1.21673] |
| 3 | [-1.00192, -0.24703, 0.66079] | [-1.00190, -0.24705, 0.66070] |
| 4 | [0.02511, -0.76860, 1.21290] | [0.02511, -0.76861, 1.21284] |

Measured obstacle positions, in meters:

| Obstacle | Flight A [x, y, z] | Flight B [x, y, z] |
| --- | --- | --- |
| 1 | [-0.06039, 0.76145, 0.43150] | [-0.06126, 0.76135, 0.43141] |
| 2 | [1.07452, 0.28028, 1.57967] | [1.07437, 0.28012, 1.57959] |
| 3 | [-1.46354, -0.30606, 1.54186] | [-1.46355, -0.30608, 1.54181] |
| 4 | [-0.49629, -0.78178, 1.54579] | [-0.49629, -0.78183, 1.54579] |

Full gate orientations and the complete configuration are preserved in the attached JSON/raw logs. The obstacle-height discrepancy and mid-race behavior are separate investigations; this issue concerns post-finish return behavior.

## Relevant code path

1. `scripts/deploy.py`: breaks the race loop on `terminated`, `truncated`, or `controller_finished`; calls `env.close()` in `finally`.
2. `lsy_drone_racing/envs/real_race_env.py::RealRaceCoreEnv.close`: calls `return_to_start()` if the drone has taken off and an active `KeyboardInterrupt` is not bypassing return.
3. `lsy_drone_racing/utils/crazyflie.py::_prepare_high_level`: submits a stop setpoint, switches commander level, submits `send_notify_setpoint_stop(0)`, and sets `commander.enHighLevel = 1`.
4. `Crazyflie.return_to_start`: generates the first target using:

```python
vel_norm = np.linalg.norm(initial_obs["vel"])
break_pos = initial_obs["pos"].copy()
if vel_norm > 1e-6:
    break_pos += initial_obs["vel"] / vel_norm * breaking_distance
break_pos[2] = return_height
self._run(self._go_to, break_pos, 0.0, breaking_duration)
```

Defaults are `breaking_distance=1.0 m`, `return_height=1.75 m`, `breaking_duration=3.0 s`, followed by 5.0 s to home and 3.0 s for landing. The generated target is not checked/clamped against the configured bounds. The race's step-based bounds check is no longer running during this blocking cleanup/return sequence.

## Suggested investigation / validation

- Make the braking target and the full braking/return trajectory respect the available flight volume; clamping only the endpoint does not guarantee the trajectory stays inside.
- Check trajectory initialization and continuity across the low-level/high-level handoff: current position, velocity, attitude, and estimator state; stop/priority-release ordering and timing.
- Determine what terminal speed, tilt, altitude margin, and available thrust the return controller can handle; add a supported transition/stabilization strategy rather than assuming a 3 s go-to always arrests motion safely.
- Monitor bounds/altitude throughout braking, return, and landing, not only during policy steps.
- Record onboard setpoint/mode, attitude/state, motor PWM, battery voltage, and supervisor status across the handoff to identify the remaining mechanism.
- Validate the two recorded terminal states as regression cases, including the absence of unintended near-floor motion before the planned landing and explicit handling of infeasible states.

## Evidence files

Attach `guide_home_evidence.zip` and `return_home_evidence.png` to this issue. The ZIP contains this description, the plot/PDF, full raw logs for both runs, and:

- `terminal_states_and_events.json`: full terminal observations, relevant events, configuration, model identities, and track geometry.
- `deploy_20261117T100017.801841Z_trajectory.csv` and `deploy_20261117T100827.697140Z_trajectory.csv`: **complete recorded mocap trajectories**, including preflight, race, return, and landing. Columns include race-relative time, time relative to first return command, phase, xyz, quaternion xyzw, and tilt.
- Each run's original `events.jsonl`, `metadata.json`, and `summary.json` under `raw/`.
- `prepare.py`: offline script regenerating the derived JSON, CSV, and plots from the repository's original `runs/` directories.

Original repository paths:

```text
runs/deploy_20261117T100017.801841Z/{events.jsonl,metadata.json,summary.json}
runs/deploy_20261117T100827.697140Z/{events.jsonl,metadata.json,summary.json}
```

Both recordings finalized with zero dropped records and no writer error. Recorded loop processing was below 5 ms with no 20 ms processing overruns during the race. Host external-pose submissions continued during return. Timestamps are host monotonic timestamps; they do not confirm radio delivery, sensor acquisition time, or onboard execution. No motor/battery/onboard estimator telemetry was recorded. Requested thrust is not measured physical thrust.
