# Recorded deployment analysis — 17 November 2026

Analyzed all three `runs/deploy_*` recordings. Times below are seconds from the first
policy observation, unless stated otherwise. Directory timestamps are UTC. Source data
were not modified. Reproduce the figures and metrics with:

```bash
MPLCONFIGDIR=/tmp/drone-analysis-matplotlib .pixi/envs/deploy/bin/python runs/deployment_analysis/analyze.py
```

## Main finding

Both flights completed the recorded five-crossing gate sequence, then descended to
approximately the resting mocap height during the first automatic return maneuver.
The repeatability, timing, and aggressive finish state point to the policy-to-return
handoff as the main failure region. The data establish when and where the descent occurs;
they do not establish the exact onboard control or motor failure mechanism.

| Recording (UTC) | Flight outcome | Race time | Finish speed | Finish tilt from vertical | First return minimum z | Delay after finish |
| --- | --- | --- | --- | --- | --- | --- |
| 10:00:17 | Five crossings completed | 5.123 s | 3.015 m/s | 69.1° | 0.0560 m | 0.658 s |
| 10:06:51 | No flight: estimator startup timeout for cf2 | — | — | — | — | — |
| 10:08:27 | Five crossings completed | 5.305 s | 3.071 m/s | 67.3° | 0.0496 m | 0.804 s |

The preflight resting mocap heights were approximately 0.0441 m and 0.0443 m. Return
minima were therefore only approximately 12 mm and 5 mm above those baselines. Together
with the user's report of ground contact, this is strongly consistent with the reported
impacts. Mocap alone does not independently measure physical contact or propeller clearance.
Both drones subsequently climbed back to approximately 1.7 m and continued the return.
The planned landing command occurred about 8 s after race completion, well after the dip.
Emergency stop was recorded after the final planned landing, not at the unplanned dip.

## The finish state already contains downward motion

At the terminal transition:

- First flight: z = 1.256 m, vertical velocity = -0.804 m/s.
- Second flight: z = 1.282 m, vertical velocity = -0.566 m/s.

The policy exits while highly tilted and moving at about 3 m/s. Completion currently
requires the gate crossing, not a low-speed, level, stabilized terminal state. The last
policy command projected onto the measured vertical axis was approximately 0.282 N and
0.209 N, compared with the configured weight of 0.426 N. These are geometric projections
of requested thrust, not measurements of actual thrust. They explain why a large total
thrust command does not necessarily prevent descent at a large tilt.

With the configured maximum collective thrust of 0.8 N and mass of 43.4 g, the nominal
maximum tilt permitting vertical force equal to weight is about 57.8°. The measured finish
tilts exceed that. This is a model-based calculation, not an experimentally calibrated
thrust measurement. In the first run, even the approximately 0.787 N final command has
insufficient nominal vertical component at the observed tilt.

## Automatic return also leaves the configured flight bounds

`Crazyflie.return_to_start()` switches to the high-level commander, constructs a point
1 m along the current velocity direction, sets its height to 1.75 m, and sends a 3 s go-to.
It does not clamp that point to the configured bounds.

- First braking target: [0.544, 1.625, 1.750] m.
- Second braking target: [0.517, 1.631, 1.750] m.
- Configured maximum y: 1.500 m.
- Maximum measured y during the first 3 s of return: 2.182 m and 2.114 m.

Both requested braking targets are outside the configured region, and the drone overshoots
further. The environment's step-based bounds check is no longer running during `close()` /
`return_to_start()`. This is a separate concrete issue even if it does not fully explain
the altitude loss. Merely clamping the target would not establish that the dynamic braking
trajectory remains inside the bounds.

## Additional low pass during the second race

Between the first visit to gate 2 and gate 3, the second flight reached z = 0.0929 m at
2.975 s, compared with 0.3098 m at 2.780 s in the first flight. It recovered and crossed
gate 3 afterward. The 9.3 cm value is a mocap reference height, not physical ground
clearance; it does not establish contact. This is a separate altitude-margin problem
before the finish handoff.

## Timing and recording integrity

Both recordings finalized with zero dropped records and no writer errors. The flights
used identical exported policy/config hashes and the same deployment configuration.

| Metric | First flight | Second flight |
| --- | --- | --- |
| Maximum recorded loop processing time | 4.63 ms | 3.96 ms |
| Recorded processing overruns of 20 ms | 0 | 0 |
| Inference p95 (includes enabled policy diagnostics) | 2.87 ms | 2.29 ms |
| Longest external-pose submission interval | 43.0 ms | 50.6 ms |
| Estimate vs latest mocap position difference, p95 | 1.60 cm | 1.65 cm |

There is no evidence here of controller deadline overruns or a long interruption in host
external-pose submissions. Small estimate/mocap differences do not independently validate
the onboard estimator. External-pose records have host submission timestamps, not Vicon
acquisition timestamps; delivery over radio and freshness cannot be confirmed from them.

## Track observation discrepancy

Obstacle 1 was measured at z ≈ 0.431 m versus nominal z = 1.550 m in both flights. Its
observed x/y were approximately [-0.06, 0.76] m. The track check validates obstacle x/y
only, so this discrepancy passes the check. Actual obstacle positions become policy
inputs once visited. Check whether this is an intended marker-frame origin, a shortened
obstacle, or a tracking/configuration mismatch. The data do not establish it as the
cause of the ground contact.

## Recommended order of work

1. Inspect and redesign the terminal stabilization / automatic-return handoff. A gate
   crossing at roughly 3 m/s and 67–69° tilt is not a stable starting point for a gentle
   return. Validate the transition in simulation or constrained tests before another
   full-speed hardware run. A terminal braking strategy must account for velocity,
   attitude, altitude margin, and thrust authority; changing only a delay is insufficient.
2. Make return trajectories respect the track bounds throughout braking and return, and
   monitor bounds/altitude during cleanup as well as during policy control.
3. Address the second flight's mid-race altitude margin, independently of the return issue.
4. Verify obstacle 1's physical geometry and coordinate-frame meaning.
5. Capture onboard attitude/state, motor PWM, battery voltage, supervisor state, and active
   setpoint/mode across the handoff. That can distinguish inadequate stabilization from
   motor inhibition, onboard estimation error, or command-mode problems. Current logs
   cannot confirm motor saturation, battery sag, or a tumble shutdown.

One recorder naming caveat: `attitude_command_sent.yaw_rate_deg_s` is misleading for the
current setup, which sets `flightmode.stabModeYaw = 1` (absolute yaw angle). Interpret the
third transmitted field according to that firmware setting. The method/field name alone
is not evidence of a yaw-rate/angle mismatch in these flights.

Artifacts: `flight_analysis.png`, `flight_analysis.pdf`, and `metrics.json` alongside this report.
