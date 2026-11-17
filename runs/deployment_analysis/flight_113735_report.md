# Latest flight after correcting obstacle 1

Run: `deploy_20261117T113735.863815Z` (approximately 12:37:35 Berlin local time).
Real track-object observations enabled; cf3; same exported model SHA256 as earlier flights.
All times below are relative to the first policy observation.

## The original obstacle-height discrepancy is gone

At 0.603 s, obstacle 1 becomes visible and its input height changes from 1.550 m to
1.58357 m, remaining constant afterward. Thrust changes from 0.7661 N to 0.7653 N;
the previous large cut associated with a 1.55 -> 0.43 m height change is absent.
Changing only its current height back to 1.55 m yields 0.7628 N in single-step replay.

## A separate command jump remains before gate 3

- 2.109 s: the measured gate 3 pose becomes visible.
- 2.149 s: no new object pose is revealed. Obstacle order changes from [4,1,3,2] to
  [4,3,1,2] (one-based object numbers), exchanging obstacle 1 and 3 in feature slots 2/3.
- In one 20 ms step, thrust falls from 0.3561 N to 0.1312 N, while commanded roll changes
  from -23.4 degrees to +22.1 degrees (about +45.5 degrees).
- The drone is already descending: z = 0.8467 m, vertical velocity approximately -1.22 m/s,
  estimated roll approximately -65.7 degrees and pitch approximately +48.4 degrees.

## Same-model, same-state counterfactuals

Verified model hashes and replayed all factual inputs; maximum absolute command error
is approximately 1.04e-6. At 2.149 s, preserving recorded state, previous action and all
other features:

| Single-step input | Thrust | Commanded roll |
| --- | --- | --- |
| Factual input | 0.1312 N | +22.1 degrees |
| Keep previous obstacle feature-slot order only | 0.3146 N | -9.3 degrees |
| Set obstacle 1 height to 1.55 m only | 0.1317 N | +21.4 degrees |
| Restore gate 3 nominal position/orientation only | 0.1413 N | +25.8 degrees |

The immediate command is sensitive to the rank exchange; the current 3.4 cm obstacle-height
variation does not explain it. The gate-pose reveal also has a much smaller immediate
thrust effect. Keeping a previous order for one input is not a validated deployment fix
or a full simulated trajectory.

## Outcome and limits

Thrust subsequently recovers, but downward velocity reaches approximately -1.9 m/s.
Minimum recorded mocap z is approximately 0.0691 m at 2.653 s. This is the tracking
reference height, not propeller/body clearance. The drone recovers, with the first policy
observation reporting gate 3 passed at approximately 3.575 s. A KeyboardInterrupt ends
the recording after the fourth crossing; the final gate 2 is not completed and no go-to
return sequence is entered in this run.

The data establish one ordering-sensitive command jump. They do not establish that sorting
caused the entire low pass: large tilt, earlier low-thrust segments and physical response
may contribute. Training and deployment use the same sorting code. See
`sim_order_check/report.md` for the later actual simulation comparison: all three simulations
also exchange the same slots and succeed, at different attitude/velocity states.

Changing only deployment to fixed object IDs changes the model's learned input semantics.
Evaluate the recorded state and ordering boundary in simulation, and validate any encoder
change together with training. No flight config, model or controller was changed during
this analysis.

Artifacts: `flight_113735.png`, `.pdf`, `_metrics.json`. Reproduce with:

```bash
.pixi/envs/deploy/bin/python runs/deployment_analysis/check_113735.py
```
