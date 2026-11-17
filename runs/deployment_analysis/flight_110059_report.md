# Gate 3 descent in the 11:00:59 UTC flight

Run: `deploy_20261117T110059.075060Z` (approximately 12:00:59 Berlin local time).
Real track-object observations enabled; cf3; same exported model SHA256 as earlier flights.
Times are relative to the first policy observation. This report examines flight before return.

## Conclusion

Large requested-thrust changes are present, and obstacle 1's abnormal height still affects
the policy. This flight also contains a rapid measured attitude excursion near gate 3.
It cannot be summarized as a new obstacle-height jump immediately before the gate causing
a thrust cut and then a fall. Height changes once, at 0.602 s, from 1.55 m to 0.43078 m,
and remains constant afterward.

## Timeline and same-state replay

| Time | Recorded event/command | Output if only current obstacle 1 z is set to 1.55 m |
| --- | --- | --- |
| 0.602 s, before gate 1 | Height reveal; thrust 0.689 -> 0.092 N | 0.697 N |
| 1.988 s, after gate 2 | Thrust 0.485 -> 0.285 N | 0.425 N |
| 2.169 s, approaching gate 3 | Thrust 0.420 -> 0.150 N | 0.334 N |
| 2.611 s, below gate 3 | Thrust 0.780 N; measured tilt about 26 degrees | 0.787 N |
| 2.631 s, rapid rotation begins | Thrust still 0.775 N; estimated tilt about 38 degrees | 0.783 N |
| 2.651 s | Thrust still 0.754 N; estimated tilt about 57 degrees | 0.756 N |
| 2.691 s | Thrust falls to 0.558 N; estimated tilt about 85 degrees | 0.455 N |
| 2.712 s | Thrust falls to 0.221 N; estimated tilt about 94 degrees | 0.166 N |

The initial cut is directly reproducible from the height input. The abnormal height also
influences some later commands without another reveal. At rotation onset, however, changing
height alters thrust by only about 0.008 N. The large cuts at 2.69-2.71 s follow the measured
attitude excursion; treating them as the initiating event would reverse that ordering.

Model SHA256 hashes match the recordings. Maximum absolute factual replay error is about
9.24e-7. Counterfactuals keep the recorded physical state and previous policy action fixed.
They are single-step evaluations, not closed-loop rollouts; they cannot exclude indirect
influence of earlier anomalous inputs on the later approach.

## The gate 3 event

Raw mocap, as well as filtered attitude, records the excursion:

| Time | Mocap z | Tilt from vertical |
| --- | --- | --- |
| 2.592 s | 0.310 m | 24.6 degrees |
| 2.625 s | 0.271 m | 40.1 degrees |
| 2.658 s | 0.232 m | 66.7 degrees |
| 2.725 s | 0.130 m | 98.3 degrees |
| 2.759 s | 0.101 m | 121.6 degrees |

Before this excursion, requested pitch is roughly 30-34 degrees and thrust roughly 0.78 N;
the policy has not first requested a greater-than-90-degree attitude or a comparable thrust
collapse. Commands then change rapidly in response to the abnormal state.

This occurs near the lower gate frame. At 2.592 s, the mocap reference in gate-local
coordinates is approximately [x=-0.076, y=0.085, z=-0.349] m. With the configured 0.72 m
outer frame and 0.40 m opening, the nominal lower-frame vertical range is [-0.36,-0.20] m.
The body/propeller envelope could contact the frame before its reference point crosses
the plane, making lower-frame contact a hypothesis worth checking against video.

Contact is not confirmed: video, contact sensing, precise physical frame/marker geometry,
and onboard telemetry are absent. Other instability mechanisms or tracking artifacts
cannot be excluded. Raw mocap and the filtered state share a measurement source and are
not independent sensor confirmations.

There are subsequent near-ground segments; minimum pre-return mocap z is about 0.0505 m
at 3.091 s. The first policy observation reporting gate 3 passed is around 4.559 s; the
race completes around 5.947 s. Mocap reference height is not body/propeller clearance.

Recording is complete with zero drops. Maximum loop processing is about 4.66 ms, with no
20 ms processing overrun. Thrust is host-requested collective thrust, not measured force
or motor output.

First resolve obstacle 1's geometry/frame/training mismatch. Separately inspect video at
2.59-2.76 s near the gate's lower frame. The height input explains some command changes,
but does not by itself explain the complete attitude excursion.

Artifacts: `flight_110059.png`, `.pdf`, `_metrics.json`. Reproduce with:

```bash
.pixi/envs/deploy/bin/python runs/deployment_analysis/check_110059.py
```

No flight configuration or controller code was changed during this analysis.
