# Race-only comparison of the 10:46 and 10:48 UTC flights

This report retains its original filename, but refers specifically to
`deploy_20261117T104622.175276Z` and `deploy_20261117T104828.785539Z`, not the newest
recordings added later. Berlin local times are approximately 11:46 and 11:48.
Times below are relative to the first policy observation.

## Conclusion

Both flights descend substantially before gate 3. Obstacle 1 cannot explain both events:
in the first flight, real track objects are disabled and its policy input remains at
1.55 m, yet the drone reaches about 0.177 m. In the second, measured object poses are
restored, the abnormal height reveal/thrust cut reappears, and the minimum is about 0.189 m.

| Metric | 10:46 UTC | 10:48 UTC |
| --- | --- | --- |
| Drone ID | 3 | 3 |
| real_track_objects | false | true |
| Obstacle 1 z after reveal | 1.550 m | 0.431 m |
| Reveal time | 0.603 s | 0.603 s |
| Requested thrust before -> at reveal | 0.712 -> 0.717 N | 0.698 -> 0.088 N |
| Mocap minimum after first 1 s, before return | 0.177 m at 2.783 s | 0.189 m at 2.860 s |
| First policy observation reporting gate 3 passed | 3.435 s | 3.495 s |
| Five crossings completed | 4.885 s | 4.923 s |
| Loop processing overruns of 20 ms | 0 | 0 |
| Dropped recording events | 0 | 0 |

These heights are mocap reference heights, not physical clearance. Both minima occur about
2 s before return begins and cannot be attributed to return control. Both races finish.

## Obstacle 1 single-step test

Verified model SHA256 hashes and replayed factual features with maximum absolute command
error around 1.4e-6. In the second flight at 0.603 s, changing only obstacle 1 z from
0.431 m to 1.55 m yields thrust 0.674 N instead of 0.088 N, and pitch -18.7 degrees instead
of +13.9 degrees. The first flight already has z=1.55 m and does not exhibit this cut.

This isolates a current-input effect, not an entire flight outcome. These flights also
change the observation treatment of all track objects and gates; they are not a controlled
experiment changing only obstacle 1 height.

## Gate 3 descent remains a separate problem

In the first flight at 1.607 s, the drone is already descending from z=1.244 m at about
70.2 degrees tilt. Requested thrust is 0.761 N, but its projection on the measured vertical
axis is about 0.258 N, below configured weight (0.426 N). At 1.828 s there is another
0.440 -> 0.120 N cut without an obstacle 1 height reveal. Higher thrust subsequently
returns, but the drone has accumulated downward velocity before recovering near 18 cm.

The second flight also reaches large tilt (71.8 degrees at 1.808 s) and descends; estimated
vertical velocity is about -1.77 m/s at 2.390 s. Even high recovery thrust requires time
to stop that descent.

Neither first approach crosses gate 3's plane; the drone is about 0.34 m below gate center
before circling back and crossing. This resembles the earlier low-pass/retry pattern,
but the first flight establishes that the abnormal obstacle 1 height is not necessary
for that pattern to occur.

The logs do not isolate learned simulation behavior, attitude/thrust model mismatch,
battery/motor changes or combinations of these effects. Simulation comparisons and onboard
telemetry are needed. See the later `sim_order_check/report.md` for an actual three-episode
simulation check.

## Ordering sensitivity differs between these flights

In the second flight at 1.908 s, an obstacle-slot exchange coincides with thrust
0.762 -> 0.595 N. Preserving the previous order at that same state yields 0.746 N;
changing only obstacle 1 height yields 0.690 N.

In the first flight, the corresponding exchange is at 1.728 s with thrust
0.750 -> 0.711 N. Keeping the previous order yields 0.725 N: a much smaller immediate
effect. It does not explain the entire descent on its own.

Treat obstacle-height semantics and the remaining gate 3 low pass separately. No flight
code or config was changed for this analysis.

Artifacts: `latest_race_only.png`, `.pdf`, `latest_race_metrics.json`,
`latest_race_counterfactuals.json`, and `latest_race_diagnostics.py`.
