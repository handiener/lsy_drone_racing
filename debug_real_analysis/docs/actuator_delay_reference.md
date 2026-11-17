**Response 0.8 plus 40/50/100 ms policy-input delay — 17 November 2026**

**Finding:** keeping controller gains, drag, mass, inertia, motor force curves, onboard sensing and all other parameters nominal, a combination of motor response multiplier 0.8, motor capacity 0.70 N and a 40 ms delay of policy angular velocity produces a gate-2-to-gate-3 ground-height failure. It does not reproduce both complete hardware trajectories. Position, orientation and translational velocity remain current in this candidate; it does not additionally apply the deployment DiffFilter.

At the exact 12:37 start, gates 1 and 2 are crossed at 0.98/1.74 s, then the model reaches ground height at 2.44 s, 0.70 s after gate 2. The minimum reference height is -0.000750 m. The lowest point is [-0.438, 0.058] m horizontally, approximately 0.620 m from the recorded policy-observation low point. Its relative timing is approximately 0.364 s earlier. This lies outside the investigation's predefined broad near-gate-3 spatial box, despite being between gates 2 and 3 in task progress. The physical pre-terminal state is recorded before the environment crash warp; this is not a [-1,-1,-1] teleport artifact.

**Withdraw one factor, same exact 12:37 start**

| Condition | Minimum height after gate 2, before gate 3 | Outcome |
|---|---:|---|
| Response 0.8 + angular-velocity age 40 ms + capacity 0.70 N | approximately ground | Ends before gate 3 |
| Remove angular-velocity delay | 0.756 m | Completes all five crossings |
| Restore 0.80 N motor capacity | 0.751 m | Completes all five crossings |
| Restore response multiplier 1.0, retain the delay/capacity | Not reached | Fails after gate 1, before gate 2 |

Thus the delayed angular-rate observation and reduced thrust margin jointly matter in this tested setting. Removing a factor can also alter earlier flight behaviour; these are not additive contributions or a parameter identification.

**Nearby starts**

Twenty identical predefined initial-state perturbations are used per scene: position standard deviations [5,5,1] mm and rotation-vector standard deviation 0.3 degrees per axis, seeds 900001 through 900020. Parameters remain fixed in every repetition. These starts were previously used elsewhere in the investigation, so this is not an independent hardware validation set.

| Start | Episodes | Reach gate 2 | Below 15 cm between gates 2/3 | Below 15 cm in the predefined near-gate-3 box | Pass gate 3 | Complete all five crossings |
|---|---:|---:|---:|---:|---:|---:|
| 12:37 | 20 | 20 | 12 | 10 | 9 | 8 |
| 12:52 | 20 | 10 | 0 | 0 | 10 | 4 |

Categories overlap: a trajectory can descend below the threshold and subsequently recover through gate 3. Eight of the 12:37 episodes reach below 1 cm somewhere in the gate-2-to-gate-3 segment. Among its twelve below-15-cm cases, median low-point horizontal error is 0.257 m; six meet the descriptive combined criteria of <0.30 m horizontal low-point error and <0.30 s relative timing error. These thresholds screen geometric/timing resemblance and are not confidence values. The second scene does not reproduce the hardware low pass; half of its nearby starts fail before gate 2.

**Broader scans**

Motor capacities 0.68, 0.70, 0.72, 0.75, 0.76, 0.78 and 0.80 N are tested from both recorded starts. The 0.80 N entries are controls without capacity reduction. Cases are run from takeoff with globally present interference.

| Delayed fields | Delay | Cases | Reach gate 2 | Pass gate 3 | Below 15 cm in gate-2-to-gate-3 segment |
|---|---:|---:|---:|---:|---:|
| Angular velocity only | 40 ms | 14 | 12 | 10 | 1 |
| Angular velocity only | 50 ms, exact high-rate history | 14 | 12 | 12 | 0 |
| Angular velocity only | 100 ms | 14 | 11 | 11 | 0 |
| Position/orientation/velocity/angular velocity | 40 ms | 14 | 6 | 2 | 0 |
| Position/orientation/velocity/angular velocity | 50 ms, exact high-rate history | 14 | 5 | 5 | 0 |
| Position/orientation/velocity/angular velocity | 100 ms | 14 | 0 | 0 | 0 |

Longer input age does not produce a monotonic sequence of later-segment failures. Earlier failures must not be counted as gate-3 reproductions.

The initial 84-case scan approximated 50 ms using midpoint interpolation of the 40/60 ms policy-history samples. A follow-up uses exact 500 Hz physics-history timestamps instead: 28 cases check both field scopes at exactly 50 ms, and 14 repeated angular-velocity 40 ms cases verify the implementation. All fourteen 40 ms cases match the original results exactly in crossing times, minimum height, completion, duration and crossing count. The exact 50 ms checks still have no below-15-cm gate-2-to-gate-3 episode. Their lowest segment heights are 0.722 m for angular velocity alone and 0.545 m for all kinematics.

**Default deployment filter plus additional measurement age**

A separate 84-case scan retains the actual locked drone-estimators 0.1.0 default DiffFilter, and crosses the same capacities/scenes with 300/120 Hz assumed mocap sampling and 40/50/100 ms additional measurement transport age. Motor response is again 0.8 and other physical/controller parameters are unchanged. No below-15-cm gate-2-to-gate-3 episode occurs. With 40 ms added age, six of 28 cases reach gate 2 and two pass gate 3; with 50 ms, none reach gate 2; with 100 ms, two reach gate 2 and none pass gate 3. Mocap rates and extra transport ages are sensitivity assumptions, not measured flight settings.

These tests distinguish a pure delay of only the angular-rate feature from delaying the entire estimated observation. The default-filter result must not be conflated with the positive angular-velocity-only result.

**Reproduction and archived data**

The current command is `python -B debug_real_analysis/scripts/reproduce.py --preset actuator_delay --scene 113735`. Its default preset is now `actuator_delay`. Add `--seed 16` for the selected nearby illustration; that seed is explicitly a post-hoc selection. Run `validate_reference.py` to regenerate all 50 exact-start, nearby-start and withdrawal episodes.

The 50 reference episodes and 42 previously recorded matching baseline controls are bundled under `results/reference/`, with per-episode trajectories, complete noise dictionaries, summaries, checksums and provenance. All 750 newer follow-up metric rows are under `results/followups/metrics.json`; their inventory distinguishes the 84 earlier actuator-only checks, 406 mild-response checks and 260 long-delay checks. Bulk non-reference trajectories remain local and are explicitly marked as omitted. The scripts reproduce them.

All experiments were run in isolated temporary checkouts. This local branch update packages their results for the user to upload; original project files and main remain unchanged.

**Policy-input trend check**

`response80_delay40_input_trends.json` compares the real 12:37 inputs, the exact-start candidate and a post-hoc selected nearby example (seed 16, chosen for combined low-point position/timing resemblance among the low near-gate-3 cases). Real/candidate angular-rate relative lag is 40 ms, as expected from the deliberately injected rate age; this is a consistency check, not independent evidence for the hypothesis. The real pitch-rate peak is 7.66 rad/s at 0.502 s after gate 2. Exact-start and selected-example peaks are respectively 8.31 and 7.84 rad/s at 0.340 and 0.360 s. The selected example's low point is 0.195 m away horizontally and 0.044 s different in relative timing, but the angular-rate peak remains about 0.14 s early. Its minimum model height is approximately -0.0006 m. This selected resemblance does not establish agreement of the full input/action histories or generalization to the second hardware scene.
