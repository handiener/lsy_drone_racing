**Actuator-only candidates and the response-0.8 follow-up — 17 November 2026**

The user requested that controller gains, drag, mass, inertia, motor force curves, onboard gyro/attitude sensing and all other settings remain nominal. Only motor response, achievable motor capacity and offboard observation processing vary. All experiments were run in the temporary investigation directory. The current local branch update now archives their metrics and scripts; the user will upload it.

**Result:** no tested response-0.8 condition reproduced the severe gate-3 ground-height failure. This follow-up contains 406 recorded episodes, including controls and repeated exact-start checks. All recorded physical states are finite. All runs use the frozen policy and the same two hardware-matched scenes/starts used in the earlier investigation.

| Phase | Recorded episodes | Reach gate 2 | Pass gate 3 | Below 15 cm between first gate 2 and gate 3 passages | Lowest height in that segment |
|---|---:|---:|---:|---:|---:|
| Independent input low-pass grid and unfiltered controls | 98 | 85 | 85 | 0 | 0.576 m |
| Actual 20/30 ms kinematic-input delay | 28 | 23 | 22 | 0 | 0.535 m |
| Four selected input-filter/capacity settings, small-start perturbations plus exact starts | 168 | 168 | 168 | 0 | 0.565 m |
| Locked deployment DiffFilter algorithm and defaults | 112 | 77 | 76 | 0 | 0.503 m |

Counts are outcomes of selected diagnostic scenarios, not estimates of hardware failure probabilities. Termination or timeout before reaching gate 3 does not establish what would happen without that event. In particular, the lowest default-filter case ends in a gate collision around 0.54 m, after previously reaching 0.503 m; it is not a demonstrated recovery or a demonstrated later ground impact.

**Independent-filter and delay grids**

Motor response multiplier is fixed to 0.8. Total motor capacities tested: 0.68, 0.70, 0.72, 0.75, 0.76, 0.78 and 0.80 N. The 0.80 N cases are controls without reduced capacity.

- Policy angular-velocity low-pass alone: time constants 20, 25 and 30 ms.
- Policy position, orientation, velocity and angular velocity all low-passed: time constants 20, 25 and 30 ms.
- An unfiltered-input control at each capacity.
- Actual 20 ms delayed kinematics, and 30 ms delayed kinematics constructed by interpolating the 20/40 ms historical samples. Position/velocity/rates use their midpoint; normalized quaternion averaging gives the short-arc rotation midpoint. Startup history is clamped to available observations.

Low-pass time constants and pure input ages are different quantities; the two types of tests are reported separately.

For full-kinematic filtering, the exact-start segment minima in centimetres were:

| Filter time constant | Motor capacity | 12:37 start | 12:52 start |
|---|---:|---:|---:|
| 20 ms | 0.70 N | 57.6 | 74.4 |
| 25 ms | 0.70 N | 67.8 | 68.1 |
| 30 ms | 0.70 N | 70.6 | 62.0 |
| 20 ms | 0.75 N | 71.2 | 71.5 |
| 25 ms | 0.75 N | 69.6 | 69.8 |
| 30 ms | 0.75 N | 66.6 | 66.5 |

All these exact-start examples pass gate 3; some fail later. Four settings (20/30 ms full-kinematic filtering crossed with 0.70/0.75 N capacity) additionally use 20 start perturbations per scene, plus the exact start. Position standard deviations are [5,5,1] mm and rotation-vector standard deviation is 0.3 degrees per axis, using the same seeds as the earlier paired validation. All 160 perturbed episodes pass gate 3, with none below 15 cm. This does not establish behaviour outside the tested nearby starts.

**Default deployment filter**

The project lockfile pins drone-estimators 0.1.0. Its public wheel was downloaded into this temporary directory without installation; SHA-256 matches the lockfile:

`e0f3be005863e2df7ba0a354815f85c2ee5a93d92a59efef6bab6f73db565a6e`

The wheel's defaults select DiffFilter and an estimator loop frequency of 300 Hz. Time constants are position 0.1 ms, velocity 7 ms, acceleration 90 ms, orientation 5 ms and angular velocity 70 ms. The implementation derives velocity/acceleration/angular velocity from pose differences, performs a prior update and corrects with the respective low-pass filters. This is more than independently filtering the simulator's true velocities. The DiffFilter implementation matches the previously inspected upstream source apart from a trailing newline.

The actual wheel implementation runs on NumPy float64 with JIT disabled, so enabling JAX float64 does not alter the simulator or policy. The estimator starts at the first recorded kinematic state. A recorder exposes the unchanged simulator's 500 Hz position/orientation trajectory. Mocap samples are interpolated from these physics states and fed to the default filter; the policy still runs at 50 Hz. An unfiltered comparison verified that adding the recorder leaves every recorded physical state, requested action, crossing time and critical metric exactly identical for a full response-0.8/capacity-0.70 episode.

The 112-case grid crosses both recorded scenes, seven motor capacities, 300/120 Hz mocap sampling and 0/10/20/30 ms additional measurement transport age. The package's 300 Hz estimator setting is known from source; actual mocap frequency and transport age during the flights are not preserved. The latter settings are sensitivity assumptions. ROS scheduling, packet loss, measurement noise and arbitrary host delays are not reproduced. Main policy gate-count/reveal logic remains based on physical position, as in the original scans; only policy kinematics are estimated.

The lowest case uses the 12:52 start, 300 Hz mocap, zero added transport age and 0.72 N capacity. It crosses gates 1/2 at 0.96/1.66 s, reaches 0.503 m before gate 3, and terminates in a gate collision. No default-filter case reaches ground height in that segment. This does not exclude an actuator mechanism; it shows that the tested response-0.8 models do not yet account for the recorded near-floor flight.

**What response multipliers mean in this model**

The rotor model already has separate spin-up/spin-down dynamics. Multiplying all four rotor dynamics coefficients by 0.8 slows the instantaneous acceleration/deceleration at equal current and target RPM. It is neither a fixed delay nor an RPM ceiling. Motor capacity is constrained through per-motor force/RPM limits while policy requests may remain higher.

A local linearization at nominal hover (approximately 15,900 RPM) gives these approximate motor-response time constants:

| Response multiplier | Spin-up | Spin-down |
|---|---:|---:|
| 1.0 | 57 ms | 62 ms |
| 0.8 | 71 ms | 78 ms |
| 0.5 | 114 ms | 124 ms |

These are model-derived small-signal values at a fixed target, not measured hardware latencies. They demonstrate that 0.5 is a substantial model change; its physical realism must be measured rather than inferred from a crash reproduction.

**Additional check of the earlier two actuator-only candidates**

Before the response-0.8 request, 84 runs checked the response-0.55/capacity-0.70 and response-0.50/capacity-0.75 presets: the exact start plus 20 small start perturbations for each scene and preset. For the 20 perturbed starts in each group:

| Candidate | Start | Reach gate 2 | Pass gate 3 | Below 15 cm between gates 2 and 3 | Complete all five crossings |
|---|---|---:|---:|---:|---:|
| 0.55 / 0.70 N | 12:37 | 15 | 11 | 4 | 6 |
| 0.55 / 0.70 N | 12:52 | 18 | 17 | 0 | 11 |
| 0.50 / 0.75 N | 12:37 | 20 | 2 | 18 | 1 |
| 0.50 / 0.75 N | 12:52 | 20 | 0 | 20 | 0 |

This demonstrates success/failure variation around the 0.55 candidate without changing gains. It does not identify the real fault. Many 0.50-candidate low points are far from the recorded low point: median horizontal error is about 0.99/1.08 m in the two scenes. Of the 0.55 candidate's four low 12:37 perturbations, two meet the descriptive thresholds of <0.30 m horizontal low-point error and <0.30 s low-point timing error; this is not an independent validation or a confidence score. The user's memory of an unrecorded successful hardware attempt is useful context but cannot be replayed or used to verify these parameters.

**Packaged files**

All 750 newer per-episode metric rows, including the 406 response-0.8 cases and 84 earlier actuator-only checks discussed here, are under `results/followups/metrics.json`. `inventory.json` gives phase counts and source hashes. The two compact CSVs retain their original phase labels. Selected current-reference trajectories are bundled in `results/reference/`; other bulk traces are explicitly omitted and can be regenerated with the scripts.

This report describes the milder-delay phase. The later 40 ms angular-velocity-delay candidate is the current reference; read `README.md` and `docs/actuator_delay_reference.md` first.
