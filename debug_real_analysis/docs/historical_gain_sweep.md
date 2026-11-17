**Historical investigation: the earlier gain-changing candidates**

The current reference is the three-factor `actuator_delay` preset described in [the main README](../README.md). This earlier report is preserved for comparison. The old `damping` parameter scales the controller feedback gain `kw`; it is not passive airframe damping.

**Gate 3 altitude-loss investigation — English handoff for a teammate or Claude**

This folder contains the English analysis, offline simulation wrappers and experiment generators from the 17 November 2026 hardware debugging session. The analysis reports the experiments already performed; bulk experiment records and derived trajectories are retained in the local investigation archive and are not part of this upload. It is an evidence and diagnostic package; the deployed controller, policy weights and competition configuration are unchanged.

Read this file first. The central distinction is **reproducing a failure mechanism versus identifying the actual hardware fault**. We achieved the former; the logs do not uniquely establish the latter.

**Main result**

With the original frozen policy, combinations of observation filtering, command timing, attitude-loop response and available motor thrust reproduce a severe altitude loss between the first passage of gate 2 and the first passage of gate 3. Disturbances are present from takeoff, rather than being triggered at gate 3.

The main scans and validation contain **5,822 recorded simulation episodes**: 888 coarse, 3,238 extended, 1,552 focused and 144 validation episodes. Two additional runs check the hardware safety bounds; eight check observation/event consistency. Matching-start runs use measured track geometry and initial states from hardware directories ending in `T113735.863815Z` and `T115247.042900Z` (approximately 12:37 and 12:52 Berlin time).

| Matched hardware start | Original simulation: minimum height before gate 3 | Disturbed simulation, preset `paired` | Hardware mocap minimum |
|---|---:|---:|---:|
| 12:37 | 0.747 m | 0.00010 m | 0.0691 m |
| 12:52 | 0.747 m | 0.0909 m | 0.0418 m |

Both exact-start `paired` runs recover and complete all five crossings. These are reference-point heights, not calibrated propeller clearance. The simulator's simple floor handling does not reproduce physical impact, ESC restart or firmware protection dynamics.

Regenerate the height and policy-input comparison locally with `scripts/plot_replays.py` after running the quick reproduction commands below.

**What was established from hardware before the disturbance search**

- There are 13 finalized hardware recordings, 11 with policy samples and nine reaching the gate-2-to-gate-3 segment. Startup failures and interrupted/ground-only records should not be treated as successful flights.
- Exported model hashes match all 11 policy recordings. Re-encoding raw observations agrees with saved policy features to about `1.3e-6`; network replay reproduces commands to below `1.5e-6`. Host collective PWM agrees with the documented conversion to integer rounding. No 20 ms host processing overrun was observed in the normal flights. This does not establish radio delivery or onboard execution.
- Early runs reveal obstacle 1 at approximately **0.431 m instead of nominal 1.55 m**. The training obstacle-height randomization is only ±0.05 m. At one recorded state, correcting only this feature changes requested thrust from **0.0867 N to 0.7339 N**. The track check validates obstacle x/y, so it does not catch this height discrepancy.
- Correcting obstacle height does not remove gate 3 altitude loss: the two corrected-height runs still reach 6.9 cm and 4.2 cm. Runs with `real_track_objects=false` also show low passes. The height error is an established input problem, not a complete explanation of all gate 3 failures.
- Obstacle features are distance-sorted into fixed MLP slots. At 2.149 s in the 12:37 run, the one-based order changes `[4,1,3,2] -> [4,3,1,2]`. Requested thrust drops from 0.356 N to 0.131 N and requested roll changes by about 45.5 degrees in one policy step. Holding only the previous ordering at the same recorded state/history yields approximately 0.315 N. This is a single-step sensitivity test, not a validated deployment modification.
- The three archived successful simulations also exhibit that same order exchange. They reach it at about 42–46 degrees tilt and high requested thrust; the corrected hardware run reaches it at **74.1 degrees tilt and -1.22 m/s vertical velocity**. Hardware divergence precedes the exchange. Sorting sensitivity can amplify the failure but does not explain its onset alone.
- Comparing angular velocity inferred offline from consecutive observed orientations with the `ang_vel` delivered to the policy suggests an approximately **40–55 ms relative lag** in recent hardware records; the analogous original simulation check is nearly synchronous. This is an empirical alignment result, not an acquisition-to-actuation latency measurement. The default offboard DiffFilter has different orientation and angular-velocity time constants (5 ms and 70 ms), but its exact runtime settings/version were not preserved.

The direct descent mechanism is well supported: a highly tilted vehicle accumulates downward velocity, receives low-thrust/rapid-reversal commands, and does not arrest descent in time. At nominal mass 43.4 g, weight is about 0.426 N. Even an ideal 0.8 N collective at 74.1 degrees tilt has only about 0.219 N vertical projection. Simulations also transiently exceed the nominal 57.8-degree hover-support limit; duration, velocity, altitude and recovery timing determine the outcome.

**Reproduction presets and what their parameters mean**

| Parameter | `paired` / B | `input_match` / A | Meaning |
|---|---:|---:|---|
| `omega_tau` | 0.050 s | 0.100 s | Policy angular-velocity first-order low-pass time constant |
| `pose_tau` | 0.010 s | 0.020 s | Policy position and orientation low-pass time constant |
| `command_substeps` | 2 | 3 | Old applied command is held for this many 2 ms physics substeps: 4 / 6 ms |
| `thrust_cap` | 0.76 N | 0.72 N | Maximum achievable collective capacity, implemented as per-motor force/RPM limits |
| `damping` | 0.70 | 0.70 | Multiplier on the simulated Mellinger angular-rate feedback gains `kw` |
| `drag_scale` | 1.40 | 1.40 | Multiplier on the physical translational drag matrix |

All other parameters are nominal. A time constant is **not** a fixed delay. `damping=0.7` is an effective response hypothesis/stress condition, not a measured firmware setting and not a recommendation to lower hardware gains.

Physical motor limits matter: the policy can still request about 0.79 N while individual achievable rotor forces are constrained. Motor dynamics and mixing can make instantaneous realized total thrust lower still. This is different from merely clipping the policy's collective request. The search includes the user's proposed 0.68 and 0.70 N limits, as well as 0.56–0.80 N and other coupled conditions.

The smaller `minimal` preset uses only `omega_tau=0.1`, `thrust_cap=0.72` and `damping=0.7`. From the 12:37 start it bottoms at 0.0844 m, recovers through gate 3, then collides later near gate 4. It fails before gate 2 from the other start, so it is not a general explanation.

**Ablations: the interaction matters**

Each row uses exactly the same hardware-matched start. Heights are physical simulation states during gate-2-to-gate-3 progress; an earlier collision is explicitly distinguished.

| Condition | 12:37 minimum | 12:52 minimum |
|---|---:|---:|
| Full `paired` preset | 0.00010 m | 0.0909 m |
| Restore normal `kw` damping | 0.744 m | 0.747 m |
| Restore 0.80 N capacity | 0.576 m | 0.712 m |
| Remove position/orientation low-pass | 0.256 m | 0.346 m |
| Remove the 4 ms command delay | 0.482 m | 0.413 m |
| Restore nominal drag | 0.181 m | 0.123 m |
| Only observation filters and 4 ms command delay | 0.735 m | 0.736 m |
| Remove angular-velocity filtering, keep other B changes | Fails before gate 2 | Fails before gate 2 |

This is a nonlinear closed loop. A disturbance can stabilize an earlier segment while making another segment vulnerable; these rows are not additive causal contributions. **Angular-velocity latency alone did not reproduce the characteristic hardware low pass.**

**Small initial-state perturbation validation**

For each of the two recorded starts, 20 paired perturbations use position standard deviations `[5,5,1]` mm and rotation-vector standard deviation 0.3 degrees per axis. Seeds are identical across baseline and disturbed conditions.

| Condition | Episodes | Reach gate 2 | Below 15 cm near gate 3 | Complete all crossings |
|---|---:|---:|---:|---:|
| Original simulation | 40 | 40 | 0 | 40 |
| `paired` disturbances | 40 | 40 | 36 | 24 |

This demonstrates repeatability near the investigated starts. Both scenes were used during search; this is not an independent hardware validation set or a population crash-probability estimate. The original 144-episode validation includes A and all exact-start ablations. Run `scripts/validate.py` to regenerate these metrics and trajectories locally.

**How close are the actual policy inputs?**

Height resemblance alone is insufficient. The scripts also compare orientation, angular velocity, vertical velocity, positions and requested thrust, aligned at the first observation after gate 2 crossing without arbitrary time warping.

Preset A is closer to some 12:37 input trends: relative angular-rate lag about 50 ms versus about 40 ms in hardware, pitch-axis rate input peak **7.78 versus 7.66 rad/s**, and peak timing **0.460 versus 0.502 s after gate 2**. Its later thrust trough occurs about 0.560 s after gate 2 at 0.142 N, versus about 0.582 s at 0.125 N in hardware. It reaches the model floor before gate 3.

However, A does not reproduce the same severe low pass from the 12:52 start. B reproduces height loss at both starts but does not simultaneously match all angular-rate, attitude and command waveforms. **No single tested parameter set has been shown to match both full hardware trajectories and all their policy inputs.** Multiple parameter combinations can produce similar failures; matching a curve after a large search does not identify the actual battery, firmware or motor fault.

Use `scripts/plot_replays.py` to compare fresh reproductions with the hardware logs already present on `debug_real`. The detailed comparison and ranking scripts can also read regenerated sweep/validation outputs.

**Checks against diagnostic artifacts**

- Terminal physical poses are retained before the official environment's `[-1,-1,-1]` crash warp. Ground-height results are not synthetic teleport coordinates.
- In all 5,822 main/validation trajectories retained in the local investigation archive, physical time, position, quaternion, velocity and angular velocity were checked to be finite. Terminal policy-input/action fields may intentionally be NaN where no next policy command exists.
- Re-running B with the hardware safety bounds leaves its two critical minima unchanged.
- Main scans score passes and reveal objects using physical position while allowing delayed policy kinematics. Eight extra runs instead compute the **policy's** gate count and reveals from the delayed/filtered estimated position. B still loses altitude severely at gate 3: the 12:37 case reaches the model floor and terminates; the 12:52 case reaches 0.0964 m and completes. The qualitative weakness is not entirely caused by mixing a fresh counter with old kinematics, but exact recovery is sensitive to event timing.
- Coarse runs used the older [harness_v1.py](../scripts/harness_v1.py). Subsequent scans used [harness.py](../scripts/harness.py), which adds optional inner-loop sensing/allocator variations and subcycle command timing. Keep these historical phase implementations distinct when comparing IDs or exact numerics.

**What remains unknown about the real drone**

The hardware recordings have host timestamps, not sensor-acquisition timestamps. They do not contain motor RPM/PWM, battery voltage, actual onboard setpoints/modes, supervisor state, complete firmware settings or a clean exact flight-software snapshot. Offboard state and mocap also share a measurement source. An apparently old angular-rate input is not evidence that the onboard gyro itself is delayed by the same amount.

There is a concrete implementation difference worth checking: the inspected Crazyflow Mellinger code clips individual mixed motor outputs, while inspected Bitcraze/LSY firmware source applies a common reduction when the highest output saturates, then an idle floor. The diagnostic harness can test this alternative locally without editing the installed library. Actual flight firmware revision, battery compensation, `powerDist.idleThrust`, gains and protection settings must be read back before assigning this difference as a real cause.

The data establish an input-height error and state-dependent sorting sensitivity. The simulations establish a reproducible **class of coupled timing/response/thrust-margin failures**. They do not prove that actual `kw` is 70%, actual drag is 1.4 times the model, or actual capacity is 0.76 N.

The finish/automatic-return problem documented elsewhere in `runs/deployment_analysis` is a separate phase and should not be conflated with the gate 3 descent.

**Recommended next simulation/training work**

1. Measure matched short-segment responses with onboard attitude, gyro, received setpoints/modes, individual motor outputs/RPM if available, battery voltage and supervisor telemetry. Distinguish estimator timing, inner-loop damping, allocation saturation and actuator recovery.
2. Add observation filtering/age/holding, attitude-loop and rotor-response uncertainty, and physical thrust authority to training **in combination**. Include nominal environments and gradually introduce stress conditions; do not replace the simulator with one fitted adversarial parameter set.
3. Include recoverable gate-2-to-gate-3 starts before the state becomes physically unrecoverable. Assess minimum height, first-attempt gate passage, vertical velocity and saturation duration, not lap completion alone.
4. Preserve the observation/model interface in an initial fine-tuning iteration. Encoder/order changes require corresponding training and validation. Fix incorrect obstacle-frame/height semantics rather than teaching a policy to depend on fictitious geometry.

This wrapper is an offline diagnostic implementation using Python history buffers. A production Brax/JAX training wrapper needs per-world JAX buffer/filter state and masked resets; this Python wrapper is not a drop-in `jit`/`vmap` training environment.

**Quick reproduction**

Use the existing project environment and the checked-in project dependency configuration. Historical runs used CPU, Crazyflow 0.4.0, JAX/JAXlib 0.11.0, NumPy 2.5.2, SciPy 1.18.0, MuJoCo 3.10.0 and Flax 0.12.8. These historical diagnostic harnesses assume the checked-in 50 Hz policy loop and 500 Hz physics/attitude loop; their time-step constants must be updated together if those rates change. The controller adapter inspects that Crazyflow implementation and fails explicitly if its expected allocator source no longer matches. Do not silently assume arbitrary newer versions reproduce these numbers.

From the repository root, with the appropriate Python environment active:

```bash
# Original simulation and the reproducible coupled-failure condition.
python -B debug_real_analysis/scripts/reproduce.py --preset baseline --scene 113735
python -B debug_real_analysis/scripts/reproduce.py --preset paired --scene 113735
python -B debug_real_analysis/scripts/reproduce.py --preset baseline --scene 115247
python -B debug_real_analysis/scripts/reproduce.py --preset paired --scene 115247

# Additional presets and the hardware safety-bound check.
python -B debug_real_analysis/scripts/reproduce.py --preset input_match --scene 113735
python -B debug_real_analysis/scripts/reproduce.py --preset minimal --scene 113735
python -B debug_real_analysis/scripts/reproduce.py --preset paired --scene 113735 --real-bounds

# Compare newly reproduced inputs, actions and heights with the existing hardware logs.
python -B debug_real_analysis/scripts/plot_replays.py --scene 113735
python -B debug_real_analysis/scripts/plot_replays.py --scene 115247
python -B debug_real_analysis/scripts/plot_replays.py --scene 113735 --preset input_match

# Re-run full grids or paired validation; these execute simulations.
python -B debug_real_analysis/scripts/scan.py
python -B debug_real_analysis/scripts/extended.py
python -B debug_real_analysis/scripts/focused.py
python -B debug_real_analysis/scripts/validate.py
python -B debug_real_analysis/scripts/check_observation_logic.py

# After validate.py, regenerate the two-scene summary and detailed A comparison.
python -B debug_real_analysis/scripts/build_summary.py
python -B debug_real_analysis/scripts/plot_candidates.py 113735 bc60e02f023f input_match_113735
```

Paths are resolved from the scripts, so commands also work from another current directory using an absolute script path. New runs/caches/figures default to the ignored `debug_real_analysis/generated/` directory. Set `DRONE_DEBUG_OUTPUT_DIR` to choose another writable output directory. The `runs/` hardware logs already present in this branch are read-only inputs. An optional local `evidence/` archive is also supported; regenerated outputs do not overwrite it.

**Package inventory and provenance**

- `README.md`: the English conclusions, measured observations, reproduction conditions, limitations and next steps.
- `scripts/`: the simulation/observation/command wrappers, scan generators, validation, consistency check, comparison and plotting scripts. Presets and complete search grids are defined in Python.
- [VERIFICATION.md](../VERIFICATION.md): checks performed before publication.
- `SHA256SUMS`: checksums for the published report and scripts, excluding this checksum file itself.

Original source baseline: `d27923be237e18a4363bf1eea77f90e955d4ad09`. Frozen deployed-model SHA-256 values:

```text
config.json     12e4613f1772d4407f441736e7e77f0296cb75c20a1b93ea99117e5fbf941c9e
policy.msgpack  9b5116f127abc0a4fa2c492c06385ce843883b18e64a0c01dd241348b2cc9394
```

The original investigation retains all 5,822 per-episode metrics, a hash inventory for all original traces, full validation trajectories, selected sweep trajectories, eight consistency-check traces, two hardware-bound checks and English-labelled figures locally. These additional data files are not included in this commit. The scripts regenerate the results from the existing branch inputs. Historical outputs and their inclusion flags belong to the local archive; the published conclusions do not depend on treating a selected subset as a fresh full-grid ranking.

`rank.py` explicitly skips unavailable trajectories. For a complete ranking, first regenerate the corresponding full sweeps. Scores are heuristic screening aids, not confidence values or unique system-identification results. The `evidence/` and `figures/` directories are ignored so that the local investigation archive is not accidentally staged with the scripts.

Selected primary references: [Crazyflow paper](https://arxiv.org/html/2606.01478v3), [Crazyflow pipeline extension](https://learnsyslab.github.io/crazyflow/user-guide/pipelines/), [default offboard estimator implementation inspected](https://github.com/learnsyslab/drone-estimators/blob/e4b0da03aca553ff99d9e5cd52d4597e17aba17c/drone_estimators/ros/node.py), [firmware motor allocation](https://github.com/bitcraze/crazyflie-firmware/blob/2026.08/src/modules/src/power_distribution_quadrotor.c).
