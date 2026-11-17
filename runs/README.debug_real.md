# Real-flight debugging evidence

This branch collects original hardware recordings, three instrumented official simulation
runs, English analysis reports, comparison plots, trajectory exports, and an issue draft.
No analysis performed here establishes a single complete cause of the gate 3 failures.

## Start here

1. [Hardware recording inventory](deployment_analysis/inventory.md): every finalized
   hardware recording available at packaging time, including startup failures and interrupted
   flights. Counts, outcomes and config flags are derived from raw data, not inferred from
   filenames. `pending_recordings.json` lists any live recording excluded from this snapshot.
2. [Simulation versus real ordering comparison](deployment_analysis/sim_order_check/report.md)
   and [comparison plot](deployment_analysis/sim_order_check/sim_vs_real_swap.png): three
   actual official simulation episodes, all completing, with the same obstacle 1/3 feature
   exchange observed in the corrected-height hardware flight.
3. [Corrected obstacle 1 flight](deployment_analysis/flight_113735_report.md): the large height
   mismatch is gone, but a rank exchange remains associated with a sharp command change.
4. [Return-home issue draft](deployment_analysis/guide_home_issue/issue.md) and
   [return plot](deployment_analysis/guide_home_issue/return_home_evidence.png): out-of-bounds
   braking targets, terminal states, command timeline, and two complete return trajectories.

## What the evidence supports

- Early flights expose obstacle 1 at about 0.43 m instead of nominal 1.55 m. Exact-model,
  same-state single-step replay attributes a large command cut to this input change.
- In `deploy_20261117T113735.863815Z`, obstacle 1 is approximately 1.584 m. At 2.149 s,
  feature order changes from [4,1,3,2] to [4,3,1,2], requested thrust falls from 0.356 to
  0.131 N, and requested roll changes by about 45.5 degrees. Keeping only the previous
  slot order at the same recorded state yields 0.315 N.
- Simulation also exhibits that exact order exchange. Three consecutive randomized resets
  with runtime seed 0 complete in 3.42, 3.56 and 3.48 s. Their exchange states have roughly
  42-46 degree tilt, compared with 74 degrees in the hardware flight. The hardware state
  diverges before the exchange. Sorting alone is not a sufficient failure explanation.
- The first two fully recorded return sequences request braking targets beyond the configured
  y boundary. Host policy commands cease and return commands are submitted, followed by
  near-floor motion and boundary violations. Host send completion does not establish
  firmware receipt, control-mode acceptance, or correct execution.

## Layout

```text
runs/deploy_<UTC timestamp>/
  events.jsonl            Original append-only host recording; never rewritten by analysis
  metadata.json          Resolved config, source/model identities and clock information
  summary.json           Recorder integrity; complete does not mean flight success
runs/deployment_analysis/
  inventory.{md,json}    Index of finalized hardware runs
  trajectories/         Mocap CSV and policy observation/action CSV for each finalized run
  report.md              Initial full-flight/return analysis
  race_only_report.md    Initial two race-only analyses
  latest_race_report.md  Specifically the 10:46 and 10:48 UTC flights (historical filename)
  flight_*_report.md     Detailed individual-flight analyses
  sim_order_check/       Three full sim records, ordering CSVs, metrics, plots and scripts
  guide_home_issue/      English issue draft and structured evidence
  guide_home_evidence.zip  Portable copy of the return-home issue evidence
  SHA256SUMS             SHA256 integrity manifest, paths relative to repository root
```

The directory named `latest_race_*` refers to a fixed historical pair; use the inventory
to find the most recent run. Some additional runs are archived without detailed causal
analysis. Policy model files already tracked in the repository are required for inference
replay; scripts check model hashes where applicable.

## Interpreting the records

- Policy observations/actions are captured at the configured 50 Hz; external-pose submissions
  are normally about 30 Hz. They have host timestamps, not sensor acquisition timestamps.
- `action` is the environment command. `policy.policy_action` is the normalized action;
  `policy.features` is before the policy's internal normalization. Previous action is recorded.
- Raw mocap and the offboard filtered state share a measurement source. Neither establishes
  onboard estimator quality or radio delivery. No individual motor, battery, or onboard
  setpoint/supervisor telemetry was captured in these recordings.
- A mocap z value is the tracking reference height, not physical propeller clearance.
- Counterfactual policy evaluations change the indicated current input while preserving the
  recorded state/history. They are not complete trajectories or validated control fixes.
- `cleanup` in exported CSV is a host phase and can include return, landing or emergency stop.
- `summary.complete` indicates recorder finalization, not race completion or return success.
- The third transmitted attitude field must be interpreted using the configured firmware
  yaw mode. A recorder field named `yaw_rate_deg_s` does not establish that mode.

## Reproduction

Run from the repository root with the project's deploy environment. These commands are
**offline** and do not connect to a drone:

```bash
# Export CSVs and refresh the inventory/integrity manifest from finalized records.
.pixi/envs/deploy/bin/python runs/deployment_analysis/build_inventory.py

# Regenerate original fixed-pair reports/plots and individual-flight diagnostics.
MPLCONFIGDIR=/tmp/drone-analysis-matplotlib .pixi/envs/deploy/bin/python runs/deployment_analysis/analyze.py
.pixi/envs/deploy/bin/python runs/deployment_analysis/race_diagnostics.py
.pixi/envs/deploy/bin/python runs/deployment_analysis/latest_race_diagnostics.py
.pixi/envs/deploy/bin/python runs/deployment_analysis/check_110059.py
.pixi/envs/deploy/bin/python runs/deployment_analysis/check_113735.py

# Re-run three simulations through official scripts.sim.simulate, or compare saved episodes.
.pixi/envs/deploy/bin/python runs/deployment_analysis/sim_order_check/capture_sim.py
.pixi/envs/deploy/bin/python runs/deployment_analysis/sim_order_check/compare.py

# Rebuild the return-home CSV/JSON/plots from the original two flights.
.pixi/envs/deploy/bin/python runs/deployment_analysis/guide_home_issue/prepare.py

# Verify the delivered snapshot before regenerating artifacts.
sha256sum -c runs/deployment_analysis/SHA256SUMS
```

Scripts pin their historical source runs so adding a new flight does not silently change
which report they reproduce. Re-running simulation may depend on installed library versions;
the saved observations and commands are the evidence for the delivered analysis. Plot
regeneration can change PDF/PNG bytes; refresh the manifest afterward if intended.

The historical raw metadata reports a dirty checkout. The recorded base commit alone is
not an exact snapshot of the original flight software. This evidence-only staging does not
include the separate working-tree changes to deployment code, tests or the current config.
The sim capture script obtains diagnostics independently of those uncommitted recorder changes.
