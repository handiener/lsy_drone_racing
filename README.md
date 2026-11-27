# Flight logs

This is a data-only branch of `lsy_drone_racing` (orphan, with no code history). It
collects every real-world deployment recording in one place. Code and experiments live on
the normal branches. Analysis scripts read the logs from a worktree of this branch.

## Setup

```bash
# from the lsy_drone_racing checkout
git fetch origin flight-logs
git worktree add ../lsy_flight_logs flight-logs
```

## Layout

```text
runs/<YYYY-MM-DD>/<run_id>/   one folder per deploy.py run, files unchanged
  events.jsonl                host recording: mocap poses sent, observations, actions,
                              attitude commands (collective PWM), loop timing, e-stop
  metadata.json               resolved config, code commit, clock anchor
  summary.json                recorder integrity (complete != flight success)
  onboard.jsonl               firmware telemetry: IMU, motor PWMs, battery,
                              controller commands (from 24 Nov on)
labels.csv                    hand-written: policy, model, action thrust_max, notes
runs.csv                      generated index (do not edit): python tools/build_index.py
notes/                        lab-day notes
```

`runs.csv` columns: `outcome` is `finished`, `safety_bound`, `ctrl_c`, `no_flight`
or `terminated_other`. `vbat_*` come from the onboard log and need motors running.

## Adding a lab day

1. Copy the new `runs/deploy_*` folders from the deployment PC to `runs/<date>/`.
2. Add one row per run to `labels.csv`.
3. Run `python tools/build_index.py`, then commit and push.

## Known issues in the data

- **24 Nov, new track:** the deploy toml had `pos_limit_high` x = 1.5 m while the split-S
  gates are at x = 2.0 m. Every run reaching the split-S ended at x ≈ 1.5 as
  `safety_bound`. This was the software limit, not a crash.
- **17 Nov:** host recordings only (no onboard log). Gate-3 debugging; see
  `notes/2026-11-17_debug_real.md`. The analysis files referenced there are on
  `origin/debug_real` under `runs/deployment_analysis/`.
- Sources: runs were copied byte for byte from `origin/debug_real`,
  `origin/logging-no-simtoreal`, `origin/simtoreal-logging-26-11-24` and
  `origin/26-11-24_log-multi-course`.
