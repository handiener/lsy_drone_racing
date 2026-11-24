# Recurrent racing and frozen-opponent league PPO

This package is independent of `rl_race`. It retains the vendored Brax PPO's
sequence minibatch loop, clipped objective, GAE, value/entropy losses, gradient
clipping and separate actor/critic optimizers. Crazyflow and official environments
are unchanged. Run commands from the repository root.

## Model and data flow

Every stage uses 57 observations: the original 49 ego features in their original
order, then body-frame relative opponent position (3), relative velocity (3),
`(opponent gates - ego gates) / sequence length` (1), and presence (1).
Solo policies always receive eight zeros. A historical multi policy observes the
other drone from its own perspective. Gate lookahead is fixed at two, with four
obstacles; incompatible dimensions are rejected.

Actor and critic have **independent LSTM(256) → MLP(512,512)** heads, LeakyReLU,
and the original tanh-normal `[roll,pitch,yaw,thrust]` distribution. Defaults are
80° roll/pitch, 70° yaw, seed 1. There is no MLP-to-LSTM weight conversion.
Architecture sweeps may override these sizes, but all opponents in a pool must
exactly match its architecture, distribution and action scaling. No padding or
silent conversion occurs.

The wrapper carries each learner's actor/critic `(cell,hidden)` state across
rollout segments. Sampling records behavior log probabilities and initial
recurrent states. PPO shuffles complete sequences, recomputes them using new
weights, and detaches their initial carry for truncated BPTT (default 32 steps).
Episode boundaries within sequences reset both carries. Physics is not
backpropagated. A time limit bootstraps the **physical terminal observation**
with its un-reset critic carry; GAE stops at the reset boundary. Observation
statistics stay fixed throughout an update and change only after all SGD epochs.
Opponent dimensions have a unit standard-deviation floor, and absence forces
those dimensions to zero, including after solo→multi transitions.

## Three stages

1. `pretrain`: solo dense-path reward and the existing path-start mechanism.
   Official-start evaluation success ≥ `switch_success` (default 0.3) switches to race.
2. `race`: solo sparse reward; success ≥ `sparse_switch_success` (default 0.7)
   freezes this policy as the solo opponent and starts multi training.
3. `multi`: one learner and one frozen opponent in the **same two-drone world**.
   Only learner trajectories enter PPO. No IPPO or shared-policy updates.

`race_time=0.05` is the default time penalty per control step in both `race`
and `multi`: 2.5 reward units per second at 50 Hz, six times lower than the
previous `0.3`. This leaves room for the longer new track's gate/finish rewards
to outweigh the time cost. Dense pretraining uses `pretrain_time=0.2`.
Explicit CLI/YAML values, including historical sweep settings, override defaults.
All three stages default to `max_episode_steps=1250`, or 25 seconds at 50 Hz,
for training and evaluation. Saved runs retain their recorded episode limits.

Both solo stages always evaluate periodically. `intermediate_eval=False`
(default) disables only periodic **multi** evaluation. Independent final
evaluation still runs. A solo budget exhausted below its threshold saves a final
checkpoint, records `success_threshold_not_reached` in `stages.json`, and stops;
it never silently advances. Low thresholds in smoke tests exercise control flow,
not successful learning. Both transitions preserve actor/LSTM weights and the
normalizer. `reset_critic=True` reinitializes the critic and freezes actor updates
for `critic_only_iterations` (default 80), then uses `policy_lr_after_switch`.

The default tracks are **new** `level2.toml` and `multi_new_level2.toml`, not the
old `multi_level2.toml`. Copies are pinned into each run. Multi training uses
50 Hz for both drones, 500 Hz physics, official starts, and optional randomized
start-slot exchange. This does not modify the official 50/100 Hz runner.

Training ends a world on the first crash, completion or timeout. A win requires
actual completion before the other drone; an opponent crash alone is a draw.
Both crashing is a draw. Reasons, actual gates, finish, crash, inter-drone contact,
bounds, timeout and outcome components are logged. Reset happens before the
next sample, so a disabled learner does not keep contributing invalid experience.

Multi reward extends sparse racing with conservative progress, change in relative
progress, proximity and outcome terms. Inter-drone contact is identified from
MuJoCo contacts before disabled drones warp; its configured total crash cost
replaces the base crash cost rather than adding a second full crash penalty.
Defaults can be changed through `competition_progress`, `ranking_reward`,
`proximity_penalty`, `proximity_distance`, `inter_collision_penalty`, `outcome_reward`.

## Train or resume

```bash
export SCIPY_ARRAY_API=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90 MUJOCO_GL=egl
# RTX 5090 recipe, calibrated with short real PPO runs (not a convergence test).
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.train \
  --config_file=lsy_drone_racing/control/rl_race_multi/configs/full.yaml \
  --run_name=recurrent_league_s1

# Small complete flow test: 2 worlds, short sequences, deliberately zero thresholds.
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.train \
  --config_file=lsy_drone_racing/control/rl_race_multi/configs/smoke.yaml \
  --run_name=recurrent_smoke
```

`--config_file` loads training keyword arguments from YAML. Nondefault CLI
arguments override file values; to explicitly set a value equal to the Python
default, set it in the YAML. Unknown keys fail validation. Without a file, all
options are available directly as `--name=value`; see `train --help`.

`configs/full.yaml` and function defaults share the calibrated recipe:
16,384 worlds, batch size 4,096, 4 minibatches, unroll 32, and 4 update epochs.
Network/action defaults remain LSTM(256), MLP(512,512), two gates, 80° roll/pitch,
70° yaw, seed 1. `configs/smoke.yaml` is independent and unchanged.

The reference `rl_race` sweep winner `1u3vhuhn_s1` used 16,384 worlds, batch 4,096,
32 minibatches, 4 epochs, learning rates 3e-4 / 1e-4 and dense/sparse budgets of
2B / 12B. Its dense stage actually needed 251,658,240 transitions to switch. The
previous 20,971,520-transition recurrent dense budget stopped without a single
successful official-start episode. The recurrent recipe keeps the reference
world count, gradient minibatch size, learning rates and four update epochs,
but collects fewer sequences per iteration to fit recurrent PPO and opponents.
The overnight recipe below reduces the reference sparse data allowance and switches at 70% to leave more wall-clock time for Multi;
it cannot guarantee reaching the threshold before the sparse cap.

One iteration collects `4096 × 4 × 32 = 524,288` learner transitions and makes
`4 × 4 = 16` optimizer updates. Budgets are aligned to host epochs:

| Stage | Transition budget | PPO iterations | Host epoch size |
| --- | ---: | ---: | ---: |
| Dense | 2,013,265,920 | 3,840 | 320 iterations |
| Sparse | 2,516,582,400 | 4,800 | 320 iterations |
| Multi | 2,684,354,560 | 5,120 | 256 iterations |

Increasing worlds only changes collection parallelism: worlds are NOT a factor
in `batch_size * num_minibatches * unroll_length`. These are maxima; real success
thresholds can switch stages earlier. The dense cap approximates the reference's
nominal 2B allowance. The sparse cap is 20% of its actual 12,582,912,000
transitions, explicitly trading solo learning allowance for the overnight target.
Sparse allows `4,800 × 16 = 76,800` optimizer updates. Restoring the reference
sparse allowance would require 24,000 iterations / 384,000 optimizer updates
and about 7.5 hours for Sparse alone. Critic-only warmup is 80 iterations / 1,280 optimizer updates /
41,943,040 transitions, matching the reference's 10 larger iterations.

Short GPU calibration on 2026-11-24 used the local RTX 5090 (32,607 MiB), the
real simulator and full recurrent PPO backward/optimizer updates. Each isolated
stage ran six iterations. Times below are medians of the last four iterations,
after compilation; memory is the process peak sampled with `nvidia-smi`, including
allocator caching, with preallocation disabled and allocator fraction `.90`.

| Workload | Process peak VRAM | Seconds / iteration | Training time at full stage cap |
| --- | ---: | ---: | ---: |
| Solo Dense | 18.69 GiB | 1.13 | 1.20 h |
| Solo Sparse | 18.69 GiB | 1.13 | 1.51 h |
| Multi, 41 distinct frozen actors | 20.88 GiB | 2.52 | 3.59 h |

The heterogeneous pool used perturbed checkpoint copies solely to exercise
different opponent parameters, not trained competitors. GPU utilization reached
100% during these probes. These short runs establish a working size and estimated
throughput, not convergence or constant utilization. Solo uses less VRAM than
Multi; `.90` limits the allocator rather than reserving unused memory.
The smaller 1,024-batch / 16-minibatch trial used only 11.17 GiB in Solo and took
about 1.39 s/iteration. At batch 4,096, Multi with 8,192 worlds used 20.85 GiB
and took about 2.51 s/iteration; doubling to 16,384 worlds was effectively tied.
The default keeps the reference single-run world count.

Full caps total about **6.3 hours of training**, leaving about 40 minutes within
a seven-hour target for compilation, evaluation, snapshot/checkpoint writes and
host overhead. Multi has more time than either solo stage. Stage switching can
shorten the run, but short calibration cannot guarantee a seven-hour deadline
or success. There is no automatic timeout or forced stage advancement.
Raw calibration measurements are in `/tmp/rl_multi_gpu_calibration_20261124/`;
temporary scripts and calibration checkpoints are outside the repository.

For an overnight run with all defaults, open `tmux new -s multi_night` from the
repository root, then run:

```bash
SCIPY_ARRAY_API=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 MUJOCO_GL=egl \
pixi run -e gpu python -u -m lsy_drone_racing.control.rl_race_multi.train \
  --run_name=20261124_5090_calibrated_s1
```

Detach with **Ctrl+B**, then **D**; SSH can disconnect while training continues.
Reconnect with `tmux attach -t multi_night`. No sweep or automatic seven-hour
cutoff is enabled. In the morning, inspect the run's `stages.json` for completed
stages and `metrics.jsonl` for recent training/evaluation metrics. A run can still
be training after seven hours with these budgets.

Solo evaluation uses 128 worlds every 167,772,160 transitions (320 iterations),
plus its initial evaluation. Multi intermediate evaluation remains off. Budgets
divide epoch sizes exactly; there is no epoch-rounding overrun in this recipe.
Solo switches require 30% / 70% success. A stage stops and saves below
threshold when its budget is exhausted; it never advances merely to use the GPU.

Terminal output is limited to short evaluation summaries and stage start/end
messages. Per-iteration metrics are still recorded in `metrics.jsonl` and W&B,
without dumping complete metric dictionaries to the terminal. With multi eval
turned off, its terminal can be quiet until the stage ends.

The short final check remains 16 episodes per opponent, seed 410000, batch 16.
A fresh complete Multi stage has Solo + 20 snapshots, hence 336 final episodes.
Increase episodes and seeds for model comparisons. Standalone evaluation defaults
remain larger. Short calibration does not establish training success.

Outputs default to `rl_race_multi/outputs/RUN/`: settings, pinned tracks,
per-stage environment/network metadata, JSONL metrics, stage status,
`checkpoints/{pretrain,race,multi}_{latest,final}.pkl`, league manifest and models,
plus final evaluation/episode reports. Use a new run name when resuming:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.train \
  --start_stage=multi --start_params=PATH/checkpoints/multi_latest.pkl \
  --solo_checkpoint=PATH/checkpoints/race_final.pkl \
  --league_manifest=PATH/league/manifest.json --run_name=league_resume
```

Resume architecture/action settings must match the checkpoint (supply the same
YAML or arguments). `start_stage=race` similarly accepts a recurrent dense
checkpoint. Weights, normalizer and league models are restored; **Adam state,
RNG, partial rollout, recurrent carry and local stage step counters restart**.
This is a warm resume, not bitwise continuation. The cumulative log/checkpoint
step counter continues from the loaded checkpoint; local stage iteration counters restart. Legacy MLP exports/checkpoints are rejected.
Each worker currently uses one local GPU; independent W&B agents support multiple
machines. Default training worlds are 16,384; the short measurements above do not
bound every long-run memory peak or state-dependent physics workload.

## League

Slot zero is the frozen solo policy. With history, the default
`solo_opponent_fraction=0.0` samples 100% historical policies, with historical
weight `k**0.9` (oldest k=1). Without history it is 100% solo. Selection changes
only at episode resets: an already running solo-opponent episode is not interrupted
when the first historical snapshot arrives. Settings: `solo_opponent_fraction`,
`sampling_exponent`, `snapshot_interval` (default 256 PPO iterations),
`league_capacity` (default 64). One snapshot interval is 134,217,728 transitions;
the full Multi budget creates exactly 20 periodic snapshots. Solo + 20 snapshots
use 21 slots, leaving 43 free in a fresh run without an external pool.
Snapshots are taken on host epoch boundaries. Multi budgets round up to whole
snapshot intervals; actual counts are recorded. Frozen opponents default to
mean actions (`deterministic_opponent=True`) and have independent actor carry
and their own normalizer. Evaluation/deployment always use mean actions.

All models load before rollout. There is no disk or W&B access inside a JIT or
per-environment Python controller loop. Snapshots use fixed-size JAX arrays. At capacity, only the oldest history slot
not referenced by any live world may be replaced; slot zero is permanent. If all
slots are occupied, the checkpoint is still archived but its pool insertion is
skipped for that interval. `league/snapshots.jsonl` records these decisions.
Chronological ranks keep newer checkpoints more likely even after slot reuse.
An opponent's weights and normalizer stay fixed for its entire episode; archive
storage grows with snapshot count, so size the interval and capacity accordingly.

A local manifest can also collect already evaluated policies, without W&B:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.league \
  --checkpoints='("RUN_A/checkpoints/multi_final.pkl","RUN_B/checkpoints/multi_final.pkl")' \
  --output=league_pool.json --minimum_success=0.5
```

Models must share architecture/action metadata and have actual final evaluation
reports. Inspect those results and trajectories for diversity; reward/seed/style
names do not prove different behavior. `league_manifest` loads this fixed pool
alongside the required frozen solo checkpoint. Manifests retain checkpoint paths,
network/reward settings and evaluation provenance; actor and normalizer are
stored together in each checkpoint.

## W&B sweeps

The existing launcher, per-machine SDK/cache directories, timeout and artifact
budget controls are retained. Final objective is `final/leaderboard_score`;
completion, collision, win/loss/draw and opponent-specific rates remain visible.
A failed solo threshold leaves `training_complete=False` in the sweep summary.
Artifacts include recurrent checkpoint, stage/settings metadata and league data.
Artifact weight-budget estimates include opponent checkpoints.

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.sweep \
  --machine MACHINE --create lsy_drone_racing/control/rl_race_multi/sweeps/L2_sweep.yaml
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.sweep \
  --machine MACHINE --sweep ENTITY/PROJECT/ID --hours 10
```

`L2_sweep.yaml` compares recurrent performance settings; `league_diversity.yaml`
generates fast/safe/balanced/conservative **candidates**, requiring evaluation.
Both keep the default 256/512/512 architecture pool-compatible. `example.yaml`
is a seed comparison. None are launched automatically.

## Evaluation, export and official simulation

```bash
# Default multi evaluation uses complete races, fixed seeds and every frozen pool member.
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.evaluation \
  --run_dir=PATH_TO_RUN --stage=multi --episodes=128 --report_name=holdout
# --official=False explicitly selects the simplified first-event TRAINING rules.

pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.export \
  --run_dir=PATH_TO_RUN --stage=multi --destination=PATH_TO_RUN/exported
```

For official `scripts/multi_sim.py`, create a separate copy of the multi TOML
and select both controller entries (absolute model directories are supported):

```toml
[[controller]]
file = "rl_race_multi_controller.py"
model_dir = "/absolute/PATH_TO_RUN/exported"
[[controller]]
file = "rl_race_multi_controller.py"
model_dir = "/absolute/PATH_TO_SOLO_RUN/exported"
```

```bash
pixi run -e gpu python scripts/multi_sim.py \
  --config=/absolute/path/to/your_multi_config.toml --n_runs=1 --render=False
```

Export `--stage=race` for a frozen solo controller (eight opponent zeros).
Each rank has independent previous action and LSTM carry. `episode_reset()`
clears both. At 100 Hz, the controller holds each 50 Hz action for two calls and
advances its LSTM only once, preserving the trained clock; unsupported frequencies
fail explicitly. The official runner's rules and scheduling are untouched.

## Validation

```bash
SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu MUJOCO_GL=egl pixi run -e gpu-tests pytest -q \
  tests/unit/control/test_rl_race_multi.py \
  tests/unit/control/test_rl_race_multi_sweep.py \
  tests/integration/test_rl_race_multi.py
```

Tests cover ego/opponent encoding, normalization transition, LSTM boundaries and
sequence replay, PPO gradients and parameter changes, physical-terminal timeout
bootstrap, critic reset without actor loss, pool sampling/compatibility, stage
switches, real joint physics, export equality and official-controller clocks.
Small smoke models are deliberately untrained; passing these tests does not
establish racing performance. No hours-long recurrent training is part of acceptance.
