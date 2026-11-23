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
2. `race`: solo sparse reward; success ≥ `sparse_switch_success` (default 0.9)
   freezes this policy as the solo opponent and starts multi training.
3. `multi`: one learner and one frozen opponent in the **same two-drone world**.
   Only learner trajectories enter PPO. No IPPO or shared-policy updates.

Both solo stages always evaluate periodically. `intermediate_eval=False`
(default) disables only periodic **multi** evaluation. Independent final
evaluation still runs. A solo budget exhausted below its threshold saves a final
checkpoint, records `success_threshold_not_reached` in `stages.json`, and stops;
it never silently advances. Low thresholds in smoke tests exercise control flow,
not successful learning. Both transitions preserve actor/LSTM weights and the
normalizer. `reset_critic=True` reinitializes the critic and freezes actor updates
for `critic_only_iterations` (default 10), then uses `policy_lr_after_switch`.

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
# Full recipe; this is a long training run, not the acceptance test.
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
machines. Default training worlds are 128, not 16,384; benchmark memory/performance
before increasing them with recurrent league inference.

## League

Slot zero is the frozen solo policy. With history, sampling is 25% slot zero and
75% history, with historical weight `k**0.9` (oldest k=1). Without history it is
100% solo. Settings: `solo_opponent_fraction`, `sampling_exponent`,
`snapshot_interval` (default 100 PPO iterations), `league_capacity` (default 16).
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
