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
Architecture sweeps may override these sizes. The learner and historical snapshots
share an architecture; manual frozen opponents may use other LSTM sizes or the
existing single-racer MLP, with their own normalizers and action mappings. MLP
opponents read the original 49 ego features. There is no weight conversion.

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

## Independent solo training

All stages use `train.py` and the same `configs/full.yaml`. The default runs
Dense → Sparse → Multi; `end_stage=pretrain` stops after Dense and `end_stage=race`
stops after Sparse. `start_stage` and `end_stage` select a contiguous part of the
curriculum. Stage selection does not change network, reward, reference, budget or
evaluation settings. This is pure simulation, with no anneal or sim-to-real adapter.

```bash
# Train/evaluate Dense + Sparse with the same settings used before Multi.
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.train \
  --config_file=lsy_drone_racing/control/rl_race_multi/configs/full.yaml \
  --end_stage=race --run_name=lstm_solo_s1

# Train only Dense and save/evaluate its checkpoint.
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.train \
  --config_file=lsy_drone_racing/control/rl_race_multi/configs/full.yaml \
  --end_stage=pretrain --run_name=lstm_dense_s1

# Train Sparse separately from a chosen Dense checkpoint (default architecture).
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.train \
  --config_file=lsy_drone_racing/control/rl_race_multi/configs/full.yaml \
  --end_stage=race \
  --start_stage=race --start_params=PATH/checkpoints/pretrain_final.pkl \
  --run_name=lstm_sparse_s1

# Start Multi directly from the chosen Sparse checkpoint (default architecture).
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.train \
  --config_file=lsy_drone_racing/control/rl_race_multi/configs/full.yaml \
  --start_stage=multi \
  --start_params=PATH_TO_SOLO_RUN/checkpoints/race_final.pkl \
  --solo_checkpoint=PATH_TO_SOLO_RUN/checkpoints/race_final.pkl \
  --run_name=lstm_multi_s1
```

For a custom solo network/action mapping, supply the same `hidden_size`,
`policy_layers`, `value_layers`, activation and action limits when starting Multi.
The checkpoint is validated before rollout.

`stop_on_success=False` lets the selected final solo stage use its full budget.
Earlier stages must still meet their thresholds before advancing.

`reference_path=hermite|dynamic` selects the same guides and path-start behavior as
`rl_race`. The shared recipe preserves the Hermite default; set `reference_path=dynamic`
in that recipe or on the CLI to select the dynamic guide for either workflow.
The bundled guide and [offline generator](env_wrapper/reference_generator/README.md)
live in this package; runtime training does not import `rl_race`. The reference
shapes Dense progress and solo path starts; it adds no dense reward to Sparse/Multi.

Architecture comparisons use fresh solo runs: sweep `hidden_size`, `policy_layers`,
`value_layers` and `activation`. When starting Sparse from a Dense checkpoint, retain
the source architecture and action mapping; resizing loaded weights is rejected.
[`sweeps/architecture.yaml`](sweeps/architecture.yaml) compares architectures/references
through the existing sweep launcher and scores the selected final stage. It loads
`configs/full.yaml` first, then applies sampled values, including values equal to
Python defaults. Set its `end_stage` to
`pretrain` for Dense-only experiments, or `race` for Dense → Sparse experiments.
Neither requires a Multi run.

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
The TOMLs remain the source for track geometry, safety bounds, simulator dynamics,
sensor range, disturbances and randomizations. Incompatible training clocks/control
modes fail explicitly; they are not silently rewritten. Headless training only
switches rendering off in the loaded config. Every run pins copies of both TOMLs.
The existing Multi start-zone exception is preserved: ±0.13 m horizontal
randomization and a wrapper minimum pair separation of 0.22 m (commit `71c89a4`).
This is a deliberate historical customization, not an untouched official start rule.

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

One PPO iteration collects `batch_size × num_minibatches × unroll_length`
learner transitions: `4096 × 4 × 32 = 524,288` with the shared recipe. It makes
`4 × 4 = 16` optimizer updates; world count controls collection parallelism.

| Stage | Maximum transitions | PPO iterations | Success gate |
| --- | ---: | ---: | ---: |
| Dense | 2,013,265,920 | 3,840 | 30% |
| Sparse | 2,516,582,400 | 4,800 | 70% |
| Multi | 2,684,354,560 | 5,120 | full budget |

Solo evaluates at the initial state and every 167,772,160 transitions, with 128
worlds. Multi periodic evaluation is off by default. Final evaluation uses 16
episodes per opponent, seed 410000 and batch 16. These are short checks; use more
episodes/seeds when comparing trained policies. Metrics and measured stage counts
are saved even when a success gate stops the curriculum.

The 80-iteration critic-only phase is 1,280 optimizer updates / 41,943,040
transitions. Multi snapshots occur every 256 PPO iterations / 134,217,728
transitions; the full budget creates 20 snapshots. Budgets round to whole host
epochs; the shared recipe is already aligned to those boundaries.

Historical RTX 5090 calibration (2026-11-24) measured roughly 1.13 s/iteration and
18.69 GiB for Solo, 2.52 s/iteration and 20.88 GiB for Multi with 41 actors sharing
one architecture. These are throughput probes, not convergence evidence or memory
bounds for new architectures. Raw records: `/tmp/rl_multi_gpu_calibration_20261124/`.
No automatic wall-clock cutoff or background training is enabled.

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
step counter continues from the loaded checkpoint; local stage iteration counters restart.
Legacy MLP weights are rejected for learner resume, but supported as frozen opponents.
Each worker currently uses one local GPU; independent W&B agents support multiple
machines. Default training worlds are 16,384; the short measurements above do not
bound every long-run memory peak or state-dependent physics workload.

## League

Drop uniquely named checkpoints into [`league_pool/weights/`](league_pool/README.md),
or select `--league_weights=OTHER_FOLDER`. Recurrent v3 `.pkl` files are self-contained;
old single-racer MLP weights need a same-stem `.json` copied from their stage config.
The pool folder contains inference adapters as well as weights; its MLP adapter also
works as a standalone official-simulator controller without importing `rl_race`.
Each opponent retains its original normalization, action limits and LSTM width.
Solo policies receive no opponent features; Multi policies observe the learner.

With manual weights, `fixed_opponent_fraction=0.25` reserves a uniform 25% share
for the original solo plus manual fixed policies; 75% samples historical snapshots
with weight `rank**sampling_exponent` (default exponent 0.9). Before history exists
all episodes use the fixed set. With an empty manual folder, original sampling is
preserved: `solo_opponent_fraction=0.0` means 100% history once history exists.

All policies load once on the host, outside rollout/JIT. One inference bank per
architecture retains stable array shapes; weights and normalizers remain frozen.
`deterministic_opponent=True` uses mean actions. Opponent changes happen only on
world reset, and each opponent has independent memory. At `league_capacity=64`,
only the oldest inactive historical slot can be replaced; fixed/active slots stay
protected. If no slot can be replaced, the snapshot is archived without insertion.
`league/snapshots.jsonl` records the result.

Each run stores fixed copies and a `league/manifest.json`. Resume/evaluation read
that recorded pool; later directory changes do not silently alter old results.
An explicit weights directory can extend a restored pool. Files added during a
running job are not hot-loaded. See the folder README for file formats and capacity.

A local manifest can also collect already evaluated policies, without W&B:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_race_multi.league \
  --checkpoints='("RUN_A/checkpoints/multi_final.pkl","RUN_B/checkpoints/multi_final.pkl")' \
  --output=league_pool.json --minimum_success=0.5
```

This older manifest builder requires shared architecture/action metadata and final evaluation
reports. Inspect those results and trajectories for diversity; reward/seed/style
names do not prove different behavior. `league_manifest` loads this recorded pool
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
  tests/unit/control/test_rl_race_multi_extensions.py \
  tests/integration/test_rl_race_multi.py \
  tests/integration/test_rl_race_multi_extensions.py
```

Tests cover ego/opponent encoding, normalization transition, LSTM boundaries and
sequence replay, PPO gradients and parameter changes, physical-terminal timeout
bootstrap, critic reset without actor loss, pool sampling/compatibility, stage
switches, real joint physics, export equality and official-controller clocks.
Small smoke models are deliberately untrained; passing these tests does not
establish racing performance. No hours-long recurrent training is part of acceptance.
