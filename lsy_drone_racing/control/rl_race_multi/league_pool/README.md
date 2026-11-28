# Frozen opponent league pool

Drop trusted local opponents into the matching category under `weights/`, or set
`--league_weights=OTHER_FOLDER`. All `*.pkl` files are scanned recursively, once
when a fresh Multi stage starts. Existing files directly under `weights/` also work.
No import/conversion command, training job or sim-to-real wrapper is needed.

* **Recurrent v3 checkpoints:** copy `race_final.pkl`, `multi_final.pkl`, etc. under unique
  names. One file contains the architecture, normalizer and action mapping. LSTM widths and
  MLP head sizes may differ from the learner. Solo checkpoints receive zero opponent features;
  Multi checkpoints receive the other drone's features.
* **Existing `rl_race` MLP weights:** copy a pickle and its original stage config as a
  same-stem pair, e.g. `H1.pkl` + `H1.json`. Rename the original `race_config.json` (or
  `anneal_config.json` for those weights) to `H1.json`; do not construct new metadata by hand.
  The old pickle has no architecture/action metadata. Each MLP uses its original 49 ego
  features, normalizer and physical action limits, including differing tilt limits.

The learner remains an LSTM. MLP weights are frozen opponents, not learner resume weights.
Supported external MLPs use the existing two-gate/four-obstacle 49D layout, tanh-normal actor,
and 50 Hz attitude commands. Incompatible metadata fails before rollout.

Weight files in `weights/` are ignored by Git. Each training run saves self-contained copies in
`RUN/league/fixed/`, folding MLP sidecar metadata into the copy, and records the pool in
`RUN/league/manifest.json`. Resume and evaluation use that recorded pool; later folder changes
do not silently alter it. Passing an explicit folder alongside a manifest extends the pool.

With manual opponents, `fixed_opponent_fraction=0.25` reserves 25% of episodes for the fixed
set (uniform among the original solo and manual opponents), and 75% for historical Multi
snapshots (`rank**sampling_exponent`). Before history exists all episodes use the fixed set.
Manual files never get replaced. `league_capacity` includes the original solo, manual files
and history slots; leave room for history. Episode opponents change only on world reset.
With no manual files, the original `solo_opponent_fraction` sampling is preserved.

Pool inference is part of training's wrapper. This folder contains no simulator calibration
or sim-to-real adaptation code. Different inference architectures add compilation and memory
cost; an empty directory preserves the original single-architecture path.

## Files and inference

```text
league_pool/
  __init__.py  # Model contracts, LSTM inference and dispatch
  brax_mlp.py  # Brax 49D MLP loading, inference and simulator controller
  weights/
    mlp/       # Brax MLP .pkl + same-stem .json in this directory
    lstm/
      solo/    # LSTM stage=pretrain or race; .pkl is self-contained
      multi/   # LSTM stage=multi; .pkl is self-contained
```

These directories organize models for people. Model kind, architecture and solo/Multi
behavior are still read from metadata; moving a file does not rewrite its stage.
Sampling shares apply to individual models, not categories: splitting directories
does not give a category an extra share. Keep each legacy MLP's JSON next to its pickle.

`../league.py` manages sampling and snapshots; the environment wrapper calls the
adapters here. Production code never imports the old `control/rl_race` package.
The MLP path matches its unclipped normalization and tanh-normal action mapping.
The LSTM path shares this project's network implementation, without copying a
second architecture. Model structure, weights, normalizer and physical action
limits stay attached to each opponent. File names do not select architecture.

A Multi-trained LSTM receives the learner's relative position, relative velocity,
progress difference and presence from the opponent's own frame. A solo LSTM gets
eight zeros. MLPs consume their original 49 ego features. Supported policies
control one drone with four actions; a central policy jointly emitting both drones'
actions needs another adapter. LSTM memory is stored at the largest pool width,
sliced to the selected network and zero-padded afterwards. It resets when that
drone finishes/crashes or the world resets, including on opponent changes.

Sampling shares refer to **episode starts**, not simulator timesteps. Fixed
opponents are uniform within their reserved share. History has chronological
weights `k**sampling_exponent`, oldest `k=1`. An already active world retains its
opponent when a new snapshot arrives. Different episode lengths can therefore
produce different proportions when counting individual timesteps.

## Run a pooled MLP in the official simulator

Use the same `.pkl` and metadata as training; no export/conversion command is
required. In a **copy** of the desired official TOML, select this file for either
controller entry (and keep a second entry for the other drone):

```toml
[[controller]]
file = "rl_race_multi/league_pool/brax_mlp.py"
weights = "/absolute/path/to/league_pool/weights/mlp/H1.pkl"
```

The same entry with `[controller]` works for single-drone simulation. Absolute
weight paths are easiest; relative paths resolve against this pool directory.
A run's self-contained `RUN/league/fixed/*.pkl` copies also work without sidecars.
The controller has independent state per rank and holds each trained 50 Hz action
for two calls when the official runner schedules that drone at 100 Hz.

```bash
pixi run -e gpu python scripts/multi_sim.py \
  --config=/tmp/pool_race.toml --n_runs=1 --render=False
```

Existing recurrent export/controller commands remain available for LSTM models.
This change does not modify official TOMLs, rewards or race termination rules.

Future distilled opponents can be added through a format-specific adapter with
an explicit observation/action/clock/memory contract. Online MPPI execution,
PyTorch checkpoint loading and distillation training are not implemented here.
