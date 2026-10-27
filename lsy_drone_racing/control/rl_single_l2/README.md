# L2 PPO Training

A single-drone Level 2 training framework using the official simulator and SBX PPO.

## Quick Start

Run all commands from the **repository root**. Install or update the official GPU environment:

```bash
pixi install --locked -e gpu
```

`--locked` uses the existing `pixi.lock` and refuses to update it during installation.

Then start GPU training with the default hyperparameters:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_single_l2.train --device=gpu
```

The official `gpu` environment is defined in `pyproject.toml`; Pixi manages it under
`.pixi/envs/gpu/`. There is no need to create a Conda environment or activate a
Pixi shell: `-e gpu` selects the environment for each command.
`pixi run` can also install a missing environment automatically.

The official `pip install -e ".[rl]"` command remains supported. `rl` is a Python
optional dependency group, not a separate Pixi environment. All PPO packages are now
in the base project dependencies, so installing `gpu` already includes them.

The run name is generated from the current time, and logs and checkpoints are saved
automatically. No other training arguments are required.

To name an experiment yourself, append `--run_name=l2_run_001`. Existing run directories
are preserved; use a new name for each experiment.

### Train with Evaluation and Save the Best Model

To use the default training settings **and enable evaluation**, run:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_single_l2.train \
  --device=gpu --eval_freq=1000000
```

This evaluates five episodes at each evaluation point and keeps the best evaluated model
in `checkpoints/best/`. It selects a model without stopping training early.
Omit `--eval_freq` or set it to `0` to disable evaluation.

### Command-Line Options

Append options after the training module name as `--name=value`. Omitted options use
the defaults below. `-e gpu` selects the Pixi environment; `--device=gpu` selects the
training simulator and JAX device. Append `--help` to the training command for CLI help.

**Run, files and evaluation**

| Option | Default | Purpose |
| --- | --- | --- |
| `--device=gpu` | `cpu` | Run the simulator and JAX policy on `gpu` or `cpu`. |
| `--run_name=l2_run_001` | Generated timestamp | Name the experiment folder; existing folders are preserved. |
| `--total_timesteps=155000000` | `155000000` | Cumulative transition target across all parallel environments. |
| `--n_envs=16384` | `16384` | Number of parallel simulation worlds. |
| `--batch_size=16384` | `16384` | Transitions in each PPO minibatch. |
| `--seed=42` | `42` | Training random seed. |
| `--eval_freq=1000000` | `0` (off) | Enable evaluation and best-model selection at this transition interval. |
| `--eval_episodes=5` | `5` | Episodes per evaluation; used when evaluation is enabled. |
| `--save_freq=20000000` | `20000000` | Interval between regular checkpoints; `0` disables periodic saving. |
| `--keep=3` | `3` | Keep this many recent regular checkpoints; the best model is kept separately. |
| `--mode=resume` | `new` | Choose `new`, `resume` or `warm_start`; the latter two require `--checkpoint`. |
| `--checkpoint="/path/to/step_..."` | None | Checkpoint directory to resume or warm-start from. |
| `--output_root="/path/to/outputs"` | Package `outputs/` | Parent directory for experiment folders. |
| `--config="config/level2.toml"` | Repository `config/level2.toml` | Path to the race TOML configuration. |

**PPO, episode and observation settings**

| Option | Default | Purpose |
| --- | --- | --- |
| `--n_steps=256` | `256` | Steps collected per environment before each PPO update. |
| `--n_epochs=3` | `3` | Training passes over each collected rollout. |
| `--learning_rate=0.0003` | `0.0003` | Optimizer learning rate. |
| `--clip_range=0.2` | `0.2` | PPO policy-ratio clipping range. |
| `--ent_coef=0.005` | `0.005` | Entropy bonus weight controlling the exploration incentive. |
| `--gamma=0.99` | `0.99` | Discount factor for future rewards. |
| `--gae_lambda=0.95` | `0.95` | GAE parameter for advantage estimation. |
| `--max_episode_steps=1500` | `1500` | Maximum control steps per episode before timeout. |
| `--eval_seed=10000` | `10000` | First evaluation seed; episodes use consecutive seeds. |
| `--gate_lookahead=2` | `2` | Number of upcoming gates included in the observation. |
| `--obstacle_count=4` | `4` | Number of nearest obstacle slots included in the observation. |
| `--alpha_max_rad=0.16` | `0.16` | Maximum proposed body rotation increment per control step, in radians. |

`batch_size` must be at least two and divide `n_envs × n_steps` exactly. Saving and
evaluation wait until a complete PPO update has finished, so their actual intervals
round up to rollout boundaries. A normal training finish saves a final checkpoint even
when periodic saving is disabled.

With the defaults, each rollout collects `16384 × 256 = 4,194,304` transitions and each
epoch has 256 minibatches. The reference budget rounds up to **37 rollouts, or
155,189,248 transitions**. Thus `--eval_freq=1000000` evaluates once per rollout at this scale.

For `resume`, saved PPO and environment settings take precedence over new values for
those settings. The total target and save/evaluation options apply to the continuation;
see [Resume or Start a New Experiment](#resume-or-start-a-new-experiment).

## Training and Simulation Environments

| Environment | Used for | Dependencies |
| --- | --- | --- |
| `gpu` | GPU training, resuming and exporting the Actor | Base project dependencies and CUDA-enabled JAX |
| `default` | CPU training, simulation and competition evaluation | Simulator, SBX, SB3, PyTorch, Flax, Optax, TFP and SciPy |

These are separate Pixi environments using the same repository and model files.
After exporting the Actor, the `default` environment loads it directly from `deploy_model/`;
no files need to be copied between environments. The commands below select each
environment explicitly. All PPO dependencies are declared in `[project.dependencies]`,
so both `default` and `gpu` install them. Update each local
environment you use with `pixi install --locked -e <environment>` after changing dependencies.

## How It Works

The official environment handles physics, Level 2 randomization, collisions and gate crossing.
The adapter supplies observations, physical actions and training rewards to SBX.

- **Observation:** 59 normalized features covering drone state, the previous command,
  race progress, the next two gates and the four nearest obstacles. Actor and critic
  receive the same sensor-limited information, including support for repeated and reversed gates.
- **Networks:** separate Actor and Critic MLPs, each with two 256-unit LeakyReLU layers.
  The Actor has a tanh mean head and a state-dependent log-standard-deviation head
  clipped to `[-2.5, 1.0]`. The Critic outputs one scalar value.
  Initialization is fixed: Flax default hidden/value kernels, a small orthogonal mean
  head, and a small random log-std head with bias `-0.5`.
- **Action:** the Actor samples `[thrust_raw, rotation_x, rotation_y, rotation_z]` during
  training. A sigmoid bounds thrust and a radial tanh bounds the body rotation increment
  to `0.16` radians (about 9.2 degrees) before the official Euler-angle clipping.
  This bounds each proposed increment relative to the current orientation, rather than
  the absolute tilt angle. The supplied reference annotates this limit as `0.16`;
  its imported constant's definition is not included in the reference snippet.
  The environment receives `[roll, pitch, yaw, total_thrust]`.
  Competition inference uses the Actor mean.
- **Normalization:** moments are collected from raw features and held fixed throughout
  each rollout and its PPO update. They are updated before the next rollout and frozen for inference.

On GPU runs, raw simulator observations, action conversion, feature encoding, rewards and
normalization stay in JAX on the GPU. Reward components are aggregated per batch and
logged once per rollout; detailed episode dictionaries are built only for finished worlds.
Standard SBX still requires NumPy observations, rewards, end flags and a CPU rollout buffer,
so training retains transfers at that interface. This uses the standard SBX collector;
it does not include the reference's custom JIT/scan collector. Normalization moments use
float32 on-device, so small numerical differences from the earlier NumPy implementation
are expected.

The baseline reward per control step is:

```text
2 × progress + 20 × finished − 10 × crash − 0.001 × squared_angular_speed − 0.01
```

Progress measures distance reduction toward the previous target using the same observed
gate center before and after the step. Timeouts are handled separately from crashes.
Curriculum training, intermediate starts and success-state caching are not included.

| Files | Responsibility |
| --- | --- |
| `train.py` | CLI, effective configuration, output paths and training startup |
| `env_wrapper.py` | Simulator adapter, observations, actions, rewards and normalization |
| `policy.py` | Actor/Critic networks, the Gaussian distribution and the SBX policy binding |
| `callbacks.py`, `checkpoint.py` | Metrics, evaluation, checkpoint timing, saving/loading and export |
| `controller.py`, `../rl_single_level2_controller.py` | Actor inference and the official loader entry point |

## Outputs and Evaluation

Results are saved relative to this package, independently of the launch directory:

```text
outputs/<generated_run_name>/
├── config.json                 # Effective training settings
├── logs/
│   ├── log.txt                 # Training metrics as text
│   ├── progress.csv
│   └── progress.json           # JSON Lines
└── checkpoints/
    ├── step_000020971520/       # First periodic checkpoint with the default rollout size
    └── best/                   # Created when separate evaluation is enabled
```

| Metric group | What to inspect |
| --- | --- |
| `rollout/`, `episodes/` | Episode return, completion rate, gate progress, crashes, timeouts and successful flight time |
| `train/` | Policy/value losses, entropy, approximate KL, clip fraction and learning rate |
| `actions/`, `reward/` | Each action head's mean/std and individual reward terms |
| `time/`, `eval/` | Training speed and, when enabled, independent evaluation results |

Saving defaults to every 20,000,000 cumulative transitions, matching the reference interval
and keeping the latest **three** regular checkpoints. Saves occur after complete PPO
updates, so at this scale the first periodic save is at 20,971,520 transitions and later
periodic saves are five rollouts apart. A final checkpoint is also saved when training
finishes normally, and its directory is printed in the terminal. Exceptions and Ctrl+C
close resources without adding a final checkpoint; resume from the last completed save.

**Separate evaluation is disabled by default.** Append `--eval_freq=1000000` to enable it.
It runs five episodes with fixed seeds, deterministic actions and frozen normalization.
Evaluation also runs after complete updates. The best checkpoint is kept separately in
`checkpoints/best/`, ranked by success rate, gate progress and then successful flight time.
It keeps one best evaluated checkpoint separately from the three recent regular saves.
The best checkpoint can be earlier than the final model; it is selected from the checkpoints
evaluated on those fixed seeds. With evaluation disabled, no `best/` directory is created.
Evaluation selects the best checkpoint without stopping training early; training continues
to the configured `total_timesteps` target.

## Resume or Start a New Experiment

Replace `/path/to/checkpoint` with a **checkpoint directory**, such as a `step_...` folder.
This command resumes training and extends the **cumulative target** to 300 million transitions:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_single_l2.train \
  --device=gpu --mode=resume --checkpoint="/path/to/checkpoint" --total_timesteps=300000000
```

| Mode | Restored state | Training settings |
| --- | --- | --- |
| `resume` | Actor, Critic, optimizers, normalization statistics, step counts and RNG state | Reuses saved PPO/environment settings; total target must exceed saved steps |
| `warm_start` | Actor, Critic and frozen normalization statistics; fresh optimizers and zero counters | Uses the new command-line settings |

Both modes create a new run directory. Physical simulator states are not saved, so
episodes restart on resume; continuation is not step-for-step identical.
Checkpoints contain `model.zip`, `runtime.pkl`, `actor.msgpack` and `metadata.json`.
Loading checks the network and environment settings.
Only load native SBX/pickle checkpoints from trusted training runs.

## Export and Run the Competition Controller

Choose a checkpoint after evaluating its flight behavior, then export it:

```bash
pixi run -e gpu python -m lsy_drone_racing.control.rl_single_l2.checkpoint \
  --checkpoint="/path/to/checkpoint"
```

This creates `deploy_model/` under this package with `actor.msgpack` and `config.json`.
The export includes Actor weights, frozen normalization and network/action configuration.
Inference constructs the Actor from `policy.py` and uses the observation/action functions
in `env_wrapper.py`. The default environment includes the SBX/SB3 dependencies they import.
Existing exports are preserved; use `--destination="/path/to/new_model"` for another export.

Export uses the same official `gpu` environment as training.

After exporting to the default directory, install the simulation environment once
(if it is not already installed), then test the controller:

```bash
pixi install -e default
pixi run -e default python scripts/sim.py --config=level2.toml --render=False
```

`config/level2.toml` already selects `rl_single_level2_controller.py` and `attitude` mode.
The controller finds `deploy_model/` relative to its own package; an export is required
before running the competition controller.

Include the inference code, selected `deploy_model/`, root `pyproject.toml` and `pixi.lock`
in your submission. Competition installs use the default Pixi environment, with the
complete PPO dependency set declared as base project dependencies.
Training artifacts in `outputs/` are ignored by Git; selected deployment files remain trackable.

## Record and Replay L2 Flights

After exporting the Actor to `deploy_model/`, record an evaluation flight with the shared tools:

```bash
pixi run -e default python personal_tools/record_run.py --config level2.toml
```

The command uses the L2 controller and attitude interface already selected in `level2.toml`;
no `--controller` override is required. Recording also works in the official `gpu`
environment, which shares the same base project dependencies.
The tools follow the official simulation loop, whose physics device defaults to CPU;
selecting the `gpu` environment alone does not change that device setting.

The recorder saves a trajectory, results and six synchronized camera views under `runs/`.
It records single-world evaluation episodes using the exported Actor. Parallel PPO training
and native training checkpoints are separate from this recording workflow.

Replace `RUN` with the printed recording directory name to render another view:

```bash
pixi run -e default python personal_tools/replay_run.py runs/RUN/trajectory.npz --camera fpv
```

Replay uses saved states and the embedded MuJoCo model, so Actor files are no longer required.
For `ffmpeg`/EGL requirements, camera options and multiple episodes, see the
[shared recording guide](../../../personal_tools/README.record_replay.md).

## Competition Evaluation

After exporting the selected Actor to this package's `deploy_model/`, run from the
repository root:

```bash
pixi run -e default evaluate
```

This runs the official `scripts/evaluate.py` using `config/level2.toml`, which already
selects `rl_single_level2_controller.py` and `attitude` mode. No controller override is
required. The current script runs **20 episodes** without rendering and, when at least
half succeed, writes the success rate and average successful flight time to
`evaluation.csv` in the repository root. If fewer than half succeed, evaluation exits
with an error before writing the results.

## Framework Checks

Run the focused tests with:

```bash
pixi run -e tests python -m pytest -q \
  tests/unit/control/test_rl_single_l2.py tests/integration/test_rl_single_l2.py
```

They cover feature/action/reward contracts, Gaussian gradients, real simulator resets,
PPO updates, checkpoint restoration, warm starts and exported-controller inference.
These checks validate the framework; successful L2 racing still needs training and evaluation.
