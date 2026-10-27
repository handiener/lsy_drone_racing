# Simulation, Recording and Replay

Run commands from the **repository root**. Update and activate the simulation environment:

```bash
pixi install --locked -e default
pixi shell -e default
```

The examples below use this shell. An existing `gpu` environment also works.
Without activating a shell, prefix a command with `pixi run -e default`.

Video recording and replay require **`ffmpeg` with the `libx264` encoder on `PATH`**
and working EGL rendering. `ffmpeg` is an external executable, not a Python package;
it is not currently installed by this project's Pixi manifest. Check with `ffmpeg -version`
and install it through your system package manager if needed.

## 1. Official simulation

Run a level without opening a viewer:

```bash
python scripts/sim.py --config level0.toml --render=False
```

Replace `level0.toml` with `level1.toml`, `level2.toml`, or `level3.toml`.
To use your controller and run three episodes:

```bash
python scripts/sim.py --config level2.toml \
  --controller my_controller.py --n_runs=3 --render=False
```

Configs are in `config/`; controllers are in `lsy_drone_racing/control/`.
Omit `--controller` to use the controller selected in the config.

## 2. Record and replay

Record one complete run with six synchronized views in one video on a remote machine:

```bash
python personal_tools/record_run.py --config level0.toml \
  --controller my_controller.py
```

### Recording artifacts

The script prints its output directory under `runs/`, containing:

- `result.json`: config, controller, finish status, gates passed and times.
- `trajectory.npz`: 3D states, actual randomized track and compiled MuJoCo model.
- `preview.mp4`: two rows of three views, arranged as shown below.
- `views/`: separate videos for each view.

| Left | Center | Right |
| --- | --- | --- |
| World | World +120° | World +240° |
| Top | FPV forward | FPV rear |

The three world cameras share the recorded overview framing and rotate around it
at 120° intervals. Individual files are `world.mp4`, `world_120.mp4`,
`world_240.mp4`, `top.mp4`, `fpv.mp4` and `fpv_rear.mp4`.

To record five runs with the same reset and randomization behavior as official `sim.py`:

```bash
python personal_tools/record_run.py --config level3.toml --n_runs 5
```

Multiple runs create `run_0001/`, `run_0002/`, etc., each with the files above,
plus a batch `summary.json`. Replay a run using `runs/RUN/run_0001/trajectory.npz`.
With `n_runs=1`, these files stay directly in the output directory.
`n_runs` goes directly to one official `simulate()` call. Seeds and resets follow
the config and official environment. A fixed config seed makes the batch reproducible.
`--n-runs` is an alias for `--n_runs`; the default is one run.

Replace `RUN` below with that directory's name. Render another view:

```bash
python personal_tools/replay_run.py runs/RUN/trajectory.npz --camera fpv
```

Camera choices: `world`, `top`, `side`, `fpv`, `fpv_rear`. The default is `world`.
Replay saves `replay_<camera>.mp4` beside the trajectory.

## 3. Other options

Choose the recording directory, FPS and resolution for each view:

```bash
python personal_tools/record_run.py --config level2.toml \
  --output-dir runs/debug --fps 30 --width 1280 --height 720
```

Width and height apply to each view; the combined video is three times that width
and twice that height.

Replay also supports `--fps`, `--width`, `--height` and `--output`.
For a custom free camera (angles in degrees):

```bash
python personal_tools/replay_run.py runs/RUN/trajectory.npz \
  --camera world --distance 5 --azimuth 180 --elevation -25 \
  --lookat 0 0 0 --output runs/RUN/custom.mp4
```

Use `--azimuth-offset 120` to rotate from the recorded world angle instead.

Run the tool tests:

```bash
python -m unittest personal_tools.test_run_tools
```

## 4. Notes

- Recording reuses official `simulate()` and uses EGL without a display or real-time pacing.
  The official human viewer (`--render=True`) needs a graphical display.
- Trajectory states are saved at the environment frequency (currently 50 Hz), including
  initial and terminal states. Video FPS only changes display sampling, not recorded states.
- The world view is captured during the official simulation. Other views are rendered
  from its saved trajectory after all runs finish, using the same state times for every frame.
- Replay restores saved states without running the controller or dynamics. Camera changes
  use the NPZ, not the MP4. Controller markers appear in the world view only.
- The embedded MJB preserves the model and assets. Replay requires the same MuJoCo version.
- `flight_time` follows official timing; `simulation_duration` includes the final step.
  Video duration matches simulation time within one video frame.
- The official environment may move a disabled drone to `[-1, -1, -1]` at termination;
  the saved terminal state preserves this behavior.
- Use new output paths: existing recording directories and videos are not overwritten.
  H.264 output requires positive, even width and height.
- The default `runs/` output folder is ignored by Git.
