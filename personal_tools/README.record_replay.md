# Simulation, Recording and Replay

Run commands from the repository root using the project's Python environment.

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

Record one complete run, including a video, on a remote machine:

```bash
python personal_tools/record_run.py --config level0.toml \
  --controller my_controller.py
```

The script prints its output directory under `runs/`, containing:

- `result.json`: config, controller, finish status, gates passed and times.
- `trajectory.npz`: 3D states, actual randomized track and compiled MuJoCo model.
- `preview.mp4`: video with controller visualization markers.

To record five runs with the same reset and randomization behavior as official `sim.py`:

```bash
python personal_tools/record_run.py --config level3.toml --n_runs 5
```

Multiple runs create `run_0001/`, `run_0002/`, etc., each with the three files above,
plus a batch `summary.json`. Replay a run using `runs/RUN/run_0001/trajectory.npz`.
With `n_runs=1`, the three files stay directly in the output directory.
`n_runs` goes directly to one official `simulate()` call. Seeds and resets follow
the config and official environment. A fixed config seed makes the batch reproducible.
`--n-runs` is an alias for `--n_runs`; the default is one run.

Replace `RUN` below with that directory's name. Render another view:

```bash
python personal_tools/replay_run.py runs/RUN/trajectory.npz --camera fpv
```

Camera choices: `world`, `top`, `side`, `fpv`. The default is `world`.
Replay saves `replay_<camera>.mp4` beside the trajectory.

## 3. Other options

Choose the recording directory, video FPS and resolution:

```bash
python personal_tools/record_run.py --config level2.toml \
  --output-dir runs/debug --fps 30 --width 1280 --height 720
```

Replay also supports `--fps`, `--width`, `--height` and `--output`.
For a custom free camera (angles in degrees):

```bash
python personal_tools/replay_run.py runs/RUN/trajectory.npz \
  --camera world --distance 5 --azimuth 180 --elevation -25 \
  --lookat 0 0 0 --output runs/RUN/custom.mp4
```

Run the tool tests:

```bash
python -m unittest personal_tools.test_run_tools
```

## 4. Notes

- Recording reuses official `simulate()` and uses EGL without a display or real-time pacing.
  The official human viewer (`--render=True`) needs a graphical display.
- Trajectory states are saved at the environment frequency (currently 50 Hz), including
  initial and terminal states. Video FPS only changes display sampling, not recorded states.
- Replay restores saved states without running the controller or dynamics. Camera changes
  use the NPZ, not the MP4. Controller markers appear in preview only.
- The embedded MJB preserves the model and assets. Replay requires the same MuJoCo version.
- `flight_time` follows official timing; `simulation_duration` includes the final step.
  Video duration matches simulation time within one video frame.
- The official environment may move a disabled drone to `[-1, -1, -1]` at termination;
  the saved terminal state preserves this behavior.
- Use new output paths: existing recording directories and videos are not overwritten.
  H.264 output requires positive, even width and height.
