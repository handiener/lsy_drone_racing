# Deployment

The idea for the deployment is to have an environment that exactly matches the interfaces and observation space of the simulation. Since the action space for the controller is identical, controllers can be directly deployed on the real drone without any modifications.

!!! warning
    Please be aware that running a controller on the real drone may still exhibit significant differences compared to the simulation due to the sim2real gap.

## Motion Tracking

We use a [Vicon](https://www.vicon.com/) motion tracking system to track the motion of the drone. The Vicon system consists of several cameras that are placed around the track, and a base station that calculates object poses by triangulation. Gates, obstacles and the drone are all equipped with reflective markers, which can be tracked by the cameras. Since we'd need to resort to numerical differentiation to get velocity information, we're running state estimators that filter the noisy Vicon measurements and provide smoother estimates of the drone's state.

After completing the [hardware setup](../getting_started/setup.md#hardware-setup-ubuntu-only), you need a total of *three* open terminals in the repository to deploy your controller. The commands below use the deploy environment.

In the first terminal, run this command to launch the translation layer from the motion capture cameras to ROS2.

```bash
pixi run mocap
```

!!! warning
    If you cannot see the drone in RVIZ, it is likely that Vicon is not turned on, or the drone is not selected for tracking in the Vicon system.

The second terminal is used to launch the estimator for the drone. To use the default settings, run `pixi run estimator`. These settings are made for `cf01`, so if you use a different drone, you need to tell the estimator which drone to track with `--drone_name`. Use the DEC number on your drone or the name shown in RVIZ. Optionally, you can choose the estimator type (`ukf`, `ekf` or `diff`) with `--estimator_type`. If the estimator works, you should see the frequency information in the terminal.

```bash
pixi run estimator --drone_name cf02
# or 
pixi run estimator --drone_name cf02 --estimator_type ukf
```

## Setting up the Track 
We provide a script that can help you set up the track. It will publish the nominal positions and orientations of the gates, obstacles and drones to RVIZ as MarkerArray.
```bash
pixi shell -e deploy
python scripts/nominal_frame_publisher.py --config level2.toml
```  
Now you need to add the MarkerArray to RVIZ as follows:
<div align="center">
  <img src="../../img/rviz_setup.png" alt="RVIZ SETUP" width="600"/>
</div>

## Deploying Your Controller

To deploy your controller on the real drone, use the deployment script in the `lsy_drone_racing/scripts` folder. Place the drone on its start position, power it on, and launch the estimators.

!!! note
    Make sure the drone has enough battery to complete the track. If a red LED is constantly turned on, the drone is low on battery. A blinking red LED indicates that the battery is sufficiently charged.

In the third terminal, start the deploy environment with `pixi shell -e deploy` and the deployment script with the correct configuration and controller.

```bash
pixi shell -e deploy
python scripts/deploy.py
# or
python scripts/deploy.py --config <config_name>.toml --controller <controller_name>.py
```

!!! note
    Be careful when flying the drone! Make sure to kill the process (**Ctrl+C**) immediately when your controller is unstable.

The deployment script will first check if the real track poses and the drone starting pose is within acceptable bounds of the configured track. If not, the script will print an error message and terminate. If the poses are correct, the drone will take off, fly through the track, print out the final lap time, and land automatically.

## Saving a Track Layout (Optional)

If you want to save the measured layout for later use, run the `save_track_as_config.py` script in the `lsy_drone_racing/scripts` folder:

```bash
pixi shell -e deploy
python scripts/save_track_as_config.py --config <config_name>.toml --save_config_to <output_name>.toml
```

The script saves the measured gate, obstacle, and drone starting poses to `config/<output_name>.toml`.

## Recording hardware deployments

Enable the host recorder in the deployment terminal:

```bash
python scripts/deploy.py --config level2.toml --record=True
```

Each invocation creates a new `runs/deploy_<UTC timestamp>/` directory. To choose the
location, pass `--output_dir runs/hardware_001` (this also enables recording).
Existing directories are refused before connecting to the drone.

The directory contains:

- `metadata.json`: resolved configuration, source/config hashes, Git revision and dirty
  status, command-line arguments, and a wall-clock/monotonic-clock anchor.
- `events.jsonl`: timestamped events, one JSON object per line. Policy-loop events share
  a zero-based `step` index. Observation events contain copied policy inputs and current
  raw mocap pose snapshots. Action events contain the controller output and inference
  start/end times. Transition events contain the immediate post-send observation, reward,
  termination flags, and environment-step timing. Loop events contain elapsed processing
  time and the nominal deadline (excluding the subsequent sleep).
- `summary.json`: writer status and record counts. `complete` means recording finalized
  without a recorder error; it does **not** mean the drone completed the track.

Controllers may optionally provide `enable_deployment_recording(obs)` and
`deployment_diagnostics()` hooks for additional model metadata and policy diagnostics.

The Crazyflie wrapper also records converted attitude/full-state commands after submission
to cflib2, external mocap poses sent to the onboard estimator, high-level return/landing
commands, and emergency stops. Submission does not confirm receipt or execution onboard.
External-pose events continue through return/landing while the connection remains open;
policy-loop observations stop when the race loop exits. Exceptions, interrupts, race-end,
and cleanup events delimit the phases.

Data is copied into a bounded queue, serialized on a background thread, and flushed roughly
every 0.2 seconds. Queue overflow or a writer failure disables further recording and emits
an error without stopping flight control. Cleanup drains queued records **after** closing
hardware. Ctrl+C is still passed through to the environment's emergency-stop handling.
Abrupt process termination or power loss can lose queued/buffered data and leave no summary;
read complete JSONL lines and ignore a partial final line. Flushes are not an fsync guarantee.

All event timestamps use `time.monotonic_ns()` on the host. Mocap samples have no sensor
acquisition timestamp or freshness guarantee here, and estimated states are asynchronous
snapshots, not atomic measurements. `next_obs` is read immediately after sending a command;
it is not a measurement of the response over a complete 20 ms control interval. Copying and
policy diagnostics add some host overhead; inspect loop timing for overruns.

Add `--onboard_log=True` to capture onboard IMU, motor outputs, battery, controller signals,
and attitude alongside the host recording in `onboard.jsonl`. Both streams use the same host
monotonic clock; onboard samples also include firmware timestamps. Firmware variables that
are unavailable are listed in the onboard metadata. Onboard logging starts after reset and
continues through return/landing until the drone closes. Its final summary is independent
of the host `summary.json`. With only `--onboard_log=True`, output stays in `logs/onboard/`.
Record ROS topics separately with `ros2 bag` for full-rate, source-stamped mocap streams.

For machine-specific settings, copy `config/level2.toml` to `config/level2.local.toml`, add
`/config/level2.local.toml` to `.git/info/exclude`, and deploy with
`--config level2.local.toml`. Keep radio ID/channel changes in that local file. It is a full
configuration copy; reconcile it manually when shared configuration defaults change.

For example, read policy actions without ROS or hardware:

```python
import json
from pathlib import Path

run = Path("runs/hardware_001")
with (run / "events.jsonl").open() as stream:
    actions = [event for line in stream if (event := json.loads(line))["kind"] == "action"]
```
