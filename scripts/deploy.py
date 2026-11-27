"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import rclpy

from lsy_drone_racing.envs.race_core import RaceCoreEnv
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.utils.deploy_recorder import DeploymentRecorder, code_identity, file_identity
from lsy_drone_racing.utils.deploy_viewer import DeployViewer

if TYPE_CHECKING:
    from lsy_drone_racing.envs.real_race_env import RealDroneRaceEnv

logger = logging.getLogger(__name__)


def main(
    config: str = "level2.toml",
    controller: str | None = None,
    render: bool = False,
    onboard_log: bool = False,
    record: bool = False,
    output_dir: str | None = None,
):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
         the controller specified in the config file is used.
        render: Show a live MuJoCo window that mirrors the real race from mocap data.
        onboard_log: Stream onboard telemetry (IMU, motor outputs, battery, controller commands) at
         100 Hz until the drone is closed. Shares the host recording directory when enabled.
        record: Save deployment data to a new directory under runs/.
        output_dir: Explicit recording directory; also enables recording. Must not exist.
    """
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config" / config
    config = load_config(config_path)
    if controller is not None:
        config.controller.file = controller

    controller_path = root / "lsy_drone_racing/control" / config.controller.file
    recorder = None
    if record or output_dir is not None:
        recorder = DeploymentRecorder(
            output_dir,
            {
                "config": config.to_dict(),
                "config_file": file_identity(config_path),
                "controller_file": file_identity(controller_path),
                "code": code_identity(root),
                "argv": sys.argv,
                "onboard_log": onboard_log,
                "notes": "Host snapshots without sensor timestamps; onboard data is separate.",
            },
        )
    env: RealDroneRaceEnv | None = None
    viewer = None
    try:
        rclpy.init()
        env = gymnasium.make(
            "RealDroneRacing-v0",
            drones=config.deploy.drones,
            freq=config.env.freq,
            track=config.env.track,
            randomizations=config.env.randomizations,
            dynamics=config.sim.dynamics,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
        )
        if recorder:
            env.unwrapped.drone.record_event = recorder.record
        if render:
            viewer_sim_config = config.sim.copy_and_resolve_references()
            viewer_sim_config.drone = config.deploy.drones[0]["drone"]
            viewer_sim_config.camera = -1
            viewer_env = RaceCoreEnv(
                n_envs=1,
                n_drones=1,
                freq=config.env.freq,
                sim_config=viewer_sim_config,
                sensor_range=config.env.sensor_range,
                track=config.env.track,
                control_mode=config.env.control_mode,
                seed=config.env.seed,
                device="cpu",
            )
            viewer = DeployViewer(viewer_env)
            viewer.warm_up()

        obs, info = env.reset(seed=config.env.seed, options=config.deploy)
        next_obs = obs  # Set next_obs to avoid errors when the loop never enters
        if onboard_log:
            if recorder:
                log_path = recorder.output_dir / "onboard.jsonl"
            else:
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                log_path = root / "logs" / "onboard" / f"onboard_{stamp}.jsonl"
            # Samples are written while flying. The log ends when the env closes the drone.
            env.unwrapped.drone.start_onboard_log(log_path)
            if recorder:
                recorder.record("onboard_log_started", path=log_path)
        if recorder:
            recorder.record(
                "reset",
                obs=obs,
                info=info,
                track={
                    "gates_pos": env.unwrapped.gates.pos,
                    "gates_quat": env.unwrapped.gates.quat,
                    "obstacles_pos": env.unwrapped.obstacles.pos,
                },
                drone_parameters=env.unwrapped.drone_parameters,
                battery_rest_v=env.unwrapped.drone.battery_rest_v,
            )

        if render:
            viewer.set_track(
                gates_pos=env.unwrapped.gates.pos,
                gates_quat=env.unwrapped.gates.quat,
                obstacles_pos=env.unwrapped.obstacles.pos,
            )
            viewer.update(obs["pos"], obs["quat"])

        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config)
        if recorder and hasattr(controller, "enable_deployment_recording"):
            recorder.record("policy_metadata", **controller.enable_deployment_recording(obs))
        start_time = time.perf_counter()
        step = 0
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            obs = {k: v[0] for k, v in obs.items()}
            if recorder:
                connector = env.unwrapped.drone._ros_connector
                name = env.unwrapped.drone_name
                recorder.record(
                    "observation",
                    step=step,
                    obs=obs,
                    info=info,
                    mocap={"pos": connector.tf_pos[name], "quat": connector.tf_quat[name]},
                )
            inference_start_ns = time.monotonic_ns()
            action = controller.compute_control(obs, info)
            inference_end_ns = time.monotonic_ns()
            if recorder:
                recorder.record(
                    "action",
                    step=step,
                    action=action,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=inference_end_ns,
                )
                if hasattr(controller, "deployment_diagnostics"):
                    recorder.record("policy", step=step, **controller.deployment_diagnostics())
            step_start_ns = time.monotonic_ns()
            next_obs, reward, terminated, truncated, info = env.step(action)
            step_end_ns = time.monotonic_ns()
            if recorder:
                recorder.record(
                    "transition",
                    step=step,
                    next_obs=next_obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    step_start_ns=step_start_ns,
                    step_end_ns=step_end_ns,
                )
            if render:
                viewer.update(next_obs["pos"], next_obs["quat"])
            controller_finished = controller.step_callback(
                action, next_obs, reward, terminated, truncated, info
            )
            if recorder:
                recorder.record(
                    "loop",
                    step=step,
                    controller_finished=controller_finished,
                    elapsed_s=time.perf_counter() - t_loop,
                    deadline_s=1 / config.env.freq,
                )
            step += 1
            if terminated or truncated or controller_finished:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        finished_track = next_obs["n_gates_passed"] == next_obs["gate_sequence"].shape[0]
        if recorder:
            recorder.record(
                "race_end",
                flight_time_s=ep_time,
                finished_track=finished_track,
                steps=step,
                ros_ok=rclpy.ok(),
            )
        logger.info(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")
    except BaseException as exc:
        if recorder:
            recorder.record("exception", type=type(exc).__name__, message=str(exc))
        raise
    finally:
        # Keep any active KeyboardInterrupt visible to env.close's emergency-stop handling.
        try:
            if recorder:
                recorder.record("cleanup_start")
            if env is not None:
                env.close()
            if recorder:
                recorder.record("cleanup_end")
        except BaseException as exc:
            if recorder:
                recorder.record("cleanup_exception", type=type(exc).__name__, message=str(exc))
            raise
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                try:
                    if rclpy.ok():
                        rclpy.shutdown()
                finally:
                    if recorder:
                        recorder.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
