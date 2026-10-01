"""Render a recorded 3D trajectory from a new camera without controller or dynamics.

Example:
    python personal_tools/replay_run.py runs/example/trajectory.npz --camera fpv --fps 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Set EGL before the first MuJoCo import, including imports through the shared helper module.
os.environ["MUJOCO_GL"] = "egl"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personal_tools._run_io import VideoWriter, load_model, load_trajectory, video_times

if TYPE_CHECKING:
    import mujoco


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate normalized quaternions along the shortest arc, in either component order."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    denominator = np.sin(angle)
    linear = (1 - fraction) * first + fraction * second
    safe_denominator = np.where(denominator > 1e-8, denominator, 1.0)
    spherical = (
        np.sin((1 - fraction) * angle) / safe_denominator * first
        + np.sin(fraction * angle) / safe_denominator * second
    )
    result = np.where(dot > 0.9995, linear, spherical)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def interpolate_state(trajectory: dict[str, np.ndarray], time: float) -> dict[str, np.ndarray]:
    """Interpolate visual poses only; this creates no new simulated or measured states."""
    times = trajectory["time"]
    right = int(np.searchsorted(times, time, side="right"))
    if right == 0:
        left = right = 0
    elif right == len(times):
        left = right = len(times) - 1
    else:
        left = right - 1
    fraction = 0.0 if left == right else float((time - times[left]) / (times[right] - times[left]))
    state = {
        key: (1 - fraction) * trajectory[key][left] + fraction * trajectory[key][right]
        for key in ("mocap_pos", "qpos", "qvel")
    }
    state["mocap_quat"] = _slerp(
        trajectory["mocap_quat"][left], trajectory["mocap_quat"][right], fraction
    )
    return state


def restore_state(
    model: mujoco.MjModel, data: mujoco.MjData, state: dict[str, np.ndarray], time: float
) -> None:
    """Restore mocap poses and update rendering geometry without advancing physics."""
    import mujoco

    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.mocap_pos[:] = state["mocap_pos"]
    data.mocap_quat[:] = state["mocap_quat"]
    data.time = time
    # This is the same rendering update sequence as Crazyflow Sim.render. Its mocap drones have
    # custom weld IDs, so a full mj_forward can fail on contacts between static bodies.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def _camera(
    model: mujoco.MjModel,
    trajectory: dict[str, np.ndarray],
    metadata: dict[str, Any],
    preset: str,
    distance: float | None,
    azimuth: float | None,
    elevation: float | None,
    lookat: list[float] | tuple[float, float, float] | None,
    azimuth_offset: float = 0.0,
) -> mujoco.MjvCamera | str:
    import mujoco

    if not np.isfinite(azimuth_offset):
        raise ValueError("azimuth_offset must be finite")
    if preset in ("fpv", "fpv_rear"):
        if (
            any(value is not None for value in (distance, azimuth, elevation, lookat))
            or azimuth_offset
        ):
            raise ValueError(
                f"Custom free-camera parameters cannot be combined with --camera {preset}"
            )
        name = "fpv_cam:0"
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(f"The recorded model has no FPV camera named {name!r}")
        if preset == "fpv_rear":
            # Rotate the existing mount around the drone's local up axis. The loaded replay
            # model is private to this video; the recorded MJB and forward view stay intact.
            if model.cam_mode[camera_id] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
                raise ValueError("Rear FPV requires a camera fixed to the drone body")
            original = model.cam_quat[camera_id].copy()
            rear = np.empty(4)
            mujoco.mju_mulQuat(rear, np.array([0.0, 0.0, 0.0, 1.0]), original)
            model.cam_quat[camera_id] = rear
            model.cam_pos[camera_id, :2] *= -1
        return name

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    points = np.concatenate(
        [trajectory[key].reshape(-1, 3) for key in ("pos", "gates_pos", "obstacles_pos")]
    )
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low)) / 2, 0.5)
    camera.lookat[:] = center
    camera.distance = max(3.0, 1.3 * radius / np.sin(np.deg2rad(model.vis.global_.fovy / 2)))
    camera.azimuth, camera.elevation = {
        "world": (135.0, -30.0),
        "top": (90.0, -90.0),
        "side": (90.0, -5.0),
    }[preset]
    if preset == "world":
        # Keep the recording's free-camera framing when the user has not selected a new preset.
        config = metadata.get("world_cam_config") or metadata.get("cam_config") or {}
        if isinstance(config, list):
            config = config[0] if config else {}
        for key in ("distance", "azimuth", "elevation", "lookat"):
            if key in config:
                if key == "lookat":
                    camera.lookat[:] = config[key]
                else:
                    setattr(camera, key, config[key])
    for key, value in (("distance", distance), ("azimuth", azimuth), ("elevation", elevation)):
        if value is not None:
            if not np.isfinite(value) or (key == "distance" and value <= 0):
                raise ValueError(
                    f"{key} must be finite" + (" and positive" if key == "distance" else "")
                )
            setattr(camera, key, value)
    camera.azimuth = (camera.azimuth + azimuth_offset) % 360.0
    if lookat is not None:
        if len(lookat) != 3 or not np.isfinite(lookat).all():
            raise ValueError("lookat must contain three finite coordinates")
        camera.lookat[:] = lookat
    return camera


def replay(
    trajectory_path: str | Path,
    *,
    camera: str = "world",
    fps: float = 60,
    width: int = 960,
    height: int = 540,
    output: str | Path | None = None,
    distance: float | None = None,
    azimuth: float | None = None,
    elevation: float | None = None,
    lookat: list[float] | tuple[float, float, float] | None = None,
    azimuth_offset: float = 0.0,
    frame_times: np.ndarray | None = None,
) -> Path:
    """Write a camera-specific video from saved poses, without starting a simulation."""
    import mujoco

    if camera not in ("world", "top", "side", "fpv", "fpv_rear"):
        raise ValueError(f"Unknown camera preset: {camera}")
    path = Path(trajectory_path)
    destination = Path(output) if output is not None else path.parent / f"replay_{camera}.mp4"
    if destination.exists():
        raise FileExistsError(
            f"Video output already exists: {destination}. Choose another --output."
        )
    trajectory, metadata = load_trajectory(path)
    model = load_model(trajectory["model_mjb"], metadata["mujoco_version"])
    dimensions = {
        "qpos": (model.nq,),
        "qvel": (model.nv,),
        "mocap_pos": (model.nmocap, 3),
        "mocap_quat": (model.nmocap, 4),
    }
    for key, expected in dimensions.items():
        if trajectory[key].shape[1:] != expected:
            raise ValueError(
                f"Saved {key} shape does not match the embedded model: expected {expected}"
            )
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    selected_camera = _camera(
        model, trajectory, metadata, camera, distance, azimuth, elevation, lookat, azimuth_offset
    )
    data = mujoco.MjData(model)
    times = video_times(float(trajectory["time"][-1]), fps)
    if frame_times is not None:
        # Recorder previews supply the exact state timestamp used by each world-view frame.
        # Ordinary replay keeps its existing interpolated display cadence.
        frame_times = np.asarray(frame_times, dtype=np.float64)
        if (
            frame_times.shape != times.shape
            or not np.isfinite(frame_times).all()
            or np.any(np.diff(frame_times) < 0)
            or frame_times[0] < 0
            or frame_times[-1] > trajectory["time"][-1]
        ):
            raise ValueError("frame_times must be ordered trajectory times, one per video frame")
        times = frame_times
    with VideoWriter(destination, fps, width, height) as writer:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for time in times:
                restore_state(model, data, interpolate_state(trajectory, float(time)), float(time))
                renderer.update_scene(data, camera=selected_camera)
                writer.write(renderer.render())
    return destination


def main(argv: list[str] | None = None) -> None:
    """Parse the replay command line and report actionable errors without a traceback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="Recorded trajectory.npz archive")
    parser.add_argument(
        "--camera", choices=("world", "top", "side", "fpv", "fpv_rear"), default="world"
    )
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--output", type=Path, help="MP4 output path; existing files are never replaced"
    )
    parser.add_argument("--distance", type=float)
    parser.add_argument("--azimuth", type=float, help="Free-camera azimuth in degrees")
    parser.add_argument(
        "--azimuth-offset",
        type=float,
        default=0.0,
        help="Rotate the free camera by this many degrees",
    )
    parser.add_argument("--elevation", type=float, help="Free-camera elevation in degrees")
    parser.add_argument("--lookat", nargs=3, type=float, metavar=("X", "Y", "Z"))
    arguments = parser.parse_args(argv)
    try:
        destination = replay(
            arguments.trajectory,
            camera=arguments.camera,
            fps=arguments.fps,
            width=arguments.width,
            height=arguments.height,
            output=arguments.output,
            distance=arguments.distance,
            azimuth=arguments.azimuth,
            elevation=arguments.elevation,
            lookat=arguments.lookat,
            azimuth_offset=arguments.azimuth_offset,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")
    print(f"Video saved to {destination}")


if __name__ == "__main__":
    main()
