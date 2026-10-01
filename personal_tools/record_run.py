"""Record official simulation episodes, including headless controller visualizations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import uuid4

# EGL must be selected before importing MuJoCo or Crazyflow.
os.environ["MUJOCO_GL"] = "egl"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from personal_tools._run_io import (  # noqa: E402
    SCHEMA_VERSION,
    TimelineVideoWriter,
    combine_views,
    snapshot_model,
    validate_trajectory,
    video_times,
)
from personal_tools.replay_run import replay  # noqa: E402
from scripts import sim as official_sim  # noqa: E402

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)
PREVIEW_VIEWS = ("world", "world_120", "world_240", "top", "fpv", "fpv_rear")


def _input_path(value: str, directory: str) -> Path:
    """Resolve official filenames and explicit paths without changing configuration files."""
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.is_file():
        path = REPO_ROOT / directory / path
    return path.resolve(strict=True)


class RecordingEnv(gymnasium.Wrapper):
    """Observe the official reset/step calls without implementing a simulation loop."""

    def __init__(self, env: gymnasium.Env, recorder: RunRecorder):
        """Attach a recorder to the environment returned by the official factory."""
        super().__init__(env)
        self.recorder = recorder

    def reset(self, **kwargs: Any) -> tuple[dict, dict]:
        """Record the initial state and initialize the viewer before controller construction."""
        self.recorder.begin_episode()
        result = self.env.reset(**kwargs)
        self.recorder.capture()
        self.recorder.warm_renderer()
        return result

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        """Record every environment state, including a step that ends the episode."""
        result = self.env.step(action)
        self.recorder.capture(terminated=result[2], truncated=result[3])
        return result

    def close(self):
        """Render the terminal state before releasing the official environment."""
        if self.recorder.closed:
            return
        try:
            self.recorder.finish_video()
        finally:
            self.recorder.closed = True
            self.env.close()


class RunRecorder:
    """Manage temporary hooks and artifacts for a single call to official simulate()."""

    def __init__(self, output_dir: Path, fps: float, width: int, height: int, n_runs: int = 1):
        """Initialize storage; the caller owns the new output directory."""
        self.output_dir = Path(output_dir)
        self.batch_dir = self.output_dir
        self.n_runs = n_runs
        self.episode_index = 0
        self.results: list[dict[str, Any]] = []
        self.render_times: list[float] = []
        self.fps, self.width, self.height = fps, width, height
        self.samples: dict[str, list] = defaultdict(list)
        self.metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mujoco_version": mujoco.__version__,
            "crazyflow_version": importlib.metadata.version("crazyflow"),
            "state_quaternion_order": "xyzw",
            "mocap_quaternion_order": "wxyz",
            "linear_velocity_frame": "world",
            "angular_velocity_frame": "body",
            "source_cadence": "env.step, plus reset",
            "preview_resampling": "zero_order_hold",
        }
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.metadata["repo_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
        except OSError:
            self.metadata["repo_commit"] = None
        self.env: Any = None
        self.wrapper: RecordingEnv | None = None
        self.config: Any = None
        self.model_mjb: np.ndarray | None = None
        self.track_ids: dict[str, np.ndarray] = {}
        self.video: TimelineVideoWriter | None = None
        self.last_frame: np.ndarray | None = None
        self.last_frame_time: float | None = None
        self.finalizing = False
        self.closed = False
        self.flight_time: float | None = None
        self.finished = False
        self.gates_passed = 0
        self.completed = False

    def begin_episode(self):
        """Finalize the previous episode before the official loop resets its environment."""
        if self.episode_index:
            self.finish_video()
            self.save()
        self.episode_index += 1
        if self.episode_index > self.n_runs:
            raise RuntimeError("The official loop reset more episodes than requested")
        self.output_dir = self.batch_dir
        if self.n_runs > 1:
            self.output_dir = self.batch_dir / f"run_{self.episode_index:04d}"
            self.output_dir.mkdir(exist_ok=False)
        self.samples = defaultdict(list)
        self.render_times = []
        self.finalizing = False
        self.last_frame = self.last_frame_time = None
        self.flight_time = None
        self.finished = self.completed = False
        self.gates_passed = 0
        self.metadata.update(config=self.config.to_dict(), episode_index=self.episode_index)
        self.metadata["preview_cameras"] = list(PREVIEW_VIEWS)
        self.model_mjb = snapshot_model(self.env.sim.mj_model)
        self.video = TimelineVideoWriter(
            self.output_dir / "views/world.mp4", self.fps, self.width, self.height
        )

    def capture(self, terminated: bool = False, truncated: bool = False):
        """Copy ground truth independently of render cadence and sensor visibility."""
        data, sim = self.env.data, self.env.sim
        states, core = data.sim_data.states, data.sim_data.core
        sample = {
            name: np.array(getattr(states, name)[0], copy=True)
            for name in ("pos", "quat", "vel", "ang_vel")
        }
        sample.update(
            {
                name: np.array(getattr(data, name)[0], copy=True)
                for name in ("gates_pos", "gates_quat", "obstacles_pos", "n_gates_passed")
            }
        )
        if not self.samples["time"]:
            # A randomized reset can change the track from the constructor's nominal layout.
            self.track_ids["gate_sequence"] = np.array(data.gate_sequence, copy=True)
            self.track_ids["gate_sequence_direction"] = np.array(
                data.gate_sequence_direction, copy=True
            )
        # This Crazyflow model uses mocap drones, not free joints. Copy the whole scene, then
        # overwrite the authoritative environment poses; mjx_data itself may not yet be synced.
        mocap_pos = np.array(sim.mjx_data.mocap_pos[0], copy=True)
        mocap_quat = np.array(sim.mjx_data.mocap_quat[0], copy=True)
        drone_ids = self.track_ids["drone_mocap_ids"]
        gate_ids = self.track_ids["gate_mocap_ids"]
        obstacle_ids = self.track_ids["obstacle_mocap_ids"]
        mocap_pos[drone_ids] = sample["pos"]
        mocap_quat[drone_ids] = np.roll(sample["quat"], 1, axis=-1)
        mocap_pos[gate_ids] = sample["gates_pos"]
        mocap_quat[gate_ids] = np.roll(sample["gates_quat"], 1, axis=-1)
        mocap_pos[obstacle_ids] = sample["obstacles_pos"]
        sim_step = int(np.asarray(core.steps).reshape(-1)[0])
        sample.update(
            time=sim_step / core.freq,
            env_step=int(np.asarray(data.steps).reshape(-1)[0]),
            sim_step=sim_step,
            obstacles_quat=np.roll(mocap_quat[obstacle_ids], -1, axis=-1),
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            qpos=np.array(sim.mjx_data.qpos[0], copy=True),
            qvel=np.array(sim.mjx_data.qvel[0], copy=True),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        for name, value in sample.items():
            self.samples[name].append(value)
        # render_callback() runs before env.render(); give it current state as well as the
        # already initialized offscreen viewer. The functional dynamics still owns env.data.
        sim.data = data.sim_data

    def warm_renderer(self):
        """Use the official render synchronization and create the offscreen marker viewer."""
        # Sim.render's decorator reads sim.data; the environment owns newer functional state.
        self.env.sim.data = self.env.data.sim_data
        self.env.render()
        viewer = getattr(self.env.sim.viewer, "viewer", None)
        camera = getattr(viewer, "cam", None)
        if camera is not None:
            # Save the actual free-camera framing, including defaults omitted by the config.
            # The other world views orbit the same lookat with the same distance/elevation.
            self.metadata["world_cam_config"] = {
                "distance": float(camera.distance),
                "azimuth": float(camera.azimuth),
                "elevation": float(camera.elevation),
                "lookat": np.asarray(camera.lookat).tolist(),
            }

    def finish_video(self):
        """Include the terminal state, which official simulate() skips rendering."""
        if self.video is None or self.finalizing:
            return
        self.finalizing = True
        try:
            if self.samples["time"]:
                self.warm_renderer()
                self.video.finish(self.samples["time"][-1], self.last_frame)
        finally:
            self.video.close()

    def run(self, simulate_module: ModuleType, config_path: Path, controller_path: Path | None):
        """Invoke the official loop once, restoring all hooks on success or failure."""
        original_make = simulate_module.gymnasium.make
        original_config = simulate_module.load_config
        original_controller = simulate_module.load_controller
        original_stats = simulate_module.log_episode_stats

        def load_config(path: Path) -> Any:
            self.config = original_config(path)
            self.metadata["config_path"] = str(path.resolve())
            self.metadata["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return self.config

        def load_controller(path: Path) -> Any:
            self.metadata["controller_path"] = str(path.resolve())
            self.metadata["controller_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return original_controller(path)

        def stats(obs: dict, curr_time: float):
            self.flight_time = float(curr_time)
            self.gates_passed = int(np.asarray(obs["n_gates_passed"]).item())
            self.finished = self.gates_passed == obs["gate_sequence"].shape[0]
            original_stats(obs, curr_time)
            self.completed = True

        with ExitStack() as hooks:

            def make(*args: Any, **kwargs: Any) -> RecordingEnv:
                if self.env is not None:
                    raise RuntimeError("record_run supports one environment per simulate() call")
                if args and str(args[0]).startswith("Real"):
                    raise ValueError("record_run only supports simulated racing environments")
                created = original_make(*args, **kwargs)
                self.wrapper = RecordingEnv(created, self)
                self.env = created.unwrapped
                sim = self.env.sim
                if sim.n_worlds != 1 or sim.n_drones != 1:
                    raise ValueError("record_run requires a single-world, single-drone environment")
                gate_ids, obstacle_ids = self.env.mocap_ids
                self.track_ids = {
                    "drone_mocap_ids": np.asarray(sim.data.core.drone_mocap_ids, dtype=np.int64),
                    "gate_mocap_ids": np.asarray(gate_ids, dtype=np.int64),
                    "obstacle_mocap_ids": np.asarray(obstacle_ids, dtype=np.int64),
                    "gate_sequence": np.array(self.env.data.gate_sequence, copy=True),
                    "gate_sequence_direction": np.array(
                        self.env.data.gate_sequence_direction, copy=True
                    ),
                }
                self.metadata.update(
                    config=self.config.to_dict(),
                    env_freq=self.env.settings.freq,
                    sim_freq=sim.freq,
                    camera=self.env.settings.camera,
                    cam_config=dict(self.env.settings.cam_config),
                )
                original_render = sim.render
                render_signature = inspect.signature(original_render)

                def render(*render_args: Any, **render_kwargs: Any) -> np.ndarray:
                    # Bind after RaceCoreEnv.render() has synchronized its latest data. This
                    # prevents requires_mujoco_sync from restoring the old initial drone pose.
                    sim.data = self.env.data.sim_data
                    arguments = render_signature.bind(*render_args, **render_kwargs).arguments
                    arguments.update(
                        mode="rgb_array", camera=-1, width=self.width, height=self.height
                    )
                    frame = original_render(**arguments)
                    timestamp = float(np.asarray(self.env.data.sim_data.core.steps).flat[0])
                    timestamp /= self.env.data.sim_data.core.freq
                    self.last_frame = np.array(frame, copy=True)
                    self.last_frame_time = timestamp
                    if not self.finalizing:
                        self.video.push(timestamp, self.last_frame)
                        self.render_times.append(timestamp)
                    return frame

                hooks.enter_context(patch.object(sim, "render", render))
                return self.wrapper

            hooks.enter_context(patch.object(simulate_module, "load_config", load_config))
            hooks.enter_context(patch.object(simulate_module, "load_controller", load_controller))
            hooks.enter_context(patch.object(simulate_module, "log_episode_stats", stats))
            hooks.enter_context(patch.object(simulate_module.gymnasium, "make", make))
            try:
                simulate_module.simulate(
                    config=str(config_path),
                    controller=str(controller_path) if controller_path else None,
                    n_runs=self.n_runs,
                    render=True,
                )
                self.completed = True
            finally:
                # Official simulate() has no try/finally; ensure headless resources are closed
                # even if a controller raises, while patches are still installed.
                if self.wrapper is not None and not self.closed:
                    if sys.exc_info()[0] is None:
                        self.wrapper.close()
                    else:
                        try:
                            self.wrapper.close()
                        except Exception:
                            logger.exception("Could not finish preview after simulation error")

    def save(self, error: BaseException | None = None):
        """Save portable state arrays and the official statistics, or a marked partial run."""
        if self.samples["time"]:
            arrays = {name: np.asarray(values) for name, values in self.samples.items()}
            arrays.update(self.track_ids, model_mjb=self.model_mjb)
            arrays["preview_render_time"] = np.asarray(self.render_times, dtype=np.float64)
            validate_trajectory(arrays, self.metadata)
            arrays["metadata_json"] = np.asarray(json.dumps(self.metadata))
            np.savez_compressed(self.output_dir / "trajectory.npz", **arrays)
            self.gates_passed = int(arrays["n_gates_passed"][-1, 0])
            self.finished = self.gates_passed == len(arrays["gate_sequence"])
        result = {
            "episode_index": self.episode_index,
            "config": self.metadata.get("config"),
            "config_path": self.metadata.get("config_path"),
            "controller": self.metadata.get("controller_path"),
            "finished": self.finished,
            "gates_passed": self.gates_passed,
            "flight_time": self.flight_time,
            "simulation_duration": self.samples["time"][-1] if self.samples["time"] else None,
            "env_frequency": self.metadata.get("env_freq"),
            "state_samples": len(self.samples["time"]),
            "status": "complete" if self.completed and error is None else "error",
            "error": f"{type(error).__name__}: {error}" if error else None,
            "video": {
                "fps": self.fps,
                "width": 3 * self.width,
                "height": 2 * self.height,
                "view_width": self.width,
                "view_height": self.height,
                "layout": [list(PREVIEW_VIEWS[:3]), list(PREVIEW_VIEWS[3:])],
            },
            "versions": {
                key: self.metadata[key]
                for key in ("mujoco_version", "crazyflow_version", "repo_commit")
            },
        }
        (self.output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        entry = {
            key: result[key]
            for key in (
                "episode_index",
                "finished",
                "gates_passed",
                "flight_time",
                "simulation_duration",
                "status",
                "error",
            )
        }
        entry["directory"] = str(self.output_dir.relative_to(self.batch_dir))
        if self.results and self.results[-1]["episode_index"] == self.episode_index:
            self.results[-1] = entry
        else:
            self.results.append(entry)
        if self.n_runs > 1:
            summary = {
                "n_runs": self.n_runs,
                "completed_runs": sum(item["status"] == "complete" for item in self.results),
                "runs": self.results,
            }
            (self.batch_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )

    def render_views(self):
        """Build synchronized additional views after the official loop and viewer have closed."""
        if not self.closed:
            raise RuntimeError("Additional views require the official environment to be closed")
        for entry in self.results:
            directory = self.batch_dir / entry["directory"]
            path = directory / "trajectory.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as archive:
                rendered = archive["preview_render_time"]
                end_time = float(archive["time"][-1])
            ticks = video_times(end_time, self.fps)
            # Match the world preview's held source frame, even if official render cadence is
            # lower than env frequency. All six panels then show the same recorded state.
            indices = np.searchsorted(rendered, ticks + 1e-9, side="right") - 1
            frame_times = rendered[np.maximum(indices, 0)].copy()
            frame_times[-1] = end_time
            for name, camera, offset in (
                ("world_120", "world", 120.0),
                ("world_240", "world", 240.0),
                ("top", "top", 0.0),
                ("fpv", "fpv", 0.0),
                ("fpv_rear", "fpv_rear", 0.0),
            ):
                logger.info("Rendering %s view for %s", name, directory)
                replay(
                    path,
                    camera=camera,
                    fps=self.fps,
                    width=self.width,
                    height=self.height,
                    output=directory / "views" / f"{name}.mp4",
                    azimuth_offset=offset,
                    frame_times=frame_times,
                )
            combine_views(
                [directory / "views" / f"{name}.mp4" for name in PREVIEW_VIEWS],
                directory / "preview.mp4",
                fps=self.fps,
                width=self.width,
                height=self.height,
            )


def record_run(
    config: str = "level0.toml",
    controller: str | None = None,
    output_dir: str | Path | None = None,
    fps: float = 60,
    width: int = 960,
    height: int = 540,
    n_runs: int = 1,
) -> Path:
    """Record episodes through one official simulate() call and return the output directory."""
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs < 1:
        raise ValueError("n_runs must be a positive integer")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even numbers for H.264")
    config_path = _input_path(config, "config")
    controller_path = _input_path(controller, "lsy_drone_racing/control") if controller else None
    if output_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "runs" / f"{stamp}_{uuid4().hex[:8]}"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    recorder = RunRecorder(output_dir, fps, width, height, n_runs=n_runs)
    try:
        recorder.run(official_sim, config_path, controller_path)
    except BaseException as error:
        try:
            recorder.save(error)
        except Exception:
            logger.exception("Could not save partial run to %s", output_dir)
        raise
    recorder.save()
    recorder.render_views()
    print(f"Recorded {n_runs} run(s): {output_dir}")
    return output_dir


def main():
    """Parse recorder options and execute the official simulation loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="level0.toml")
    parser.add_argument("--controller")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-runs", "--n_runs", dest="n_runs", type=int, default=1)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    record_run(**vars(args))


if __name__ == "__main__":
    main()
