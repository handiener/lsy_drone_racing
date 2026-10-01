"""Focused recording/replay checks without a controller flight or an EGL context.

Run with the project's Python: ``python -m unittest personal_tools.test_run_tools``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from personal_tools import _run_io, replay_run

_REAL_VIDEO_WRITER = _run_io.VideoWriter


def small_model() -> mujoco.MjModel:
    """Create a model with the features an XML-only snapshot could lose."""
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody>
            <camera name="overview" pos="0 -3 2"/>
            <body name="drone" pos="0 0 0.7">
              <freejoint/>
              <geom type="sphere" size="0.03" mass="0.03"/>
              <camera name="fpv_cam:0" pos="0.05 0 0" euler="0 -90 -90"/>
            </body>
            <body name="drone_mocap" mocap="true" pos="0 0 0.7"/>
            <body name="gate:0" mocap="true" pos="0.5 0 0.7">
              <geom type="box" size="0.01 0.3 0.3"/>
            </body>
            <body name="obstacle:0" mocap="true" pos="0.5 0.6 0.8">
              <geom type="cylinder" size="0.02 0.3"/>
            </body>
          </worldbody>
          <equality><weld body1="drone" body2="drone_mocap"/></equality>
        </mujoco>
        """
    )


def sample_trajectory(model: mujoco.MjModel) -> tuple[dict, dict]:
    """Provide a flight whose actual track differs from its nominal track."""
    times = np.array([0.0, 0.02, 0.04])
    pos = np.array([[[0.0, 0.0, 0.7]], [[0.1, 0.0, 0.7]], [[0.2, 0.0, 0.7]]])
    half_yaw = np.array([0.0, np.pi / 4, np.pi / 2])
    quat = np.zeros((3, 1, 4))
    quat[:, 0, 2] = np.sin(half_yaw)
    quat[:, 0, 3] = np.cos(half_yaw)
    gate_pos = np.broadcast_to([[[0.4, 0.6, 0.8]]], (3, 1, 3)).copy()
    obstacle_pos = np.broadcast_to([[[0.8, -0.6, 0.8]]], (3, 1, 3)).copy()
    mocap_pos = np.concatenate((pos, gate_pos, obstacle_pos), axis=1)
    mocap_quat = np.tile([1.0, 0.0, 0.0, 0.0], (3, 3, 1))
    mocap_quat[:, 0] = np.roll(quat[:, 0], 1, axis=-1)
    qpos = np.concatenate((pos[:, 0], np.roll(quat[:, 0], 1, axis=-1)), axis=1)
    arrays = {
        "model_mjb": _run_io.snapshot_model(model),
        "time": times,
        "env_step": np.arange(3),
        "sim_step": np.array([0, 10, 20]),
        "pos": pos,
        "quat": quat,
        "vel": np.tile([[[5.0, 0.0, 0.0]]], (3, 1, 1)),
        "ang_vel": np.tile([[[0.0, 0.0, np.pi / 0.04]]], (3, 1, 1)),
        "gates_pos": gate_pos,
        "gates_quat": np.tile([[[0.0, 0.0, 0.0, 1.0]]], (3, 1, 1)),
        "obstacles_pos": obstacle_pos,
        "obstacles_quat": np.tile([[[0.0, 0.0, 0.0, 1.0]]], (3, 1, 1)),
        "mocap_pos": mocap_pos,
        "mocap_quat": mocap_quat,
        "qpos": qpos,
        "qvel": np.zeros((3, model.nv)),
        "n_gates_passed": np.array([[0], [0], [1]]),
        "terminated": np.array([False, False, True]),
        "truncated": np.zeros(3, dtype=bool),
        "gate_sequence": np.array([0]),
        "gate_sequence_direction": np.array([1]),
        "drone_mocap_ids": np.array([0]),
        "gate_mocap_ids": np.array([1]),
        "obstacle_mocap_ids": np.array([2]),
    }
    metadata = {
        "schema_version": _run_io.SCHEMA_VERSION,
        "mujoco_version": mujoco.__version__,
        "crazyflow_version": "test",
        "repo_commit": None,
        "config": {"controller": {"file": "must-not-load.py"}},
        "config_path": "test.toml",
        "controller_path": "must-not-load.py",
        "env_freq": 50,
        "sim_freq": 500,
        "camera": -1,
        "cam_config": {"lookat": [0, 0, 0.7], "distance": 4},
    }
    return arrays, metadata


class MemoryVideoWriter:
    """Collect RGB frames in memory while retaining the writer's lifecycle."""

    instances: list = []

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        """Store encoder parameters and retain frames for assertions."""
        self.frames = []
        self.closed = False
        self.path = Path(path)
        self.fps = fps
        self.width, self.height = width, height
        self.instances.append(self)

    def __enter__(self) -> MemoryVideoWriter:
        """Return the memory encoder."""
        return self

    def __exit__(self, *args: object) -> None:
        """Finish the memory encoder."""
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """Retain an owned copy of each RGB frame."""
        self.frames.append(np.array(frame, copy=True))

    def _check_frame(self, frame: np.ndarray) -> np.ndarray:
        return _REAL_VIDEO_WRITER._check_frame(self, frame)

    def close(self) -> None:
        """Record encoder closure."""
        self.closed = True


class TrajectoryAndReplayTests(unittest.TestCase):
    """Check model fidelity, source time and dynamics-free replay."""

    def setUp(self) -> None:
        """Create an independent compiled model and frame collection."""
        self.model = small_model()
        MemoryVideoWriter.instances.clear()

    def test_mjb_preserves_weld_mocap_and_cameras(self) -> None:
        """A compiled snapshot retains features required by the Crazyflow model."""
        model = _run_io.load_model(_run_io.snapshot_model(self.model), mujoco.__version__)
        self.assertEqual((model.nq, model.nv, model.nmocap, model.neq), (7, 6, 3, 1))
        np.testing.assert_array_equal(model.eq_type, self.model.eq_type)
        np.testing.assert_array_equal(model.eq_obj1id, self.model.eq_obj1id)
        np.testing.assert_array_equal(model.eq_obj2id, self.model.eq_obj2id)
        np.testing.assert_array_equal(model.body_mocapid, self.model.body_mocapid)
        np.testing.assert_array_equal(model.cam_bodyid, self.model.cam_bodyid)
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cam:0"), 0
        )
        with self.assertRaises(ValueError):
            _run_io.load_model(_run_io.snapshot_model(self.model), "different-version")

    def test_preview_50hz_source_60fps_uses_simulation_time(self) -> None:
        """One second of env states yields a timed preview with its terminal frame."""
        with mock.patch.object(_run_io, "VideoWriter", MemoryVideoWriter):
            writer = _run_io.TimelineVideoWriter(Path("unused.mp4"), 60, 2, 2)
            for step in range(51):
                frame = np.full((2, 2, 3), step, dtype=np.uint8)
                writer.push(step / 50, frame)
            writer.finish(1.0, frame)
        frames = MemoryVideoWriter.instances[-1].frames
        self.assertEqual(len(frames), 61)
        self.assertEqual(int(frames[0][0, 0, 0]), 0)
        self.assertEqual(int(frames[1][0, 0, 0]), 0)
        self.assertEqual(int(frames[2][0, 0, 0]), 1)
        self.assertEqual(int(frames[-1][0, 0, 0]), 50)
        self.assertTrue(MemoryVideoWriter.instances[-1].closed)

    def test_interpolation_uses_slerp_and_preserves_terminal_track(self) -> None:
        """Display interpolation gives a 45-degree halfway pose without changing samples."""
        arrays, _ = sample_trajectory(self.model)
        original = arrays["mocap_quat"].copy()
        halfway = replay_run.interpolate_state(arrays, 0.01)
        np.testing.assert_allclose(halfway["mocap_pos"][0], [0.05, 0, 0.7])
        np.testing.assert_allclose(
            halfway["mocap_quat"][0], [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]
        )
        data = mujoco.MjData(self.model)
        final = replay_run.interpolate_state(arrays, 0.04)
        with mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("dynamics")):
            replay_run.restore_state(self.model, data, final, 0.04)
        np.testing.assert_allclose(data.mocap_pos, arrays["mocap_pos"][-1])
        np.testing.assert_allclose(data.mocap_quat, arrays["mocap_quat"][-1])
        np.testing.assert_array_equal(arrays["mocap_quat"], original)
        self.assertEqual(data.time, 0.04)

    def test_world_cameras_orbit_the_recorded_framing_at_equal_angles(self) -> None:
        """World cameras share their framing and rotate around the actual recorded azimuth."""
        arrays, metadata = sample_trajectory(self.model)
        metadata["cam_config"] = {
            "lookat": [0.3, -0.7, 0.8],
            "distance": 5.5,
            "azimuth": 310,
            "elevation": -24,
        }
        cameras = [
            replay_run._camera(
                self.model, arrays, metadata, "world", None, None, None, None, azimuth_offset=offset
            )
            for offset in (0, 120, 240)
        ]
        np.testing.assert_allclose([camera.azimuth for camera in cameras], [310, 70, 190])
        for camera in cameras:
            np.testing.assert_allclose(camera.lookat, [0.3, -0.7, 0.8])
            self.assertEqual(camera.distance, 5.5)
            self.assertEqual(camera.elevation, -24)
        np.testing.assert_allclose(
            np.mod(np.diff([camera.azimuth for camera in cameras] + [cameras[0].azimuth]), 360),
            [120, 120, 120],
        )
        custom = replay_run._camera(
            self.model, arrays, metadata, "world", None, 45, None, None, azimuth_offset=120
        )
        self.assertEqual(custom.azimuth, 165)

        metadata["world_cam_config"] = dict(metadata["cam_config"], azimuth=280)
        actual = replay_run._camera(
            self.model, arrays, metadata, "world", None, None, None, None, azimuth_offset=120
        )
        self.assertEqual(actual.azimuth, 40)

    def test_rear_fpv_rotates_the_saved_camera_mount_without_mutating_snapshot(self) -> None:
        """Rear FPV faces backward while keeping the onboard camera's height and up axis."""
        front_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cam:0")
        self.model.cam_pos[front_id] = [0.05, 0.02, 0.03]
        arrays, metadata = sample_trajectory(self.model)
        snapshot = arrays["model_mjb"].copy()
        replay_model = _run_io.load_model(snapshot, mujoco.__version__)
        rear = replay_run._camera(
            replay_model, arrays, metadata, "fpv_rear", None, None, None, None
        )
        rear_id = mujoco.mj_name2id(replay_model, mujoco.mjtObj.mjOBJ_CAMERA, rear)
        self.assertEqual(replay_model.cam_bodyid[rear_id], self.model.cam_bodyid[front_id])
        np.testing.assert_allclose(replay_model.cam_pos[rear_id], [-0.05, -0.02, 0.03])
        front_data, rear_data = mujoco.MjData(self.model), mujoco.MjData(replay_model)
        body_id = self.model.cam_bodyid[front_id]
        # Camera -Z looks forward and camera +Y points up. Inspect MuJoCo's world transforms
        # at two drone headings, then express both axes in the actual drone body frame.
        for time in (0.0, 0.02):
            state = replay_run.interpolate_state(arrays, time)
            replay_run.restore_state(self.model, front_data, state, time)
            replay_run.restore_state(replay_model, rear_data, state, time)
            body_rotation = front_data.xmat[body_id].reshape(3, 3)
            front_rotation = front_data.cam_xmat[front_id].reshape(3, 3)
            rear_rotation = rear_data.cam_xmat[rear_id].reshape(3, 3)
            front_direction = body_rotation.T @ -front_rotation[:, 2]
            rear_direction = body_rotation.T @ -rear_rotation[:, 2]
            front_up = body_rotation.T @ front_rotation[:, 1]
            rear_up = body_rotation.T @ rear_rotation[:, 1]
            np.testing.assert_allclose(front_direction, [1, 0, 0], atol=1e-12)
            np.testing.assert_allclose(rear_direction, -front_direction, atol=1e-12)
            np.testing.assert_allclose(front_up, [0, 0, 1], atol=1e-12)
            np.testing.assert_allclose(rear_up, front_up, atol=1e-12)
            np.testing.assert_allclose(
                body_rotation.T @ (front_data.cam_xpos[front_id] - front_data.xpos[body_id]),
                [0.05, 0.02, 0.03],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                body_rotation.T @ (rear_data.cam_xpos[rear_id] - rear_data.xpos[body_id]),
                [-0.05, -0.02, 0.03],
                atol=1e-12,
            )
        np.testing.assert_array_equal(arrays["model_mjb"], snapshot)
        np.testing.assert_allclose(self.model.cam_pos[front_id], [0.05, 0.02, 0.03])
        with self.assertRaises(ValueError):
            replay_run._camera(replay_model, arrays, metadata, "fpv_rear", 5, None, None, None)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required to inspect the video layout")
    def test_six_view_video_keeps_two_rows_and_synchronized_frame_order(self) -> None:
        """Decoded video places all six views correctly and preserves frame order."""
        width, height, fps = 32, 16, 60
        colors = np.array(
            [
                [240, 20, 20],
                [20, 240, 20],
                [20, 20, 240],
                [240, 240, 20],
                [240, 20, 240],
                [20, 240, 240],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index, color in enumerate(colors):
                path = root / f"view_{index}.mp4"
                with _REAL_VIDEO_WRITER(path, fps, width, height) as writer:
                    for scale in (0.5, 1.0):
                        writer.write(
                            np.broadcast_to((color * scale).astype(np.uint8), (height, width, 3))
                        )
                sources.append(path)
            output = _run_io.combine_views(
                sources, root / "preview.mp4", fps=fps, width=width, height=height
            )
            decoded = subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(output),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            ).stdout
        frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 2 * height, 3 * width, 3)
        self.assertEqual(len(frames), 2)
        centers = [(height // 2, width // 2 + index * width) for index in range(3)]
        centers.extend([(height + height // 2, width // 2 + index * width) for index in range(3)])
        for frame, scale in zip(frames, (0.5, 1.0), strict=True):
            np.testing.assert_allclose(
                [frame[y, x] for y, x in centers], colors.astype(float) * scale, atol=15
            )

    def test_replay_renders_saved_states_without_controller_or_dynamics(self) -> None:
        """The complete replay path loads the snapshot, not a new race/controller."""
        arrays, metadata = sample_trajectory(self.model)
        rendered_states = []

        class Renderer:
            def __init__(self, model: mujoco.MjModel, **kwargs: object) -> None:
                self.model = model

            def update_scene(self, data: mujoco.MjData, **kwargs: object) -> None:
                rendered_states.append((data.time, data.mocap_pos.copy()))

            def render(self) -> np.ndarray:
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def close(self) -> None:
                pass

            def __enter__(self) -> Renderer:
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez_compressed(path, **arrays, metadata_json=json.dumps(metadata))
            for index, frame_times in enumerate((None, np.array([0.0, 0.0, 0.04]))):
                with self.subTest(held_frames=frame_times is not None):
                    rendered_states.clear()
                    with (
                        mock.patch.object(mujoco, "Renderer", Renderer),
                        mock.patch.object(
                            mujoco, "mj_step", side_effect=AssertionError("dynamics")
                        ),
                        mock.patch.object(
                            mujoco, "mj_forward", side_effect=AssertionError("dynamics")
                        ),
                        mock.patch.object(replay_run, "VideoWriter", MemoryVideoWriter),
                    ):
                        result = replay_run.replay(
                            path,
                            fps=60,
                            width=2,
                            height=2,
                            output=path.parent / f"replay_{index}.mp4",
                            frame_times=frame_times,
                        )
                    self.assertEqual(Path(result).parent, path.parent)
                    expected = [0, 1 / 60, 0.04] if frame_times is None else frame_times
                    np.testing.assert_allclose([state[0] for state in rendered_states], expected)
                    np.testing.assert_allclose(rendered_states[-1][1], arrays["mocap_pos"][-1])
                    if frame_times is not None:
                        np.testing.assert_array_equal(rendered_states[0][1], rendered_states[1][1])
                    self.assertTrue(MemoryVideoWriter.instances[-1].closed)


class OfficialLoopRecordingTests(unittest.TestCase):
    """Exercise the recorder through the real official loop, using a small fake race."""

    def make_race(
        self, *, callback_error: bool = False, controller_stop: bool = False, error_episode: int = 1
    ) -> tuple:
        """Build CPU-only state and rendering doubles with a real compiled scene."""
        from personal_tools import record_run

        events = []
        model = small_model()

        class Sim:
            def __init__(self) -> None:
                self.n_worlds = self.n_drones = 1
                self.freq = 500
                self.mj_model = model
                self.viewer = None
                self.marker_count = 0
                self.mjx_data = SimpleNamespace(
                    mocap_pos=np.zeros((1, model.nmocap, 3)),
                    mocap_quat=np.tile([1.0, 0.0, 0.0, 0.0], (1, model.nmocap, 1)),
                    qpos=mujoco.MjData(model).qpos[None].copy(),
                    qvel=np.zeros((1, model.nv)),
                )

            def render(
                self,
                mode: str = "human",
                camera: int = -1,
                width: int = 960,
                height: int = 540,
                cam_config: dict | None = None,
            ) -> np.ndarray:
                self.viewer = SimpleNamespace() if self.viewer is None else self.viewer
                events.append(("render", env.tick))
                env.render_cameras.append(camera)
                if mode != "rgb_array":
                    raise AssertionError("The official render call must become headless RGB")
                if self.data is not env.data.sim_data:
                    raise AssertionError("Render decorator would see stale simulation state")
                frame = np.full((height, width, 3), env.tick, dtype=np.uint8)
                frame[:, :, 1] = self.marker_count
                return frame

        class Env(record_run.gymnasium.Env):
            def __init__(self) -> None:
                self.sim = Sim()
                self.tick = 0
                self.closed = False
                self.render_cameras = []
                self.reset_count = 0
                self.reset_seeds = []
                self.controller_seeds = []
                self.track_offset = 0.0
                self.mocap_ids = (np.array([1]), np.array([2]))
                self.settings = SimpleNamespace(freq=50, camera=-1, cam_config={})
                self.action_space = record_run.gymnasium.spaces.Box(-1, 1, shape=(1,))
                self.observation_space = record_run.gymnasium.spaces.Dict({})
                self.set_state()
                self.sim.data = self.data.sim_data

            def set_state(self) -> None:
                terminal = self.tick == 2 and not controller_stop
                pos = [-1.0, -1.0, -1.0] if terminal else [self.tick / 10, 0.0, 0.7]
                states = SimpleNamespace(
                    pos=np.array([[pos]]),
                    quat=np.array([[[0.0, 0.0, 0.0, 1.0]]]),
                    vel=np.zeros((1, 1, 3)),
                    ang_vel=np.zeros((1, 1, 3)),
                )
                core = SimpleNamespace(
                    steps=np.array([self.tick * 10]),
                    freq=500,
                    drone_mocap_ids=np.array([0]),
                    mjx_synced=False,
                )
                self.data = SimpleNamespace(
                    sim_data=SimpleNamespace(states=states, core=core),
                    steps=np.array([self.tick]),
                    gates_pos=np.array([[[0.4 + self.track_offset, 0.6, 0.8]]]),
                    gates_quat=np.array([[[0.0, 0.0, 0.0, 1.0]]]),
                    obstacles_pos=np.array([[[0.8, -0.6, 0.8]]]),
                    n_gates_passed=np.array([[int(terminal)]]),
                    gate_sequence=np.array([0]),
                    gate_sequence_direction=np.array([1]),
                )

            def observation(self) -> dict:
                return {
                    "pos": self.data.sim_data.states.pos[0, 0].copy(),
                    "n_gates_passed": self.data.n_gates_passed[0, 0],
                    "gate_sequence": np.array([0]),
                    "gates_pos": np.array([[-99.0, -99.0, -99.0]]),
                }

            def reset(self, **kwargs: object) -> tuple[dict, dict]:
                seed = kwargs.get("seed")
                self.reset_count += 1
                self.reset_seeds.append(seed)
                self.track_offset = (
                    float(seed) / 100 if seed is not None else (self.reset_count - 1) / 10
                )
                self.tick = 0
                self.set_state()
                events.append(("reset", 0))
                return self.observation(), {}

            def step(self, action: object) -> tuple:
                # Mimic buffer reuse/donation: recordings must already own their old arrays.
                self.data.sim_data.states.pos.fill(99)
                self.tick += 1
                self.set_state()
                events.append(("step", self.tick))
                return self.observation(), 0.0, self.tick == 2 and not controller_stop, False, {}

            def render(self) -> None:
                if not self.data.sim_data.core.mjx_synced:
                    self.sim.mjx_data.mocap_pos[0, 0] = self.data.sim_data.states.pos[0, 0]
                    self.data.sim_data.core.mjx_synced = True
                self.sim.render(camera=self.settings.camera, cam_config=self.settings.cam_config)

            def close(self) -> None:
                self.closed = True
                events.append(("close", self.tick))

        env = Env()

        class Controller:
            def __init__(self, obs: dict, info: dict, config: object) -> None:
                if env.sim.viewer is None:
                    raise AssertionError("Renderer must initialize before controller construction")
                events.append(("constructor", env.tick))
                env.controller_seeds.append(config.env.seed)

            def compute_control(self, obs: dict, info: dict) -> np.ndarray:
                events.append(("compute", env.tick))
                return np.zeros(1)

            def step_callback(self, *args: object) -> bool:
                events.append(("step_callback", env.tick))
                if callback_error and env.reset_count == error_episode:
                    raise RuntimeError("intentional callback failure")
                return env.tick == 2 and controller_stop

            def render_callback(self, sim: Sim) -> None:
                if sim.viewer is None or sim.data is not env.data.sim_data:
                    raise AssertionError("First marker callback needs initialized, current viewer")
                sim.marker_count += 1
                events.append(("marker", env.tick))

            def episode_callback(self) -> None:
                events.append(("episode_callback", env.tick))

            def episode_reset(self) -> None:
                events.append(("episode_reset", env.tick))

        return env, Controller, events

    @staticmethod
    def configuration() -> object:
        """Return a fresh official config with a stable seed for each test."""
        from ml_collections import ConfigDict

        return ConfigDict(
            {
                "controller": {"file": "fake.py"},
                "sim": {"render": False},
                "env": {
                    "id": "DroneRacing-v0",
                    "freq": 50,
                    "sensor_range": 0.7,
                    "control_mode": "state",
                    "track": {},
                    "seed": 7,
                },
            }
        )

    def execute_recording(
        self,
        *,
        callback_error: bool = False,
        controller_stop: bool = False,
        n_runs: int = 1,
        error_episode: int = 1,
    ) -> tuple:
        """Run official simulate and verify hooks restore while their outer mocks remain."""
        from personal_tools import record_run

        official = record_run.official_sim
        env, controller, events = self.make_race(
            callback_error=callback_error,
            controller_stop=controller_stop,
            error_episode=error_episode,
        )
        configuration = self.configuration()
        MemoryVideoWriter.instances.clear()
        original_render = env.sim.render
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory)
            config_path, controller_path = output / "fake.toml", output / "fake.py"
            config_path.write_text("# test config\n")
            controller_path.write_text("# test controller\n")
            factory = stack.enter_context(
                mock.patch.object(official.gymnasium, "make", return_value=env)
            )
            config_loader = stack.enter_context(
                mock.patch.object(official, "load_config", return_value=configuration)
            )
            controller_loader = stack.enter_context(
                mock.patch.object(official, "load_controller", return_value=controller)
            )
            stats = stack.enter_context(
                mock.patch.object(
                    official,
                    "log_episode_stats",
                    side_effect=lambda *args: events.append(("stats", env.tick)),
                )
            )
            stack.enter_context(mock.patch.object(_run_io, "VideoWriter", MemoryVideoWriter))
            recorder = record_run.RunRecorder(output, 60, 2, 2, n_runs=n_runs)
            official_loop = official.simulate
            failure = None
            try:
                recorder.run(official, config_path, controller_path)
            except RuntimeError as error:
                if not callback_error:
                    raise
                failure = error
            recorder.save(failure)
            self.assertIs(official.simulate, official_loop)
            self.assertIs(official.gymnasium.make, factory)
            self.assertIs(official.load_config, config_loader)
            self.assertIs(official.load_controller, controller_loader)
            self.assertIs(official.log_episode_stats, stats)
            self.assertEqual(env.sim.render, original_render)
            factory.assert_called_once()
            controller_loader.assert_called_once()
            if n_runs > 1:
                episodes = []
                for directory in sorted(output.glob("run_*")):
                    arrays, metadata = _run_io.load_trajectory(directory / "trajectory.npz")
                    result = json.loads((directory / "result.json").read_text())
                    episodes.append((arrays, metadata, result))
                summary = json.loads((output / "summary.json").read_text())
            else:
                arrays, metadata = _run_io.load_trajectory(output / "trajectory.npz")
                result = json.loads((output / "result.json").read_text())
        self.assertTrue(env.closed)
        self.assertTrue(all(writer.closed for writer in MemoryVideoWriter.instances))
        if n_runs > 1:
            return episodes, summary, events, failure, env.reset_seeds, env.controller_seeds
        return arrays, metadata, result, events, failure

    def test_official_callbacks_state_truth_and_terminal_frame(self) -> None:
        """Retain official callbacks, actual randomized track and disabled terminal pose."""
        arrays, _, result, events, failure = self.execute_recording()
        self.assertIsNone(failure)
        self.assertEqual(
            events,
            [
                ("reset", 0),
                ("render", 0),
                ("constructor", 0),
                ("compute", 0),
                ("step", 1),
                ("step_callback", 1),
                ("marker", 1),
                ("render", 1),
                ("compute", 1),
                ("step", 2),
                ("step_callback", 2),
                ("episode_callback", 2),
                ("stats", 2),
                ("episode_reset", 2),
                ("render", 2),
                ("close", 2),
            ],
        )
        np.testing.assert_allclose(arrays["time"], [0, 0.02, 0.04])
        np.testing.assert_allclose(arrays["pos"][:, 0], [[0, 0, 0.7], [0.1, 0, 0.7], [-1, -1, -1]])
        np.testing.assert_allclose(arrays["gates_pos"][0, 0], [0.4, 0.6, 0.8])
        self.assertTrue(arrays["terminated"][-1])
        self.assertTrue(result["finished"])
        self.assertEqual(result["flight_time"], 0.02)
        self.assertEqual(result["simulation_duration"], 0.04)
        self.assertEqual(result["status"], "complete")
        frames = MemoryVideoWriter.instances[-1].frames
        self.assertEqual(len(frames), 3)
        self.assertEqual(int(frames[-1][0, 0, 0]), 2)
        self.assertEqual(int(frames[-1][0, 0, 1]), 1)

    def test_controller_termination_is_owned_by_official_loop(self) -> None:
        """A controller can stop the episode even when env termination flags are false."""
        arrays, _, result, events, _ = self.execute_recording(controller_stop=True)
        self.assertEqual(len(arrays["time"]), 3)
        self.assertFalse(arrays["terminated"].any())
        self.assertFalse(arrays["truncated"].any())
        self.assertFalse(result["finished"])
        self.assertIn(("episode_callback", 2), events)
        self.assertNotIn(("marker", 2), events)

    def test_exception_restores_hooks_and_keeps_partial_state(self) -> None:
        """Callback failure closes resources, restores hooks and preserves the completed step."""
        arrays, _, result, events, failure = self.execute_recording(callback_error=True)
        self.assertIsInstance(failure, RuntimeError)
        np.testing.assert_allclose(arrays["time"], [0, 0.02])
        self.assertEqual(result["status"], "error")
        self.assertIn("intentional callback failure", result["error"])
        self.assertNotIn(("episode_callback", 1), events)
        self.assertEqual(events[-2:], [("render", 1), ("close", 1)])

    def test_multiple_runs_preserve_official_rng_stream_and_separate_timelines(self) -> None:
        """One environment serves three episodes with independent videos and actual tracks."""
        episodes, summary, events, failure, seeds, controller_seeds = self.execute_recording(
            n_runs=3
        )
        self.assertIsNone(failure)
        self.assertEqual(seeds, [None, None, None])
        self.assertEqual(controller_seeds, [7, 7, 7])
        self.assertEqual(summary["completed_runs"], 3)
        self.assertEqual(
            [item["directory"] for item in summary["runs"]], ["run_0001", "run_0002", "run_0003"]
        )
        for index, (arrays, metadata, result) in enumerate(episodes):
            np.testing.assert_allclose(arrays["time"], [0, 0.02, 0.04])
            np.testing.assert_allclose(arrays["gates_pos"][0, 0, 0], 0.4 + index / 10)
            self.assertEqual(metadata["config"]["env"]["seed"], 7)
            self.assertEqual(result["status"], "complete")
        self.assertEqual(events.count(("constructor", 0)), 3)
        self.assertEqual(events.count(("marker", 1)), 3)
        self.assertEqual(events.count(("close", 2)), 1)
        self.assertEqual([len(writer.frames) for writer in MemoryVideoWriter.instances], [3, 3, 3])

    def test_later_episode_failure_keeps_completed_runs_and_partial_run(self) -> None:
        """An exception in episode two preserves episode one and stops the official batch."""
        episodes, summary, events, failure, seeds, _ = self.execute_recording(
            n_runs=3, callback_error=True, error_episode=2
        )
        self.assertIsInstance(failure, RuntimeError)
        self.assertEqual(seeds, [None, None])
        self.assertEqual(summary["n_runs"], 3)
        self.assertEqual(summary["completed_runs"], 1)
        self.assertEqual([result["status"] for _, _, result in episodes], ["complete", "error"])
        np.testing.assert_allclose(episodes[0][0]["time"], [0, 0.02, 0.04])
        np.testing.assert_allclose(episodes[1][0]["time"], [0, 0.02])
        self.assertEqual(events.count(("close", 1)), 1)

    def test_public_recording_synchronizes_six_views_after_official_batch(self) -> None:
        """One official call per batch supplies all synchronized camera videos."""
        from personal_tools import record_run

        for n_runs in (1, 3):
            with self.subTest(n_runs=n_runs):
                official = record_run.official_sim
                env, controller, events = self.make_race()
                # The recording's overview must stay a free world camera even if the
                # environment config normally selects its onboard camera.
                env.settings.camera = "fpv"
                snapshots = {}
                MemoryVideoWriter.instances.clear()

                def replay_saved(path: Path, **kwargs: object) -> Path:
                    self.assertTrue(env.closed)
                    events.append(("replay", kwargs["camera"]))
                    path = Path(path)
                    snapshots.setdefault(path, path.read_bytes())
                    self.assertEqual(path.read_bytes(), snapshots[path])
                    world = next(
                        writer
                        for writer in MemoryVideoWriter.instances
                        if writer.path == path.parent / "views" / "world.mp4"
                    )
                    np.testing.assert_allclose(
                        kwargs["frame_times"], [int(frame[0, 0, 0]) / 50 for frame in world.frames]
                    )
                    self.assertEqual(kwargs["fps"], 60)
                    self.assertEqual((kwargs["width"], kwargs["height"]), (2, 2))
                    destination = Path(kwargs["output"])
                    offset = kwargs.get("azimuth_offset", 0)
                    name = f"world_{offset:g}" if kwargs["camera"] == "world" else kwargs["camera"]
                    self.assertEqual(destination, path.parent / "views" / f"{name}.mp4")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(b"test replay")
                    return destination

                def combine_saved(
                    paths: list[Path], output: Path, fps: float, width: int, height: int
                ) -> Path:
                    self.assertTrue(env.closed)
                    events.append(("combine", "preview"))
                    output = Path(output)
                    self.assertEqual(
                        [Path(path) for path in paths],
                        [
                            output.parent / "views" / f"{view}.mp4"
                            for view in (
                                "world",
                                "world_120",
                                "world_240",
                                "top",
                                "fpv",
                                "fpv_rear",
                            )
                        ],
                    )
                    self.assertEqual(output.name, "preview.mp4")
                    self.assertEqual((fps, width, height), (60, 2, 2))
                    output.write_bytes(b"test combined preview")
                    return output

                with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                    root = Path(temporary)
                    config_path, controller_path = root / "fake.toml", root / "fake.py"
                    config_path.write_text("# test config\n")
                    controller_path.write_text("# test controller\n")
                    factory = stack.enter_context(
                        mock.patch.object(official.gymnasium, "make", return_value=env)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            official, "load_config", return_value=self.configuration()
                        )
                    )
                    loader = stack.enter_context(
                        mock.patch.object(official, "load_controller", return_value=controller)
                    )
                    stack.enter_context(mock.patch.object(official, "log_episode_stats"))
                    loop = stack.enter_context(
                        mock.patch.object(official, "simulate", wraps=official.simulate)
                    )
                    stack.enter_context(
                        mock.patch.object(_run_io, "VideoWriter", MemoryVideoWriter)
                    )
                    replay = stack.enter_context(
                        mock.patch.object(record_run, "replay", side_effect=replay_saved)
                    )
                    combine = stack.enter_context(
                        mock.patch.object(record_run, "combine_views", side_effect=combine_saved)
                    )
                    output = record_run.record_run(
                        config=str(config_path),
                        controller=str(controller_path),
                        output_dir=root / "recorded",
                        fps=60,
                        width=2,
                        height=2,
                        n_runs=n_runs,
                    )
                    loop.assert_called_once_with(
                        config=str(config_path),
                        controller=str(controller_path),
                        n_runs=n_runs,
                        render=True,
                    )
                    factory.assert_called_once()
                    loader.assert_called_once()
                    self.assertEqual(replay.call_count, 5 * n_runs)
                    self.assertEqual(combine.call_count, n_runs)
                    self.assertEqual(
                        [
                            (call.kwargs["camera"], call.kwargs.get("azimuth_offset", 0))
                            for call in replay.call_args_list
                        ],
                        [("world", 120), ("world", 240), ("top", 0), ("fpv", 0), ("fpv_rear", 0)]
                        * n_runs,
                    )
                    close_index = events.index(("close", 2))
                    self.assertTrue(
                        all(
                            index > close_index
                            for index, event in enumerate(events)
                            if event[0] in ("replay", "combine")
                        )
                    )
                    directories = (
                        [output]
                        if n_runs == 1
                        else [output / f"run_{index:04d}" for index in range(1, n_runs + 1)]
                    )
                    for index, directory in enumerate(directories):
                        trajectory = directory / "trajectory.npz"
                        self.assertEqual(trajectory.read_bytes(), snapshots[trajectory])
                        arrays, metadata = _run_io.load_trajectory(trajectory)
                        self.assertEqual(
                            metadata["preview_cameras"],
                            ["world", "world_120", "world_240", "top", "fpv", "fpv_rear"],
                        )
                        result = json.loads((directory / "result.json").read_text())
                        self.assertEqual(
                            result["video"]["layout"],
                            [["world", "world_120", "world_240"], ["top", "fpv", "fpv_rear"]],
                        )
                        self.assertEqual(
                            (result["video"]["width"], result["video"]["height"]), (6, 4)
                        )
                        np.testing.assert_allclose(arrays["time"], [0, 0.02, 0.04])
                        np.testing.assert_allclose(
                            arrays["pos"][:, 0], [[0, 0, 0.7], [0.1, 0, 0.7], [-1, -1, -1]]
                        )
                        np.testing.assert_allclose(arrays["gates_pos"][0, 0, 0], 0.4 + index / 10)
                        self.assertTrue((directory / "preview.mp4").is_file())
                        self.assertTrue((directory / "views" / "world_120.mp4").is_file())
                        self.assertTrue((directory / "views" / "world_240.mp4").is_file())
                        self.assertTrue((directory / "views" / "top.mp4").is_file())
                        self.assertTrue((directory / "views" / "fpv.mp4").is_file())
                        self.assertTrue((directory / "views" / "fpv_rear.mp4").is_file())
                self.assertEqual(env.reset_seeds, [None] * n_runs)
                self.assertEqual(env.controller_seeds, [7] * n_runs)
                self.assertTrue(all(camera == -1 for camera in env.render_cameras))


if __name__ == "__main__":
    unittest.main()
