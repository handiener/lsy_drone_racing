"""Focused recording/replay checks without a controller flight or an EGL context.

Run with the project's Python: ``python -m unittest personal_tools.test_run_tools``.
"""

from __future__ import annotations

import json
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
              <camera name="fpv_cam:0" pos="0.05 0 0"/>
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
            with (
                mock.patch.object(mujoco, "Renderer", Renderer),
                mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("dynamics")),
                mock.patch.object(mujoco, "mj_forward", side_effect=AssertionError("dynamics")),
                mock.patch.object(replay_run, "VideoWriter", MemoryVideoWriter),
            ):
                result = replay_run.replay(path, fps=60, width=2, height=2)
            self.assertEqual(Path(result).parent, path.parent)
        self.assertEqual(len(rendered_states), 3)
        self.assertEqual(rendered_states[0][0], 0.0)
        self.assertEqual(rendered_states[-1][0], 0.04)
        np.testing.assert_allclose(rendered_states[-1][1], arrays["mocap_pos"][-1])
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

    def execute_recording(
        self,
        *,
        callback_error: bool = False,
        controller_stop: bool = False,
        n_runs: int = 1,
        error_episode: int = 1,
    ) -> tuple:
        """Run official simulate and verify hooks restore while their outer mocks remain."""
        from ml_collections import ConfigDict

        from personal_tools import record_run

        official = record_run.official_sim
        env, controller, events = self.make_race(
            callback_error=callback_error,
            controller_stop=controller_stop,
            error_episode=error_episode,
        )
        configuration = ConfigDict(
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


if __name__ == "__main__":
    unittest.main()
