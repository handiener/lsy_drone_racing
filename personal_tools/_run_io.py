"""Portable trajectory storage and simulation-time video encoding for run tools."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    import mujoco

SCHEMA_VERSION = 1


def snapshot_model(model: mujoco.MjModel) -> np.ndarray:
    """Serialize the compiled model, including mesh/texture assets and runtime edits."""
    import mujoco

    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return buffer


def load_model(model_mjb: np.ndarray, mujoco_version: str) -> mujoco.MjModel:
    """Load an embedded MJB with the same MuJoCo version that recorded it."""
    import mujoco

    if mujoco_version != mujoco.__version__:
        raise ValueError(
            f"This trajectory was recorded with MuJoCo {mujoco_version}, but the active "
            f"environment has {mujoco.__version__}. Activate the recording environment before "
            "replaying: compiled MJB models are version-specific."
        )
    if model_mjb.dtype != np.uint8 or model_mjb.ndim != 1 or not model_mjb.size:
        raise ValueError("model_mjb must be a nonempty one-dimensional uint8 array")
    return mujoco.MjModel.from_binary_path(
        "trajectory.mjb", assets={"trajectory.mjb": model_mjb.tobytes()}
    )


def validate_trajectory(arrays: Mapping[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Reject malformed archives before loading their model or rendering their states."""
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trajectory schema {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    for key in ("mujoco_version", "env_freq", "sim_freq"):
        if key not in metadata:
            raise ValueError(f"Trajectory metadata is missing {key!r}")
    for key in ("env_freq", "sim_freq"):
        value = metadata[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"metadata.{key} must be positive and finite")

    required = {
        "time",
        "env_step",
        "sim_step",
        "pos",
        "quat",
        "vel",
        "ang_vel",
        "gates_pos",
        "gates_quat",
        "obstacles_pos",
        "obstacles_quat",
        "mocap_pos",
        "mocap_quat",
        "qpos",
        "qvel",
        "n_gates_passed",
        "terminated",
        "truncated",
        "gate_sequence",
        "gate_sequence_direction",
        "drone_mocap_ids",
        "gate_mocap_ids",
        "obstacle_mocap_ids",
        "model_mjb",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"Trajectory is missing arrays: {', '.join(missing)}")
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(f"{key} must be an ndarray without pickled objects")

    times = arrays["time"]
    if times.ndim != 1 or not times.size or not np.isfinite(times).all():
        raise ValueError("time must be a nonempty finite one-dimensional array")
    if abs(float(times[0])) > 1e-9 or np.any(np.diff(times) <= 0):
        raise ValueError("time must start at zero and increase strictly")
    count = times.size
    for key in ("env_step", "sim_step", "terminated", "truncated"):
        if arrays[key].shape != (count,):
            raise ValueError(f"{key} must have shape ({count},)")
    for key in ("env_step", "sim_step"):
        steps = arrays[key]
        if not np.issubdtype(steps.dtype, np.integer):
            raise ValueError(f"{key} must contain integers")
        if steps[0] != 0 or np.any(np.diff(steps) <= 0):
            raise ValueError(f"{key} must start at zero and increase strictly")
    if not np.allclose(times, arrays["sim_step"] / metadata["sim_freq"], atol=1e-8):
        raise ValueError("time must match sim_step / metadata.sim_freq")
    if not np.array_equal(arrays["env_step"], np.arange(count)):
        raise ValueError("env_step must contain every step, starting at zero without gaps")
    if not np.allclose(times, arrays["env_step"] / metadata["env_freq"], rtol=0, atol=1e-8):
        raise ValueError("time must match env_step / metadata.env_freq (the source cadence)")
    for key in ("terminated", "truncated"):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"{key} must contain booleans")

    entity_counts: dict[str, int] = {}
    for entity, id_key in (
        ("drone", "drone_mocap_ids"),
        ("gate", "gate_mocap_ids"),
        ("obstacle", "obstacle_mocap_ids"),
    ):
        ids = arrays[id_key]
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"{id_key} must be a one-dimensional integer array")
        entity_counts[entity] = len(ids)
    if not entity_counts["drone"]:
        raise ValueError("The trajectory must contain at least one drone")
    dimensions = {
        "pos": (count, entity_counts["drone"], 3),
        "quat": (count, entity_counts["drone"], 4),
        "vel": (count, entity_counts["drone"], 3),
        "ang_vel": (count, entity_counts["drone"], 3),
        "gates_pos": (count, entity_counts["gate"], 3),
        "gates_quat": (count, entity_counts["gate"], 4),
        "obstacles_pos": (count, entity_counts["obstacle"], 3),
        "obstacles_quat": (count, entity_counts["obstacle"], 4),
        "n_gates_passed": (count, entity_counts["drone"]),
    }
    for key, shape in dimensions.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}; got {arrays[key].shape}")
    for key in ("qpos", "qvel"):
        if arrays[key].ndim != 2 or arrays[key].shape[0] != count:
            raise ValueError(f"{key} must have shape (number of samples, model dimension)")
    mocap = arrays["mocap_pos"]
    if mocap.ndim != 3 or mocap.shape[0] != count or mocap.shape[2] != 3:
        raise ValueError("mocap_pos must have shape (number of samples, model.nmocap, 3)")
    if arrays["mocap_quat"].shape != (count, mocap.shape[1], 4):
        raise ValueError("mocap_quat must match mocap_pos with four quaternion components")
    ids = np.concatenate(
        [arrays[key] for key in ("drone_mocap_ids", "gate_mocap_ids", "obstacle_mocap_ids")]
    )
    if np.any(ids < 0) or np.any(ids >= mocap.shape[1]) or len(np.unique(ids)) != len(ids):
        raise ValueError("Drone, gate and obstacle mocap IDs must be unique valid model IDs")

    numeric_keys = required - {"model_mjb", "terminated", "truncated"}
    for key in numeric_keys:
        if not np.issubdtype(arrays[key].dtype, np.number) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} must contain finite numeric values")
    for key in ("quat", "gates_quat", "obstacles_quat", "mocap_quat"):
        if not np.allclose(np.linalg.norm(arrays[key], axis=-1), 1.0, atol=1e-3):
            raise ValueError(f"{key} must contain unit quaternions")
    sequence, directions = arrays["gate_sequence"], arrays["gate_sequence_direction"]
    if sequence.ndim != 1 or directions.shape != sequence.shape:
        raise ValueError("gate_sequence and gate_sequence_direction must be matching 1D arrays")
    if not np.issubdtype(sequence.dtype, np.integer):
        raise ValueError("gate_sequence must contain integers")
    if np.any(sequence < 0) or np.any(sequence >= entity_counts["gate"]):
        raise ValueError("gate_sequence contains an invalid gate index")
    if np.any(~np.isin(directions, [-1, 1])):
        raise ValueError("gate_sequence_direction must contain only -1 or 1")
    if not np.issubdtype(arrays["n_gates_passed"].dtype, np.integer):
        raise ValueError("n_gates_passed must contain integers")
    if np.any(arrays["n_gates_passed"] < 0) or np.any(arrays["n_gates_passed"] > sequence.size):
        raise ValueError("n_gates_passed must be between zero and the gate sequence length")
    binary = arrays["model_mjb"]
    if binary.dtype != np.uint8 or binary.ndim != 1 or not binary.size:
        raise ValueError("model_mjb must be a nonempty one-dimensional uint8 array")


def load_trajectory(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a schema-v1 archive using NumPy's pickle-free format."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    try:
        raw = arrays["metadata_json"]
        if raw.ndim != 0 or raw.dtype.kind != "U":
            raise ValueError("metadata_json must be a scalar Unicode JSON string")
        metadata = json.loads(str(raw.item()))
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode a JSON object")
    except KeyError as exc:
        raise ValueError("Trajectory is missing metadata_json") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Trajectory metadata_json is invalid JSON") from exc
    validate_trajectory(arrays, metadata)
    return arrays, metadata


def _video_parameters(fps: float, width: int, height: int) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even integers for yuv420p MP4")


def video_times(end_time: float, fps: float, start_time: float = 0.0) -> np.ndarray:
    """Return FPS sample times, retaining the terminal state as the last video frame.

    Encoding uses a fixed frame rate. The terminal frame replaces the last regular sample, so the
    video duration differs from the recorded duration by at most one output frame interval.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not math.isfinite(end_time) or not math.isfinite(start_time) or end_time < start_time:
        raise ValueError("Video times must be finite, with end_time >= start_time")
    count = math.floor((end_time - start_time) * fps + 1e-9) + 1
    times = start_time + np.arange(count, dtype=np.float64) / fps
    times[-1] = end_time
    return times


class VideoWriter:
    """Stream RGB frames to ffmpeg without a shell or real-time pacing."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        """Open an ffmpeg encoder and refuse to overwrite an existing output."""
        _video_parameters(fps, width, height)
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"Video output already exists: {self.path}")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found on PATH")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.fps = width, height, fps
        self.frames_written = 0
        self._closed = False
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-f",
                    "mp4",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
                stderr=self._stderr,
            )
        except BaseException:
            self._stderr.close()
            raise

    def _check_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 RGB frame of shape {(self.height, self.width, 3)}; "
                f"got {frame.shape}, {frame.dtype}"
            )
        return np.ascontiguousarray(frame)

    def write(self, frame: np.ndarray) -> None:
        """Encode one frame at the configured output FPS."""
        if self._closed:
            raise RuntimeError("Video writer is closed")
        frame = self._check_frame(frame)
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            try:
                self.close()
            except RuntimeError as encode_error:
                raise encode_error from exc
            raise RuntimeError("ffmpeg closed its input unexpectedly") from exc
        self.frames_written += 1

    def close(self) -> None:
        """Finish the MP4 and report encoder errors with ffmpeg's diagnostic text."""
        if self._closed:
            return
        self._closed = True
        assert self._process.stdin is not None
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        returncode = self._process.wait()
        self._stderr.seek(0)
        diagnostic = self._stderr.read().decode("utf-8", errors="replace").strip()
        self._stderr.close()
        if returncode:
            raise RuntimeError(f"ffmpeg failed with exit code {returncode}: {diagnostic}")

    def __enter__(self) -> VideoWriter:
        """Return this encoder for use in a with statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the file without replacing an exception raised by its caller."""
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except RuntimeError:
                pass


class TimelineVideoWriter:
    """Resample rendered images onto simulation-time FPS ticks using zero-order hold."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        """Create an encoder, retaining one pending output frame for the terminal state."""
        self.writer = VideoWriter(path, fps, width, height)
        self.fps = fps
        self._next_index = 0
        self._previous_time: float | None = None
        self._previous_frame: np.ndarray | None = None
        self._pending_frame: np.ndarray | None = None
        self._finished = False

    def _queue(self, frame: np.ndarray) -> None:
        if self._pending_frame is not None:
            self.writer.write(self._pending_frame)
        self._pending_frame = frame
        self._next_index += 1

    def push(self, time: float, frame: np.ndarray) -> None:
        """Supply a rendered frame and its actual simulation time; do not sleep."""
        if self._finished:
            raise RuntimeError("Video timeline is finished")
        if not math.isfinite(time) or time < 0:
            raise ValueError("Frame time must be finite and nonnegative")
        if self._previous_time is not None and time < self._previous_time:
            raise ValueError("Frame times must not go backwards")
        frame = self.writer._check_frame(frame).copy()
        if self._previous_frame is None:
            if abs(time) > 1e-9:
                raise ValueError("The first rendered frame must be at simulation time zero")
        else:
            while self._next_index / self.fps < time - 1e-9:
                self._queue(self._previous_frame)
        self._previous_frame = frame
        self._previous_time = time

    def finish(self, end_time: float, final_frame: np.ndarray) -> None:
        """Encode the terminal state and cap the output length to the simulation duration."""
        if self._finished:
            return
        if not math.isfinite(end_time) or end_time < 0:
            raise ValueError("end_time must be finite and nonnegative")
        if self._previous_time is not None and end_time < self._previous_time:
            raise ValueError("end_time precedes the most recently rendered frame")
        final_frame = self.writer._check_frame(final_frame).copy()
        if self._previous_frame is None:
            self._previous_frame = final_frame
        count = math.floor(end_time * self.fps + 1e-9) + 1
        if self._next_index > count:
            raise ValueError("Rendered timeline extends beyond the requested video duration")
        while self._next_index < count:
            frame = final_frame if self._next_index == count - 1 else self._previous_frame
            self._queue(frame)
        # The last scheduled frame has not yet reached ffmpeg, so it can always be terminal.
        self._pending_frame = final_frame
        self.writer.write(self._pending_frame)
        self._pending_frame = None
        self._finished = True
        self.writer.close()

    def close(self) -> None:
        """Close an unfinished encoder; callers should use finish for a complete run."""
        self.writer.close()

    def __enter__(self) -> TimelineVideoWriter:
        """Return this timeline for use in a with statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the encoder, preserving an existing caller exception."""
        self.writer.__exit__(exc_type, exc, traceback)
