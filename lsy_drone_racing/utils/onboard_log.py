"""Onboard telemetry logging helpers for real Crazyflie deployments.

The radio link carries firmware log blocks of at most ``LOG_BLOCK_BYTES`` data bytes each. This
module packs log variables into blocks, stores received samples, and writes and reads the log file.
It has no dependency on ``cflib2``, so it can be tested without hardware. The radio side lives in
[Crazyflie][lsy_drone_racing.utils.crazyflie.Crazyflie].

The log file is JSON Lines. The first line holds metadata (firmware version, firmware parameters,
logged variables and their blocks, the host clock anchor). Every further line is one sample of one
log block::

    {"t_host_ns": ..., "t_onboard_ms": ..., "block": 0, "values": {"gyro.x": ..., ...}}

``t_host_ns`` is ``time.monotonic_ns()`` on the host when the sample arrived, ``t_onboard_ms`` the
timestamp of the log packet. Samples of different blocks share the onboard clock. The host time
allows aligning them with host-side recordings (commands sent, motion capture).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_LOG_PARAMS",
    "DEFAULT_LOG_VARIABLES",
    "LOG_BLOCK_BYTES",
    "OnboardLog",
    "load_onboard_log",
    "pack_log_blocks",
]

# Data bytes available in one log packet: 30 byte CRTP payload - 1 byte block ID - 3 byte timestamp.
LOG_BLOCK_BYTES = 26

# Signals for system identification and sim-to-real analysis. Names follow the Crazyflie firmware
# log table of contents. Variables that the connected firmware does not expose are skipped and
# recorded as missing in the metadata.
DEFAULT_LOG_VARIABLES: tuple[str, ...] = (
    # IMU
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "acc.x",
    "acc.y",
    "acc.z",
    # Motor outputs and battery
    "motor.m1",
    "motor.m2",
    "motor.m3",
    "motor.m4",
    "pm.vbatMV",
    # Commands of the Mellinger controller
    "ctrlMel.cmd_roll",
    "ctrlMel.cmd_pitch",
    "ctrlMel.cmd_yaw",
    "ctrlMel.cmd_thrust",
    # Onboard attitude estimate and supervisor state
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "supervisor.info",
)

# Firmware parameters recorded once per log. Missing parameters are stored as None.
DEFAULT_LOG_PARAMS: tuple[str, ...] = (
    "stabilizer.controller",
    "stabilizer.estimator",
    "flightmode.stabModeRoll",
    "flightmode.stabModePitch",
    "flightmode.stabModeYaw",
    "supervisor.tmblChckEn",
    "powerDist.idleThrust",
    "ctrlMel.mass",
    "ctrlMel.massThrust",
    "ctrlMel.kR_xy",
    "ctrlMel.kw_xy",
    "ctrlMel.kR_z",
    "ctrlMel.kw_z",
    "pm.lowVoltage",
)

_TYPE_BYTES = {
    "u8": 1,
    "i8": 1,
    "u16": 2,
    "i16": 2,
    "fp16": 2,
    "u32": 4,
    "i32": 4,
    "f32": 4,
    "float": 4,
}


def pack_log_blocks(
    variables: list[str] | tuple[str, ...], types: dict[str, str], max_bytes: int = LOG_BLOCK_BYTES
) -> list[list[str]]:
    """Pack log variables into blocks that fit into one log packet each.

    Variables keep their order. A new block starts when the next variable does not fit anymore.

    Args:
        variables: Log variable names, e.g. ``"gyro.x"``.
        types: Firmware type of each variable, e.g. ``"f32"``. Unknown types count as 4 bytes.
        max_bytes: Data bytes per block.

    Returns:
        The variable names of each block.
    """
    blocks: list[list[str]] = []
    used = max_bytes  # Forces a new block for the first variable
    for name in variables:
        size = _TYPE_BYTES.get(types.get(name, ""), 4)
        if size > max_bytes:
            raise ValueError(f"Log variable {name} ({size} bytes) exceeds the block size")
        if used + size > max_bytes:
            blocks.append([])
            used = 0
        blocks[-1].append(name)
        used += size
    return blocks


@dataclass
class OnboardLog:
    """Samples and metadata of one onboard logging session."""

    path: Path
    period_ms: int
    blocks: list[list[str]] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    firmware: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    samples: list[tuple[int, int, int, dict[str, float]]] = field(default_factory=list)
    wall_time_ns: int = field(default_factory=time.time_ns)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)

    def append(self, t_onboard_ms: int, block: int, values: dict[str, float]) -> None:
        """Store one received sample with its host arrival time."""
        self.samples.append((time.monotonic_ns(), int(t_onboard_ms), block, dict(values)))

    def metadata(self) -> dict[str, Any]:
        """Return the metadata written as the first line of the log file."""
        return {
            "kind": "onboard_log_metadata",
            "format_version": 1,
            "period_ms": self.period_ms,
            "blocks": self.blocks,
            "types": self.types,
            "missing_variables": self.missing,
            "firmware": self.firmware,
            "errors": self.errors,
            "n_samples": len(self.samples),
            "clock_anchor": {"wall_time_ns": self.wall_time_ns, "monotonic_ns": self.monotonic_ns},
        }

    def save(self) -> Path:
        """Write metadata and samples to ``path`` as JSON Lines and return the path."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            f.write(json.dumps(self.metadata()) + "\n")
            for t_host_ns, t_onboard_ms, block, values in self.samples:
                sample = {
                    "t_host_ns": t_host_ns,
                    "t_onboard_ms": t_onboard_ms,
                    "block": block,
                    "values": values,
                }
                f.write(json.dumps(sample) + "\n")
        return self.path


def load_onboard_log(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Load a log file written by [OnboardLog.save][lsy_drone_racing.utils.onboard_log.OnboardLog].

    Returns:
        The metadata, and per variable the arrays ``t_onboard_ms``, ``t_host_ns`` and ``value``.
    """
    with Path(path).open() as f:
        metadata = json.loads(f.readline())
        columns: dict[str, dict[str, list[float]]] = {}
        for line in f:
            sample = json.loads(line)
            for name, value in sample["values"].items():
                column = columns.setdefault(
                    name, {"t_onboard_ms": [], "t_host_ns": [], "value": []}
                )
                column["t_onboard_ms"].append(sample["t_onboard_ms"])
                column["t_host_ns"].append(sample["t_host_ns"])
                column["value"].append(value)
    data = {
        name: {
            "t_onboard_ms": np.asarray(column["t_onboard_ms"], dtype=np.int64),
            "t_host_ns": np.asarray(column["t_host_ns"], dtype=np.int64),
            "value": np.asarray(column["value"], dtype=np.float64),
        }
        for name, column in columns.items()
    }
    return metadata, data
