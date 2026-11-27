import json
from pathlib import Path

import numpy as np
import pytest

from lsy_drone_racing.utils.onboard_log import (
    LOG_BLOCK_BYTES,
    OnboardLog,
    battery_voltage,
    load_onboard_log,
    pack_log_blocks,
)


@pytest.mark.unit
def test_pack_log_blocks_respects_packet_size():
    names = [f"v{i}" for i in range(15)]
    blocks = pack_log_blocks(names, {name: "f32" for name in names})
    assert [len(block) for block in blocks] == [6, 6, 3]  # 6 floats = 24 of 26 bytes
    assert [name for block in blocks for name in block] == names  # Order is kept


@pytest.mark.unit
def test_pack_log_blocks_mixed_sizes():
    types = {"a": "f32", "b": "u16", "c": "u8", "d": "f32"}
    blocks = pack_log_blocks(["a", "b", "c", "d"] * 3, types)
    sizes = {"f32": 4, "u16": 2, "u8": 1}
    for block in blocks:
        assert sum(sizes[types[name]] for name in block) <= LOG_BLOCK_BYTES


@pytest.mark.unit
def test_pack_log_blocks_rejects_oversized_variable():
    with pytest.raises(ValueError):
        pack_log_blocks(["big"], {"big": "f32"}, max_bytes=2)


def _make_log(path: Path, flush_every: int = 64) -> OnboardLog:
    log = OnboardLog(path=path, period_ms=10, flush_every=flush_every)
    log.blocks = [["gyro.x", "gyro.y"], ["pm.vbatMV"]]
    log.firmware = {"version": "test", "params": {"stabilizer.controller": 2}}
    log.open()
    return log


@pytest.mark.unit
def test_onboard_log_roundtrip(tmp_path: Path):
    log = _make_log(tmp_path / "log.jsonl")
    log.append(100, 0, {"gyro.x": 1.0, "gyro.y": 2.0})
    log.append(105, 1, {"pm.vbatMV": 4100})
    log.append(110, 0, {"gyro.x": 3.0, "gyro.y": 4.0})
    log.errors.append("link hiccup")
    path = log.close()

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["kind"] == "onboard_log_metadata"
    assert lines[-1] == {"kind": "onboard_log_summary", "n_samples": 3, "errors": ["link hiccup"]}

    metadata, data = load_onboard_log(path)
    assert metadata["complete"] and metadata["n_samples"] == 3
    assert metadata["errors"] == ["link hiccup"]
    assert metadata["firmware"]["params"]["stabilizer.controller"] == 2
    np.testing.assert_array_equal(data["gyro.x"]["t_onboard_ms"], [100, 110])
    np.testing.assert_array_equal(data["gyro.x"]["value"], [1.0, 3.0])
    np.testing.assert_array_equal(data["pm.vbatMV"]["value"], [4100])
    assert np.all(np.diff(data["gyro.x"]["t_host_ns"]) >= 0)


@pytest.mark.unit
def test_onboard_log_survives_crash(tmp_path: Path):
    """Samples are on disk before close, and a truncated file without summary still loads."""
    log = _make_log(tmp_path / "log.jsonl", flush_every=2)
    for t in range(5):
        log.append(10 * t, 0, {"gyro.x": float(t), "gyro.y": 0.0})
    # Simulate the process dying: no close(), only what was flushed is on disk, plus a torn line
    flushed = log.path.read_text()
    assert len(flushed.splitlines()) == 1 + 4  # Metadata + 4 flushed samples
    crashed = tmp_path / "crashed.jsonl"
    crashed.write_text(flushed + '{"t_host_ns": 1, "t_onb')

    metadata, data = load_onboard_log(crashed)
    assert not metadata["complete"] and metadata["n_samples"] == 4
    np.testing.assert_array_equal(data["gyro.x"]["value"], [0.0, 1.0, 2.0, 3.0])
    log.close()


@pytest.mark.unit
def test_battery_voltage_is_median_in_volts():
    assert battery_voltage([3950, 3960, 3940, 0, 3955]) == pytest.approx(3.95)


@pytest.mark.unit
@pytest.mark.parametrize("samples", [[], [0, 0, 0], [12000]])
def test_battery_voltage_rejects_implausible_readings(samples: list[float]):
    with pytest.raises(ValueError):
        battery_voltage(samples)
