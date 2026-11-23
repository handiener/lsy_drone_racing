import json
from pathlib import Path

import numpy as np
import pytest

from lsy_drone_racing.utils.onboard_log import (
    LOG_BLOCK_BYTES,
    OnboardLog,
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


@pytest.mark.unit
def test_onboard_log_roundtrip(tmp_path: Path):
    log = OnboardLog(path=tmp_path / "log.jsonl", period_ms=10)
    log.blocks = [["gyro.x", "gyro.y"], ["pm.vbatMV"]]
    log.firmware = {"version": "test", "params": {"stabilizer.controller": 2}}
    log.append(100, 0, {"gyro.x": 1.0, "gyro.y": 2.0})
    log.append(105, 1, {"pm.vbatMV": 4100})
    log.append(110, 0, {"gyro.x": 3.0, "gyro.y": 4.0})
    path = log.save()

    first = json.loads(path.read_text().splitlines()[0])
    assert first["kind"] == "onboard_log_metadata" and first["n_samples"] == 3

    metadata, data = load_onboard_log(path)
    assert metadata["firmware"]["params"]["stabilizer.controller"] == 2
    np.testing.assert_array_equal(data["gyro.x"]["t_onboard_ms"], [100, 110])
    np.testing.assert_array_equal(data["gyro.x"]["value"], [1.0, 3.0])
    np.testing.assert_array_equal(data["pm.vbatMV"]["value"], [4100])
    assert np.all(np.diff(data["gyro.x"]["t_host_ns"]) >= 0)
