"""Recording durability and failure isolation without drone hardware."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from lsy_drone_racing.utils.deploy_recorder import DeploymentRecorder


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_snapshots_and_close_drain(tmp_path: Path):
    path = tmp_path / "run"
    recorder = DeploymentRecorder(path, {"config": {"freq": 50}})
    observation = {"pos": np.array([1.0, 2.0, 3.0])}
    for step in range(100):
        recorder.record("observation", step=step, obs=observation, terminated=np.bool_(False))
    observation["pos"][:] = -99
    recorder.close()
    recorder.close()
    events = read_events(path)
    assert len(events) == 100
    assert events[0]["obs"]["pos"] == [1, 2, 3]
    assert [event["step"] for event in events] == list(range(100))
    assert all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:]))
    assert json.loads((path / "summary.json").read_text())["complete"]
    assert "clock_anchor" in json.loads((path / "metadata.json").read_text())


def test_existing_directory_refused(tmp_path: Path):
    with pytest.raises(FileExistsError):
        DeploymentRecorder(tmp_path, {})


def test_serialization_failure_does_not_escape(tmp_path: Path):
    path = tmp_path / "run"
    recorder = DeploymentRecorder(path, {})
    recorder.record("unsupported", value=object())
    recorder.close()
    summary = json.loads((path / "summary.json").read_text())
    assert not summary["complete"]
    assert "Writer failed" in summary["recording_error"]


def test_full_queue_does_not_block_flight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "run"
    gate = threading.Event()
    original = DeploymentRecorder._write

    def delayed_write(self: DeploymentRecorder):
        gate.wait(timeout=5)
        original(self)

    monkeypatch.setattr(DeploymentRecorder, "_write", delayed_write)
    recorder = DeploymentRecorder(path, {}, queue_size=1)
    try:
        recorder.record("first")
        recorder.record("overflow")
        recorder.record("after_failure")
    finally:
        gate.set()
        recorder.close()
    assert [event["kind"] for event in read_events(path)] == ["first"]
    summary = json.loads((path / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["records_dropped"] == 2
