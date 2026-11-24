"""Exercise deployment shutdown paths with mocked ROS and hardware."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lsy_drone_racing.utils.onboard_log import OnboardLog, load_onboard_log
from tests.unit.utils.test_deploy_recorder import read_events


@pytest.mark.parametrize("failure", [None, KeyboardInterrupt, RuntimeError])
@pytest.mark.parametrize("onboard_log", [False, True])
def test_deployment_recording_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException] | None,
    onboard_log: bool,
):
    pytest.importorskip("rclpy")
    from scripts import deploy

    obs = {
        "pos": np.zeros((1, 3), np.float32),
        "quat": np.array([[0, 0, 0, 1]], np.float32),
        "n_gates_passed": np.array([0]),
        "gate_sequence": np.array([[0]]),
    }
    closed = []
    connector = SimpleNamespace(tf_pos={"cf2": np.zeros(3)}, tf_quat={"cf2": np.zeros(4)})
    drone = SimpleNamespace(_ros_connector=connector)
    onboard = None

    def start_onboard_log(path: Path):
        nonlocal onboard
        onboard = OnboardLog(path, period_ms=10)
        onboard.open()

    drone.start_onboard_log = start_onboard_log

    class FakeEnv:
        unwrapped = None
        drone_name = "cf2"
        drone_parameters = {"thrust_max": 0.1}
        gates = SimpleNamespace(pos=np.zeros((1, 3)), quat=np.zeros((1, 4)))
        obstacles = SimpleNamespace(pos=np.zeros((1, 3)))

        def __init__(self):
            self.unwrapped = self
            self.drone = drone

        def reset(self, **kwargs: Any) -> tuple:
            return {key: value[0] for key, value in obs.items()}, {}

        def obs(self) -> dict:
            return obs

        def info(self) -> dict:
            return {}

        def step(self, action: np.ndarray) -> tuple:
            # Emit a hardware event through the callback installed by deploy.main.
            drone.record_event("attitude_command_sent", collective_pwm=12345)
            if onboard is not None:
                onboard.append(10, 0, {"gyro.x": 1.0})
            if failure:
                raise failure("interrupted flight")
            obs["n_gates_passed"][:] = 1
            return {key: value[0] for key, value in obs.items()}, 0, True, False, {}

        def close(self):
            closed.append(type(sys.exception()))
            drone.record_event("emergency_stop_sent")
            if onboard is not None:
                onboard.close()

    class FakeController:
        def __init__(self, *args: Any):
            pass

        def compute_control(self, observation: dict, info: dict) -> np.ndarray:
            # Recorder must preserve input even if a controller mutates it.
            observation["pos"][:] = 42
            return np.array([0, 0, 0, 0.3], dtype=np.float32)

        def step_callback(self, *args: Any) -> bool:
            return False

    monkeypatch.setattr(deploy.rclpy, "init", lambda: None)
    monkeypatch.setattr(deploy.rclpy, "ok", lambda: True)
    monkeypatch.setattr(deploy.rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(deploy.gymnasium, "make", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(deploy, "load_controller", lambda path: FakeController)
    output = tmp_path / "flight"
    if failure:
        with pytest.raises(failure, match="interrupted flight"):
            deploy.main(output_dir=str(output), onboard_log=onboard_log)
    else:
        deploy.main(output_dir=str(output), onboard_log=onboard_log)
    events = read_events(output)
    assert closed == [failure or type(None)]
    assert next(event for event in events if event["kind"] == "observation")["obs"]["pos"] == [
        0,
        0,
        0,
    ]
    assert "attitude_command_sent" in [event["kind"] for event in events]
    assert "emergency_stop_sent" in [event["kind"] for event in events]
    assert events[-1]["kind"] == "cleanup_end"
    assert ("exception" in [event["kind"] for event in events]) == bool(failure)
    assert json.loads((output / "summary.json").read_text())["complete"]
    assert (output / "onboard.jsonl").exists() == onboard_log
    if onboard_log:
        metadata, data = load_onboard_log(output / "onboard.jsonl")
        assert metadata["complete"]
        assert metadata["n_samples"] == 1
        action_time = next(e["monotonic_ns"] for e in events if e["kind"] == "action")
        cleanup_time = events[-1]["monotonic_ns"]
        assert action_time <= data["gyro.x"]["t_host_ns"][0] <= cleanup_time
