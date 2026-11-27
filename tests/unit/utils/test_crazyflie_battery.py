"""Battery-at-rest reading of the Crazyflie wrapper, with a fake cflib2 link."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("cflib2")
pytest.importorskip("drone_estimators")

from lsy_drone_racing.utils.crazyflie import Crazyflie  # noqa: E402


class _FakeStream:
    def __init__(self, values: list[float]):
        self.values = list(values)
        self.stopped = False

    async def next(self) -> SimpleNamespace:
        return SimpleNamespace(timestamp=0, data={"pm.vbatMV": self.values.pop(0)})

    async def stop(self) -> None:
        self.stopped = True


class _FakeBlock:
    def __init__(self, stream: _FakeStream):
        self.stream = stream
        self.variables: list[str] = []

    async def add_variable(self, name: str) -> None:
        self.variables.append(name)

    async def start(self, period_ms: int) -> _FakeStream:
        return self.stream


def _wrapper(values: list[float]) -> tuple[Crazyflie, _FakeBlock, list]:
    """A Crazyflie wrapper without radio or ROS whose link returns the given readings."""
    drone = Crazyflie.__new__(Crazyflie)
    drone._loop = asyncio.new_event_loop()
    drone._loop_thread = None
    block = _FakeBlock(_FakeStream(values))

    async def create_block() -> _FakeBlock:
        return block

    drone._cf = SimpleNamespace(log=lambda: SimpleNamespace(create_block=create_block))
    events: list = []
    drone.record_event = lambda kind, **payload: events.append((kind, payload))
    drone.battery_rest_v = None
    return drone, block, events


@pytest.mark.unit
def test_read_battery_voltage_uses_and_stops_a_temporary_block():
    drone, block, _ = _wrapper([3900.0 + i for i in range(10)])
    assert drone.read_battery_voltage() == pytest.approx(3.9045)
    assert block.variables == ["pm.vbatMV"]
    assert block.stream.stopped


@pytest.mark.unit
def test_reset_records_rest_voltage_before_arming_and_survives_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    async def step(name: str) -> None:
        calls.append(name)

    for name in ("_apply_settings", "_reset_estimator", "_arm", "_unlock_thrust"):
        monkeypatch.setattr(Crazyflie, name, lambda self, n=name: step(n))
    drone, _, events = _wrapper([3950.0] * 10)
    original = drone.read_battery_voltage
    drone.read_battery_voltage = lambda: (calls.append("battery"), original())[1]
    drone.reset(arm=True)
    assert calls == ["_apply_settings", "_reset_estimator", "battery", "_arm", "_unlock_thrust"]
    assert drone.battery_rest_v == pytest.approx(3.95)
    assert events == [("battery_rest", {"volt": pytest.approx(3.95), "armed": False})]

    drone, _, events = _wrapper([])  # The stream yields nothing: the read fails
    drone.reset(arm=False)
    assert drone.battery_rest_v is None
    assert events == [("battery_rest", {"volt": None, "armed": False})]
