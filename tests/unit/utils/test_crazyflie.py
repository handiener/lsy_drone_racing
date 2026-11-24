"""Check the pinned cflib2 API and shutdown without ROS or a radio connection."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cflib2 import Crazyflie as CflibCrazyflie
from cflib2.supervisor import Supervisor

from lsy_drone_racing.utils.crazyflie import Crazyflie


def make_drone() -> Crazyflie:
    """Construct only the wrapper state, avoiding hardware and ROS initialization."""
    drone = object.__new__(Crazyflie)
    supervisor = Mock(spec_set=Supervisor)
    supervisor.send_arming_request = AsyncMock()
    supervisor.send_emergency_stop = AsyncMock()
    drone._cf = Mock(spec_set=CflibCrazyflie)
    drone._cf.supervisor.return_value = supervisor
    drone._cf.disconnect = AsyncMock()
    drone._commander_level = None
    drone._loop = asyncio.new_event_loop()
    drone._loop_thread = None
    drone._estimator_future = None
    drone._estimator_stop_event = None
    drone._onboard_log = None
    drone._ros_connector = SimpleNamespace(close=Mock())
    drone.record_event = Mock()
    drone.update_freq = 30
    return drone


def test_arm_uses_installed_supervisor_api(monkeypatch: pytest.MonkeyPatch):
    drone = make_drone()
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    try:
        drone._run(drone._arm)
        drone.cf.supervisor().send_arming_request.assert_awaited_once_with(do_arm=True)
    finally:
        drone.close(emergency_stop=False)


@pytest.mark.parametrize("stop_fails", [False, True])
def test_close_stops_updater_and_disconnects_even_if_emergency_stop_fails(stop_fails: bool):
    drone = make_drone()
    cf = drone.cf
    if stop_fails:
        cf.supervisor().send_emergency_stop.side_effect = RuntimeError("link lost")
    stop_event = threading.Event()
    drone._estimator_stop_event = stop_event
    drone._loop_thread = threading.Thread(target=drone._loop.run_forever)
    drone._loop_thread.start()

    async def updater():
        while not stop_event.is_set():
            await asyncio.sleep(0.001)

    future = asyncio.run_coroutine_threadsafe(updater(), drone._loop)
    drone._estimator_future = future
    drone._onboard_log = object()
    drone.stop_onboard_log = Mock()
    if stop_fails:
        with pytest.raises(RuntimeError, match="link lost"):
            drone.close()
    else:
        drone.close()
        drone.record_event.assert_called_once_with("emergency_stop_sent")
    cf.supervisor().send_emergency_stop.assert_awaited_once_with()
    drone.stop_onboard_log.assert_called_once_with()
    cf.disconnect.assert_awaited_once_with()
    assert stop_event.is_set()
    assert future.done()
    assert not drone._loop.is_running()
    assert drone._loop.is_closed()
    assert drone._loop_thread is None
    assert drone.cf is None
    drone._ros_connector.close.assert_called_once_with()
    drone.close()  # Repeated cleanup is harmless.
