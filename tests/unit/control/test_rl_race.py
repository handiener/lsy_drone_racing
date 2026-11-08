"""Pure functions of the Brax racing pipeline: path, observations, actions, rewards."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.rl_race import path, rewards
from lsy_drone_racing.control.rl_race.actions import ActionConfig, to_command
from lsy_drone_racing.control.rl_race.observations import ObsConfig, encode

# Two gates on the x axis, both passed in +x, plus one reversed entry.
GATES_POS = jnp.array([[[1.0, 0.0, 1.0], [3.0, 0.0, 1.0]]])
GATES_QUAT = jnp.array([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
SEQUENCE = jnp.array([0, 1])
DIRECTION = jnp.array([1.0, 1.0])
START = jnp.array([-1.0, 0.0, 1.0])


def test_hermite_hits_control_points_and_tangents():
    p0, t0, p1, t1 = (
        jnp.array([[0.0, 0, 0]]),
        jnp.array([[1.0, 2, 0]]),
        jnp.array([[2.0, 0, 1]]),
        jnp.array([[0.0, 1, 1]]),
    )
    for s, point, tangent in ((0.0, p0, t0), (1.0, p1, t1)):
        got_point, got_tangent = path.hermite(p0, t0, p1, t1, jnp.array([s]))
        np.testing.assert_allclose(got_point, point, atol=1e-6)
        np.testing.assert_allclose(got_tangent, tangent, atol=1e-6)


def test_leg_controls_use_start_then_previous_gate():
    for leg, p0 in ((0, START), (1, GATES_POS[0, 0])):
        c = path.leg_control_points(
            GATES_POS, GATES_QUAT, SEQUENCE, DIRECTION, jnp.array([leg]), START
        )
        np.testing.assert_allclose(c[0][0], p0, atol=1e-6)
        np.testing.assert_allclose(c[2][0], GATES_POS[0, leg], atol=1e-6)
        # Straight course along +x: both tangents point along +x with the chord length.
        np.testing.assert_allclose(c[1][0], [2.0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(c[3][0], [2.0, 0, 0], atol=1e-6)


def test_straight_leg_arc_length_projection_and_progress():
    controls = path.leg_control_points(
        GATES_POS, GATES_QUAT, SEQUENCE, DIRECTION, jnp.array([1]), START
    )
    points, cumulative = path.polyline(*controls)
    assert float(cumulative[0, -1]) == pytest.approx(2.0, rel=1e-4)
    arc, dist = path.project(points, cumulative, jnp.array([[2.0, 0.3, 1.0]]))
    assert float(arc[0]) == pytest.approx(1.0, abs=1e-3)
    assert float(dist[0]) == pytest.approx(0.3, abs=1e-3)
    gain = path.arc_progress(
        GATES_POS,
        GATES_QUAT,
        SEQUENCE,
        DIRECTION,
        jnp.array([1]),
        START,
        jnp.array([[1.5, 0, 1.0]]),
        jnp.array([[1.7, 0.1, 1.0]]),
    )
    assert float(gain[0]) == pytest.approx(0.2, abs=1e-3)


def test_reversed_gate_flips_the_passing_normal():
    quats = jnp.repeat(GATES_QUAT, 2, axis=0)  # two worlds
    normal = path.passing_normals(quats, jnp.array([0, 1]), jnp.array([1.0, -1.0]))
    np.testing.assert_allclose(normal, [[1, 0, 0], [-1, 0, 0]], atol=1e-6)


def _raw(pos: list[float], yaw: float = 0.0, passed: int = 0) -> dict[str, jnp.ndarray]:
    quat = R.from_euler("z", yaw).as_quat()
    return {
        "pos": jnp.array([pos]),
        "quat": jnp.array([quat]),
        "vel": jnp.array([[1.0, 0, 0]]),
        "ang_vel": jnp.zeros((1, 3)),
        "n_gates_passed": jnp.array([passed]),
        "gate_sequence": SEQUENCE[None],
        "gate_sequence_direction": DIRECTION[None],
        "gates_pos": GATES_POS,
        "gates_quat": GATES_QUAT,
        "obstacles_pos": jnp.array([[[0.0, 3.0, 1.5], [0.5, 0.2, 1.5]]]),
    }


def test_observation_layout_body_frame_and_padding():
    cfg = ObsConfig(n_gates=2, n_obstacles=2)
    prev = jnp.zeros((1, 3))
    level = np.asarray(encode(_raw([0.0, 0, 1.0]), prev, cfg))[0]
    assert level.shape == (cfg.size,)
    np.testing.assert_allclose(level[22:25], [1.0, 0, 0], atol=1e-6)  # gate 0 center, relative
    np.testing.assert_allclose(level[25:28], [1.0, 0, 0], atol=1e-6)  # gate 0 normal
    assert level[28] == 1.0  # valid
    np.testing.assert_allclose(level[36:39], [0.5, 0.2, 0.5], atol=1e-6)  # nearest obstacle first
    # Yawed by +90 degrees, world +x appears as body -y.
    yawed = np.asarray(encode(_raw([0.0, 0, 1.0], yaw=np.pi / 2), prev, cfg))[0]
    np.testing.assert_allclose(yawed[22:25], [0, -1.0, 0], atol=1e-6)
    # After the last gate of the sequence, the second slot is zero padding.
    last = np.asarray(encode(_raw([2.0, 0, 1.0], passed=1), prev, cfg))[0]
    assert last[28] == 1.0 and np.all(last[29:36] == 0)


def test_action_mapping_limits_and_yaw():
    cfg = ActionConfig(theta_max_deg=70.0, thrust_min=0.1, thrust_max=0.8)
    cmd = np.asarray(to_command(jnp.array([[1.0, -1.0, -1.0], [0.0, 0.5, 1.0]]), cfg))
    np.testing.assert_allclose(cmd[0], [np.deg2rad(70), -np.deg2rad(70), 0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(cmd[1], [0.0, np.deg2rad(35), 0.0, 0.8], atol=1e-6)
    with pytest.raises(ValueError):
        ActionConfig(theta_max_deg=95)


def test_reward_events_and_components():
    prev = jnp.array([0, 1, 4, 2])
    passed = jnp.array([1, 1, 5, 3])
    terminated = jnp.array([False, True, False, True])
    ev = rewards.events(prev, passed, terminated, n_seq=5)
    np.testing.assert_array_equal(ev["gates"], [1, 0, 1, 0])  # crash step pays no gate
    np.testing.assert_array_equal(ev["finished"], [False, False, True, False])
    np.testing.assert_array_equal(ev["crashed"], [False, True, False, True])
    cfg = rewards.preset("dense_path")
    total, comp = rewards.compute(cfg, ev, jnp.array([[0, 0, 2.0]] * 4), jnp.array([0.1] * 4))
    np.testing.assert_allclose(comp["progress"], [1.5, 0.0, 1.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(comp["ang_vel"], [-0.01] * 4, atol=1e-7)
    np.testing.assert_allclose(total[2], 10 + 50 - 0.05 - 0.01 + 1.5, atol=1e-5)
    np.testing.assert_allclose(total[1], -20 - 0.05 - 0.01, atol=1e-5)
    assert rewards.preset("race").progress == 0.0


def test_path_potential_penalizes_passing_beside_the_gate():
    args = (GATES_POS, GATES_QUAT, SEQUENCE, DIRECTION, jnp.array([1]), START)
    before = jnp.array([[2.5, 0.0, 1.0]])  # on the path, 0.5 m before gate 1 (x = 3)
    through = jnp.array([[3.2, 0.0, 1.0]])  # beyond the plane, through the opening
    beside = jnp.array([[3.2, 1.4, 1.0]])  # beyond the plane, 1.4 m beside the opening
    plain = [float(path.arc_progress(*args, before, p)[0]) for p in (through, beside)]
    assert plain[0] == pytest.approx(plain[1], abs=1e-3)  # arc alone cannot tell them apart
    shaped = [
        float(path.arc_progress(*args, before, p, lateral_weight=1.0)[0]) for p in (through, beside)
    ]
    assert shaped[0] - shaped[1] == pytest.approx(np.hypot(0.2, 1.4) - 0.2, abs=1e-2)
    # A constant offset parallel to the path costs nothing.
    side_a, side_b = jnp.array([[1.8, 0.3, 1.0]]), jnp.array([[2.2, 0.3, 1.0]])
    gain = float(path.arc_progress(*args, side_a, side_b, lateral_weight=1.0)[0])
    assert gain == pytest.approx(0.4, abs=1e-2)
