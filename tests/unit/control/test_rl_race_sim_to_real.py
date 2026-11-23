"""Per-world delay, episode isolation and physical parameter invariants for fine-tuning."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.rl_race.env_wrapper.env import EnvConfig, RaceEnv
from lsy_drone_racing.control.rl_race.env_wrapper.sim_to_real import SimToReal, SimToRealConfig


def processor(preset: str = "reference", n: int = 2) -> SimToReal:
    """Exercise observation history without constructing a simulator."""
    obj = SimToReal.__new__(SimToReal)
    obj.config = SimToRealConfig(preset, nominal_prob=0.0)
    obj.n = n
    return obj


def test_delay_and_masked_episode_reset():
    obj = processor()
    key = jax.random.PRNGKey(1)
    state = obj.update(
        None, {"ang_vel": jnp.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])}, jnp.ones(2, bool), key
    )
    for step in range(1, 4):
        rate = jnp.array([[step] * 3, [step + 10] * 3], dtype=jnp.float32)
        state = obj.update(state, {"ang_vel": rate}, jnp.zeros(2, bool), key)
        np.testing.assert_array_equal(
            state["omega"], np.array([[max(0, step - 2)] * 3, [10 + max(0, step - 2)] * 3])
        )
    state = obj.update(
        state,
        {"ang_vel": jnp.array([[100.0, 100.0, 100.0], [14.0, 14.0, 14.0]])},
        jnp.array([True, False]),
        key,
    )
    np.testing.assert_array_equal(state["history"][0], np.full((4, 3), 100.0))
    np.testing.assert_array_equal(state["omega"], [[100.0] * 3, [12.0] * 3])


def test_parameters_hold_until_that_world_resets():
    obj = processor("estimator_mix")
    raw = {"ang_vel": jnp.zeros((2, 3))}
    a = obj.update(None, raw, jnp.ones(2, bool), jax.random.PRNGKey(1))
    b = obj.update(a, raw, jnp.array([True, False]), jax.random.PRNGKey(2))
    for key in ["rate", "cap", "delay", "tau"]:
        np.testing.assert_array_equal(a[key][1], b[key][1])
    assert float(a["cap"][0]) != float(b["cap"][0])


def test_physical_limits_survive_autoresets_without_compounding():
    env = RaceEnv(
        2,
        EnvConfig(max_episode_steps=2, sim_to_real=SimToRealConfig("reference", 0.0)),
        device="cpu",
    )
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.split(jax.random.PRNGKey(5), 2))
    base = np.asarray(env._s2r.rotor_base)
    for _ in range(6):
        state = step(state, jnp.zeros((2, 4)))
        np.testing.assert_allclose(
            state.pipeline_state.sim_data.params.rotor_dyn_coef,
            np.broadcast_to(base * 0.8, (2, 1, *base.shape)),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            state.pipeline_state.sim_data.controls.force_torque.params["thrust_max"], 0.175
        )
        assert np.isfinite(state.obs).all()
    assert state.obs.shape == (2, 49)


@pytest.mark.parametrize("preset,prob", [("invalid", 0.5), ("reference", -1), ("reference", 1.1)])
def test_invalid_configuration(preset: str, prob: float):
    with pytest.raises(ValueError):
        SimToRealConfig(preset, prob)


def test_legacy_positional_env_config():
    """The new optional field must not shift the old positional constructor."""
    cfg = EnvConfig(123)
    assert cfg.max_episode_steps == 123
    assert cfg.sim_to_real.preset == "none"


def test_estimator_mixture_chooses_family_per_episode():
    obj = processor("estimator_mix", 512)
    raw = {"ang_vel": jnp.zeros((512, 3))}
    state = obj.update(None, raw, jnp.ones(512, bool), jax.random.PRNGKey(3))
    smooth = np.asarray(state["tau"]) > 0
    assert smooth.any() and (~smooth).any()
    assert np.all(np.asarray(state["delay"])[smooth] == 0)
    assert np.all(np.asarray(state["delay"])[~smooth] >= 1)
    assert np.all(np.asarray(state["tau"])[smooth] >= 0.02)
    assert np.all(np.asarray(state["tau"])[smooth] < 0.06)
    later = obj.update(state, raw, jnp.zeros(512, bool), jax.random.PRNGKey(8))
    for key in ["rate", "cap", "delay", "tau"]:
        np.testing.assert_array_equal(state[key], later[key])


def test_nominal_episodes_keep_fresh_rates_and_original_actuation():
    obj = processor("estimator_mix", 32)
    obj.config = SimToRealConfig("estimator_mix", nominal_prob=1.0)
    raw = {"ang_vel": jnp.ones((32, 3))}
    state = obj.update(None, raw, jnp.ones(32, bool), jax.random.PRNGKey(2))
    raw = {"ang_vel": jnp.full((32, 3), 7.0)}
    state = obj.update(state, raw, jnp.zeros(32, bool), jax.random.PRNGKey(3))
    np.testing.assert_array_equal(state["rate"], 1.0)
    np.testing.assert_array_equal(state["cap"], np.float32(0.8))
    np.testing.assert_array_equal(state["delay"], 0)
    np.testing.assert_array_equal(state["tau"], 0)
    np.testing.assert_array_equal(state["omega"], raw["ang_vel"])
