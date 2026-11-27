"""JAX-native per-world actuator/observation hypotheses for optional fine-tuning.

Main presets change rotor response, achievable thrust and the policy angular-rate input.
Only the fixed reference and mixed estimator profiles are retained. Reward, geometry and
controller gains are unchanged. These are stress hypotheses, not hardware calibration.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

PRESETS = ("none", "reference", "estimator_mix")
HISTORY = 4  # current + 20/40/60 ms, at the checked 50 Hz control rate


@dataclass(frozen=True)
class SimToRealConfig:
    """Select a stress profile and the fraction of unchanged episodes."""

    preset: str = "none"
    nominal_prob: float = 0.35

    def __post_init__(self) -> None:
        """Reject unknown profiles and invalid mixture probabilities."""
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown sim-to-real preset: {self.preset}")
        if not 0 <= self.nominal_prob <= 1:
            raise ValueError("nominal_prob must lie in [0, 1]")


class SimToReal:
    """Install physical actuator limits and keep masked, per-world observation history."""

    def __init__(self, core: Any, config: SimToRealConfig) -> None:
        """Prepare stable parameter shapes and rebuild only this environment’s pipeline."""
        from crazyflow.control.transform import motor_force2rotor_vel
        from crazyflow.sim.pipeline import insert_fn_after

        self.config = config
        self.n = core.data.sim_data.core.n_worlds
        if core.settings.freq != 50:
            raise ValueError("Sim-to-real history currently requires a 50 Hz policy")
        self.rotor_base = core.data.sim_data.params.rotor_dyn_coef

        # Give the parameters stable per-world shapes before compiling resets/steps.
        def expand(data: Any) -> Any:
            ft = data.controls.force_torque
            params = dict(ft.params)
            params["thrust_max"] = jnp.full((self.n, 1, 1), 0.2, jnp.float32)
            return data.replace(
                params=data.params.replace(
                    rotor_dyn_coef=jnp.broadcast_to(
                        self.rotor_base, (self.n, 1, *self.rotor_base.shape)
                    )
                ),
                controls=data.controls.replace(force_torque=ft.replace(params=params)),
            )

        core.data = core.data.replace(sim_data=expand(core.data.sim_data))
        core.sim.default_data = expand(core.sim.default_data)

        def achievable_rpm(data: Any) -> Any:
            maximum = motor_force2rotor_vel(
                data.controls.force_torque.params["thrust_max"], data.params.rpm2thrust
            )
            return data.replace(
                controls=data.controls.replace(
                    rotor_vel=jnp.minimum(data.controls.rotor_vel, maximum)
                )
            )

        insert_fn_after(core.sim.step_pipeline, "force_torque_controller", achievable_rpm)
        core._reset = core.build_reset_fn()

    def update(
        self, old: dict[str, Array] | None, raw: dict[str, Array], mask: Array, key: Array
    ) -> dict[str, Array]:
        """Advance history and resample only reset worlds; reset history never crosses episodes."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        shape = (self.n,)
        nominal = jax.random.uniform(k1, shape) < self.config.nominal_prob
        if self.config.preset == "reference":
            rate, cap, delay, tau = (
                jnp.full(shape, 0.8),
                jnp.full(shape, 0.70),
                jnp.full(shape, 2, jnp.int32),
                jnp.zeros(shape),
            )
        else:
            # Sample bounded uncertainty around the reference stress profile.
            # Fixed within an episode: coherent plant uncertainty, not white noise.
            rate = jax.random.uniform(k2, shape, minval=0.75, maxval=0.90)
            cap = jax.random.uniform(k3, shape, minval=0.68, maxval=0.74)
            delay = jax.random.randint(k4, shape, 1, HISTORY)
            tau = jnp.zeros(shape)
            # Choose pure age or smoothing; do not stack both worst cases.
            smooth = jax.random.bernoulli(jax.random.fold_in(k5, 1), 0.5, shape)
            delay = jnp.where(smooth, 0, delay)
            tau = jnp.where(smooth, jax.random.uniform(k5, shape, minval=0.02, maxval=0.06), 0.0)
        sampled = dict(
            rate=jnp.where(nominal, 1.0, rate),
            cap=jnp.where(nominal, 0.8, cap),
            delay=jnp.where(nominal, 0, delay),
            tau=jnp.where(nominal, 0.0, tau),
        )
        if old is None:
            values = sampled
            history = jnp.repeat(raw["ang_vel"][:, None], HISTORY, axis=1)
            omega = raw["ang_vel"]
        else:
            values = {k: jnp.where(mask, v, old[k]) for k, v in sampled.items()}
            history = jnp.concatenate((raw["ang_vel"][:, None], old["history"][:, :-1]), axis=1)
            history = jnp.where(mask[:, None, None], raw["ang_vel"][:, None], history)
            delayed = jnp.take_along_axis(history, values["delay"][:, None, None], axis=1)[:, 0]
            alpha = -jnp.expm1(-0.02 / jnp.maximum(values["tau"], 1e-8))
            omega = jnp.where(
                values["tau"][:, None] > 0,
                old["omega"] + alpha[:, None] * (delayed - old["omega"]),
                delayed,
            )
            omega = jnp.where(mask[:, None], raw["ang_vel"], omega)
        return dict(**values, history=history, omega=omega)

    def apply(self, data: Any, state: dict[str, Array]) -> Any:
        """Set absolute physical parameters, never compound a previous multiplier."""
        sd = data.sim_data
        ft = sd.controls.force_torque
        params = dict(ft.params)
        params["thrust_max"] = state["cap"][:, None, None] / 4
        return data.replace(
            sim_data=sd.replace(
                params=sd.params.replace(
                    rotor_dyn_coef=self.rotor_base * state["rate"][:, None, None, None]
                ),
                controls=sd.controls.replace(force_torque=ft.replace(params=params)),
            )
        )

    @staticmethod
    def observe(raw: dict[str, Array], state: dict[str, Array]) -> dict[str, Array]:
        """Expose perturbed rate to the policy while leaving true simulator state untouched."""
        return dict(raw, ang_vel=state["omega"])
