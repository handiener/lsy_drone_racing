"""Phase 1 gate-entry initialization for training resets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ResetConfig:
    """Phase 1 gate-entry starts; probability zero preserves official resets."""

    segment_init_prob: float = 0.40
    segment_init_offset_m: float = 0.50
    segment_init_vel_mps: float = 2.50
    segment_init_perturb_m: float = 0.10

    def __post_init__(self) -> None:
        """Validate probabilities and finite position/velocity scales."""
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Reset parameters must be finite")
        if not 0 <= self.segment_init_prob <= 1:
            raise ValueError("segment_init_prob must be between zero and one")
        if self.segment_init_offset_m <= 0:
            raise ValueError("segment_init_offset_m must be positive")
        if self.segment_init_vel_mps < 0 or self.segment_init_perturb_m < 0:
            raise ValueError("Segment speed and position perturbation must be nonnegative")


def build_segment_reset(
    config: ResetConfig, contact_check: Callable[[Any], Array]
) -> Callable[[Any, Array], tuple[Any, Array, Array]]:
    """Compile Phase 1 proposals, with official collision checks and fallback.

    The reference's 0.5 m nominal gate-entry offset, +/-0.1 m position noise
    and 2.5 m/s through velocity are retained. Sequence entries determine the
    physical gate and signed direction. Level roll/pitch and a yaw clipped to
    the official attitude range avoid an initial out-of-range heading.
    """
    from lsy_drone_racing.envs.race_core import _reset_env_data

    @jax.jit
    def segment_reset(data: Any, mask: Array) -> tuple[Any, Array, Array]:
        count = data.steps.shape[0]
        key, choose_key, target_key, noise_key = jax.random.split(data.sim_data.core.rng_key, 4)
        data = data.replace(
            sim_data=data.sim_data.replace(core=data.sim_data.core.replace(rng_key=key))
        )
        requested = mask & (jax.random.uniform(choose_key, (count,)) < config.segment_init_prob)
        target = jax.random.randint(
            target_key, (count,), 0, data.gate_sequence.shape[0], dtype=data.n_gates_passed.dtype
        )

        def apply(candidate_data: Any) -> tuple[Any, Array, Array]:
            rows = jnp.arange(count)
            gate_id = candidate_data.gate_sequence[target]
            direction = candidate_data.gate_sequence_direction[target, None]
            center = candidate_data.nominal_gates_pos[rows, gate_id]
            normal = (
                Rotation.from_quat(candidate_data.nominal_gates_quat[rows, gate_id]).apply(
                    jnp.array([1.0, 0.0, 0.0])
                )
                * direction
            )
            noise = jax.random.uniform(
                noise_key,
                (count, 3),
                minval=-config.segment_init_perturb_m,
                maxval=config.segment_init_perturb_m,
            )
            position = center - config.segment_init_offset_m * normal + noise
            yaw = jnp.clip(jnp.arctan2(normal[:, 1], normal[:, 0]), -jnp.pi / 2, jnp.pi / 2)
            quat = jnp.stack(
                [jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(yaw / 2), jnp.cos(yaw / 2)],
                axis=-1,
            )
            states = candidate_data.sim_data.states
            proposed = states.replace(
                pos=jnp.where(requested[:, None, None], position[:, None], states.pos),
                vel=jnp.where(
                    requested[:, None, None],
                    (config.segment_init_vel_mps * normal)[:, None],
                    states.vel,
                ),
                quat=jnp.where(requested[:, None, None], quat[:, None], states.quat),
                ang_vel=jnp.where(requested[:, None, None], 0, states.ang_vel),
            )
            candidate = candidate_data.replace(
                sim_data=candidate_data.sim_data.replace(
                    states=proposed, core=candidate_data.sim_data.core.replace(mjx_synced=False)
                )
            )
            in_bounds = jnp.all(
                (position > candidate.pos_limit_low) & (position < candidate.pos_limit_high),
                axis=-1,
            )
            true_normal = (
                Rotation.from_quat(candidate.gates_quat[rows, gate_id]).apply(
                    jnp.array([1.0, 0.0, 0.0])
                )
                * direction
            )
            on_entry_side = (
                jnp.sum((position - candidate.gates_pos[rows, gate_id]) * true_normal, axis=-1) < 0
            )
            collision = contact_check(candidate).reshape(count, -1).any(axis=1)
            accepted = requested & in_bounds & on_entry_side & ~collision
            mask3 = accepted[:, None, None]
            states = states.replace(
                pos=jnp.where(mask3, proposed.pos, states.pos),
                vel=jnp.where(mask3, proposed.vel, states.vel),
                quat=jnp.where(mask3, proposed.quat, states.quat),
                ang_vel=jnp.where(mask3, proposed.ang_vel, states.ang_vel),
            )
            result = candidate_data.replace(
                sim_data=candidate_data.sim_data.replace(
                    states=states, core=candidate_data.sim_data.core.replace(mjx_synced=False)
                )
            )
            # Recompute sensor masks from range, without revealing unseen geometry.
            result = _reset_env_data(result, mask=accepted)
            result = result.replace(
                n_gates_passed=jnp.where(accepted[:, None], target[:, None], result.n_gates_passed)
            )
            return result, accepted, jnp.where(accepted, target, 0)

        return jax.lax.cond(
            requested.any(),
            apply,
            lambda d: (
                d,
                jnp.zeros(count, dtype=bool),
                jnp.zeros(count, dtype=data.n_gates_passed.dtype),
            ),
            data,
        )

    return segment_reset
