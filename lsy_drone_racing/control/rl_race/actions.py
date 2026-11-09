"""Absolute attitude action: policy output in [-1, 1]^3 -> official attitude command.

The policy outputs ``[roll, pitch, thrust]`` (Brax ``tanh_normal`` actions are already in
[-1, 1]). Roll and pitch are absolute angles scaled to +/- theta_max; yaw is not controlled and is
always commanded as 0; thrust maps affinely onto the full collective-thrust range.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

ACTION_SIZE = 3


@dataclass(frozen=True)
class ActionConfig:
    """Attitude limits; frozen so it can be a static jit argument."""

    theta_max_deg: float = 70.0
    thrust_min: float = 0.0
    thrust_max: float = 1.0

    def __post_init__(self) -> None:
        """Check limits: the official attitude interface clips at +/- 90 degrees."""
        if not 0 < self.theta_max_deg <= 90:
            raise ValueError("theta_max_deg must be in (0, 90]")
        if not self.thrust_max > self.thrust_min >= 0:
            raise ValueError("Need thrust_max > thrust_min >= 0")


def to_command(action: Array, cfg: ActionConfig) -> Array:
    """Map (n, 3) actions in [-1, 1] to (n, 4) commands [roll, pitch, yaw, total_thrust]."""
    action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
    theta = np.deg2rad(cfg.theta_max_deg)
    thrust = cfg.thrust_min + 0.5 * (action[..., 2] + 1.0) * (cfg.thrust_max - cfg.thrust_min)
    yaw = jnp.zeros_like(thrust)
    return jnp.stack([theta * action[..., 0], theta * action[..., 1], yaw, thrust], axis=-1)
