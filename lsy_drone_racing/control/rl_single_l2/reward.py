"""Six-term JAX reward for single-drone Level 2 PPO training.

This adapts the reference's distance progress, L2 angular-speed penalty and
terminal rewards, and explicitly includes a fixed bonus per official gate pass.
It uses sensor-masked observations and the official gate-sequence counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients at the default 50 Hz control frequency.

    Progress is measured in meters per step and angular speed in radians per
    second. Gate and terminal bonuses are paid once on the corresponding event;
    the time penalty is charged on every control step, including the last.
    """

    progress: float = 15.0
    gate_pass: float = 10.0
    completion: float = 100.0
    crash: float = 15.0
    angular_velocity: float = 0.01
    time: float = 0.10


@partial(jax.jit, static_argnames=("config",))
def compute_reward(
    previous_obs: dict[str, Any],
    current_obs: dict[str, Any],
    terminated: Array,
    truncated: Array,
    config: RewardConfig | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute all six components before the adapter resets completed worlds.

    Reward = 15 * distance_progress + 10 * gate_passes + 100 * finished
             - 15 * crash - 0.01 * angular_speed_L2 - 0.10.

    Both distances use the pre-step target and its pre-step observed center, so
    target changes and newly revealed geometry do not create artificial progress.
    Crash-step progress is zero because the simulator warps crashed positions.
    Gate events come from n_gates_passed, which already accounts for required
    direction and repeated gates; no separate geometric pass detector is added.
    The last pass earns both the gate bonus and the one-off completion bonus.
    Timeouts receive no extra crash penalty. Single observations and batches are
    accepted; rewards and component arrays always have shape (n_envs,).

    Unlike the reference's experimental branches, every component returned here
    is included in the sum. There is no per-index gate-bonus scaling, guidance
    field, path construction, obstacle barrier or action-smoothness term.
    """
    del truncated  # SBX handles timeout bootstrapping separately.
    cfg = config or RewardConfig()
    previous_pos = jnp.asarray(previous_obs["pos"]).reshape(-1, 3)
    pos = jnp.asarray(current_obs["pos"]).reshape(-1, 3)
    count = previous_pos.shape[0]
    sequence = jnp.asarray(previous_obs["gate_sequence"], dtype=jnp.int32).reshape(count, -1)
    sequence_length = sequence.shape[1]
    previous_passed = jnp.asarray(previous_obs["n_gates_passed"], dtype=jnp.int32).reshape(count)
    passed = jnp.asarray(current_obs["n_gates_passed"], dtype=jnp.int32).reshape(count)
    previous_passed = jnp.clip(previous_passed, 0, sequence_length)
    passed = jnp.clip(passed, 0, sequence_length)
    active = previous_passed < sequence_length
    finished = active & (passed >= sequence_length)
    crashed = jnp.asarray(terminated, dtype=bool).reshape(count) & ~finished & active
    gate_passes = jnp.maximum(passed - previous_passed, 0)
    gate_passes = jnp.where(crashed, 0, gate_passes)

    rows = jnp.arange(count)
    gate_ids = sequence[rows, jnp.minimum(previous_passed, sequence_length - 1)]
    gates_pos = jnp.asarray(previous_obs["gates_pos"]).reshape(count, -1, 3)
    centers = gates_pos[rows, gate_ids]
    progress = jnp.linalg.norm(previous_pos - centers, axis=-1)
    progress -= jnp.linalg.norm(pos - centers, axis=-1)
    progress = jnp.where(crashed | ~active, 0.0, progress)
    angular_speed = jnp.linalg.norm(jnp.asarray(current_obs["ang_vel"]).reshape(count, 3), axis=-1)

    components = {
        "progress": cfg.progress * progress,
        "gate_pass": cfg.gate_pass * gate_passes.astype(jnp.float32),
        "completion": cfg.completion * finished.astype(jnp.float32),
        "crash": -cfg.crash * crashed.astype(jnp.float32),
        "angular_velocity": -cfg.angular_velocity * angular_speed,
        "time": jnp.full(count, -cfg.time, dtype=jnp.float32),
    }
    components = {key: value.astype(jnp.float32) for key, value in components.items()}
    reward = jnp.sum(jnp.stack(list(components.values())), axis=0)
    return reward, components
