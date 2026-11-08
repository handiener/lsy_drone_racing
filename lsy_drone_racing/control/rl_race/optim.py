"""Optimizer with separate policy/value learning rates and an optional critic-only phase.

Brax PPO trains both networks with one Adam and one learning rate. After the switch from the
pre-training reward to the race reward the critic is re-initialized; it is then trained alone for
a few PPO iterations (policy learning rate 0) before the policy resumes with a lower rate.
"""

from __future__ import annotations

from typing import Any

import jax
import optax


def iteration_updates(num_updates_per_batch: int, num_minibatches: int) -> int:
    """Optimizer steps per PPO iteration (epochs x minibatches)."""
    return num_updates_per_batch * num_minibatches


def _labels(params: Any) -> Any:
    """Label every leaf of PPONetworkParams(policy=..., value=...) with its network."""
    return type(params)(
        policy=jax.tree.map(lambda _: "policy", params.policy),
        value=jax.tree.map(lambda _: "value", params.value),
    )


def separate_lr(
    policy_lr: float,
    value_lr: float,
    max_grad_norm: float | None = 0.5,
    freeze_policy_steps: int = 0,
) -> optax.GradientTransformation:
    """Global-norm clipping, then Adam per network; the policy is frozen for the first steps."""
    policy_schedule: Any = policy_lr
    if freeze_policy_steps > 0:
        policy_schedule = optax.join_schedules(
            [optax.constant_schedule(0.0), optax.constant_schedule(policy_lr)],
            boundaries=[freeze_policy_steps],
        )
    per_network = optax.multi_transform(
        {"policy": optax.adam(policy_schedule), "value": optax.adam(value_lr)}, _labels
    )
    if max_grad_norm is None:
        return per_network
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), per_network)
