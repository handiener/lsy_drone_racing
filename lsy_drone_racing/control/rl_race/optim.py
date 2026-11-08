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
    decay_steps: int = 0,
    final_lr_factor: float = 1.0,
) -> optax.GradientTransformation:
    """Global-norm clipping, then Adam per network.

    The policy is frozen (learning rate 0) for the first ``freeze_policy_steps`` optimizer steps.
    With ``decay_steps > 0`` both learning rates then decay linearly to ``final_lr_factor`` times
    their initial value over ``decay_steps`` optimizer steps.
    """

    def schedule(lr: float) -> Any:
        if decay_steps > 0 and final_lr_factor != 1.0:
            return optax.linear_schedule(lr, lr * final_lr_factor, decay_steps)
        return optax.constant_schedule(lr)

    policy_schedule = schedule(policy_lr)
    if freeze_policy_steps > 0:
        policy_schedule = optax.join_schedules(
            [optax.constant_schedule(0.0), policy_schedule], boundaries=[freeze_policy_steps]
        )
    per_network = optax.multi_transform(
        {"policy": optax.adam(policy_schedule), "value": optax.adam(schedule(value_lr))}, _labels
    )
    if max_grad_norm is None:
        return per_network
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), per_network)
