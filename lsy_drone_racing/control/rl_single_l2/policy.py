"""Flax Actor/Critic networks and their SBX PPO policy for Level 2 racing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from sbx.ppo.policies import PPOPolicy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

NET_ARCH = (256, 256)
LOG_STD_INIT = -0.5
LOG_STD_MIN = -2.5
LOG_STD_MAX = 1.0


class Actor(nn.Module):
    """Gaussian over raw thrust/rotation with a bounded mean and state-dependent std."""

    action_dim: int = 4
    net_arch: Sequence[int] = NET_ARCH
    log_std_init: float = LOG_STD_INIT
    activation_fn: Callable = nn.leaky_relu
    # Accepted by SBX's constructor; Level2Policy fixes this to False.
    # The network uses the explicit, fixed initializers below.
    ortho_init: bool = False
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Any:
        """Return a Gaussian with a tanh mean and state-dependent clipped log std."""
        x = obs
        for width in self.net_arch:
            x = self.activation_fn(nn.Dense(width)(x))
        mean = nn.tanh(
            nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), name="mean")(x)
        )
        # A small nonzero kernel makes exploration state-dependent from initialization.
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.constant(self.log_std_init),
            name="log_std_head",
        )(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))

    def get_std(self) -> jnp.ndarray:
        """Provide SBX API parity; actual standard deviations require observations."""
        return jnp.exp(jnp.array(self.log_std_init))


class Critic(nn.Module):
    """Estimate V(o) from the same features using entirely separate parameters."""

    net_arch: Sequence[int] = NET_ARCH
    activation_fn: Callable = nn.leaky_relu
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Return one scalar per observation, with no output activation."""
        x = obs
        for width in self.net_arch:
            x = self.activation_fn(nn.Dense(width)(x))
        return nn.Dense(1, name="value")(x)


def network_config() -> dict[str, Any]:
    """Describe the inference network without serializing Python callables."""
    return {
        "net_arch": list(NET_ARCH),
        "activation": "leaky_relu",
        "action_dim": 4,
        "std_type": "state_dependent",
        "log_std_init": LOG_STD_INIT,
        "log_std_min": LOG_STD_MIN,
        "log_std_max": LOG_STD_MAX,
    }


class Level2Policy(PPOPolicy):
    """Use SBX's native optimizers and PPO loop with the Level 2 networks."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable,
        **kwargs: Any,
    ) -> None:
        """Validate flat observations and the four-dimensional raw action contract."""
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ValueError("Level2Policy requires a flat Box observation")
        if not isinstance(action_space, gym.spaces.Box) or action_space.shape != (4,):
            raise ValueError("Level2Policy requires four continuous raw actions")
        if kwargs.get("use_sde", False):
            raise ValueError("State-dependent Gaussian heads do not use gSDE")
        if kwargs.get("ortho_init", False):
            raise ValueError("Level2Policy fixes ortho_init=False and uses fixed initializers")
        kwargs["ortho_init"] = False
        fixed = {
            "actor_class": Actor,
            "critic_class": Critic,
            "net_arch": list(NET_ARCH),
            "activation_fn": nn.leaky_relu,
            "log_std_init": LOG_STD_INIT,
        }
        if fixed.keys() & kwargs.keys():
            raise ValueError(
                "Network architecture and distribution settings are fixed by Level2Policy"
            )
        super().__init__(observation_space, action_space, lr_schedule, **fixed, **kwargs)
