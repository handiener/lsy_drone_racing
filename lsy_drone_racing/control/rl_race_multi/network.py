"""Independent actor/critic LSTMs and a single shared inference contract."""

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.training import distribution
from brax.training.networks import FeedForwardNetwork


def zeros(batch: tuple[int, ...], hidden: int) -> tuple[jax.Array, jax.Array]:
    """LSTM carry is (cell, hidden), never trainable optimizer state."""
    return (jnp.zeros((*batch, hidden)), jnp.zeros((*batch, hidden)))


def reset_carry(carry: Any, done: jax.Array) -> Any:
    return jax.tree.map(lambda x: jnp.where(done[..., None], 0.0, x), carry)


def normalize(obs: jax.Array, stats: Any) -> jax.Array:
    """Opponent dimensions use a unit std floor; absence always means exactly zero."""
    std = jnp.maximum(stats.std, 1e-3)
    std = std.at[..., 49:].set(jnp.maximum(std[..., 49:], 1.0))
    result = jnp.clip((obs - stats.mean) / std, -10.0, 10.0)
    return result.at[..., 49:].set(jnp.where(obs[..., -1:] > 0, result[..., 49:], 0.0))


class LSTMHead(nn.Module):
    hidden_size: int
    layers: tuple[int, ...]
    output_size: int
    activation: str

    @nn.compact
    def __call__(self, obs: jax.Array, carry: Any) -> tuple[Any, jax.Array]:
        carry, x = nn.LSTMCell(self.hidden_size, name="lstm")(carry, obs)
        activation = getattr(nn, self.activation)
        for index, width in enumerate(self.layers):
            x = activation(nn.Dense(width, name=f"hidden_{index}")(x))
        x = nn.Dense(self.output_size, name="output")(x)
        return carry, x


@dataclass(frozen=True)
class RecurrentNetworks:
    policy_network: Any
    value_network: Any
    parametric_action_distribution: Any
    hidden_size: int


@dataclass(frozen=True)
class NetworkConfig:
    policy_layers: tuple[int, ...] = (512, 512)
    value_layers: tuple[int, ...] = (512, 512)
    activation: str = "leaky_relu"
    noise_scale: float = 1.0
    hidden_size: int = 256

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or not self.policy_layers or not self.value_layers:
            raise ValueError("Network layers and hidden size must be positive")
        if min((*self.policy_layers, *self.value_layers)) < 1 or self.noise_scale <= 0:
            raise ValueError("Invalid network widths/noise scale")
        if self.activation not in {"leaky_relu", "relu", "swish", "elu", "tanh"}:
            raise ValueError("Unsupported activation")

    def metadata(self) -> dict:
        return dict(
            type="independent_lstm",
            hidden_size=self.hidden_size,
            policy_layers=list(self.policy_layers),
            value_layers=list(self.value_layers),
            activation=self.activation,
            noise_scale=self.noise_scale,
            distribution="tanh_normal",
            observation_dim=57,
            action_dim=4,
        )

    def factory(self) -> Any:
        def build(observation_size: Any, action_size: int, **kwargs: Any) -> RecurrentNetworks:
            if observation_size not in (57, (57,)) or action_size != 4:
                raise ValueError("Recurrent policy requires 57 observations and four actions")

            def network(layers: tuple[int, ...], width: int) -> FeedForwardNetwork:
                model = LSTMHead(self.hidden_size, layers, width, self.activation)

                def init(key: jax.Array) -> Any:
                    return model.init(key, jnp.zeros((1, 57)), zeros((1,), self.hidden_size))

                def apply(stats: Any, params: Any, obs: jax.Array, carry: Any) -> Any:
                    carry, result = model.apply(params, normalize(obs, stats), carry)
                    return carry, result[..., 0] if width == 1 else result

                return FeedForwardNetwork(init=init, apply=apply)

            return RecurrentNetworks(
                network(self.policy_layers, 8),
                network(self.value_layers, 1),
                distribution.NormalTanhDistribution(event_size=4, var_scale=self.noise_scale),
                self.hidden_size,
            )

        return build


def network_from_metadata(metadata: dict) -> NetworkConfig:
    """Restore the shared architecture contract for training, opponents and deployment."""
    if (
        metadata.get("type") != "independent_lstm"
        or metadata.get("observation_dim") != 57
        or metadata.get("action_dim") != 4
    ):
        raise ValueError("Expected independent recurrent 57D/four-action architecture")
    return NetworkConfig(
        tuple(metadata["policy_layers"]),
        tuple(metadata["value_layers"]),
        metadata["activation"],
        metadata["noise_scale"],
        metadata["hidden_size"],
    )


def inference(
    networks: RecurrentNetworks,
    params: Any,
    obs: jax.Array,
    carry: Any,
    key: jax.Array,
    deterministic: bool = False,
) -> tuple[Any, jax.Array, dict]:
    actor, critic = carry
    next_actor, logits = networks.policy_network.apply(params[0], params[1], obs, actor)
    next_critic, value = networks.value_network.apply(params[0], params[2], obs, critic)
    dist = networks.parametric_action_distribution
    raw = logits[..., :4] if deterministic else dist.sample_no_postprocessing(logits, key)
    return (
        (next_actor, next_critic),
        dist.postprocess(raw),
        dict(
            raw_action=raw,
            log_prob=dist.log_prob(logits, raw),
            value=value,
            distribution_params=logits,
        ),
    )


def make_inference_fn(networks: RecurrentNetworks, **kwargs: Any) -> Any:
    def make_policy(params: Any, deterministic: bool = False) -> Any:
        def policy(obs: jax.Array, carry: Any, key: jax.Array) -> Any:
            return inference(networks, params, obs, carry, key, deterministic)

        policy.hidden_size = networks.hidden_size
        return policy

    return make_policy
