"""Trusted-local recurrent checkpoints with explicit, validated model contracts."""

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .network import NetworkConfig

FORMAT_VERSION = 3


def save(
    params: Any,
    target: str | Path,
    *,
    network: dict,
    env: dict,
    stage: str,
    steps: int,
    league: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    payload = dict(
        format_version=FORMAT_VERSION,
        params=jax.device_get(params),
        network=network,
        env=env,
        stage=stage,
        steps=int(steps),
        league=league or {},
        metadata=metadata or {},
        resume="Weights/normalizer restored; optimizer, rollout, RNG and LSTM carry restart",
    )
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(payload))
    temporary.replace(path)
    return payload


def load(path: str | Path, network: dict | None = None, action: dict | None = None) -> dict:
    payload = pickle.loads(Path(path).read_bytes())
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Expected recurrent checkpoint v3; legacy MLP weights are incompatible")
    if payload["network"].get("observation_dim") != 57 or payload["network"].get("action_dim") != 4:
        raise ValueError("Checkpoint must have 57 observations and four actions")
    if network is not None and payload["network"] != network:
        raise ValueError("Incompatible league network architecture/distribution")
    if action is not None and payload["env"]["action"] != action:
        raise ValueError("Incompatible league action mapping")
    _validate_params(payload)
    return payload


@lru_cache(maxsize=16)
def _expected_shapes(config: NetworkConfig) -> tuple:
    networks = config.factory()(57, 4)
    key = jax.random.PRNGKey(0)
    return (
        jax.eval_shape(networks.policy_network.init, key),
        jax.eval_shape(networks.value_network.init, key),
    )


def _validate_params(payload: dict) -> None:
    metadata = payload["network"]
    config = NetworkConfig(
        tuple(metadata["policy_layers"]),
        tuple(metadata["value_layers"]),
        metadata["activation"],
        metadata["noise_scale"],
        metadata["hidden_size"],
    )
    if config.metadata() != metadata:
        raise ValueError("Invalid recurrent network metadata")
    stats, actor, critic = payload["params"]
    for value in (stats.mean, stats.std):
        if np.shape(value) != (57,) or not np.isfinite(np.asarray(value)).all():
            raise ValueError("Invalid recurrent observation normalizer")
    if np.any(np.asarray(stats.std) < 0):
        raise ValueError("Normalizer standard deviation cannot be negative")
    for actual, expected in zip((actor, critic), _expected_shapes(config)):
        if jax.tree.structure(actual) != jax.tree.structure(expected):
            raise ValueError("Checkpoint parameter structure does not match architecture")
        for value, shape in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            if np.shape(value) != shape.shape or not np.isfinite(np.asarray(value)).all():
                raise ValueError("Checkpoint parameter shape/values do not match architecture")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
