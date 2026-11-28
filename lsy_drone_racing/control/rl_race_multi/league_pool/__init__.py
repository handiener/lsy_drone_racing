"""Frozen inference contracts for recurrent checkpoints and single-racer MLP weights."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import checkpoint
from ..env_wrapper.actions import ActionConfig
from ..network import network_from_metadata
from . import brax_mlp


def load(path: str | Path, network: dict | None = None, action: dict | None = None) -> dict:
    """Load trusted local weights, requiring metadata for a bare single-racer pickle."""
    path = Path(path).expanduser().resolve()
    payload = pickle.loads(path.read_bytes())
    if isinstance(payload, dict) and payload.get("format_version") == 3:
        model = checkpoint.validate(payload, network, action)
        ActionConfig(**model["env"]["action"])
        return dict(model, kind="lstm", freq=50)
    if network is not None:
        raise ValueError("Expected recurrent checkpoint v3; legacy MLP weights are incompatible")
    return brax_mlp.load(path, payload)


def signature(model: dict) -> str:
    """Identify an inference bank; action scaling and normalizers remain per opponent."""
    return json.dumps([model["kind"], model["network"]], sort_keys=True)


def command_affine(action: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return per-model normalized-action scale and offset in physical units."""
    cfg = ActionConfig(**action)
    half = 0.5 * (cfg.thrust_max - cfg.thrust_min)
    scale = np.array(
        [*np.deg2rad([cfg.roll_limit_deg, cfg.pitch_limit_deg, cfg.yaw_limit_deg]), half],
        np.float32,
    )
    return scale, np.array([0, 0, 0, cfg.thrust_min + half], np.float32)


def predict_bank(
    model: dict, bank: dict, index: Any, obs: Any, carry: Any, key: Any, deterministic: bool
) -> tuple:
    """Evaluate a static-shape bank using each selected policy's original inference rules."""
    if model["kind"] == "lstm":
        actor = jax.tree.map(lambda x: x[index], bank["actor"])
        mean, std = bank["mean"][index], bank["std"][index]
        nets = network_from_metadata(model["network"]).factory()(57, 4)
        hidden = model["network"]["hidden_size"]
        local_carry = jax.tree.map(lambda x: x[..., :hidden], carry)

        def forward(
            params: Any, mu: Any, sigma: Any, observation: Any, memory: Any, rng: Any
        ) -> tuple:
            memory, logits = nets.policy_network.apply(
                SimpleNamespace(mean=mu, std=sigma), params, observation, memory
            )
            raw = (
                logits[..., :4]
                if deterministic
                else nets.parametric_action_distribution.sample_no_postprocessing(logits, rng)
            )
            return memory, jnp.tanh(raw)

        memory, action = jax.vmap(forward)(
            actor, mean, std, obs, local_carry, jax.random.split(key, obs.shape[0])
        )
        memory = jax.tree.map(
            lambda x: jnp.pad(x, [(0, 0), (0, carry[0].shape[-1] - hidden)]), memory
        )
        return memory, action
    action = brax_mlp.predict(model, bank, index, obs[..., :49], key, deterministic)
    return jax.tree.map(jnp.zeros_like, carry), action
