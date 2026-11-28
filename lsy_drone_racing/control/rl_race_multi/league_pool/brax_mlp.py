"""Portable Brax MLP opponent inference and official-simulator controller.

This module does not import the original rl_race package. The training pool and
standalone simulator use the same normalization, network and action mapping.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.rl_race_multi.env_wrapper.actions import ActionConfig, to_command
from lsy_drone_racing.control.rl_race_multi.env_wrapper.observations import ObsConfig, encode_ego


def load(path: str | Path, payload: Any = None) -> dict:
    """Read a self-contained pool model, or legacy weights plus their original stage JSON."""
    path = Path(path).expanduser().resolve()
    if payload is None:
        payload = pickle.loads(path.read_bytes())
    if isinstance(payload, dict) and payload.get("opponent_format_version") == 1:
        model = payload
    else:
        if not isinstance(payload, tuple) or len(payload) != 3:
            raise ValueError(f"Unsupported Brax MLP checkpoint: {path}")
        candidates = [path.with_suffix(".json")]
        stage = path.stem.removesuffix("_final").removesuffix("_latest")
        candidates += [
            path.parent / f"{stage}_config.json",
            path.parent.parent / f"{stage}_config.json",
        ]
        config_path = next((p for p in candidates if p.is_file()), None)
        if config_path is None:
            raise ValueError(
                f"Bare single-racer weights {path.name} need {path.with_suffix('.json').name} "
                "(copy the original race_config.json or anneal_config.json)"
            )
        config = json.loads(config_path.read_text())
        action = dict(config["env"]["action"])
        # Old single-run env records contain placeholders; these top-level values are physical.
        action.update(thrust_min=config["thrust_min"], thrust_max=config["thrust_max"])
        model = dict(
            opponent_format_version=1,
            kind="mlp",
            params=payload,
            network=dict(
                config["network"],
                noise_scale=config["network"].get(
                    "noise_scale", config.get("settings", {}).get("noise_scale", 1.0)
                ),
            ),
            env=dict(config["env"], action=action),
            freq=config["freq"],
            stage=config.get("stage", stage),
            steps=0,
            metadata=dict(source_config=str(config_path)),
        )
    if model.get("kind") != "mlp" or model.get("freq") != 50:
        raise ValueError("External MLP opponents must use 50 Hz attitude control")
    if model["env"]["obs"] != dict(n_gates=2, n_obstacles=4):
        raise ValueError("Single-racer opponents need the 49D two-gate/four-obstacle layout")
    stats, actor, _ = model["params"]
    if np.shape(stats.mean) != (49,) or np.shape(stats.std) != (49,):
        raise ValueError("Single-racer normalizer must have 49 observations")
    if (
        not np.isfinite(np.asarray(stats.mean)).all()
        or not np.isfinite(np.asarray(stats.std)).all()
    ):
        raise ValueError("Opponent normalizer must be finite")
    if np.any(np.asarray(stats.std) <= 0):
        raise ValueError("Opponent normalizer standard deviations must be positive")
    network = model["network"]
    noise = float(network.get("noise_scale", 1.0))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("MLP noise_scale must be positive and finite")
    network["noise_scale"] = noise
    if network.get("distribution") != "tanh_normal" or network.get("activation") not in {
        "swish",
        "elu",
        "relu",
        "tanh",
    }:
        raise ValueError("Unsupported MLP opponent network/distribution")
    layers = actor["params"]
    names = sorted(layers, key=lambda name: int(name.split("_")[-1]))
    widths = [49, *network["policy_layers"], 8]
    if len(names) != len(widths) - 1:
        raise ValueError("MLP opponent layer metadata does not match weights")
    for i, name in enumerate(names):
        layer = layers[name]
        if np.shape(layer["kernel"]) != (widths[i], widths[i + 1]) or np.shape(layer["bias"]) != (
            widths[i + 1],
        ):
            raise ValueError("MLP opponent weight shapes do not match metadata")
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actor)):
        raise ValueError("MLP opponent weights must be finite")
    ActionConfig(**model["env"]["action"])
    return model


def predict(
    model: dict, bank: dict, index: Any, features: Any, key: Any, deterministic: bool = True
) -> Any:
    """Batch inference with per-opponent weights and the original unclipped normalizer."""
    actor = jax.tree.map(lambda x: x[index], bank["actor"])
    x = (features - bank["mean"][index]) / bank["std"][index]
    names = sorted(actor["params"], key=lambda name: int(name.split("_")[-1]))
    activation = getattr(jax.nn, model["network"]["activation"])
    for i, name in enumerate(names):
        layer = actor["params"][name]
        x = jnp.einsum("bi,bij->bj", x, layer["kernel"]) + layer["bias"]
        if i < len(names) - 1:
            x = activation(x)
    raw = x[..., :4]
    if not deterministic:
        from brax.training.distribution import NormalTanhDistribution

        raw = NormalTanhDistribution(
            event_size=4, var_scale=model["network"].get("noise_scale", 1.0)
        ).sample_no_postprocessing(x, key)
    return jnp.tanh(raw)


class RaceController(Controller):
    """Use pool MLP weights in single- or two-drone official simulation."""

    def __init__(self, obs: dict, info: dict, config: Any) -> None:
        """Load an explicitly selected pickle and compile outside the timed control loop."""
        super().__init__(obs, info, config)
        self.rank = int(info.get("rank", 0))
        entries = config.controller
        entry = entries[self.rank] if isinstance(entries, (list, tuple)) else entries
        if not entry.get("weights"):
            raise ValueError("Set controller.weights to the pool MLP .pkl file")
        path = Path(entry["weights"]).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        model = load(path)
        agent = config.env.kwargs[self.rank] if "kwargs" in config.env else config.env
        mode = config.env.get("control_mode", agent["control_mode"])
        frequency = int(config.env.get("freq", agent["freq"]))
        if mode != "attitude" or frequency < model["freq"] or frequency % model["freq"]:
            raise ValueError("Pool MLP controller requires attitude control at a multiple of 50 Hz")
        self.period = frequency // model["freq"]
        self.multi = np.asarray(obs["pos"]).ndim == 2
        action_cfg = ActionConfig(**model["env"]["action"])
        obs_cfg = ObsConfig(**model["env"]["obs"])
        stats, actor, _ = model["params"]
        bank = dict(
            mean=jnp.asarray(stats.mean)[None],
            std=jnp.asarray(stats.std)[None],
            actor=jax.tree.map(lambda x: jnp.asarray(x)[None], actor),
        )

        def infer(raw: dict, previous: Any) -> tuple:
            features = encode_ego(raw, previous, obs_cfg)
            action = predict(model, bank, jnp.zeros(1, jnp.int32), features, jax.random.PRNGKey(0))
            return action, to_command(action, action_cfg)[0]

        self._infer = jax.jit(infer)
        self.episode_reset()
        self._infer(self._ego(obs), self._previous)[1].block_until_ready()

    def _ego(self, obs: dict) -> dict:
        return {k: jnp.asarray(v[self.rank] if self.multi else v)[None] for k, v in obs.items()}

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Keep each rank's previous action and preserve the trained clock at 100 Hz."""
        if self._calls % self.period == 0:
            self._previous, command = self._infer(self._ego(obs), self._previous)
            self._command = np.asarray(command, np.float32)
        self._calls += 1
        return self._command.copy()

    def episode_reset(self) -> None:
        """Reset this drone's action history between races."""
        self._previous = jnp.zeros((1, 4), jnp.float32)
        self._command = np.zeros(4, np.float32)
        self._calls = 0

    def step_callback(self, *args: Any, **kwargs: Any) -> bool:
        """Leave episode termination to the simulator."""
        return False

    def reset(self) -> None:
        """Reset history for runners using the shorter hook name."""
        self.episode_reset()
