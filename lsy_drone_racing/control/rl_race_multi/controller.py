"""Recurrent official controller, rank-aware with a 50 Hz internal policy clock."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from lsy_drone_racing.control.controller import Controller

from .env_wrapper.actions import ActionConfig, to_command
from .env_wrapper.observations import ObsConfig, encode
from .network import NetworkConfig, zeros


def network_from_metadata(metadata: dict) -> NetworkConfig:
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


def load_policy(model_dir: Path) -> tuple:
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("format_version") != 3:
        raise ValueError("Expected recurrent export v3")
    network_from_metadata(config["network"])
    arrays = serialization.msgpack_restore((model_dir / "policy.msgpack").read_bytes())
    if arrays["mean"].shape != (57,) or arrays["std"].shape != (57,):
        raise ValueError("Invalid recurrent normalizer dimensions")
    return config, arrays


class RaceController(Controller):
    def __init__(self, obs: dict, info: dict, config: object) -> None:
        super().__init__(obs, info, config)
        self.rank = int(info.get("rank", 0))
        entries = config.controller
        entry = entries[self.rank] if isinstance(entries, (list, tuple)) else entries
        path = Path(entry.get("model_dir", "deploy_model")).expanduser()
        if not path.is_absolute():
            path = Path(__file__).parent / path
        bundle, arrays = load_policy(path)
        self.solo = bundle["solo"]
        self.multi = np.asarray(obs["pos"]).ndim == 2
        mode = config.env.get("control_mode", None)
        if mode is None:
            mode = config.env.kwargs[self.rank]["control_mode"]
        if mode != "attitude":
            raise ValueError("Recurrent controller requires attitude mode")
        frequency = int(config.env.freq)
        if frequency < bundle["freq"] or frequency % bundle["freq"]:
            raise ValueError("Controller frequency must be an integer multiple of trained 50 Hz")
        self.period = frequency // bundle["freq"]
        architecture = network_from_metadata(bundle["network"])
        nets = architecture.factory()(57, 4)
        stats = SimpleNamespace(mean=jnp.asarray(arrays["mean"]), std=jnp.asarray(arrays["std"]))
        actor = jax.tree.map(jnp.asarray, arrays["actor"])
        self.obs_cfg = ObsConfig(**bundle["obs"])
        action_cfg = ActionConfig(**bundle["action"])

        def predict(features, carry):
            carry, logits = nets.policy_network.apply(stats, actor, features, carry)
            action = jnp.tanh(logits[..., :4])
            return carry, action, to_command(action, action_cfg)

        self._predict = jax.jit(predict)
        self.hidden_size = architecture.hidden_size
        self.episode_reset()
        self._predict(jnp.zeros((1, 57)), self._carry)[2].block_until_ready()

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        if self._calls % self.period == 0:
            if self.multi:
                ego = {k: jnp.asarray(v[self.rank])[None] for k, v in obs.items()}
                opponent = {k: jnp.asarray(v[1 - self.rank])[None] for k, v in obs.items()}
                # Official environments warp disabled drones below ground.
                presence = jnp.asarray([obs["pos"][1 - self.rank, 2] >= 0])
            else:
                ego = {k: jnp.asarray(v)[None] for k, v in obs.items()}
                opponent, presence = None, None
            if self.solo:
                opponent = None
            features = encode(ego, self._previous, self.obs_cfg, opponent, presence)
            self._carry, self._previous, command = self._predict(features, self._carry)
            self._command = np.asarray(command[0], np.float32)
        self._calls += 1
        return self._command.copy()

    def episode_reset(self) -> None:
        self._carry = zeros((1,), self.hidden_size)
        self._previous = jnp.zeros((1, 4))
        self._command = np.zeros(4, np.float32)
        self._calls = 0

    def step_callback(self, *args, **kwargs) -> bool:
        return False

    def reset(self) -> None:
        self.episode_reset()
