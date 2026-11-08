"""Export a trained policy to plain arrays for the competition controller (no Brax needed).

    python -m lsy_drone_racing.control.rl_race.export --run_dir=outputs/<run> [--stage=race]
        [--checkpoint=final|latest] [--destination=deploy_model]

The export holds the observation normalizer (mean, std), the policy MLP layers and a JSON config
with the observation/action layout. ``controller.policy_mean_action`` reproduces Brax's
deterministic ``tanh_normal`` action: normalize, run the MLP, take tanh of the mean half.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from flax import serialization

RL_DIR = Path(__file__).resolve().parent


def policy_arrays(params: Any) -> dict[str, Any]:
    """Extract (normalizer, policy) from Brax params ``(normalizer, policy, value)`` as numpy."""
    normalizer, policy, _ = params
    layers = policy["params"]
    names = sorted(layers, key=lambda name: int(name.split("_")[-1]))
    return {
        "mean": np.asarray(normalizer.mean, np.float32),
        "std": np.asarray(normalizer.std, np.float32),
        "layers": [
            {
                "kernel": np.asarray(layers[n]["kernel"], np.float32),
                "bias": np.asarray(layers[n]["bias"], np.float32),
            }
            for n in names
        ],
    }


def export(
    run_dir: str,
    stage: str = "race",
    checkpoint: str = "final",
    destination: str | None = None,
    overwrite: bool = False,
) -> str:
    """Write ``config.json`` and ``policy.msgpack`` for a run's checkpoint; return the path."""
    from lsy_drone_racing.control.rl_race.train import load_params

    run = Path(run_dir).expanduser().resolve()
    config = json.loads((run / f"{stage}_config.json").read_text())
    params = load_params(run / "checkpoints" / f"{stage}_{checkpoint}.pkl")
    target = Path(destination).expanduser().resolve() if destination else RL_DIR / "deploy_model"
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite=True to replace it")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    bundle = {
        "obs": config["env"]["obs"],
        "action": {
            "theta_max_deg": config["env"]["action"]["theta_max_deg"],
            "thrust_min": config["thrust_min"],
            "thrust_max": config["thrust_max"],
        },
        "network": config["network"],
        "freq": config["freq"],
        "source": {"run_dir": str(run), "stage": stage, "checkpoint": checkpoint},
    }
    (target / "config.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (target / "policy.msgpack").write_bytes(serialization.msgpack_serialize(policy_arrays(params)))
    return str(target)


if __name__ == "__main__":
    fire.Fire(export)
