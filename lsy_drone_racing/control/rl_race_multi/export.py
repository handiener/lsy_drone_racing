"""Export recurrent actor and its own normalizer, without optimizer or physics state."""

import shutil
from pathlib import Path

import fire
import jax
from flax import serialization

from . import checkpoint as checkpoints


def export(
    run_dir: str,
    stage: str = "multi",
    checkpoint: str = "final",
    destination: str | None = None,
    overwrite: bool = False,
) -> str:
    run = Path(run_dir).expanduser().resolve()
    target = Path(destination).expanduser().resolve() if destination else run / "exported"
    if target == run or target in run.parents:
        raise ValueError("Export destination must not contain the training run")
    model = checkpoints.load(run / "checkpoints" / f"{stage}_{checkpoint}.pkl")
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        shutil.rmtree(target)
    target.mkdir(parents=True)
    stats, actor, _ = model["params"]
    data = dict(
        actor=jax.device_get(actor), mean=jax.device_get(stats.mean), std=jax.device_get(stats.std)
    )
    (target / "policy.msgpack").write_bytes(serialization.msgpack_serialize(data))
    checkpoints.write_json(
        target / "config.json",
        dict(
            format_version=3,
            network=model["network"],
            obs=model["env"]["obs"],
            action=model["env"]["action"],
            freq=50,
            solo=model["stage"] != "multi",
            source=dict(run_dir=str(run), stage=stage, checkpoint=checkpoint),
        ),
    )
    return str(target)


if __name__ == "__main__":
    fire.Fire(export)
