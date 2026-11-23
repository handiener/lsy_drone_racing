"""Bounded in-memory opponent pool with safe inactive-slot replacement; policies change only at resets."""

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from . import checkpoint


def probabilities(
    count: Any, capacity: int, solo_fraction: float, exponent: float, ranks: Any = None
) -> Any:
    index = jnp.arange(capacity)
    ranks = index if ranks is None else ranks
    historical = jnp.where((index > 0) & (index < count), ranks.astype(float) ** exponent, 0.0)
    historical = historical / jnp.maximum(historical.sum(), 1.0)
    p = (1 - solo_fraction) * historical
    return p.at[0].set(jnp.where(count > 1, solo_fraction, 1.0))


class League:
    def __init__(
        self,
        solo_checkpoint: str,
        network: dict,
        action: dict,
        directory: str | Path,
        capacity: int = 16,
        solo_fraction: float = 0.25,
        exponent: float = 0.9,
        manifest: str | None = None,
    ):
        if capacity < 2 or not 0 <= solo_fraction <= 1 or exponent < 0:
            raise ValueError("Invalid league capacity/probabilities")
        # Read resume metadata before add() can write into the same directory.
        document = json.loads(Path(manifest).read_text()) if manifest else None
        self._initializing = True
        self.network, self.action = network, action
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity, self.solo_fraction, self.exponent = capacity, solo_fraction, exponent
        self.entries, self.models = [], []
        self.next_generation = 0
        self.add(solo_checkpoint, solo=True)
        if document is not None:
            for entry in document["entries"]:
                path = Path(entry["checkpoint"])
                if not path.is_absolute():
                    path = Path(manifest).resolve().parent / path
                if str(path.resolve()) == str(Path(solo_checkpoint).resolve()):
                    continue
                if not self.add(str(path), solo=entry["solo"]):
                    raise ValueError("Fixed league manifest exceeds pool capacity")
                self.entries[-1]["generation"] = entry.get(
                    "generation", self.entries[-1]["generation"]
                )
                self.next_generation = max(self.next_generation, self.entries[-1]["generation"] + 1)
                if entry.get("evaluation") is not None:
                    self.entries[-1]["evaluation"] = entry["evaluation"]
        self._initializing = False
        self._persist()

    def add(
        self, path: str | Path, solo: bool = False, active_slots: set[int] | None = None
    ) -> bool:
        payload = checkpoint.load(path, self.network, self.action)
        slot = len(self.entries)
        if slot >= self.capacity:
            candidates = (
                []
                if active_slots is None
                else [i for i in range(1, self.capacity) if i not in active_slots]
            )
            if not candidates:
                return False
            slot = min(candidates, key=lambda i: self.entries[i]["generation"])
        entry = dict(
            checkpoint=str(Path(path).resolve()),
            solo=bool(solo),
            stage=payload["stage"],
            steps=payload["steps"],
            network=payload["network"],
            reward=payload["env"]["reward"],
            normalizer="stored with actor in checkpoint",
            seed=payload.get("metadata", {}).get("seed"),
            run_id=payload.get("metadata", {}).get("run_id"),
            evaluation=payload.get("metadata", {}).get("evaluation"),
            generation=self.next_generation,
        )
        self.next_generation += 1
        if slot == len(self.entries):
            self.entries.append(entry)
            self.models.append(payload["params"])
        else:
            self.entries[slot] = entry
            self.models[slot] = payload["params"]
        self._persist()
        return True

    def ranks(self) -> Any:
        ranks = [0] * self.capacity
        order = sorted(range(1, len(self.entries)), key=lambda i: self.entries[i]["generation"])
        for rank, index in enumerate(order, 1):
            ranks[index] = rank
        return jnp.asarray(ranks, jnp.float32)

    def _persist(self) -> None:
        if self._initializing:
            return
        checkpoint.write_json(self.directory / "manifest.json", self.metadata())

    def metadata(self) -> dict:
        return dict(
            entries=self.entries,
            capacity=self.capacity,
            solo_fraction=self.solo_fraction,
            exponent=self.exponent,
            replacement="oldest inactive history slot; never replace slot zero or an active opponent",
        )

    def packed(self) -> dict:
        # Only actor and the matching normalizer are needed for a frozen opponent.
        models = self.models + [self.models[0]] * (self.capacity - len(self.models))
        return dict(
            stats=jax.tree.map(lambda *x: jnp.stack(x), *[m[0] for m in models]),
            actor=jax.tree.map(lambda *x: jnp.stack(x), *[m[1] for m in models]),
            solo=jnp.array(
                [e["solo"] for e in self.entries] + [True] * (self.capacity - len(self.entries))
            ),
            count=jnp.array(len(self.entries), jnp.int32),
            ranks=self.ranks(),
        )

    def sample(self, key: Any, count: int) -> Any:
        p = probabilities(
            len(self.entries), self.capacity, self.solo_fraction, self.exponent, self.ranks()
        )
        return jax.random.choice(key, self.capacity, (count,), p=p)


def build_manifest(checkpoints: tuple[str, ...], output: str, minimum_success: float = 0.5) -> str:
    """Build a local pool only from checkpoints accompanied by measured final reports.

    Reports are retained for human diversity review; reward names are not proof of style.
    """
    entries = []
    network = action = None
    for path in checkpoints:
        model = checkpoint.load(path, network, action)
        network, action = model["network"], model["env"]["action"]
        report_path = Path(path).resolve().parent.parent / "final_evaluation.json"
        report = json.loads(report_path.read_text())
        fingerprint = report.get("checkpoint_sha256")
        if fingerprint:
            if fingerprint != hashlib.sha256(Path(path).read_bytes()).hexdigest():
                raise ValueError("Evaluation report belongs to different checkpoint weights")
        elif Path(report["checkpoint"]).resolve() != Path(path).resolve():
            raise ValueError("Evaluation report does not belong to this checkpoint")
        if report.get("success_rate", report.get("completion_rate", 0)) < minimum_success:
            continue
        entries.append(
            dict(
                checkpoint=str(Path(path).resolve()),
                solo=model["stage"] != "multi",
                network=network,
                reward=model["env"]["reward"],
                evaluation=report,
                normalizer="stored with checkpoint actor weights",
                seed=model.get("metadata", {}).get("seed"),
                run_id=model.get("metadata", {}).get("run_id"),
            )
        )
    if not entries:
        raise ValueError("No evaluated checkpoint meets the success threshold")
    checkpoint.write_json(
        output,
        dict(
            entries=entries,
            minimum_success=minimum_success,
            diversity="Inspect measured lap time, collision rate and trajectories before choosing styles",
        ),
    )
    return output


if __name__ == "__main__":
    import fire

    fire.Fire(build_manifest)
