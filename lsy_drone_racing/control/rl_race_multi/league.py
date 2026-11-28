"""Frozen opponent pool with bounded history and safe inactive-slot replacement."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from . import checkpoint
from . import league_pool as opponents


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
    """Keep homogeneous history and manually supplied frozen opponents in separate banks."""

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
        weights_dir: str | Path | None = None,
        fixed_fraction: float = 0.25,
    ):
        if capacity < 2 or not 0 <= solo_fraction <= 1 or exponent < 0:
            raise ValueError("Invalid league capacity/probabilities")
        if not 0 <= fixed_fraction <= 1:
            raise ValueError("fixed_fraction must be in [0, 1]")
        document = json.loads(Path(manifest).read_text()) if manifest else None
        self._initializing = True
        self.network, self.action = network, action
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity, self.solo_fraction, self.exponent = capacity, solo_fraction, exponent
        self.fixed_fraction = fixed_fraction
        self.entries, self.models, self.specs = [], [], []
        self.next_generation = 0
        self._sources = set()
        self.add(solo_checkpoint, solo=True, fixed=True)
        if document is not None:
            for entry in document["entries"]:
                path = Path(entry["checkpoint"])
                if not path.is_absolute():
                    path = Path(manifest).resolve().parent / path
                # Identical weights may share one frozen blob but remain distinct pool entries.
                source = Path(entry.get("source_checkpoint") or path).resolve()
                if source in self._sources:
                    continue
                if not self.add(path, solo=entry["solo"], fixed=entry.get("fixed", entry["solo"])):
                    raise ValueError("Fixed league manifest exceeds pool capacity")
                self.entries[-1]["generation"] = entry.get(
                    "generation", self.entries[-1]["generation"]
                )
                self.next_generation = max(self.next_generation, self.entries[-1]["generation"] + 1)
                for key in ("evaluation", "source_checkpoint"):
                    if entry.get(key) is not None:
                        self.entries[-1][key] = entry[key]
                if entry.get("source_checkpoint"):
                    self._sources.add(Path(entry["source_checkpoint"]).resolve())
        # A recorded pool is immutable unless the caller explicitly extends it.
        weights = (
            Path(weights_dir).expanduser()
            if weights_dir
            else (Path(__file__).parent / "league_pool" / "weights" if document is None else None)
        )
        if weights_dir and not weights.is_dir():
            raise ValueError(f"league_weights directory does not exist: {weights}")
        if weights is not None and weights.is_dir():
            for path in sorted(weights.rglob("*.pkl")):
                if path.resolve() in self._sources:
                    continue
                if not self.add(path, solo=None, fixed=True):
                    raise ValueError("Manual league weights exceed league_capacity")
        self._initializing = False
        self._persist()

    @property
    def hidden_size(self) -> int:
        """Maximum frozen LSTM width, fixed before environment compilation."""
        return max(
            [
                self.network["hidden_size"],
                *[
                    model["network"]["hidden_size"]
                    for model in self.specs
                    if model["kind"] == "lstm"
                ],
            ]
        )

    def add(
        self,
        path: str | Path,
        solo: bool | None = False,
        active_slots: set[int] | None = None,
        fixed: bool = False,
    ) -> bool:
        """Load once on the host; never replace a fixed or currently active opponent."""
        source = Path(path).resolve()
        strict = not self.entries or not fixed
        payload = opponents.load(
            source, self.network if strict else None, self.action if strict else None
        )
        if solo is None:
            solo = payload["kind"] == "mlp" or payload["stage"] != "multi"
        key = opponents.signature(payload)
        signatures = [opponents.signature(model) for model in self.specs]
        if key not in signatures:
            if not self._initializing:
                raise ValueError("New inference banks can only be installed before training")
            self.specs.append(payload)
            signatures.append(key)
        bank = signatures.index(key)
        slot = len(self.entries)
        if slot >= self.capacity:
            candidates = (
                []
                if active_slots is None
                else [
                    i
                    for i, entry in enumerate(self.entries)
                    if not entry.get("fixed", i == 0) and i not in active_slots
                ]
            )
            if not candidates:
                return False
            slot = min(candidates, key=lambda i: self.entries[i]["generation"])
        target = source
        if fixed and self.entries:
            # A run owns a self-contained frozen copy, including an old MLP's sidecar metadata.
            encoded = pickle.dumps(payload)
            target = self.directory / "fixed" / (hashlib.sha256(encoded).hexdigest()[:20] + ".pkl")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(encoded)
        entry = dict(
            checkpoint=str(target),
            source_checkpoint=str(source),
            solo=bool(solo),
            fixed=bool(fixed),
            bank=bank,
            stage=payload["stage"],
            steps=payload["steps"],
            network=payload["network"],
            action=payload["env"]["action"],
            reward=payload["env"].get("reward", {}),
            normalizer="stored with actor in checkpoint",
            seed=payload.get("metadata", {}).get("seed"),
            run_id=payload.get("metadata", {}).get("run_id"),
            evaluation=payload.get("metadata", {}).get("evaluation"),
            generation=self.next_generation,
        )
        self.next_generation += 1
        self._sources.update([source, target.resolve()])
        if slot == len(self.entries):
            self.entries.append(entry)
            self.models.append(payload["params"])
        else:
            self.entries[slot] = entry
            self.models[slot] = payload["params"]
        self._persist()
        return True

    def ranks(self) -> Any:
        """Chronological sampling ranks, including after inactive history slots are reused."""
        ranks = [0] * self.capacity
        order = sorted(
            (i for i in range(1, len(self.entries)) if not self.entries[i]["fixed"]),
            key=lambda i: self.entries[i]["generation"],
        )
        for rank, index in enumerate(order, 1):
            ranks[index] = rank
        return jnp.asarray(ranks, jnp.float32)

    def _persist(self) -> None:
        if not self._initializing:
            checkpoint.write_json(self.directory / "manifest.json", self.metadata())

    def metadata(self) -> dict:
        """Record portable opponent metadata alongside the learner checkpoint."""
        return dict(
            entries=self.entries,
            capacity=self.capacity,
            solo_fraction=self.solo_fraction,
            exponent=self.exponent,
            fixed_fraction=self.fixed_fraction,
            replacement="oldest inactive history slot; never replace fixed or active opponents",
        )

    def packed(self) -> dict:
        """Pack stable-shape inference banks; each actor retains its own normalizer and scaling."""
        slots = self.entries + [self.entries[0]] * (self.capacity - len(self.entries))
        models = self.models + [self.models[0]] * (self.capacity - len(self.models))
        banks = []
        for i, spec in enumerate(self.specs):
            dummy = spec["params"]
            selected = [m if entry["bank"] == i else dummy for entry, m in zip(slots, models)]
            banks.append(
                dict(
                    mean=jnp.stack([m[0].mean for m in selected]),
                    std=jnp.stack([m[0].std for m in selected]),
                    actor=jax.tree.map(lambda *x: jnp.stack(x), *[m[1] for m in selected]),
                )
            )
        scales, offsets = zip(*[opponents.command_affine(e["action"]) for e in slots])
        return dict(
            banks=tuple(banks),
            kind=jnp.asarray([e["bank"] for e in slots], jnp.int32),
            scale=jnp.asarray(scales),
            offset=jnp.asarray(offsets),
            fixed=jnp.asarray([e["fixed"] for e in slots]),
            solo=jnp.asarray([e["solo"] for e in slots]),
            count=jnp.array(len(self.entries), jnp.int32),
            ranks=self.ranks(),
        )

    def probabilities(self, pool: dict) -> Any:
        """Split fixed/history mass only when a manual fixed pool is present."""
        if not any(e["fixed"] for e in self.entries[1:]):
            return probabilities(
                pool["count"], self.capacity, self.solo_fraction, self.exponent, pool["ranks"]
            )
        valid = jnp.arange(self.capacity) < pool["count"]
        fixed = valid & pool["fixed"]
        history = valid & ~pool["fixed"]
        fp = fixed / jnp.maximum(fixed.sum(), 1)
        hp = jnp.where(history, pool["ranks"] ** self.exponent, 0.0)
        hp = hp / jnp.maximum(hp.sum(), 1.0)
        fraction = jnp.where(history.any(), self.fixed_fraction, 1.0)
        return fraction * fp + (1 - fraction) * hp

    def predict(
        self, pool: dict, index: Any, obs: Any, carry: Any, key: Any, deterministic: bool = True
    ) -> tuple:
        """Return frozen memory, normalized actions, and each opponent's physical command."""
        memory = jax.tree.map(jnp.zeros_like, carry)
        action = jnp.zeros((obs.shape[0], 4), jnp.float32)
        selected = pool["kind"][index]
        for i, spec in enumerate(self.specs):
            local, values = opponents.predict_bank(
                spec,
                pool["banks"][i],
                index,
                obs,
                carry,
                jax.random.fold_in(key, i) if i else key,
                deterministic,
            )
            memory = jax.tree.map(
                lambda a, b: jnp.where((selected == i)[:, None], b, a), memory, local
            )
            action = jnp.where((selected == i)[:, None], values, action)
        command = action * pool["scale"][index] + pool["offset"][index]
        return memory, action, command

    def sample(self, key: Any, count: int) -> Any:
        """Sample one opponent per newly starting episode."""
        return jax.random.choice(key, self.capacity, (count,), p=self.probabilities(self.packed()))


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
            diversity=(
                "Inspect measured lap time, collision rate and trajectories before choosing styles"
            ),
        ),
    )
    return output


if __name__ == "__main__":
    import fire

    fire.Fire(build_manifest)
