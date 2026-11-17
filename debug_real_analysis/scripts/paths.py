"""Resolve repository inputs, archived evidence and writable diagnostic outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
EVIDENCE = BASE / "evidence"
OUT = Path(os.environ.get("DRONE_DEBUG_OUTPUT_DIR", str(BASE / "generated"))).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / "cache"))


def read_metrics(phase: str) -> list[dict]:
    """Prefer a regenerated phase; otherwise use the original archived metrics."""
    for root in (OUT, EVIDENCE):
        path = root / phase / "metrics.json"
        if path.is_file():
            return json.loads(path.read_text())
        path = root / phase / "metrics.jsonl"
        if path.is_file():
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return []


def resolve_trajectory(path: str | Path) -> Path | None:
    """Find generated or bundled traces; bulk non-selected traces are intentionally omitted."""
    relative = Path(path)
    if relative.is_absolute():
        return relative if relative.is_file() else None
    for root in (OUT, EVIDENCE):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None
