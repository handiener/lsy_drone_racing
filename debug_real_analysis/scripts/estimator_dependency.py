"""Load the exact lockfile estimator wheel into the diagnostic process, without installing it."""

from __future__ import annotations
from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
import sys
import urllib.request
from paths import OUT

WHEEL_NAME = "drone_estimators-0.1.0-py3-none-any.whl"
WHEEL_SHA256 = "e0f3be005863e2df7ba0a354815f85c2ee5a93d92a59efef6bab6f73db565a6e"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/1a/44/9c4a3aa8d79ed2b030a19842c8e416a8c7f2fed42fbf0b7a0764ec6c9765/"
    + WHEEL_NAME
)


@lru_cache(maxsize=1)
def load_diff_filter():
    override = os.environ.get("DRONE_ESTIMATOR_WHEEL")
    wheel = Path(override).expanduser().resolve() if override else OUT / "dependencies" / WHEEL_NAME
    if not wheel.is_file():
        if override:
            raise FileNotFoundError(f"DRONE_ESTIMATOR_WHEEL does not exist: {wheel}")
        try:
            with urllib.request.urlopen(WHEEL_URL, timeout=30) as response:
                content = response.read(1024 * 1024)
        except OSError as error:
            raise RuntimeError(
                "Could not download the pinned estimator wheel. Download the URL in "
                "estimator_dependency.py and set DRONE_ESTIMATOR_WHEEL to that file."
            ) from error
        if hashlib.sha256(content).hexdigest() != WHEEL_SHA256:
            raise ValueError("Downloaded estimator wheel does not match the project lockfile hash.")
        wheel.parent.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(content)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError(f"Estimator wheel does not match the project lockfile hash: {wheel}")
    existing = sys.modules.get("drone_estimators")
    if existing is not None and not str(getattr(existing, "__file__", "")).startswith(str(wheel)):
        raise RuntimeError(
            "A different estimator package is already imported; run this diagnostic in a fresh Python process."
        )
    sys.path.insert(0, str(wheel))
    return importlib.import_module("drone_estimators.estimator_diff").DiffFilter
