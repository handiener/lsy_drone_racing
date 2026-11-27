"""Buffered, append-only deployment recording without ROS dependencies."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import queue
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot record {type(value).__name__}")


def file_identity(path: Path) -> dict:
    """Identify a source file or model artifact by content."""
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def code_identity(root: Path) -> dict:
    """Capture the checkout revision and whether it has local modifications."""
    try:

        def git(*args: str) -> str:
            return (
                subprocess.check_output(
                    ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=2
                )
                .decode()
                .strip()
            )

        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


class DeploymentRecorder:
    """Copy records into a bounded queue; serialize and flush on a writer thread.

    Queue overflow and write failures disable further recording, never flight control. The
    summary reports incomplete recordings. A process/power failure can lose buffered records;
    complete lines in events.jsonl remain readable without a final summary.
    """

    def __init__(self, output_dir: Path | str | None, metadata: dict, queue_size: int = 2048):
        """Create a new output directory and write metadata before connecting to hardware."""
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            output_dir = Path("runs") / f"deploy_{stamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: str | None = None
        self._dropped = 0
        self._written = 0
        self._closed = False
        self._start_ns = time.monotonic_ns()
        metadata = {
            **metadata,
            "format_version": 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "clock_anchor": {"wall_time_ns": time.time_ns(), "monotonic_ns": self._start_ns},
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, default=_json_default, indent=2) + "\n"
        )
        self._stream = (self.output_dir / "events.jsonl").open("x")
        self._thread = threading.Thread(target=self._write, name="deployment-recorder", daemon=True)
        self._thread.start()
        logger.info("Recording deployment to %s", self.output_dir.resolve())

    def record(self, kind: str, **payload: Any) -> None:
        """Snapshot mutable data immediately without waiting for disk I/O."""
        if self._closed:
            return
        if self._error is not None:
            self._dropped += 1
            return
        try:
            self._queue.put_nowait(
                {"kind": kind, "monotonic_ns": time.monotonic_ns(), **copy.deepcopy(payload)}
            )
        except Exception as exc:
            self._dropped += 1
            self._error = f"Could not enqueue record: {exc}"
            logger.error("Deployment recording disabled: %s", self._error)

    def _write(self) -> None:
        last_flush = time.monotonic()
        failed = False
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                item = ...
            try:
                if not failed:
                    if item is not ... and item is not None:
                        line = json.dumps(item, default=_json_default)
                        self._stream.write(line + "\n")
                        self._written += 1
                    if time.monotonic() - last_flush >= 0.2 or item is None:
                        self._stream.flush()
                        last_flush = time.monotonic()
                elif item is not ... and item is not None:
                    self._dropped += 1
            except Exception as exc:
                failed = True
                if item is not ... and item is not None:
                    self._dropped += 1
                self._error = f"Writer failed: {exc}"
                logger.error("Deployment recording disabled: %s", self._error)
            if item is None:
                break

    def close(self) -> None:
        """Drain the queue after hardware cleanup, preserving any flight exception."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        try:
            self._stream.close()
        except Exception as exc:
            self._error = f"Closing event stream failed: {exc}"
            logger.exception("Could not close deployment event stream")
        try:
            summary = {
                "records_written": self._written,
                "records_dropped": self._dropped,
                "recording_error": self._error,
                "complete": self._error is None,
            }
            (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        except Exception:
            logger.exception("Could not finalize deployment recording")
