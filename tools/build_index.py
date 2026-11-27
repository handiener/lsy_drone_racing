"""Build runs.csv: one row per recorded flight, derived from the raw logs plus labels.csv.

Usage (from the repository root, any Python >= 3.10, standard library only):
    python tools/build_index.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "date", "run_id", "track", "policy", "model", "action_thrust_max", "outcome",
    "gates_passed", "gates_total", "steps", "duration_s", "max_x", "max_collective_pwm",
    "has_onboard", "vbat_start_v", "vbat_min_v", "notes",
]  # fmt: skip


def track_name(cfg: dict) -> str:
    """Name the track by its first gate (single-drone level 2 vs. new split-S course)."""
    gates = cfg["env"]["track"]["gates"]
    return "newtrack_splits" if gates[0]["pos"][0] == 2.0 else "level2_single"


def outside(pos: list[float], lo: list[float], hi: list[float]) -> bool:
    """Whether a position lies outside the axis-aligned safety box."""
    return any(p < a or p > b for p, a, b in zip(pos, lo, hi))


def host_stats(run: Path) -> dict:
    """Outcome, progress and command statistics from events.jsonl and metadata.json."""
    meta = json.loads((run / "metadata.json").read_text())
    limits = meta["config"]["env"]["track"]["safety_limits"]
    lo, hi = limits["pos_limit_low"], limits["pos_limit_high"]
    transitions, pwm, exceptions = [], [], []
    for line in (run / "events.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:  # truncated last line of an aborted recording
            continue
        kind = event["kind"]
        if kind == "transition":
            transitions.append(event)
        elif kind == "attitude_command_sent":
            pwm.append(event["collective_pwm"])
        elif kind == "exception":
            exceptions.append(event.get("type", "?"))
    row = {"track": track_name(meta["config"]), "gates_total": "", "gates_passed": 0}
    if not transitions:
        return row | {"outcome": "no_flight", "steps": 0}
    last = transitions[-1]
    obs = last["next_obs"]
    n_total = len(obs.get("gate_sequence", [])) or ""
    n_passed = obs["n_gates_passed"]
    if n_total and n_passed >= n_total:
        outcome = "finished"
    elif "KeyboardInterrupt" in exceptions:
        outcome = "ctrl_c"
    elif last["terminated"] and outside(obs["pos"], lo, hi):
        outcome = "safety_bound"
    elif last["terminated"]:
        outcome = "terminated_other"
    else:
        outcome = "other:" + ",".join(exceptions) if exceptions else "other"
    t0 = transitions[0]["step_start_ns"]
    return row | {
        "outcome": outcome,
        "gates_passed": n_passed,
        "gates_total": n_total,
        "steps": len(transitions),
        "duration_s": round((last["step_end_ns"] - t0) / 1e9, 2),
        "max_x": round(max(t["next_obs"]["pos"][0] for t in transitions), 2),
        "max_collective_pwm": max(pwm) if pwm else "",
    }


def battery(run: Path) -> dict:
    """Battery voltage at start and minimum while the motors run, from onboard.jsonl."""
    path = run / "onboard.jsonl"
    if not path.exists():
        return {"has_onboard": False}
    volts, motors_on = [], False
    for line in path.read_text().splitlines():
        try:
            values = json.loads(line).get("values", {})
        except json.JSONDecodeError:
            continue
        if "motor.m2" in values:
            motors_on = values["motor.m2"] > 15000
        if "pm.vbatMV" in values:
            volts.append((values["pm.vbatMV"] / 1000, motors_on))
    flying = [v for v, on in volts if on]
    return {
        "has_onboard": True,
        "vbat_start_v": round(volts[0][0], 2) if volts else "",
        "vbat_min_v": round(min(flying), 2) if flying else "",
    }


def main() -> None:
    """Write runs.csv next to labels.csv."""
    with (ROOT / "labels.csv").open() as f:
        labels = {row["run_id"]: row for row in csv.DictReader(f)}
    rows = []
    for run in sorted((ROOT / "runs").glob("*/deploy_*")):
        label = labels.get(run.name, {})
        row = {"date": run.parent.name, "run_id": run.name}
        row |= {k: label.get(k, "") for k in ("policy", "model", "action_thrust_max", "notes")}
        row |= host_stats(run) | battery(run)
        rows.append(row)
    missing = sorted(set(labels) - {r["run_id"] for r in rows})
    if missing:
        raise SystemExit(f"labels.csv lists runs that do not exist: {missing}")
    with (ROOT / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    unlabelled = [r["run_id"] for r in rows if not r["policy"]]
    print(f"{len(rows)} runs written to runs.csv; unlabelled: {unlabelled or 'none'}")


if __name__ == "__main__":
    main()
