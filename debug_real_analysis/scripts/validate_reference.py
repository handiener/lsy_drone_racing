"""Reproduce the current reference's 42 exact/nearby-start and eight withdrawal episodes."""

from __future__ import annotations
import json
from dataclasses import asdict
import numpy as np
from harness import Runner, OUT
from initial_state import add_start_perturbations
from presets import PRESETS, REFERENCE_PRESET, REFERENCE_ABLATIONS


def main():
    dest = OUT / "reference"
    (dest / "trajectories").mkdir(parents=True, exist_ok=True)
    metrics_path = dest / "metrics.jsonl"
    rows = (
        [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if metrics_path.exists()
        else []
    )
    known = {r["run_id"]: r for r in rows}
    for scene in ["113735", "115247"]:
        runner = Runner(scene)
        add_start_perturbations(runner)
        jobs = [("ablation", name, cfg, 0) for name, cfg in REFERENCE_ABLATIONS.items()]
        jobs += [
            ("jitter", REFERENCE_PRESET, PRESETS[REFERENCE_PRESET], seed) for seed in range(21)
        ]
        try:
            for role, name, cfg, seed in jobs:
                run_id = f"{role}_{scene}_{name}_{seed:02d}"
                if run_id in known:
                    if known[run_id]["noise"] != asdict(cfg):
                        raise ValueError(f"Existing output has a different configuration: {run_id}")
                    continue
                metric, records = runner.run(cfg, seed)
                metric.update(
                    role=role,
                    preset=name,
                    run_id=run_id,
                    initial_position_sd_m=[0.005, 0.005, 0.001] if seed else [0, 0, 0],
                    initial_rotation_sd_deg=0.3 if seed else 0,
                )
                arrays = {
                    "t": np.asarray([r["t"] for r in records], np.float32),
                    "passed": np.asarray([r["passed"] for r in records], np.int16),
                }
                for field in [
                    "pos",
                    "quat",
                    "vel",
                    "ang_vel",
                    "seen_pos",
                    "seen_quat",
                    "seen_vel",
                    "seen_ang_vel",
                    "requested",
                    "applied",
                    "motor_thrust",
                    "rotor_rpm",
                ]:
                    shape = np.asarray(records[0][field]).shape
                    arrays[field] = np.asarray(
                        [r.get(field, np.full(shape, np.nan)) for r in records], np.float32
                    )
                for field in ["t", "pos", "quat", "vel", "ang_vel"]:
                    if not np.isfinite(arrays[field]).all():
                        raise ValueError(f"Nonfinite physical state: {run_id}/{field}")
                path = dest / "trajectories" / f"{run_id}.npz"
                np.savez_compressed(path, **arrays)
                metric["trajectory"] = str(path.relative_to(dest))
                rows.append(metric)
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(metric) + "\n")
                print(
                    run_id, "height", metric["g2g3_min_z"], "passed", metric["passed"], flush=True
                )
        finally:
            runner.env.close()
    (dest / "metrics.json").write_text(json.dumps(rows, indent=2))
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
