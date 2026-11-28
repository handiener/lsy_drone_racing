"""Reproduce the bundled reference with the original staged optimization recipe."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import toml
from scipy.spatial.transform import Rotation

from .model import ROOT, reduced_seed, solve_model

DESCRIPTION = (
    "Offline reduced-dynamics guide. This simplified route has known simulator geometry "
    "contacts; it is not a collision-free or optimality certificate. "
    "Training uses only its geometry."
)


def refine_initial_guess(solution: dict, nodes: int, scale: float) -> list[np.ndarray]:
    """Resample, slow down and reconstruct the historical normalized-Euler warm start."""
    legs = len(solution["leg_times"])
    old = (len(solution["t"]) - 1) // legs
    dt = np.diff(solution["t"])[::old] * old / nodes
    pos, vel, thrust = (np.asarray(solution[k]).T for k in ("pos", "vel", "thrust"))
    positions, velocities, forces = [], [], []
    s0, s1 = np.linspace(0, 1, old + 1), np.linspace(0, 1, nodes + 1)

    def interp(array: np.ndarray) -> np.ndarray:
        return np.array([np.interp(s1, s0, row) for row in array])

    for k in range(legs):
        positions.append(interp(pos[:, k * old : (k + 1) * old + 1]))
        velocities.append(interp(vel[:, k * old : (k + 1) * old + 1]))
        force = np.c_[
            thrust[:, k * old : (k + 1) * old],
            thrust[:, min((k + 1) * old, len(solution["thrust"]) - 1)],
        ]
        forces.append(interp(force)[:, :-1])
    dt *= scale
    velocities = [v / scale for v in velocities]
    hover = np.array([[0], [0], [9.81 * solution["settings"]["mass"]]])
    forces = [(f - hover) / scale**2 + hover for f in forces]
    magnitudes = [np.linalg.norm(f, axis=0)[None] for f in forces]
    all_force = np.hstack(forces)
    direction = all_force / np.linalg.norm(all_force, axis=0)
    steps = np.repeat(dt, nodes)
    rates = []
    for i in range(direction.shape[1] - 1):
        a, b = direction[:, i], direction[:, i + 1]
        rates.append(np.cross(a, b) / (np.dot(a, b) * steps[i]))
    rates.append(rates[-1])
    rates = np.array(rates).T
    last = direction[:, -1] + steps[-1] * np.cross(rates[:, -1], direction[:, -1])
    direction = np.c_[direction, last / np.linalg.norm(last)]
    rates = np.c_[rates, rates[:, -1]]
    acceleration = np.diff(rates) / steps
    directions = [direction[:, k * nodes : (k + 1) * nodes + 1] for k in range(legs)]
    rates = [rates[:, k * nodes : (k + 1) * nodes + 1] for k in range(legs)]
    acceleration = [acceleration[:, k * nodes : (k + 1) * nodes] for k in range(legs)]
    return [dt, *positions, *velocities, *forces, *directions, *rates, *acceleration, *magnitudes]


def make_asset(solution: dict, config: dict, tilt_limit_deg: float = 80.0) -> dict:
    """Sample each constant-acceleration leg at 33 times, as in the original export."""
    if not solution["converged"] or solution["max_constraint_violation"] > 1e-5:
        raise ValueError("Refusing to export an unconverged or infeasible optimizer result")
    track = config["env"]["track"]
    p, v, f, t = (np.asarray(solution[k]) for k in ("pos", "vel", "thrust", "t"))
    if not all(np.isfinite(a).all() for a in (p, v, f, t)) or not np.all(np.diff(t) > 0):
        raise ValueError("Optimizer output must be finite with increasing times")
    nodes = (len(t) - 1) // len(track["gate_order"])
    mass = solution["settings"]["mass"]
    points = []
    for k in range(len(track["gate_order"])):
        samples = np.linspace(t[k * nodes], t[(k + 1) * nodes], 33)
        index = np.minimum(np.searchsorted(t, samples, side="right") - 1, (k + 1) * nodes - 1)
        index = np.maximum(index, k * nodes)
        elapsed = (samples - t[index])[:, None]
        points.append(
            p[index]
            + v[index] * elapsed
            + 0.5 * (f[index] / mass - [0, 0, 9.81]) * elapsed * elapsed
        )
    return dict(
        description=DESCRIPTION,
        gate_order=track["gate_order"],
        start_pos=track["drones"][0]["pos"],
        gates_pos=[g["pos"] for g in track["gates"]],
        gates_quat=Rotation.from_euler("xyz", [g["rpy"] for g in track["gates"]])
        .as_quat()
        .tolist(),
        model=dict(
            mass_kg=mass,
            thrust_max_N=solution["settings"]["thrust_max"],
            tilt_limit_deg=(
                int(tilt_limit_deg) if float(tilt_limit_deg).is_integer() else tilt_limit_deg
            ),
            thrust_direction_rate_limit_rad_s=8,
            thrust_direction_acceleration_limit_rad_s2=160,
            point_clearance_margin_m=0.05,
            estimated_time_s=solution["lap_time"],
            nodes_per_leg=nodes,
        ),
        points=np.round(points, 7).tolist(),
    )


def check_asset(asset: dict, reference: dict) -> dict[str, Any]:
    """Check geometry and timing against an existing reference, allowing solver roundoff."""
    for key in ("gate_order", "start_pos", "gates_pos"):
        if asset[key] != reference[key]:
            raise ValueError(f"Reference comparison has different {key}")
    if not np.allclose(asset["gates_quat"], reference["gates_quat"], atol=1e-12, rtol=0):
        raise ValueError("Reference comparison has different gates_quat")
    for key, value in asset["model"].items():
        if key != "estimated_time_s" and value != reference["model"][key]:
            raise ValueError(f"Reference comparison has different model.{key}")
    a, b = np.asarray(asset["points"]), np.asarray(reference["points"])
    if a.shape != b.shape:
        raise ValueError("Reference comparison has different point counts")
    error = float(np.linalg.norm(a - b, axis=-1).max())
    time_error = abs(asset["model"]["estimated_time_s"] - reference["model"]["estimated_time_s"])
    if error > 1e-4 or time_error > 1e-4:
        raise ValueError(
            f"Reference mismatch: maximum point error={error:.6g} m, time error={time_error:.6g} s"
        )
    return dict(max_point_error_m=error, time_error_s=time_error)


def generate(
    config: Path,
    work_dir: Path,
    tilt_limit_deg: float = 80.0,
    verbose: bool = False,
    max_cpu_time: float = 600.0,
) -> dict:
    """Run the original point-mass -> reduced seed -> exact 12 -> exact 24 recipe."""
    if not 0 < tilt_limit_deg < 90:
        raise ValueError("tilt_limit_deg must be in (0, 90)")
    if max_cpu_time <= 0:
        raise ValueError("max_cpu_time must be positive")
    problem = toml.load(config)
    if any(g["rpy"][:2] != [0.0, 0.0] for g in problem["env"]["track"]["gates"]):
        raise ValueError("The historical optimizer assumes upright gates")
    options = dict(config=config, theta_max_deg=tilt_limit_deg, verbose=verbose)

    def save(name: str, result: dict) -> None:
        (work_dir / f"{name}.json").write_text(json.dumps(result))
        print(f"{name}: {result['lap_time']:.12f} s", flush=True)

    print("1/4: point mass, 24 intervals per leg", flush=True)
    point = solve_model(**options, nodes=24, rate_max=None, max_iter=5000, max_cpu_time=None)
    save("point24", point)
    print("2/4: reduced 12-interval seed, direction rate 40 / acceleration 1000", flush=True)
    seed = reduced_seed(point, config, tilt_limit_deg, verbose, max_cpu_time)
    # Carry the same physical parameters into the subsequent warm start.
    seed["settings"] = point["settings"]
    save("seed12", seed)
    print("3/4: full reduced-attitude constraints at 12 intervals, time scale 1.8", flush=True)
    coarse = solve_model(
        **options,
        nodes=12,
        initial_guess=refine_initial_guess(seed, 12, 1.8),
        max_cpu_time=max_cpu_time,
    )
    save("attitude12", coarse)
    print("4/4: refine to 24 intervals, time scale 1.08", flush=True)
    fine = solve_model(
        **options,
        nodes=24,
        initial_guess=refine_initial_guess(coarse, 24, 1.08),
        max_cpu_time=max_cpu_time,
    )
    save("attitude24", fine)
    return make_asset(fine, problem, tilt_limit_deg)


def main() -> None:
    """Generate a new JSON; optionally verify that it reproduces the bundled guide."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/level2.toml")
    parser.add_argument("--tilt-limit-deg", type=float, default=80.0)
    parser.add_argument("--work-dir", type=Path, help="Keep intermediate optimizer solutions here")
    parser.add_argument("--check", action="store_true", help="Compare with the bundled JSON")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-cpu-time", type=float, default=600.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.work_dir is not None:
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            raise FileExistsError(f"Use an empty work directory: {args.work_dir}")
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rl_reference_") as temporary:
        asset = generate(
            args.config.resolve(),
            args.work_dir or Path(temporary),
            args.tilt_limit_deg,
            args.verbose,
            args.max_cpu_time,
        )
    text = json.dumps(asset, indent=2) + "\n"
    if args.check:
        bundled = Path(__file__).resolve().parent.parent / "dynamic_reference.json"
        comparison = check_asset(asset, json.loads(bundled.read_text()))
        print(
            json.dumps(dict(**comparison, byte_identical=text.encode() == bundled.read_bytes())),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(
        f"Wrote {args.output}; estimated time {asset['model']['estimated_time_s']:.12f} s",
        flush=True,
    )
