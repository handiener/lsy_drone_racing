"""Offline export must preserve acceleration, reject invalid solves and protect cached guides."""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from lsy_drone_racing.control.rl_race.env_wrapper.reference_generator.generate import (  # noqa: E402
    check_asset,
    main,
    make_asset,
)


def example() -> tuple[dict, dict]:
    """A two-second constant horizontal acceleration with gravity cancelled by thrust."""
    mass = 0.0434
    solution = dict(
        converged=True,
        max_constraint_violation=0.0,
        lap_time=2.0,
        settings=dict(mass=mass, thrust_max=0.8),
        t=[0.0, 1.0, 2.0],
        pos=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vel=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        thrust=[[mass, 0.0, mass * 9.81], [mass, 0.0, mass * 9.81]],
    )
    track = dict(
        gate_order=[1],
        drones=[dict(pos=[0.0, 0.0, 0.0])],
        gates=[dict(pos=[2.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0])],
    )
    return solution, dict(env=dict(track=track))


def test_export_integrates_acceleration_between_nodes_and_at_final_endpoint():
    solution, config = example()
    asset = make_asset(solution, config)
    p = np.asarray(asset["points"])
    assert p.shape == (1, 33, 3)
    np.testing.assert_allclose(p[0, :, 0], 0.5 * np.linspace(0, 2, 33) ** 2, atol=5e-8)
    np.testing.assert_array_equal(p[0, :, 1:], 0.0)
    assert asset["model"]["tilt_limit_deg"] == 80
    assert isinstance(asset["model"]["tilt_limit_deg"], int)
    assert asset["model"]["nodes_per_leg"] == 2
    assert check_asset(asset, copy.deepcopy(asset))["max_point_error_m"] == 0


@pytest.mark.parametrize("failure", ["unconverged", "infeasible", "nan"])
def test_bad_optimizer_results_are_not_exported(failure: str):
    solution, config = example()
    if failure == "unconverged":
        solution["converged"] = False
    elif failure == "infeasible":
        solution["max_constraint_violation"] = 0.01
    else:
        solution["pos"][1][0] = float("nan")
    with pytest.raises(ValueError):
        make_asset(solution, config)


@pytest.mark.parametrize("change", ["point", "model", "gate_orientation"])
def test_reproduction_check_detects_changed_guide(change: str):
    solution, config = example()
    original = make_asset(solution, config)
    changed = copy.deepcopy(original)
    if change == "point":
        changed["points"][0][10][0] += 0.001
    elif change == "model":
        changed["model"]["thrust_max_N"] = 0.7
    else:
        changed["gates_quat"][0] = [0.0, 0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        check_asset(changed, original)


def test_existing_output_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "reference.json"
    target.write_text("existing reference")
    monkeypatch.setattr("sys.argv", ["reference_generator", "--output", str(target)])
    with pytest.raises(FileExistsError):
        main()
    assert target.read_text() == "existing reference"
