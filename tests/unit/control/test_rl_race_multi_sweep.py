"""Sweep adapter runs one trial and attaches reproducible outputs without network calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsy_drone_racing.control.rl_race_multi import sweep
from lsy_drone_racing.control.rl_race_multi import train as training


class WandbSummary(dict):
    """Match W&B's mapping-only update signature."""

    def update(self, values: dict) -> None:
        """Update summary values using W&B's mapping-only API."""
        super().update(values)


@pytest.mark.parametrize("budget,uploaded", [(50, True), (1e-12, False)])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("stage", ["multi", "race"])
@pytest.mark.parametrize("base_config", [False, True])
def test_trial_forwards_config_and_publishes_final_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    budget: float,
    uploaded: bool,
    grouped: bool,
    stage: str,
    base_config: bool,
):
    calls = {}
    checkpoint = tmp_path / "race_final.pkl"
    checkpoint.write_bytes(b"test")
    report = {
        "stage": stage,
        "success_rate": 0.8,
        "mean_success_time": 4.0,
        "checkpoint": str(checkpoint),
    }
    (tmp_path / "final_evaluation.json").write_text(json.dumps(report))
    (tmp_path / f"{stage}_config.json").write_text("{}")
    (tmp_path / "stages.json").write_text("[]")
    (tmp_path / "settings.json").write_text("{}")

    def train(
        seed: int = 3,
        policy_layers: tuple[int, ...] = (32, 32),
        run_name: str | None = None,
        wandb_project: str | None = None,
        output_root: str | None = None,
        roll_limit_deg: float | None = None,
        pitch_limit_deg: float | None = None,
        end_stage: str = "multi",
    ) -> str:
        calls.update(seed=seed, policy_layers=policy_layers, run_name=run_name)
        assert roll_limit_deg == pitch_limit_deg == 75
        assert end_stage == stage
        assert output_root.endswith("sweeps/standalone/runs")
        return str(tmp_path)

    logs, attachments, finished = [], [], []
    run = SimpleNamespace(
        config={
            "seed": 3,
            "policy_layers": [32, 32],
            "cloud_checkpoint_budget_gb": budget,
            "tilt_limit_deg": 75,
            "end_stage": stage,
        },
        id="offline-test",
        log=lambda x: logs.append(x),
        summary=WandbSummary(),
        log_artifact=lambda x: None,
        finish=lambda exit_code: finished.append(exit_code),
    )
    if grouped:
        run.config["experiment"] = {
            "name": "tilt75",
            "tilt_limit_deg": run.config.pop("tilt_limit_deg"),
        }
    config_path = tmp_path / "shared.yaml"
    if base_config:
        config_path.write_text(
            "seed: 99\npolicy_layers: [64]\nend_stage: pretrain\n"
            "roll_limit_deg: 60\npitch_limit_deg: 60\n"
        )
        run.config["config_file"] = str(config_path)
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(sweep.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(
        sweep.wandb,
        "Artifact",
        lambda *a, **k: SimpleNamespace(add_file=lambda p: attachments.append(p)),
    )
    sweep.run_trial()
    assert calls == {"seed": 3, "policy_layers": (32, 32), "run_name": "offline-test_s3"}
    manifest = json.loads((tmp_path / "trial_manifest.json").read_text())
    assert manifest["experiment"] == ("tilt75" if grouped else None)
    assert manifest["config_file"] == (str(config_path) if base_config else None)
    if grouped:
        assert run.summary["experiment"] == "tilt75"
    assert logs[0]["final/success_rate"] == 0.8
    assert (str(checkpoint) in attachments) == uploaded
    assert len(attachments) == (5 if uploaded else 4)
    assert run.summary["checkpoint_uploaded"] == uploaded
    assert run.summary["checkpoint_bytes"] == len(b"test")
    assert run.summary["estimated_sweep_checkpoint_gb"] == len(b"test") / 1e9
    assert run.summary["training_complete"] and finished == [0]


def test_trial_failure_is_reported_to_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    finished = []
    run = SimpleNamespace(
        config={},
        id="failed",
        summary=WandbSummary(),
        finish=lambda exit_code: finished.append(exit_code),
    )
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setattr(sweep.wandb, "init", lambda **kwargs: run)

    def broken_train(
        output_root: str | None = None,
        run_name: str | None = None,
        wandb_project: str | None = None,
    ):
        raise RuntimeError("training or evaluation failed")

    monkeypatch.setattr(training, "train", broken_train)
    with pytest.raises(RuntimeError, match="training or evaluation failed"):
        sweep.run_trial()
    assert finished == [1]


def test_machine_directories_are_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(sweep, "RL_DIR", tmp_path)
    a = sweep.machine_environment("9-3-2")
    b = sweep.machine_environment("9-3-3")
    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        assert a[key] != b[key]
        assert Path(a[key]).is_dir()
    assert a["RL_RACE_MACHINE"] == "9-3-2"
    with pytest.raises(ValueError):
        sweep.machine_environment("../../escape")


def test_default_entrypoint_runs_assigned_trial(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(sweep, "run_trial", lambda: calls.append(True))
    sweep.main([])
    assert calls == [True]


def test_launcher_preserves_machine_and_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(sweep, "RL_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(sweep.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    with pytest.raises(SystemExit) as result:
        sweep.main(
            ["--machine", "9-3-2", "--sweep", "team/project/id", "--hours", "10", "--count", "3"]
        )
    assert result.value.code == 0
    cmd, options = calls[0]
    assert cmd[:4] == ["timeout", "--signal=INT", "--kill-after=120s", "10.0h"]
    assert cmd[-4:] == ["agent", "--count", "3", "team/project/id"]
    assert options["env"]["RL_RACE_MACHINE"] == "9-3-2"


def test_create_sweep_resolves_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(sweep, "RL_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(sweep.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    with pytest.raises(SystemExit) as result:
        sweep.main(["--machine", "9-3-2", "--create", "grid.yaml"])
    assert result.value.code == 0
    assert calls[0][-4:] == [
        "sweep",
        "--project",
        "lsy-drone-racing",
        str(sweep.Path("grid.yaml").resolve()),
    ]


@pytest.mark.parametrize("args", [["--machine", "9-3-2"], ["--sweep", "team/project/id"]])
def test_incomplete_launcher_cannot_start_a_trial(args: list[str]):
    with pytest.raises(SystemExit) as result:
        sweep.main(args)
    assert result.value.code == 2


@pytest.mark.parametrize("budget,uploaded", [(50, True), (1e-12, False)])
def test_league_artifact_budget_and_portable_paths(monkeypatch, tmp_path, budget, uploaded):
    main = tmp_path / "multi_final.pkl"
    main.write_bytes(b"learner")
    solo = tmp_path / "solo.pkl"
    solo.write_bytes(b"solo")
    history = tmp_path / "history.pkl"
    history.write_bytes(b"history")
    (tmp_path / "league").mkdir()
    (tmp_path / "league/manifest.json").write_text(
        json.dumps(
            dict(
                entries=[
                    dict(checkpoint=str(solo), solo=True),
                    dict(checkpoint=str(history), solo=False),
                ]
            )
        )
    )
    (tmp_path / "final_evaluation.json").write_text(
        json.dumps(dict(stage="multi", checkpoint=str(main), win_rate=0.5))
    )
    for file in ["multi_config.json", "settings.json", "stages.json"]:
        (tmp_path / file).write_text("{}")

    def fake_train(output_root=None, run_name=None, wandb_project=None):
        return str(tmp_path)

    summary = WandbSummary()
    attachments = []
    run = SimpleNamespace(
        config=dict(cloud_checkpoint_budget_gb=budget),
        id="league",
        summary=summary,
        log=lambda x: None,
        log_artifact=lambda x: None,
        finish=lambda **kw: None,
    )
    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr(sweep.wandb, "init", lambda **kw: run)
    monkeypatch.setattr(
        sweep.wandb,
        "Artifact",
        lambda *a, **kw: SimpleNamespace(
            add_file=lambda path, name=None: attachments.append((path, name))
        ),
    )
    sweep.run_trial()
    assert (
        summary["estimated_sweep_checkpoint_gb"]
        == (len(b"learner") + len(b"solo") + len(b"history")) / 1e9
    )
    assert summary["checkpoint_uploaded"] == uploaded
    portable = [path for path, name in attachments if name == "league_manifest.json"]
    if uploaded:
        entries = json.loads(Path(portable[0]).read_text())["entries"]
        assert [e["checkpoint"] for e in entries] == ["opponents/000.pkl", "opponents/001.pkl"]
        assert {name for _, name in attachments if name and name.startswith("opponents/")} == {
            "opponents/000.pkl",
            "opponents/001.pkl",
        }
    else:
        assert not portable
