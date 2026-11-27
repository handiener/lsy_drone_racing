"""Fine-tuning must keep each selected policy's problem and use its actual final stage."""

import json
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import wandb

from lsy_drone_racing.control.rl_race import finetune_sim_to_real as finetune
from lsy_drone_racing.control.rl_race import train as training
from lsy_drone_racing.control.rl_race.network import NetworkConfig


@pytest.mark.parametrize(
    "reference,stage,tilt", [("hermite", "anneal", 80), ("dynamic", "race", 85)]
)
@pytest.mark.parametrize("from_file", [False, True])
def test_selected_model_keeps_problem_and_final_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
    stage: str,
    tilt: int,
    from_file: bool,
):
    source = tmp_path / "selected_model_s2"
    (source / "checkpoints").mkdir(parents=True)
    checkpoint = source / "checkpoints" / f"{stage}_final.pkl"
    checkpoint.touch()
    # Recorded absolute paths can be stale after moving a run into the league pool.
    (source / "final_evaluation.json").write_text(
        json.dumps(dict(stage=stage, checkpoint="/old/location/anneal_final.pkl"))
    )
    cfg = training.env_config(
        "race", tilt, tilt, 70, 0.75, 1250, race_reward={"time": 0.025}, reference_path=reference
    )
    (source / f"{stage}_config.json").write_text(
        json.dumps(
            dict(
                env=asdict(cfg),
                network=NetworkConfig().metadata(),
                freq=50,
                settings=dict(
                    discounting=0.9975, gae_lambda=0.98, run_name="old", output_root="/old"
                ),
            )
        )
    )
    calls, evaluations, exports = [], [], []

    @wraps(training.train)
    def fake_train(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "fine_tuned")

    monkeypatch.setattr(finetune, "train", fake_train)
    monkeypatch.setattr(finetune, "evaluate", lambda *a, **kw: evaluations.append(kw))
    monkeypatch.setattr(finetune, "export", lambda *a, **kw: exports.append(kw))
    finetune.run(str(checkpoint if from_file else source))
    options = calls[0]
    assert options["start_params"] == str(checkpoint)
    assert options["start_stage"] == "race" and options["continue_race"]
    assert options["reference_path"] == reference
    assert options["roll_limit_deg"] == options["pitch_limit_deg"] == tilt
    assert options["race_time"] == 0.025
    assert options["official_start_prob"] == 0.75
    assert options["discounting"] == 0.9975 and options["gae_lambda"] == 0.98
    assert options["policy_layers"] == (256, 256, 256)
    assert options["sim_to_real"] == "estimator_mix"
    assert options["sim_to_real_nominal_prob"] == 0.35
    assert options["anneal_steps"] == 0
    assert options["race_steps"] == 512 * 4_194_304
    assert options["eval_every_steps"] == 128 * 4_194_304
    assert options["policy_lr_after_switch"] == 2e-5
    assert options["learning_rate"] == 1e-4
    assert options["lr_decay_to"] == 0.5
    assert options["entropy_cost"] == 0.001
    assert not options["intermediate_eval"]
    assert options["final_eval"] and options["save_step_checkpoints"]
    assert evaluations[0]["sim_to_real"] == "none"
    assert evaluations[0]["report_name"] == "nominal"
    assert exports[0]["destination"] == str(tmp_path / "fine_tuned/exported")


def test_race_continuation_restores_critic_and_installs_mixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "source.pkl"
    training.save_params((1, 2, 3), checkpoint)
    configs, calls = [], []
    monkeypatch.setattr(wandb, "run", None)

    def fake_env(n: int, cfg: Any, **kwargs: Any) -> SimpleNamespace:
        configs.append(cfg)
        return SimpleNamespace(
            freq=50, action_cfg=SimpleNamespace(thrust_min=0.085, thrust_max=0.8)
        )

    def fake_train(**kwargs: Any) -> tuple:
        calls.append(kwargs)
        kwargs["policy_params_fn"](100, None, (1, 2, 3))
        return None, (1, 2, 3), {}

    monkeypatch.setattr(training, "RaceEnv", fake_env)
    monkeypatch.setattr(training.ppo, "train", fake_train)
    run = Path(
        training.train(
            run_name="mixed",
            output_root=str(tmp_path),
            start_params=str(checkpoint),
            start_stage="race",
            continue_race=True,
            reference_path="dynamic",
            sim_to_real="estimator_mix",
            intermediate_eval=False,
            final_eval=False,
            save_step_checkpoints=True,
        )
    )
    assert calls[0]["restore_params"] == (1, 2, 3)
    assert "restore_value_fn" not in calls[0]
    assert not calls[0]["run_evals"]
    assert all(c.sim_to_real.preset == "estimator_mix" for c in configs)
    assert all(c.reference_path == "dynamic" for c in configs)
    assert (run / "checkpoints/race_step_000000000100.pkl").is_file()
    assert (run / "checkpoints/race_final.pkl").is_file()


@pytest.mark.parametrize("option", [dict(sim_to_real="none"), dict(steps=1), dict(steps=0)])
def test_invalid_finetune_fails_before_creating_output(tmp_path: Path, option: dict):
    with pytest.raises(ValueError):
        finetune.run("nonexistent", output_root=str(tmp_path / "outputs"), **option)
    assert not (tmp_path / "outputs").exists()
