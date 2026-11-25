"""Stage evaluation controls must preserve reward switching and final evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import wandb

from lsy_drone_racing.control.rl_race import evaluation
from lsy_drone_racing.control.rl_race import train as training


@pytest.mark.parametrize("intermediate_eval", [False, True])
@pytest.mark.parametrize("stage_intervals", ["defaults", "pretrain_only", "explicit"])
def test_stage_evaluation_and_reward_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intermediate_eval: bool, stage_intervals: str
):
    """Pretrain can switch with the flag off; race/anneal obey it, final eval still runs."""
    calls = []
    schedules = []
    final_calls = []
    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(
        training,
        "RaceEnv",
        lambda n, cfg, **kwargs: SimpleNamespace(
            freq=50, action_cfg=SimpleNamespace(thrust_min=0.1, thrust_max=0.8)
        ),
    )

    def fake_train(**kwargs: Any) -> tuple:
        calls.append(kwargs["run_evals"])
        schedules.append((kwargs["num_evals"], kwargs["eval_every_steps"]))
        kwargs["policy_params_fn"](100, None, (1, 2, 3))
        if len(calls) == 1:
            assert kwargs["run_evals"]
            kwargs["progress_fn"](
                100,
                {
                    "eval/episode_reward": 1.0,
                    "eval/episode_official_done": 1.0,
                    "eval/episode_success": 1.0,
                    "eval/episode_lap_time": 3.0,
                },
            )
        return None, (1, 2, 3), {}

    monkeypatch.setattr(training.ppo, "train", fake_train)
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: final_calls.append(kw) or {})
    run = training.train(
        output_root=str(tmp_path),
        run_name="stage_eval",
        intermediate_eval=intermediate_eval,
        pretrain_eval_every_steps=20_000_000 if stage_intervals != "defaults" else None,
        race_eval_every_steps=800_000_000 if stage_intervals == "explicit" else None,
        anneal_steps=1_600_000_000,
    )
    assert calls == [True, intermediate_eval, intermediate_eval]
    # Race/anneal retain exactly the old training chunks and budget rounding. Only their
    # evaluation rollouts are skipped; pretrain can check its switching threshold more often.
    expected = {
        "defaults": [(26, 80_000_000), (151, 1_600_000_000), (21, 1_600_000_000)],
        "pretrain_only": [(101, 20_000_000), (151, 400_000_000), (21, 400_000_000)],
        "explicit": [(101, 20_000_000), (151, 800_000_000), (21, 800_000_000)],
    }
    assert schedules == expected[stage_intervals]
    stages = json.loads((tmp_path / "stage_eval" / "stages.json").read_text())
    assert [s["stage"] for s in stages] == ["pretrain", "race", "anneal"]
    assert stages[0]["switched_on_success"]
    assert len(final_calls) == 1
    assert final_calls[0]["stage"] == "anneal"
    assert run == str(tmp_path / "stage_eval")


@pytest.mark.parametrize("name", ["pretrain_eval_every_steps", "race_eval_every_steps"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_stage_interval_does_not_create_run(tmp_path: Path, name: str, value: int):
    with pytest.raises(ValueError, match=name):
        training.train(output_root=str(tmp_path), run_name="invalid", **{name: value})
    assert not (tmp_path / "invalid").exists()
