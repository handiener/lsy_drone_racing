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
def test_stage_evaluation_and_reward_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intermediate_eval: bool
):
    """Pretrain can switch with the flag off; race/anneal obey it, final eval still runs."""
    calls = []
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
        anneal_steps=100,
    )
    assert calls == [True, intermediate_eval, intermediate_eval]
    stages = json.loads((tmp_path / "stage_eval" / "stages.json").read_text())
    assert [s["stage"] for s in stages] == ["pretrain", "race", "anneal"]
    assert stages[0]["switched_on_success"]
    assert len(final_calls) == 1
    assert final_calls[0]["stage"] == "anneal"
    assert run == str(tmp_path / "stage_eval")
