"""Solo dense -> solo sparse -> frozen-opponent league, using the vendored Brax PPO."""

import inspect
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import numpy as np

from . import brax_ppo_train as ppo
from . import checkpoint
from .env_wrapper import rewards
from .env_wrapper.actions import ActionConfig
from .env_wrapper.env import EnvConfig, RaceEnv
from .env_wrapper.multi import MultiRaceEnv
from .env_wrapper.observations import ObsConfig
from .env_wrapper.starts import StartConfig
from .league import League
from .network import NetworkConfig
from .optim import iteration_updates, lr_schedules, separate_lr

RL_DIR = Path(__file__).resolve().parent


class _SwitchReward(Exception):
    """Stop at a successfully evaluated stage boundary, preserving latest actor weights."""


def eval_summary(metrics: dict) -> dict:
    count = float(metrics.get("eval/episode_official_done", 0))
    success = float(metrics.get("eval/episode_success", 0))
    return {
        "eval/success_rate": success / count if count else 0.0,
        "eval/lap_time": float(metrics.get("eval/episode_lap_time", 0)) / success
        if success
        else 0.0,
    }


def train(
    run_name: str | None = None,
    output_root: str | None = None,
    seed: int = 1,
    device: str = "gpu",
    config_file: str | None = None,
    roll_limit_deg: float = 80.0,
    pitch_limit_deg: float = 80.0,
    yaw_limit_deg: float = 70.0,
    thrust_min_fraction: float = 0.0,
    thrust_max_fraction: float = 1.0,
    n_gates: int = 2,
    official_start_prob: float = 0.5,
    max_episode_steps: int = 1250,
    pretrain_max_steps: int = 2_013_265_920,
    race_steps: int = 2_516_582_400,
    multi_steps: int = 2_684_354_560,
    switch_success: float = 0.3,
    sparse_switch_success: float = 0.7,
    critic_only_iterations: int = 80,
    reset_critic: bool = True,
    start_params: str | None = None,
    start_stage: str = "pretrain",
    end_stage: str = "multi",
    stop_on_success: bool = True,
    reference_path: str = "hermite",
    solo_checkpoint: str | None = None,
    league_manifest: str | None = None,
    league_weights: str | None = None,
    fixed_opponent_fraction: float = 0.25,
    solo_config: str | None = None,
    multi_config: str | None = None,
    pretrain_crash: float = 20.0,
    pretrain_time: float = 0.2,
    pretrain_progress: float = 15.0,
    pretrain_path_lateral: float = 1.0,
    race_time: float = 0.05,
    race_crash: float = 20.0,
    race_gate: float = 5.0,
    race_lap: float = 10.0,
    race_lap_bonus: float = 0.0,
    race_lap_ref: float = 4.0,
    competition_progress: float = 0.1,
    ranking_reward: float = 0.1,
    proximity_penalty: float = 0.02,
    proximity_distance: float = 0.4,
    inter_collision_penalty: float = 25.0,
    outcome_reward: float = 1.0,
    num_envs: int = 16384,
    unroll_length: int = 32,
    batch_size: int = 4096,
    num_minibatches: int = 4,
    num_updates_per_batch: int = 4,
    learning_rate: float = 3e-4,
    policy_lr_after_switch: float = 1e-4,
    entropy_cost: float = 5e-3,
    noise_scale: float = 1.0,
    discounting: float = 0.99,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.2,
    max_grad_norm: float = 1.0,
    hidden_size: int = 256,
    policy_layers: tuple[int, ...] = (512, 512),
    value_layers: tuple[int, ...] = (512, 512),
    activation: str = "leaky_relu",
    eval_every_steps: int = 167_772_160,
    num_eval_envs: int = 128,
    intermediate_eval: bool = False,
    final_eval_episodes: int = 16,
    final_eval_seeds: tuple[int, ...] = (410000,),
    final_eval_batch_size: int = 16,
    snapshot_interval: int = 256,
    league_capacity: int = 64,
    solo_opponent_fraction: float = 0.0,
    sampling_exponent: float = 0.9,
    swap_starts: bool = True,
    deterministic_opponent: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
) -> str:
    """Train the selected curriculum stages; unmet success gates never silently advance.

    Resume restores model/normalizer, not Adam, RNG or recurrent rollout state.
    Solo stages always evaluate; intermediate_eval controls only league evaluation.
    """
    settings = {k: v for k, v in locals().copy().items()}
    if config_file:
        import yaml

        configured = yaml.safe_load(Path(config_file).read_text())
        signature = inspect.signature(train)
        if not isinstance(configured, dict) or set(configured) - set(signature.parameters):
            raise ValueError("Unknown train keys in config_file")
        if "config_file" in configured:
            raise ValueError("Nested configuration files are unsupported")
        overrides = {
            k: v
            for k, v in settings.items()
            if k != "config_file" and v != signature.parameters[k].default
        }
        return train(**{**configured, **overrides})
    order = ["pretrain", "race", "multi"]
    if start_stage not in order or end_stage not in order:
        raise ValueError("Invalid start_stage/end_stage")
    if order.index(start_stage) > order.index(end_stage):
        raise ValueError("start_stage must not follow end_stage")
    if reference_path not in {"hermite", "dynamic"}:
        raise ValueError("reference_path must be hermite or dynamic")
    if not 0 <= fixed_opponent_fraction <= 1:
        raise ValueError("fixed_opponent_fraction must be in [0, 1]")
    if start_stage != "pretrain" and not start_params:
        raise ValueError("Stage resume requires start_params")
    if start_stage == "multi" and not solo_checkpoint:
        raise ValueError("Multi resume needs frozen solo_checkpoint")
    if not (0 <= switch_success <= 1 and 0 <= sparse_switch_success <= 1):
        raise ValueError("Invalid stage thresholds")
    if n_gates != 2:
        raise ValueError("57D league observation requires n_gates=2")
    if (
        min(
            num_envs,
            batch_size,
            num_minibatches,
            num_updates_per_batch,
            unroll_length,
            eval_every_steps,
            num_eval_envs,
            max_episode_steps,
            snapshot_interval,
            final_eval_episodes,
            final_eval_batch_size,
        )
        < 1
        or not final_eval_seeds
    ):
        raise ValueError("Counts, horizons and seeds must be positive")
    if min(pretrain_max_steps, race_steps, multi_steps) < 1:
        raise ValueError("All stage budgets must be positive")
    if (batch_size * num_minibatches) % num_envs:
        raise ValueError("Sequence count must be divisible by num_envs")
    if not 0 <= solo_opponent_fraction <= 1 or sampling_exponent < 0 or league_capacity < 2:
        raise ValueError("Invalid league sampling settings")
    if (
        min(
            race_crash,
            competition_progress,
            ranking_reward,
            proximity_penalty,
            inter_collision_penalty,
            outcome_reward,
        )
        < 0
        or proximity_distance <= 0
    ):
        raise ValueError(
            "Competition coefficients must be nonnegative; proximity distance positive"
        )
    network = NetworkConfig(
        tuple(policy_layers), tuple(value_layers), activation, noise_scale, hidden_size
    )
    factory = network.factory()
    nets = factory(57, 4)
    run_name = run_name or datetime.now().strftime("recurrent_%Y%m%d_%H%M%S")
    root = Path(output_root).expanduser().resolve() if output_root else RL_DIR / "outputs"
    run = root / run_name
    run.mkdir(parents=True, exist_ok=False)
    settings.update(run_name=run_name, output_root=str(root))
    # Pin both track definitions for stage changes, evaluation and reproducibility.
    import shutil

    from .env_wrapper.env import DEFAULT_CONFIG as solo_default
    from .env_wrapper.multi import DEFAULT_CONFIG as multi_default

    for name, path in [
        ("solo_track.toml", solo_config or solo_default),
        ("multi_track.toml", multi_config or multi_default),
    ]:
        shutil.copy2(Path(path).expanduser(), run / name)
    settings.update(
        solo_config=str(run / "solo_track.toml"), multi_config=str(run / "multi_track.toml")
    )
    checkpoint.write_json(run / "settings.json", settings)
    import wandb

    wandb_run = wandb.run
    if wandb_project:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=run_name,
            dir=str(run),
            config=settings,
        )
    loaded = checkpoint.load(start_params, network.metadata()) if start_params else None
    params = loaded["params"] if loaded else None
    if (
        start_stage == "multi"
        and not league_manifest
        and loaded
        and loaded["league"].get("entries")
    ):
        checkpoint.write_json(run / "resume_league.json", loaded["league"])
        league_manifest = str(run / "resume_league.json")
    stages = order[order.index(start_stage) : order.index(end_stage) + 1]
    stage_log = []
    offset = int(loaded["steps"]) if loaded else 0
    settings["resume_from_steps"] = offset
    checkpoint.write_json(run / "settings.json", settings)
    pool = None
    transitions = batch_size * num_minibatches * unroll_length
    updates = iteration_updates(num_updates_per_batch, num_minibatches)
    started = time.time()
    try:
        for stage in stages:
            dense = stage == "pretrain"
            reward = rewards.RewardConfig(
                gate=race_gate,
                lap=race_lap,
                time=pretrain_time if dense else race_time,
                crash=pretrain_crash if dense else race_crash,
                progress=pretrain_progress if dense else 0.0,
                path_lateral=pretrain_path_lateral,
                lap_bonus=race_lap_bonus,
                lap_ref=race_lap_ref,
                competition_progress=competition_progress,
                ranking=ranking_reward,
                proximity=proximity_penalty,
                proximity_distance=proximity_distance,
                inter_collision=inter_collision_penalty,
                outcome=outcome_reward,
            )
            cfg = EnvConfig(
                max_episode_steps=max_episode_steps,
                hidden_size=hidden_size,
                reference_path=reference_path,
                action=ActionConfig(
                    roll_limit_deg,
                    pitch_limit_deg,
                    yaw_limit_deg,
                    thrust_min_fraction,
                    thrust_max_fraction,
                ),
                obs=ObsConfig(n_gates),
                reward=reward,
                start=StartConfig(official_prob=1.0 if stage == "multi" else official_start_prob),
            )
            eval_cfg = replace(cfg, start=StartConfig(official_prob=1.0))
            if stage == "multi":
                solo = checkpoint.load(solo_checkpoint, network.metadata())
                pool = League(
                    solo_checkpoint,
                    network.metadata(),
                    solo["env"]["action"],
                    run / "league",
                    league_capacity,
                    solo_opponent_fraction,
                    sampling_exponent,
                    league_manifest,
                    weights_dir=league_weights,
                    fixed_fraction=fixed_opponent_fraction,
                )
                env = MultiRaceEnv(
                    num_envs,
                    cfg,
                    pool,
                    nets,
                    settings["multi_config"],
                    device,
                    seed,
                    swap_starts,
                    snapshot_interval,
                    transitions,
                    deterministic_opponent=deterministic_opponent,
                )
                evaluation_env = MultiRaceEnv(
                    num_eval_envs,
                    eval_cfg,
                    pool,
                    nets,
                    settings["multi_config"],
                    device,
                    seed + 10000,
                    False,
                )
            else:
                env = RaceEnv(num_envs, cfg, settings["solo_config"], device, seed)
                evaluation_env = RaceEnv(
                    num_eval_envs, eval_cfg, settings["solo_config"], device, seed + 10000
                )
            try:
                if stage == "multi":
                    env.step_offset = offset
                env_record = asdict(cfg)
                env_record["action"] = asdict(env.action_cfg)
                if loaded and loaded["env"]["action"] != env_record["action"]:
                    raise ValueError("Resume action mapping differs from checkpoint")
                if pool and env_record["action"] != pool.action:
                    raise ValueError("Solo and multi action mappings differ")
                checkpoint.write_json(
                    run / f"{stage}_config.json",
                    dict(
                        stage=stage,
                        env=env_record,
                        network=network.metadata(),
                        freq=50,
                        settings=settings,
                        config_path=settings["multi_config" if stage == "multi" else "solo_config"],
                    ),
                )
                latest = {}
                actor_lr = learning_rate if dense and params is None else policy_lr_after_switch
                schedules = lr_schedules(
                    actor_lr,
                    learning_rate,
                    freeze_policy_steps=0 if params is None else critic_only_iterations * updates,
                )

                def save_checkpoint(label: str, current: Any, step: int) -> Path:
                    path = run / "checkpoints" / f"{stage}_{label}.pkl"
                    checkpoint.save(
                        current,
                        path,
                        network=network.metadata(),
                        env=env_record,
                        stage=stage,
                        steps=offset + step,
                        league=pool.metadata() if pool else None,
                        metadata=dict(
                            seed=seed,
                            run_id=run_name,
                            evaluation=latest.get("evaluation"),
                            settings=settings,
                        ),
                    )
                    return path

                def keep(step: int, _make: Any, current: Any) -> None:
                    latest.update(params=current, step=step)
                    save_checkpoint("latest", current, step)

                def progress(step: int, metrics: dict) -> None:
                    values = {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}
                    if "eval/episode_success" in values:
                        values.update(eval_summary(values))
                        latest["evaluation"] = dict(values)
                    opt_step = (step // transitions) * updates
                    values.update(
                        stage=float(order.index(stage)),
                        lr_policy=float(schedules[0](opt_step)),
                        lr_value=float(schedules[1](opt_step)),
                    )
                    record = dict(
                        stage=stage,
                        step=offset + step,
                        wall_seconds=time.time() - started,
                        metrics=values,
                    )
                    with (run / "metrics.jsonl").open("a") as out:
                        out.write(json.dumps(record, allow_nan=False) + "\n")
                    if wandb_run is not None:
                        wandb_run.log(values, step=offset + step)
                    if "eval/episode_success" in values:
                        lap = values["eval/lap_time"]
                        lap_text = f"{lap:.2f}s" if values["eval/success_rate"] > 0 else "--"
                        print(
                            f"[{stage} eval] step {offset + step:,} | "
                            f"success {values['eval/success_rate']:.1%} | "
                            f"gates {values.get('eval/episode_gates', 0):.2f} | "
                            f"crash {values.get('eval/episode_crashed', 0):.1%} | "
                            f"lap {lap_text} | {(time.time() - started) / 60:.1f} min",
                            flush=True,
                        )
                    threshold = switch_success if dense else sparse_switch_success
                    if (
                        stage != "multi"
                        and (stage != end_stage or stop_on_success)
                        and step > 0
                        and values.get("eval/success_rate", -1) >= threshold
                    ):
                        raise _SwitchReward

                steps = {"pretrain": pretrain_max_steps, "race": race_steps, "multi": multi_steps}[
                    stage
                ]
                print(
                    f"[{stage} start] budget {steps:,} transitions | "
                    f"{num_envs:,} worlds | {transitions:,} transitions/iteration | "
                    f"intermediate eval {'on' if stage != 'multi' or intermediate_eval else 'off'}",
                    flush=True,
                )
                kwargs = {}
                if params is not None:
                    kwargs.update(restore_params=params, restore_value_fn=not reset_critic)
                    kwargs["optimizer_override"] = separate_lr(
                        actor_lr,
                        learning_rate,
                        max_grad_norm=max_grad_norm,
                        freeze_policy_steps=critic_only_iterations * updates,
                    )
                switched = False
                try:
                    _, params, _ = ppo.train(
                        environment=env,
                        num_timesteps=steps,
                        wrap_env=False,
                        max_devices_per_host=1,
                        num_envs=num_envs,
                        episode_length=max_episode_steps,
                        num_eval_envs=num_eval_envs,
                        unroll_length=unroll_length,
                        batch_size=batch_size,
                        num_minibatches=num_minibatches,
                        num_updates_per_batch=num_updates_per_batch,
                        learning_rate=learning_rate,
                        entropy_cost=entropy_cost,
                        discounting=discounting,
                        gae_lambda=gae_lambda,
                        clipping_epsilon=clipping_epsilon,
                        max_grad_norm=max_grad_norm,
                        normalize_observations=True,
                        network_factory=factory,
                        seed=seed,
                        eval_env=evaluation_env,
                        num_evals=max(steps // eval_every_steps, 1) + 1,
                        iterations_per_epoch=snapshot_interval if stage == "multi" else None,
                        eval_every_steps=eval_every_steps if stage == "multi" else None,
                        deterministic_eval=True,
                        run_evals=stage != "multi" or intermediate_eval,
                        log_training_metrics=True,
                        progress_fn=progress,
                        policy_params_fn=keep,
                        **kwargs,
                    )
                except _SwitchReward:
                    switched = True
                    params = latest["params"]
                end = latest["step"]
                path = save_checkpoint("final", params, end)
                stage_log.append(
                    dict(
                        stage=stage,
                        steps=end,
                        switched_on_success=switched,
                        checkpoint=str(path),
                        reason="threshold_reached"
                        if switched
                        else (
                            "budget_complete"
                            if stage == "multi" or (stage == end_stage and not stop_on_success)
                            else "success_threshold_not_reached"
                        ),
                    )
                )
                offset += end
                checkpoint.write_json(run / "stages.json", stage_log)
                print(
                    f"[{stage} end] stage steps {end:,} | total steps {offset:,} | "
                    f"{stage_log[-1]['reason']} | checkpoint {path}",
                    flush=True,
                )
                if stage == "race" and switched:
                    solo_checkpoint = str(path)
                if stage != "multi" and not switched:
                    break
            finally:
                env.close()
                evaluation_env.close()
        if stage_log:
            from .evaluation import evaluate

            report = evaluate(
                str(run),
                stage=stage_log[-1]["stage"],
                episodes=final_eval_episodes,
                seeds=tuple(final_eval_seeds),
                batch_size=final_eval_batch_size,
                device=device,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"final/{k}": v for k, v in report.items() if isinstance(v, (float, int))},
                    step=offset,
                )
                wandb_run.summary.update(
                    dict(
                        stage_complete=stage_log[-1]["stage"] == end_stage,
                        failure_reason=stage_log[-1]["reason"],
                    )
                )
    finally:
        checkpoint.write_json(run / "stages.json", stage_log)
        if wandb_run is not None and wandb_project:
            wandb_run.finish(exit_code=bool(sys.exc_info()[0]))
    return str(run)


if __name__ == "__main__":
    fire.Fire(train)
