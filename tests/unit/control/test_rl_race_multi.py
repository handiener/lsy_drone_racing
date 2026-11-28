"""Recurrent PPO mathematics, state boundaries and immutable league contracts."""

import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from brax.envs.base import State
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo.losses import PPONetworkParams

from lsy_drone_racing.control.rl_race_multi import checkpoint
from lsy_drone_racing.control.rl_race_multi.env_wrapper.env import EnvConfig
from lsy_drone_racing.control.rl_race_multi.env_wrapper.multi import outcome
from lsy_drone_racing.control.rl_race_multi.env_wrapper.observations import (
    ObsConfig,
    encode,
    encode_ego,
)
from lsy_drone_racing.control.rl_race_multi.league import League, probabilities
from lsy_drone_racing.control.rl_race_multi.network import (
    NetworkConfig,
    make_inference_fn,
    normalize,
    zeros,
)
from lsy_drone_racing.control.rl_race_multi.recurrent import (
    compute_ppo_loss,
    generate_unroll,
    sequence_apply,
)


def raw():
    return dict(
        pos=jnp.array([[1.0, 0.0, 1.0]]),
        quat=jnp.array([[0.0, 0.0, 0.0, 1.0]]),
        vel=jnp.array([[1.0, 0.0, 0.0]]),
        ang_vel=jnp.zeros((1, 3)),
        n_gates_passed=jnp.array([1]),
        gate_sequence=jnp.array([[0, 1, 2, 3]]),
        gate_sequence_direction=jnp.ones((1, 4)),
        gates_pos=jnp.ones((1, 4, 3)),
        gates_quat=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (1, 4, 1)),
        obstacles_pos=jnp.ones((1, 4, 3)),
    )


def model(hidden=8):
    config = NetworkConfig((16,), (16,), hidden_size=hidden)
    net = config.factory()(57, 4)
    stats = running_statistics.init_state(specs.Array((57,), jnp.float32))
    params = (
        stats,
        net.policy_network.init(jax.random.PRNGKey(1)),
        net.value_network.init(jax.random.PRNGKey(2)),
    )
    return config, net, params


class ToyRace:
    """Analytic, three-step episodes for testing recurrent sequence math without physics."""

    observation_size = 57
    action_size = 4

    def __init__(self, hidden=8):
        self.hidden = hidden

    def reset(self, rng):
        b = rng.shape[0]
        z = jnp.zeros(b)
        return State(
            pipeline_state=z,
            obs=jnp.ones((b, 57)) * 0.2,
            reward=z,
            done=z,
            metrics={},
            info=dict(
                recurrent=(zeros((b,), self.hidden), zeros((b,), self.hidden)),
                terminal_obs=jnp.zeros((b, 57)),
                truncation=z,
                steps=z,
                episode_done=z,
                episode_metrics=dict(sum_reward=z, length=z),
            ),
        )

    def step(self, state, action):
        step = state.pipeline_state + 1
        done = step >= 3
        terminal = jnp.broadcast_to((step / 10)[:, None], state.obs.shape)
        obs = jnp.where(done[:, None], 0.2, terminal)
        reward = action[:, 0] + 0.1 * step
        info = dict(
            state.info,
            terminal_obs=terminal,
            truncation=jnp.zeros_like(step),
            steps=step,
            episode_done=done.astype(float),
        )
        return state.replace(
            pipeline_state=jnp.where(done, 0, step),
            obs=obs,
            reward=reward,
            done=done.astype(float),
            info=info,
        )


def trajectory(length=5):
    config, net, params = model()
    env = ToyRace()
    state = env.reset(jax.random.split(jax.random.PRNGKey(4), 2))
    policy = make_inference_fn(net)(params)
    state, data = generate_unroll(
        env, state, policy, jax.random.PRNGKey(5), length, ("truncation",)
    )
    return config, net, params, env, state, data


def test_57d_preserves_ego_and_relative_features():
    a = raw()
    b = dict(
        a,
        pos=a["pos"] + jnp.array([[2.0, 3.0, 4.0]]),
        vel=a["vel"] + 2,
        n_gates_passed=jnp.array([3]),
    )
    solo = encode(a, jnp.zeros((1, 4)), ObsConfig())
    multi = encode(a, jnp.zeros((1, 4)), ObsConfig(), b)
    assert solo.shape == (1, 57)
    np.testing.assert_allclose(solo[:, :49], encode_ego(a, jnp.zeros((1, 4)), ObsConfig()))
    np.testing.assert_array_equal(solo[:, 49:], 0)
    np.testing.assert_allclose(multi[:, 49:], [[2, 3, 4, 2, 2, 2, 0.5, 1]])
    absent = encode(a, jnp.zeros((1, 4)), ObsConfig(), b, jnp.array([False]))
    np.testing.assert_array_equal(absent[:, 49:], 0)
    # Ego yaw +90 degrees: world +x becomes body -y.
    a["quat"] = jnp.array([[0.0, 0.0, jnp.sqrt(0.5), jnp.sqrt(0.5)]])
    rotated = encode(a, jnp.zeros((1, 4)), ObsConfig(), b)
    np.testing.assert_allclose(rotated[0, 49:52], [3, -2, 4], atol=1e-5)


def test_normalization_solo_to_multi_is_finite_and_masked():
    _, _, params = model()
    stats = params[0].replace(mean=jnp.zeros(57), std=jnp.zeros(57))
    obs = jnp.ones((2, 57)) * 10
    obs = obs.at[0, -1].set(0)
    result = normalize(obs, stats)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[0, 49:], 0)
    np.testing.assert_allclose(result[1, 49:55], 10.0)


def test_network_default_and_independent_lstm_parameters():
    default = NetworkConfig()
    assert default.hidden_size == 256 and default.policy_layers == (512, 512)
    _, net, params = model()
    carry = (zeros((2,), 8), zeros((2,), 8))
    hidden, action, extras = make_inference_fn(net)(params)(
        jnp.ones((2, 57)), carry, jax.random.PRNGKey(0)
    )
    assert action.shape == (2, 4) and extras["value"].shape == (2,)
    assert hidden[0][0].shape == (2, 8)
    assert not np.array_equal(
        params[1]["params"]["lstm"]["ii"]["kernel"], params[2]["params"]["lstm"]["ii"]["kernel"]
    )


def test_hidden_crosses_rollout_and_resets_inside_sequence():
    _, net, params, env, state, data = trajectory(2)
    actor_before = np.asarray(state.info["recurrent"][0][0])
    assert np.any(actor_before != 0)
    policy = make_inference_fn(net)(params)
    final, continued = generate_unroll(
        env, state, policy, jax.random.PRNGKey(8), 2, ("truncation",)
    )
    np.testing.assert_allclose(continued.extras["policy_extras"]["actor_c"][0], actor_before)
    # Third episode step terminates, so next step starts with zero carry.
    np.testing.assert_array_equal(continued.extras["policy_extras"]["actor_c"][1], 0)
    assert np.any(np.asarray(final.info["recurrent"][0][0]) != 0)


def test_recomputed_sequence_matches_behavior_logprob_and_bptt():
    _, net, params, _, _, data = trajectory()
    e = data.extras["policy_extras"]
    _, (logits, _) = sequence_apply(
        net.policy_network,
        params[0],
        params[1],
        data.observation,
        (e["actor_c"][0], e["actor_h"][0]),
        data.discount == 0,
    )
    logprob = net.parametric_action_distribution.log_prob(logits, e["raw_action"])
    np.testing.assert_allclose(logprob, e["log_prob"], atol=2e-6)
    np.testing.assert_allclose(logits, e["distribution_params"], atol=2e-6)

    # TBPTT detaches the saved initial state but gradients flow through time inside a segment.
    def initial_grad(initial):
        return sequence_apply(
            net.policy_network,
            params[0],
            params[1],
            data.observation,
            (initial, e["actor_h"][0]),
            data.discount == 0,
        )[1][0].sum()

    np.testing.assert_array_equal(jax.grad(initial_grad)(e["actor_c"][0]), 0)

    def temporal(obs):
        return sequence_apply(
            net.policy_network,
            params[0],
            params[1],
            obs,
            (e["actor_c"][0], e["actor_h"][0]),
            data.discount == 0,
        )[1][0][1].sum()

    assert np.linalg.norm(jax.grad(temporal)(data.observation)[0]) > 0


def test_ppo_sequence_loss_has_finite_gradients_and_updates_both_networks():
    _, net, params, _, _, data = trajectory()
    batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)
    networks = PPONetworkParams(params[1], params[2])

    def loss(p):
        return compute_ppo_loss(
            p,
            params[0],
            batch,
            jax.random.PRNGKey(9),
            net,
            discounting=0.99,
            entropy_cost=0.005,
            clipping_epsilon=0.2,
        )

    (value, metrics), grads = jax.value_and_grad(loss, has_aux=True)(networks)
    assert np.isfinite(value)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(grads.policy)) > 0
    assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(grads.value)) > 0
    optimizer = optax.adam(1e-3)
    update, _ = optimizer.update(grads, optimizer.init(networks))
    new = optax.apply_updates(networks, update)
    assert any(
        not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(networks), jax.tree.leaves(new))
    )
    # Brax KL adds 1e-5 inside log(scale/old_scale+eps), even for identical policies.
    np.testing.assert_allclose(metrics["kl_mean"], 4 * np.log1p(1e-5), atol=1e-6)


def test_sequence_shuffle_preserves_time_order():
    _, _, _, _, _, data = trajectory()
    batched = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)
    order = jax.random.permutation(jax.random.PRNGKey(10), 2)
    shuffled = jax.tree.map(lambda x: x[order], batched)
    for index in range(2):
        np.testing.assert_array_equal(
            shuffled.observation[index], batched.observation[order[index]]
        )
        np.testing.assert_array_equal(shuffled.discount[index], batched.discount[order[index]])


@pytest.mark.parametrize(
    "finished,crashed,expected",
    [
        ([[True, False]], [[False, False]], (1, 1, 0, 0)),
        ([[False, True]], [[False, False]], (1, 0, 1, 0)),
        ([[False, False]], [[False, True]], (1, 0, 0, 1)),
        ([[False, False]], [[True, True]], (1, 0, 0, 1)),
        ([[True, True]], [[False, False]], (1, 0, 0, 1)),
    ],
)
def test_outcomes_do_not_turn_opponent_crashes_into_wins(finished, crashed, expected):
    result = outcome(jnp.array(finished), jnp.array(crashed), jnp.array([False]))
    assert tuple(int(x[0]) for x in result) == expected


def test_checkpoint_league_validation_sampling_and_frozen_slots(tmp_path):
    config, net, params = model()
    env = asdict(EnvConfig(hidden_size=8))
    path = tmp_path / "solo.pkl"
    checkpoint.save(params, path, network=config.metadata(), env=env, stage="race", steps=4)
    restored = checkpoint.load(path, config.metadata(), env["action"])
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored["params"])):
        np.testing.assert_array_equal(a, b)
    pool = League(str(path), config.metadata(), env["action"], tmp_path / "league", capacity=3)
    np.testing.assert_allclose(probabilities(1, 3, 0.25, 0.9), [1, 0, 0])
    frozen = jax.tree.map(np.array, pool.packed())
    for i in (1, 2):
        item = tmp_path / f"history{i}.pkl"
        modified = (params[0], jax.tree.map(lambda x: x + i, params[1]), params[2])
        checkpoint.save(modified, item, network=config.metadata(), env=env, stage="multi", steps=i)
        assert pool.add(item)
    packed = pool.packed()
    for a, b in zip(
        jax.tree.leaves(frozen["banks"][0]["actor"]), jax.tree.leaves(packed["banks"][0]["actor"])
    ):
        np.testing.assert_array_equal(a[0], b[0])
    assert not pool.add(path)
    prob = np.asarray(probabilities(3, 3, 0.25, 0.9))
    assert prob[2] > prob[1] and np.isclose(prob[0], 0.25)
    assert np.asarray(pool.sample(jax.random.PRNGKey(0), 100)).max() < 3
    replay = League(
        str(path),
        config.metadata(),
        env["action"],
        tmp_path / "replay",
        capacity=3,
        manifest=str(tmp_path / "league/manifest.json"),
    )
    assert len(replay.entries) == 3
    with pytest.raises(ValueError, match="architecture"):
        checkpoint.load(path, NetworkConfig((16,), (16,), hidden_size=9).metadata())
    bad = dict(env["action"], roll_limit_deg=85)
    with pytest.raises(ValueError, match="action"):
        checkpoint.load(path, config.metadata(), bad)


@pytest.mark.parametrize("timeout,expected", [(True, 0.15625), (False, 0.0)])
def test_timeout_bootstraps_physical_terminal_not_reset_observation(timeout, expected):
    from brax.training import distribution, types
    from brax.training.networks import FeedForwardNetwork

    from lsy_drone_racing.control.rl_race_multi.network import RecurrentNetworks

    dist = distribution.NormalTanhDistribution(event_size=4)
    actor = FeedForwardNetwork(
        init=lambda _: None, apply=lambda s, p, o, c: (c, jnp.zeros((*o.shape[:-1], 8)))
    )
    critic = FeedForwardNetwork(init=lambda _: None, apply=lambda s, p, o, c: (c, o[..., 0]))
    net = RecurrentNetworks(actor, critic, dist, 1)
    # B=1,T=2: reset observation has value 100, physical terminal value is 2.
    obs = jnp.zeros((1, 2, 57))
    terminal = obs.at[0, -1, 0].set(2.0)
    logits = jnp.zeros((1, 2, 8))
    raw_action = jnp.zeros((1, 2, 4))
    hidden = jnp.zeros((1, 2, 1))
    data = types.Transition(
        obs,
        raw_action,
        jnp.zeros((1, 2)),
        jnp.array([[1.0, 0.0]]),
        obs.at[0, -1, 0].set(100.0),
        dict(
            policy_extras=dict(
                raw_action=raw_action,
                log_prob=dist.log_prob(logits, raw_action),
                distribution_params=logits,
                value=jnp.zeros((1, 2)),
                actor_c=hidden,
                actor_h=hidden,
                critic_c=hidden,
                critic_h=hidden,
            ),
            state_extras=dict(terminal_obs=terminal, truncation=jnp.array([[0.0, float(timeout)]])),
        ),
    )
    _, metrics = compute_ppo_loss(
        PPONetworkParams({}, {}),
        None,
        data,
        jax.random.PRNGKey(0),
        net,
        entropy_cost=0.0,
        discounting=0.5,
        gae_lambda=1.0,
        vf_coefficient=0.5,
    )
    np.testing.assert_allclose(metrics["v_loss"], expected, atol=1e-6)


def test_critic_reset_preserves_actor_and_normalizer():
    from lsy_drone_racing.control.rl_race_multi import brax_ppo_train as ppo

    config, net, params = model()
    stats = params[0].replace(mean=jnp.ones(57))
    params = (stats, params[1], params[2])
    _, restored, _ = ppo.train(
        ToyRace(),
        num_timesteps=0,
        wrap_env=False,
        num_envs=2,
        batch_size=2,
        num_minibatches=1,
        network_factory=config.factory(),
        restore_params=params,
        restore_value_fn=False,
        seed=100,
    )
    for a, b in zip(jax.tree.leaves(params[1]), jax.tree.leaves(restored[1])):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(restored[0].mean, 1)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(params[2]), jax.tree.leaves(restored[2]))
    )


@pytest.mark.parametrize("failure_stage", ["pretrain", "race"])
def test_unmet_solo_threshold_never_advances(
    tmp_path, monkeypatch, failure_stage, capsys: pytest.CaptureFixture[str]
):
    import wandb

    from lsy_drone_racing.control.rl_race_multi import evaluation
    from lsy_drone_racing.control.rl_race_multi import train as training

    config, net, params = model()
    calls = []

    class FakeEnvironment:
        def __init__(self, n, cfg, *args, **kwargs):
            self.cfg, self.action_cfg = cfg, cfg.action

        def close(self):
            pass

    monkeypatch.setattr(training, "RaceEnv", FakeEnvironment)
    monkeypatch.setattr(wandb, "run", None)

    def ppo(**kwargs):
        stage = "pretrain" if not calls else "race"
        calls.append(stage)
        assert kwargs["run_evals"] is True
        kwargs["policy_params_fn"](4, None, params)
        kwargs["progress_fn"](0, {"episode/policy_loss": 1.0})
        kwargs["progress_fn"](
            4,
            {
                "eval/episode_official_done": 1.0,
                "eval/episode_success": 0.0 if stage == failure_stage else 1.0,
            },
        )
        return None, params, {}

    monkeypatch.setattr(training.ppo, "train", ppo)
    evaluations = []
    monkeypatch.setattr(
        evaluation, "evaluate", lambda *a, **kw: evaluations.append(kw["stage"]) or {}
    )
    output = Path(
        training.train(
            output_root=str(tmp_path),
            run_name="budget",
            device="cpu",
            hidden_size=8,
            policy_layers=(16,),
            value_layers=(16,),
            num_envs=2,
            batch_size=2,
            num_minibatches=1,
            num_updates_per_batch=1,
            unroll_length=2,
            pretrain_max_steps=4,
            race_steps=4,
            multi_steps=4,
            eval_every_steps=4,
            intermediate_eval=False,
        )
    )
    stages = json.loads((output / "stages.json").read_text())
    assert stages[-1]["stage"] == failure_stage
    assert stages[-1]["reason"] == "success_threshold_not_reached"
    assert not stages[-1]["switched_on_success"]
    assert evaluations == [failure_stage]
    assert calls == (["pretrain"] if failure_stage == "pretrain" else ["pretrain", "race"])
    assert (output / "checkpoints" / f"{failure_stage}_final.pkl").exists()
    terminal = capsys.readouterr().out
    assert "[pretrain eval]" in terminal
    assert "episode/policy_loss" not in terminal
    assert "eval/episode_success" not in terminal
    assert "success_threshold_not_reached" in terminal
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert any("episode/policy_loss" in record["metrics"] for record in records)


def test_default_256_actor_critic_forward():
    config = NetworkConfig()
    networks = config.factory()(57, 4)
    stats = running_statistics.init_state(specs.Array((57,), jnp.float32))
    params = (
        stats,
        networks.policy_network.init(jax.random.PRNGKey(21)),
        networks.value_network.init(jax.random.PRNGKey(22)),
    )
    hidden, action, extra = make_inference_fn(networks)(params)(
        jnp.zeros((1, 57)), (zeros((1,), 256), zeros((1,), 256)), jax.random.PRNGKey(23)
    )
    assert action.shape == (1, 4) and extra["value"].shape == (1,)
    for leaf in jax.tree.leaves(hidden):
        assert leaf.shape == (1, 256)


def test_corrupt_checkpoint_and_misattributed_evaluation_are_rejected(tmp_path):
    import pickle

    from lsy_drone_racing.control.rl_race_multi.league import build_manifest

    config, _, params = model()
    path = tmp_path / "checkpoints/multi_final.pkl"
    checkpoint.save(
        params,
        path,
        network=config.metadata(),
        env=asdict(EnvConfig(hidden_size=8)),
        stage="multi",
        steps=4,
    )
    checkpoint.write_json(
        tmp_path / "final_evaluation.json",
        dict(checkpoint=str(path), success_rate=0.9, checkpoint_sha256="incorrect"),
    )
    with pytest.raises(ValueError, match="different checkpoint weights"):
        build_manifest((str(path),), str(tmp_path / "pool.json"))
    payload = checkpoint.load(path)
    payload["params"][1]["params"]["lstm"]["ii"]["kernel"] = np.zeros((56, 8))
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="shape"):
        checkpoint.load(path)


def test_pool_reuses_only_inactive_slots_and_ranks_by_recency(tmp_path):
    config, _, params = model()
    env = asdict(EnvConfig(hidden_size=8))
    files = []
    for i in range(5):
        path = tmp_path / f"{i}.pkl"
        checkpoint.save(
            (params[0], jax.tree.map(lambda x: x + i, params[1]), params[2]),
            path,
            network=config.metadata(),
            env=env,
            stage="race" if i == 0 else "multi",
            steps=i,
        )
        files.append(path)
    pool = League(str(files[0]), config.metadata(), env["action"], tmp_path / "pool", capacity=3)
    assert pool.add(files[1]) and pool.add(files[2])
    frozen = jax.tree.map(np.array, pool.packed())
    assert not pool.add(files[3], active_slots={0, 1, 2})
    assert pool.add(files[3], active_slots={0, 1})
    for a, b in zip(
        jax.tree.leaves(frozen["banks"][0]["actor"]),
        jax.tree.leaves(pool.packed()["banks"][0]["actor"]),
    ):
        np.testing.assert_array_equal(a[:2], b[:2])
    assert pool.add(files[4], active_slots={0, 2})
    p = np.asarray(probabilities(3, 3, 0.25, 0.9, pool.ranks()))
    assert p[1] > p[2]  # physical slot 1 is now newer than slot 2
    reloaded = League(
        str(files[0]),
        config.metadata(),
        env["action"],
        tmp_path / "reload",
        capacity=3,
        manifest=str(tmp_path / "pool/manifest.json"),
    )
    np.testing.assert_array_equal(reloaded.ranks(), pool.ranks())
    # Explicit resume into the same directory must read the whole manifest before writing.
    same = League(
        str(files[0]),
        config.metadata(),
        env["action"],
        tmp_path / "pool",
        capacity=3,
        manifest=str(tmp_path / "pool/manifest.json"),
    )
    assert [e["checkpoint"] for e in same.entries] == [e["checkpoint"] for e in pool.entries]
    np.testing.assert_array_equal(same.ranks(), pool.ranks())


def test_official_full_race_env_cannot_accidentally_feed_invalid_ppo_samples():
    from lsy_drone_racing.control.rl_race_multi import brax_ppo_train as ppo

    env = ToyRace()
    env.full_race = True
    with pytest.raises(ValueError, match="evaluation-only"):
        ppo.train(env, num_timesteps=4)


def test_warm_resume_retains_global_progress_but_starts_new_rollouts(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
):
    import wandb

    from lsy_drone_racing.control.rl_race_multi import evaluation
    from lsy_drone_racing.control.rl_race_multi import train as training
    from lsy_drone_racing.control.rl_race_multi.env_wrapper.actions import ActionConfig

    config, _, params = model()
    env_record = asdict(EnvConfig(hidden_size=8, action=ActionConfig(80, 80, 70)))
    solo = tmp_path / "solo.pkl"
    source = tmp_path / "source.pkl"
    checkpoint.save(params, solo, network=config.metadata(), env=env_record, stage="race", steps=90)
    checkpoint.save(
        params, source, network=config.metadata(), env=env_record, stage="multi", steps=100
    )
    monkeypatch.setattr(wandb, "run", None)

    class FakeMulti:
        def __init__(self, n, cfg, *args, **kwargs):
            self.action_cfg = cfg.action

        def close(self):
            pass

    monkeypatch.setattr(training, "MultiRaceEnv", FakeMulti)

    def ppo(**kwargs):
        for a, b in zip(jax.tree.leaves(params[1]), jax.tree.leaves(kwargs["restore_params"][1])):
            np.testing.assert_array_equal(a, b)
        kwargs["policy_params_fn"](4, None, params)
        kwargs["progress_fn"](4, {"training/total_loss": 1.0})
        return None, params, {}

    monkeypatch.setattr(training.ppo, "train", ppo)
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs: {})
    output = Path(
        training.train(
            run_name="resume",
            output_root=str(tmp_path),
            device="cpu",
            start_stage="multi",
            start_params=str(source),
            solo_checkpoint=str(solo),
            hidden_size=8,
            policy_layers=(16,),
            value_layers=(16,),
            num_envs=2,
            batch_size=2,
            num_minibatches=1,
            num_updates_per_batch=1,
            unroll_length=2,
            multi_steps=4,
            eval_every_steps=4,
            snapshot_interval=1,
        )
    )
    result = checkpoint.load(output / "checkpoints/multi_final.pkl")
    assert result["steps"] == 104
    assert json.loads((output / "settings.json").read_text())["resume_from_steps"] == 100
    terminal = capsys.readouterr().out
    assert "training/total_loss" not in terminal
    assert "[multi start]" in terminal and "[multi end]" in terminal
