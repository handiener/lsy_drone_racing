"""Solo stage selection and frozen heterogeneous-opponent inference contracts."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from lsy_drone_racing.control.rl_race.controller import policy_mean_action
from lsy_drone_racing.control.rl_race_multi import checkpoint
from lsy_drone_racing.control.rl_race_multi import league_pool as opponents
from lsy_drone_racing.control.rl_race_multi.env_wrapper.actions import ActionConfig, to_command
from lsy_drone_racing.control.rl_race_multi.env_wrapper.env import EnvConfig
from lsy_drone_racing.control.rl_race_multi.league import League
from lsy_drone_racing.control.rl_race_multi.network import NetworkConfig, zeros


def recurrent(path, hidden=8, stage="race", **metadata):
    config = NetworkConfig((16,), (16,), hidden_size=hidden)
    nets = config.factory()(57, 4)
    params = (
        running_statistics.init_state(specs.Array((57,), jnp.float32)),
        nets.policy_network.init(jax.random.PRNGKey(1)),
        nets.value_network.init(jax.random.PRNGKey(2)),
    )
    env = asdict(EnvConfig(hidden_size=hidden, action=ActionConfig(80, 80, 70)))
    return checkpoint.save(
        params, path, network=config.metadata(), env=env, stage=stage, steps=100, metadata=metadata
    )


def mlp(path, layers=(7,)):
    rng = np.random.default_rng(3)
    stats = running_statistics.init_state(specs.Array((49,), jnp.float32)).replace(
        mean=jnp.linspace(-0.3, 0.2, 49), std=jnp.linspace(0.5, 2.0, 49)
    )
    actor = dict(
        params={
            f"hidden_{i}": dict(
                kernel=rng.normal(0, 0.1, (a, b)).astype(np.float32),
                bias=rng.normal(0, 0.2, b).astype(np.float32),
            )
            for i, (a, b) in enumerate(zip([49, *layers], [*layers, 8]))
        }
    )
    path.write_bytes(pickle.dumps(jax.device_get((stats, actor, {}))))
    config = dict(
        env=dict(obs=dict(n_gates=2, n_obstacles=4), action=asdict(ActionConfig(85, 85, 70))),
        network=dict(
            policy_layers=list(layers),
            value_layers=list(layers),
            activation="tanh",
            distribution="tanh_normal",
            noise_scale=0.5,
        ),
        freq=50,
        stage="race",
        thrust_min=0.08545,
        thrust_max=0.8,
    )
    path.with_suffix(".json").write_text(json.dumps(config))
    return stats, actor, config


def test_mlp_pool_matches_single_controller_and_own_action_mapping(tmp_path):
    primary = tmp_path / "solo.pkl"
    model = recurrent(primary)
    weights = tmp_path / "weights"
    weights.mkdir()
    stats, actor, config = mlp(weights / "single.pkl")
    original = (weights / "single.pkl").read_bytes()
    pool = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "pool",
        capacity=4,
        weights_dir=weights,
    )
    assert len(pool.entries) == 2 and pool.entries[1]["fixed"] and pool.entries[1]["solo"]
    observations = jax.random.normal(jax.random.PRNGKey(8), (4, 57)) * 20
    carry = zeros((4,), pool.hidden_size)
    predict = jax.jit(
        lambda obs: pool.predict(
            pool.packed(), jnp.ones(4, jnp.int32), obs, carry, jax.random.PRNGKey(0)
        )
    )
    memory, action, command = predict(observations)
    layers = [actor["params"][k] for k in ["hidden_0", "hidden_1"]]
    expected = policy_mean_action(
        dict(mean=stats.mean, std=stats.std, layers=layers), observations[:, :49], "tanh"
    )
    np.testing.assert_allclose(action, expected, atol=2e-6)
    cfg = ActionConfig(**dict(config["env"]["action"], thrust_min=0.08545, thrust_max=0.8))
    np.testing.assert_allclose(command, to_command(expected, cfg), atol=2e-6)
    for leaf in jax.tree.leaves(memory):
        np.testing.assert_array_equal(leaf, 0)
    assert (weights / "single.pkl").read_bytes() == original
    # Run-local frozen copies include their old MLP sidecars and survive source removal.
    (weights / "single.pkl").unlink()
    (weights / "single.json").unlink()
    copied = opponents.load(pool.entries[1]["checkpoint"])
    assert copied["env"]["action"]["thrust_max"] == 0.8
    restored = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "restored",
        capacity=4,
        manifest=str(tmp_path / "pool/manifest.json"),
    )
    np.testing.assert_allclose(
        restored.predict(
            restored.packed(), jnp.ones(4, jnp.int32), observations, carry, jax.random.PRNGKey(0)
        )[2],
        command,
        atol=2e-6,
    )


def test_pool_different_recurrent_width_and_multi_observation_contract(tmp_path):
    primary = tmp_path / "solo.pkl"
    model = recurrent(primary)
    weights = tmp_path / "weights"
    weights.mkdir()
    foreign = recurrent(weights / "multi.pkl", hidden=12, stage="multi")
    pool = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "pool",
        capacity=4,
        weights_dir=weights,
    )
    assert pool.hidden_size == 12 and not pool.entries[1]["solo"]
    obs = jnp.ones((2, 57)).at[:, -1].set(1)
    carry = zeros((2,), 12)
    memory, action, command = jax.jit(pool.predict)(
        pool.packed(), jnp.array([0, 1]), obs, carry, jax.random.PRNGKey(9)
    )
    from lsy_drone_racing.control.rl_race_multi.controller import network_from_metadata

    for i, spec in enumerate([model, foreign]):
        nets = network_from_metadata(spec["network"]).factory()(57, 4)
        h = spec["network"]["hidden_size"]
        expected_memory, logits = nets.policy_network.apply(
            spec["params"][0], spec["params"][1], obs[i : i + 1], zeros((1,), h)
        )
        np.testing.assert_allclose(action[i], jnp.tanh(logits[0, :4]), atol=1e-6)
        for actual, expected in zip(memory, expected_memory):
            np.testing.assert_allclose(actual[i, :h], expected[0], atol=1e-6)
            np.testing.assert_array_equal(actual[i, h:], 0)
    assert np.isfinite(command).all()


def test_manual_opponents_remain_sampled_and_never_replaced(tmp_path):
    primary = tmp_path / "solo.pkl"
    model = recurrent(primary)
    weights = tmp_path / "weights"
    weights.mkdir()
    mlp(weights / "single.pkl")
    pool = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "pool",
        capacity=3,
        solo_fraction=0,
        fixed_fraction=0.4,
        weights_dir=weights,
    )
    np.testing.assert_allclose(pool.probabilities(pool.packed()), [0.5, 0.5, 0])
    history = tmp_path / "history.pkl"
    recurrent(history, stage="multi")
    pool.add(history)
    np.testing.assert_array_equal(pool.ranks(), [0, 0, 1])
    np.testing.assert_allclose(pool.probabilities(pool.packed()), [0.2, 0.2, 0.6])
    fixed = [dict(e) for e in pool.entries[:2]]
    assert not pool.add(history, active_slots={2})
    assert pool.add(history, active_slots=set())
    assert pool.entries[:2] == fixed


@pytest.mark.parametrize("solo_stage", ["pretrain", "race"])
def test_categorized_weights_keep_model_roles_sampling_and_portable_restore(tmp_path, solo_stage):
    primary = tmp_path / "primary.pkl"
    model = recurrent(primary)
    weights = tmp_path / "weights"
    for category in ["mlp", "lstm/solo", "lstm/multi"]:
        (weights / category).mkdir(parents=True)
    # Identical basenames in separate categories must remain distinct entries.
    mlp(weights / "mlp/best.pkl")
    recurrent(weights / "lstm/solo/best.pkl", hidden=10, stage=solo_stage)
    recurrent(weights / "lstm/multi/best.pkl", hidden=12, stage="multi")
    mlp(weights / "legacy.pkl")  # Flat-directory input remains supported.
    pool = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "pool",
        capacity=6,
        weights_dir=weights,
    )
    entries = {
        Path(e["source_checkpoint"]).relative_to(weights).as_posix(): e for e in pool.entries[1:]
    }
    assert set(entries) == {
        "mlp/best.pkl",
        "lstm/solo/best.pkl",
        "lstm/multi/best.pkl",
        "legacy.pkl",
    }
    assert entries["lstm/solo/best.pkl"]["solo"]
    assert entries["lstm/solo/best.pkl"]["stage"] == solo_stage
    assert not entries["lstm/multi/best.pkl"]["solo"]
    for name in ["mlp/best.pkl", "legacy.pkl"]:
        assert entries[name]["solo"] and pool.specs[entries[name]["bank"]]["kind"] == "mlp"
    assert pool.hidden_size == 12
    history = tmp_path / "history.pkl"
    recurrent(history, stage="multi")
    pool.add(history)
    expected = np.array([0.05] * 5 + [0.75])
    np.testing.assert_allclose(pool.probabilities(pool.packed()), expected)
    for path in weights.rglob("*"):
        if path.is_file():
            path.unlink()
    restored = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "restored",
        capacity=6,
        manifest=str(pool.directory / "manifest.json"),
    )
    assert len(restored.entries) == 6
    assert [e["solo"] for e in restored.entries] == [e["solo"] for e in pool.entries]
    np.testing.assert_allclose(restored.probabilities(restored.packed()), expected)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0])
def test_wrapper_sampling_frequencies_match_fixed_and_historical_shares(tmp_path, fraction):
    from lsy_drone_racing.control.rl_race_multi.env_wrapper.multi import MultiRaceEnv

    primary = tmp_path / "solo.pkl"
    model = recurrent(primary)
    weights = tmp_path / "weights"
    weights.mkdir()
    mlp(weights / "mlp_small.pkl")
    mlp(weights / "mlp_deep.pkl", layers=(9, 5))
    recurrent(weights / "multi.pkl", hidden=12, stage="multi")
    pool = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "pool",
        capacity=8,
        weights_dir=weights,
        fixed_fraction=fraction,
    )
    np.testing.assert_allclose(pool.probabilities(pool.packed()), [0.25] * 4 + [0.0] * 4)
    for i in range(2):
        history = tmp_path / f"history_{i}.pkl"
        recurrent(history, stage="multi")
        pool.add(history)
    relative = np.array([1.0, 2.0]) ** 0.9
    expected = np.r_[np.full(4, fraction / 4), (1 - fraction) * relative / relative.sum(), 0, 0]
    np.testing.assert_allclose(pool.probabilities(pool.packed()), expected, atol=1e-7)
    # Exercise the actual wrapper selector without allocating any physics worlds.
    env = object.__new__(MultiRaceEnv)
    env.n_envs, env.league, env.fixed_opponent = 100_000, pool, None
    indices = jax.jit(env._sample)(jax.random.PRNGKey(7), pool.packed())
    observed = np.bincount(np.asarray(indices), minlength=8) / env.n_envs
    np.testing.assert_allclose(observed, expected, atol=0.005, rtol=0)
    np.testing.assert_array_equal(observed[6:], 0)
    if fraction in (0.0, 1.0):
        np.testing.assert_array_equal(observed[expected == 0], 0)


def official_multi_raw():
    """Small official-format observation with different ego poses for the two ranks."""
    return dict(
        pos=np.array([[0.5, -1.0, 0.5], [0.7, -0.8, 0.6]], np.float32),
        quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (2, 1)),
        vel=np.zeros((2, 3), np.float32),
        ang_vel=np.zeros((2, 3), np.float32),
        n_gates_passed=np.array([0, 1]),
        gate_sequence=np.tile(np.arange(4), (2, 1)),
        gate_sequence_direction=np.ones((2, 4)),
        gates_pos=np.ones((2, 4, 3)),
        gates_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 4, 1)),
        obstacles_pos=np.ones((2, 4, 3)),
    )


def test_pool_mlp_official_controller_matches_training_for_each_rank_and_clock(tmp_path):
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.rl_race_multi.env_wrapper.observations import (
        ObsConfig,
        encode_ego,
    )
    from lsy_drone_racing.control.rl_race_multi.league_pool import brax_mlp
    from lsy_drone_racing.utils import load_controller

    primary = tmp_path / "solo.pkl"
    model = recurrent(primary)
    weights = tmp_path / "weights"
    weights.mkdir()
    mlp(weights / "single.pkl")
    pool = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "pool",
        capacity=3,
        weights_dir=weights,
    )
    # The simulator can use the run-local portable copy without the original JSON/source.
    source = pool.entries[1]["checkpoint"]
    (weights / "single.pkl").unlink()
    (weights / "single.json").unlink()
    controller_class = load_controller(Path(brax_mlp.__file__))
    config = ConfigDict(
        dict(
            controller=[dict(weights=source), dict(weights=source)],
            env=dict(
                kwargs=[
                    dict(freq=50, control_mode="attitude"),
                    dict(freq=100, control_mode="attitude"),
                ]
            ),
        )
    )
    raw = official_multi_raw()
    for rank in [0, 1]:
        ctrl = controller_class(raw, dict(rank=rank), config)
        ego = {k: jnp.asarray(v[rank])[None] for k, v in raw.items()}
        features = encode_ego(ego, jnp.zeros((1, 4)), ObsConfig())
        _, _, expected = pool.predict(
            pool.packed(),
            jnp.ones(1, jnp.int32),
            jnp.pad(features, [(0, 0), (0, 8)]),
            zeros((1,), pool.hidden_size),
            jax.random.PRNGKey(0),
        )
        command = ctrl.compute_control(raw)
        np.testing.assert_allclose(command, expected[0], atol=1e-6)
        if rank == 1:
            changed = dict(raw, pos=raw["pos"] + 0.1)
            np.testing.assert_array_equal(ctrl.compute_control(changed), command)
        ctrl.episode_reset()
        np.testing.assert_allclose(ctrl.compute_control(raw), command, atol=1e-6)
        assert ctrl.step_callback() is False
    # Single-drone observations use the same controller and the same ego encoder.
    config = ConfigDict(
        dict(controller=dict(weights=source), env=dict(freq=50, control_mode="attitude"))
    )
    single = {k: v[0] for k, v in raw.items()}
    ctrl = controller_class(single, {}, config)
    assert np.isfinite(ctrl.compute_control(single)).all()


def test_pool_can_load_and_infer_with_original_rl_race_package_blocked(tmp_path):
    import subprocess
    import sys
    import textwrap

    source = tmp_path / "single.pkl"
    mlp(source)
    program = textwrap.dedent("""
        import importlib.abc
        import sys
        class BlockOriginal(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "lsy_drone_racing.control.rl_race" or fullname.startswith(
                    "lsy_drone_racing.control.rl_race."):
                    raise ImportError("Original rl_race is unavailable")
        sys.meta_path.insert(0, BlockOriginal())
        import jax
        import jax.numpy as jnp
        import numpy as np
        from lsy_drone_racing.control.rl_race_multi import league_pool
        model = league_pool.load(sys.argv[1])
        stats, actor, _ = model["params"]
        bank = dict(mean=jnp.asarray(stats.mean)[None], std=jnp.asarray(stats.std)[None],
            actor=jax.tree.map(lambda x: jnp.asarray(x)[None], actor))
        _, action = league_pool.predict_bank(model, bank, jnp.zeros(2, jnp.int32),
            jnp.ones((2, 57)), (jnp.zeros((2, 8)), jnp.zeros((2, 8))),
            jax.random.PRNGKey(0), True)
        assert np.isfinite(action).all()
        assert "lsy_drone_racing.control.rl_race" not in sys.modules
        print("independent inference passed")
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "independent inference passed" in result.stdout


def test_old_manifest_is_not_extended_by_new_default_folder_files(tmp_path, monkeypatch):
    from lsy_drone_racing.control.rl_race_multi import league

    model = recurrent(tmp_path / "solo.pkl")
    pool = League(
        str(tmp_path / "solo.pkl"),
        model["network"],
        model["env"]["action"],
        tmp_path / "original",
        capacity=3,
    )
    # Simulate a later changed default folder. Evaluation must use the recorded pool.
    monkeypatch.setattr(league, "__file__", str(tmp_path / "league.py"))
    weights = tmp_path / "league_pool/weights"
    weights.mkdir(parents=True)
    (weights / "bad.pkl").write_bytes(b"new unrelated file")
    restored = League(
        str(tmp_path / "solo.pkl"),
        model["network"],
        model["env"]["action"],
        tmp_path / "restored",
        capacity=3,
        manifest=str(pool.directory / "manifest.json"),
    )
    assert len(restored.entries) == 1


def test_restore_preserves_sampling_when_fixed_files_have_identical_weights(tmp_path):
    primary = tmp_path / "solo.pkl"
    model = recurrent(primary)
    weights = tmp_path / "weights"
    weights.mkdir()
    recurrent(weights / "a.pkl", hidden=12, stage="multi")
    (weights / "b.pkl").write_bytes((weights / "a.pkl").read_bytes())
    pool = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "pool",
        capacity=4,
        weights_dir=weights,
    )
    history = tmp_path / "history.pkl"
    recurrent(history, stage="multi")
    pool.add(history)
    expected = np.asarray(pool.probabilities(pool.packed()))
    assert pool.entries[1]["checkpoint"] == pool.entries[2]["checkpoint"]
    (weights / "a.pkl").unlink()
    (weights / "b.pkl").unlink()
    restored = League(
        str(primary),
        model["network"],
        model["env"]["action"],
        tmp_path / "restored",
        capacity=4,
        manifest=str(pool.directory / "manifest.json"),
    )
    assert len(restored.entries) == 4
    np.testing.assert_array_equal(restored.probabilities(restored.packed()), expected)


def test_legacy_mlp_requires_metadata_and_rejects_wrong_clock(tmp_path):
    path = tmp_path / "single.pkl"
    _, _, config = mlp(path)
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="need single.json"):
        opponents.load(path)
    config["freq"] = 100
    path.with_suffix(".json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="50 Hz"):
        opponents.load(path)


@pytest.mark.parametrize("drones", [1, 2])
def test_default_track_comes_from_toml_without_physical_overrides(drones):
    from lsy_drone_racing.control.rl_race_multi.env_wrapper.env import DEFAULT_CONFIG, load_track
    from lsy_drone_racing.control.rl_race_multi.env_wrapper.multi import (
        DEFAULT_CONFIG as MULTI_CONFIG,
    )
    from lsy_drone_racing.utils import load_config

    path = MULTI_CONFIG if drones == 2 else DEFAULT_CONFIG
    before = path.read_bytes()
    expected = load_config(path).to_dict()
    expected["sim"]["render"] = False
    assert load_track(path, drones).to_dict() == expected
    assert path.read_bytes() == before


@pytest.mark.parametrize("drones", [1, 2])
@pytest.mark.parametrize("field,value", [("freq", 100), ("control_mode", "state")])
def test_incompatible_toml_control_settings_are_rejected(tmp_path, drones, field, value):
    import toml

    from lsy_drone_racing.control.rl_race_multi.env_wrapper.env import DEFAULT_CONFIG, load_track
    from lsy_drone_racing.control.rl_race_multi.env_wrapper.multi import (
        DEFAULT_CONFIG as MULTI_CONFIG,
    )

    document = toml.load(MULTI_CONFIG if drones == 2 else DEFAULT_CONFIG)
    agent = document["env"]["kwargs"][-1] if drones == 2 else document["env"]
    agent[field] = value
    path = tmp_path / "unsupported.toml"
    path.write_text(toml.dumps(document))
    with pytest.raises(ValueError, match="50 Hz attitude"):
        load_track(path, drones)


def test_thrust_mapping_preserves_fraction_metadata_across_stages():
    action = ActionConfig(80, 80, 70, thrust_min_fraction=0.1, thrust_max_fraction=0.9)
    mapped = action.with_thrust_bounds(0.08, 0.8)
    assert mapped.thrust_min_fraction == 0.1 and mapped.thrust_max_fraction == 0.9
    np.testing.assert_allclose(
        to_command(jnp.array([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]), mapped)[:, -1],
        [0.152, 0.728],
    )


def test_shared_recipe_keeps_python_defaults_and_sweeps_in_sync():
    import inspect

    import yaml

    from lsy_drone_racing.control.rl_race_multi.train import RL_DIR, train

    recipe = yaml.safe_load((RL_DIR / "configs/full.yaml").read_text())
    signature = inspect.signature(train)
    for key, value in recipe.items():
        if key == "run_name":
            continue
        default = signature.parameters[key].default
        assert (tuple(value) if isinstance(default, tuple) else value) == default, key
    for path in (RL_DIR / "sweeps").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert config["parameters"]["config_file"]["value"] == str(
            RL_DIR / "configs/full.yaml"
        ).removeprefix(str(RL_DIR.parents[2]) + "/")


def test_dynamic_reference_matches_single_and_saved_config_is_backward_compatible():
    from lsy_drone_racing.control.rl_race.env_wrapper import path as single
    from lsy_drone_racing.control.rl_race_multi.env_wrapper import path
    from lsy_drone_racing.control.rl_race_multi.env_wrapper.env import DEFAULT_CONFIG
    from lsy_drone_racing.control.rl_race_multi.evaluation import config_from_record
    from lsy_drone_racing.utils import load_config

    track = load_config(DEFAULT_CONFIG).env.track
    assert Path(single.__file__).with_name("dynamic_reference.json").read_bytes() == (
        Path(path.__file__).with_name("dynamic_reference.json").read_bytes()
    )
    asset = json.loads(Path(path.__file__).with_name("dynamic_reference.json").read_text())
    count = len(asset["gate_order"])
    positions = jnp.broadcast_to(jnp.array(asset["gates_pos"]), (count, len(asset["gates_pos"]), 3))
    quats = jnp.broadcast_to(jnp.array(asset["gates_quat"]), (count, len(asset["gates_quat"]), 4))
    order = jnp.array(track.gate_order)
    args = (
        positions,
        quats,
        jnp.abs(order) - 1,
        jnp.sign(order),
        jnp.arange(count),
        jnp.array(asset["start_pos"]),
    )
    actual = jax.jit(path.build_dynamic_polyline(track))(*args)
    expected = single.build_dynamic_polyline(track)(*args)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-6)
    saved = asdict(EnvConfig(reference_path="dynamic"))
    assert config_from_record(saved).reference_path == "dynamic"
    del saved["reference_path"]
    assert config_from_record(saved).reference_path == "hermite"


@pytest.mark.parametrize("end_stage,stop_on_success", [("pretrain", True), ("race", False)])
@pytest.mark.parametrize("resume", [False, True])
def test_selected_solo_stages_and_fixed_budget_do_not_start_multi(
    tmp_path, monkeypatch, end_stage, stop_on_success, resume
):
    import wandb

    from lsy_drone_racing.control.rl_race_multi import evaluation
    from lsy_drone_racing.control.rl_race_multi import train as training

    model = recurrent(tmp_path / "model.pkl", stage="pretrain")
    calls = []

    class Env:
        def __init__(self, n, cfg, *args, **kwargs):
            self.action_cfg = cfg.action

        def close(self):
            pass

    def ppo(**kwargs):
        calls.append(kwargs["num_timesteps"])
        for step in [4, 8]:
            kwargs["policy_params_fn"](step, None, model["params"])
            kwargs["progress_fn"](
                step, {"eval/episode_official_done": 1.0, "eval/episode_success": 1.0}
            )
        return None, model["params"], {}

    monkeypatch.setattr(training, "RaceEnv", Env)
    monkeypatch.setattr(training.ppo, "train", ppo)
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: {})
    monkeypatch.setattr(wandb, "run", None)
    run = Path(
        training.train(
            config_file=str(training.RL_DIR / "configs/smoke.yaml"),
            output_root=str(tmp_path),
            run_name="selected",
            end_stage=end_stage,
            stop_on_success=stop_on_success,
            start_params=str(tmp_path / "model.pkl") if resume else None,
            learning_rate=0.003,
            policy_lr_after_switch=0.00002,
            critic_only_iterations=0,
            pretrain_max_steps=8,
            race_steps=8,
            hidden_size=8,
            policy_layers=(16,),
            value_layers=(16,),
            num_envs=2,
            batch_size=2,
            num_minibatches=1,
            unroll_length=2,
        )
    )
    stages = json.loads((run / "stages.json").read_text())
    assert [s["stage"] for s in stages] == (
        ["pretrain"] if end_stage == "pretrain" else ["pretrain", "race"]
    )
    if not stop_on_success:
        assert stages[-1]["steps"] == 8 and stages[-1]["reason"] == "budget_complete"
    assert not (run / "league").exists()
    metrics = json.loads((run / "metrics.jsonl").read_text().splitlines()[0])["metrics"]
    assert metrics["lr_policy"] == pytest.approx(0.00002 if resume else 0.003)
