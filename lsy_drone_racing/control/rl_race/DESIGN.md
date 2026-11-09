# GPU-native racing RL (`rl_race`): design note

Status: draft, 2026-11-08. Replaces the SBX prototype in `rl_single_l2/`. Phase 1 there is frozen
and documented in `rl_single_l2/SPARSE_REWARD.md`.

## Why

1. **Training is host-bound.** The SBX/SB3 pipeline does about 50k steps/s per run, and about 280k
   steps/s for the whole machine regardless of how many runs share it. The GPU sits around 25 % busy.
   Every one of the 128 rollout steps crosses CPU↔GPU several times:
   - policy inference;
   - reward and done readback;
   - the Python `info` loop;
   - the NumPy rollout buffer.

   GAE also runs in NumPy.
2. **The action design caps agility.** The prototype commands a rotation increment δ ≤ α on the
   *current* attitude. The onboard Mellinger controller is a spring-damper,
   torque = −kR·e_R − kw·ω (rate setpoint 0 in attitude mode). Its error never exceeds α, so the
   turn rate saturates at ω ≈ (kR/kw)·2·sin α ≈ 7·sin α:
   - 1.4 rad/s at α = 0.16;
   - 2.5 rad/s at α = 0.3;
   - about 7–8 rad/s for absolute attitude commands.

   This is why laps take about 6–7 s while the leaderboard's top teams are at about 3.3 s. The drone
   overshoots gate 3, then creeps back to it for 0.2–4 s.

## Benchmark (2026-11-08, RTX 5090, GPU shared with 3 other runs, so lower bounds)

| | 8192 envs | 32768 envs |
| --- | --- | --- |
| Bare `race_core` step (jitted) | 3.7 M steps/s | 3.9 M steps/s |
| Full rollout in one `lax.scan` (actor → decode → physics → obs encoding) | 0.81 M steps/s | 1.32 M steps/s |
| Current SBX pipeline | ~0.05 M steps/s per run | – |

The gap between the bare step and the full rollout comes from our own feature code (attitude
conversions, obstacle selection), so it can be optimized later.

## Trainer: Brax PPO

- `brax==0.14.2` (`brax.training.agents.ppo`). This is the fully jitted PPO used by MuJoCo
  Playground. It provides:
  - auto-reset and time-limit bootstrapping (`bootstrap_on_timeout`);
  - observation normalization;
  - a state-dependent std;
  - separate policy and critic observation keys (asymmetric critic in phase 2);
  - checkpoints;
  - `restore_params` / `restore_value_fn`, used to reset the critic when switching rewards;
  - `progress_fn`, which feeds W&B.
- **Version: Brax from GitHub, pinned to commit `c8605294264a` (main, 2026-10-20).**
  - jax removed `jax.device_put_replicated` in 0.10.0. The latest Brax release, 0.14.2 (2026-03-15),
    still calls it, so **no released Brax works with jax ≥ 0.10**. Our flax 0.12.8 needs jax ≥ 0.10.
  - Brax fixed this on `main` in `303d89d` (2026-03-25) but hasn't released since.
  - The pinned commit works unmodified with our pins (jax 0.11.0, flax 0.12.8, optax 0.2.8,
    orbax 0.12.2), plus `jaxopt`, `tensorboardX` and `mujoco-mjx==3.10.0`. A toy PPO run learned on
    the GPU (reward −94 → −17 in 2M steps, 11 s).
  - Switch to the next Brax release once one appears.
  - Rejected alternatives:
    - **Downgrading jax below 0.10:** this forces an older flax too. All pixi environments share one
      solve group, so it would also change the simulator stack in the `default` environment that the
      leaderboard installs.
    - **A compatibility shim around the released 0.14.2:** it works but patches jax.
- **Dependencies** are added to the `gpu` (training) environment only. The `default` environment,
  which the leaderboard installs, doesn't need Brax: the deployed controller is a plain MLP.

## Package layout (new, `lsy_drone_racing/control/rl_race/`)

| Module | Content |
| --- | --- |
| `env.py` | Brax `Env` adapter over the course's functional `race_core` reset/step (`EnvData` pytree). It keeps the drone axis, so phase 2 needs no new env. |
| `actions.py` | Action mapping (see below) |
| `observations.py` | Feature encoding from the simulator's own observation fields |
| `rewards.py` | Pure reward functions: `dense_path`, `race`; `dense`/`sparse` ported for comparison |
| `starts.py` | Official start and path-based starts between gates |
| `train.py` | Launcher: flags → Brax PPO, compatibility shim, W&B, staged reward schedule |
| `export.py`, `controller.py` | Plain-array export (actor + normalizer) and the competition controller |
| `evaluate.py` | Jitted paper-protocol evaluation (success, lap time, gates, crash causes) |

## Decisions

**Observations.** Only fields the course simulator provides, with light preprocessing:
- drone: position, attitude (rotation matrix), body-frame velocity, angular velocity, previous action;
- the next two gates in `gate_sequence` (from `n_gates_passed`): center and passing normal
  (`gates_quat` x-axis × `gate_sequence_direction`), both relative to the drone, in the body frame,
  plus a valid flag;
- the nearest obstacles (`obstacles_pos`), relative to the drone, in the body frame;
- gates and obstacles use the sensor-masked positions (nominal until within 0.7 m).

**Actions.** 3-D action `[roll, pitch, thrust]` from Brax's `tanh_normal` distribution, each in [−1, 1]:
- roll and pitch: absolute commands, scaled to ±θ_max with **θ_max = 70°** (flag `--theta_max_deg`, to
  be tuned later);
- thrust: affine map onto the full collective-thrust range of the attitude interface;
- **yaw is dropped from the action and always commanded as 0** (the start heading). A quadrotor's
  translation doesn't depend on yaw, and holding it fixed keeps roll/pitch aligned with fixed world
  axes, which is easier to learn.

There is no incremental mapping, `alpha_max_rad`/`alpha_gain_rad` or `mean_bound`: deterministic
actions reach the actuator limits by construction. The old incremental action exists only as a
validation mode for the trainer (see Validation).

**Starts.**
- **50 % official ground starts** at first (flag `--official_start_prob`, can be lowered later).
- The rest start on a nominal path for leg k: a cubic Hermite curve from the exit side of gate k−1 to
  the entry side of gate k. Its tangents follow each gate's passing normal, built from the provided
  positions and quaternions.
- Sample a point on the curve with about ±0.1–0.2 m position noise, and set `n_gates_passed = k`.
- Set the velocity along the curve's tangent with a random speed (0–4 m/s) and small directional
  noise, so it always points toward the next gate's correct side. Attitude is near level with yaw 0 (the
  same as the commanded yaw).
- The collision check is kept; a start inside an obstacle falls back to the official start.

**Reward schedule.**
1. **Pre-training with `dense_path`:**
   - progress = arc length gained along the path above (not the distance to the gate center, which
     also pays for approaching from the wrong side);
   - +10 per gate, +50 per lap, −0.05 per step, −0.005·‖ω_measured‖, −20 per crash.
2. **Switch** once jitted evaluation from the official start reaches **≥ 30 % success**.
3. **Critic reset:** the value network is re-initialized (`restore_value_fn=False`) and trained alone
   for **10 PPO iterations** (policy learning rate 0). After that the policy resumes with a **lower
   learning rate** than in pre-training (flag `--policy_lr_after_switch`, e.g. 1e-4 vs 3e-4).
   - Brax uses one Adam optimizer with one learning rate for policy and critic, so this needs separate
     learning rates. Implementation: `optax.multi_transform` with a schedule on the policy part
     (0 for 10 iterations, then the lower rate).
   - Brax's `train()` has no optimizer hook. We keep a copy of the PPO `train.py` from the pinned
     commit with one added `optimizer` argument. The diff to upstream is kept minimal and covered by a
     test showing that, without the hook, it matches upstream exactly.
4. **Semi-sparse `race` reward:** +10 per gate (5 if trailing, phase 2), +50 per lap, −0.05 per step,
   −0.005·‖ω_measured‖, −20 per crash. There is **no fast-lap bonus for now**.

**Evaluation.** The jitted paper protocol from the official start (many episodes) runs every few
iterations and is logged to W&B. The official `pixi run evaluate` stays the final check.

**Phase 2 readiness.**
- The env keeps the agent axis of `MultiDroneRacing-v0`.
- With a shared policy, agents flatten into the batch dimension (self-play IPPO).
- Privileged critic inputs go under `value_obs_key`.

## Validation plan

1. **Env equivalence:** with the same seed and actions, adapter observations, rewards and dones match
   the prototype step by step, across resets.
2. **Trainer:** DS from scratch with the old incremental action and the old dense reward, 3 seeds,
   100M steps, must land within the prototype's seed spread (14 ± 5 % success at 100M; similar return
   and episode-length curves).
3. **Export:** export → `controller.py` → `scripts/sim.py` / `pixi run evaluate` works in the
   `default` environment.
4. **New design:** then switch to absolute attitude + `dense_path` → `race`, and compare lap time and
   success against the prototype's 1B-step runs (DS 52 % / 6.0 s, SS 71 % / 7.6 s).

## Open questions

- Speed range of the path starts (0–4 m/s proposed).
- Final value of the post-switch policy learning rate.
- θ_max beyond 70° (later sweep).

## Implementation notes (2026-11-08)

- **Throughput:** about 1 M steps/s for one run at 8192 envs, with the GPU at ~96 %. Three runs in
  parallel share the GPU (~330k steps/s each). The old SBX pipeline did ~50k per run.
- **Progress potential:** arc length alone let the policy fly past a gate *beside* the opening and
  still collect the full leg progress. Any point beyond the gate's plane projects onto the end of
  the leg. The first run learned exactly that, then hovered until the time limit. The progress
  term is therefore Δ(arc − path_lateral · distance-to-path), with `path_lateral = 1`. As a
  potential difference it only rewards changes, so a constant offset (a cut corner) costs nothing.
- **Pre-training reward:** with crash −20 and time −0.05 per step, sitting on the takeoff platform
  is worth only about −5 (discounted). The policy learned not to take off before discovering that
  a controlled takeoff pays. Three variants were compared for 150M steps (one seed each), all of
  which learn:

  | Variant | Success | Lap time |
  | --- | --- | --- |
  | crash −5, time −0.05 | 59 % | 4.44 s |
  | **crash −20, time −0.2** | **81 %** | 4.55 s |
  | crash −5, time −0.1 | 73 % | 4.50 s |

  The chosen variant keeps the crash penalty and raises the time penalty only during
  pre-training (`--pretrain_crash=20 --pretrain_time=0.2`). The race reward is unchanged.
- **Key types:** Brax passes raw `uint32` keys, while crazyflow stores typed keys; `env.reset` wraps
  them with `jax.random.wrap_key_data`.
- **Brax eval** reads `info["steps"]`. The adapter provides it as the episode length including the
  current step.

## Current state and next steps (2026-11-09)

**On the leaderboard** (`main`, commit `0006646`): `night_cont_theta85` with
`controller.file = "rl_race_controller.py"`.
- Official simulator, 100 runs: 96/100 successful, mean 3.714 s, best 3.32 s.
- `scripts/evaluate.py`: 17/20, mean 3.71 s.
- Top of the leaderboard: 3.27 s.

**How it was trained** (seed 3, 3.85B cumulative steps):
1. `race_v3_s3`: `dense_path` pre-training (crash 20, time 0.2) until 30 % success, then the race reward
   for 1B steps (θ 70°, 8192 envs).
2. `cont_s3_theta85`: +300M steps with θ 85°.
3. `night_cont_theta85`: +1B steps with official starts only, 16384 envs, batch 2048 × 32,
   entropy 1e-3, LR decay to 10 %.

**Other runs**, each 100 official runs:

| Run | Description | Success | Mean lap |
| --- | --- | --- | --- |
| `night_cont_theta85_2` | +1B more at lower LR; plateaued | 96/100 | 3.719 s |
| `night_g3_paper_ft` | 3 gates, paper nets, γ 0.995, race time 0.1, same fine-tuning | **100/100** | 3.785 s |

The effects of the four changes in `night_g3_paper_ft` are not separated.

**Single-change effects**, each a 300M-step continuation of seed 3 starting at 4.04 s:

| Change | Lap time |
| --- | --- |
| entropy 1e-3 | 3.91 s |
| θ 85° | 3.92 s |
| LR decay | 3.97 s |
| time penalty 0.1 | 4.00 s |
| baseline | 3.99 s |

The fine-tuning recipe above gained another ~0.2 s.

**Locations** (all under `rl_race/outputs/`, git-ignored):
- runs: `<run>/` with `{stage}_config.json` and `checkpoints/{stage}_{final,latest}.pkl`;
- exports: `exports/<run>`;
- logs: `logs/`;
- 100-run evaluations: `eval100/`;
- overnight scripts: `night_chain*.sh`;
- videos: `videos/race_v3_s3/`.

W&B project `lsy-drone-racing`, groups `rl-race`, `rl-race-cont`, `rl-race-night`.

**Gotchas:**
- Evaluate in a git worktree only with `PYTHONPATH=$PWD`; otherwise the main checkout's package and
  model are used.
- `deploy_model/config.json` needs the `.gitignore` exception (on `main`).
- `scripts/evaluate.py` always uses `level2.toml`'s controller; for other controllers use
  `scripts/sim.py --controller=...`.

**Next steps (ideas, not started):**
1. **Trade reliability for speed.** The leaderboard only needs ≥ 50 % success; the policy succeeds
   96 % of the time and its best laps already reach 3.3 s. Candidates: a larger race time penalty
   together with γ 0.995, a smaller race crash penalty, or a lower entropy bonus. These change the
   race reward, so they need the user's approval.
2. **Separate the 3-gate run's changes** (lookahead, paper nets, γ 0.995, time 0.1): which ones buy
   reliability (100/100), and which cost ~0.07 s?
3. **Larger θ_max** (90°, the interface maximum) for the two tight turns (gate 1 → 2, gate 4 → 2),
   where 85° still saturates.
4. Phase 2 (IPPO): the env keeps the drone axis of `MultiDroneRacing-v0`; flatten agents into the
   batch and use `value_obs_key` for a privileged critic.
