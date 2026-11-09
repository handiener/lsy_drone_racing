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
3. **Larger θ_max** (90°, the interface maximum) for the two tight turns (gate 1 → 2, gate 4 → 2).
   At 70° the angle saturated in ~20 % of their steps; saturation at 85° has not been measured yet
   (`outputs/debug/saturation.py`).
4. Phase 2 (IPPO): the env keeps the drone axis of `MultiDroneRacing-v0`; flatten agents into the
   batch and use `value_obs_key` for a privileged critic.

## Speed experiments (2026-11-09)

**Where the lap time goes** (`outputs/debug/lap_analysis.py`, `night_cont_theta85`, 2048 official
episodes in the training sim; matches the official simulator: 95.8 %, mean 3.711 s):

| Leg | Gate | Mean | Std | Share of lap variance | Angle at limit | Thrust > 95 % |
| --- | --- | --- | --- | --- | --- | --- |
| start → 0 | 0 | 0.95 s | 0.04 | 8 % | 0 % | 32 % |
| 0 → 1 | 1 | 0.65 s | 0.09 | 32 % | 26 % | 3 % |
| 1 → 2 | 2 | 0.78 s | 0.06 | 6 % | 1 % | 10 % |
| 2 → 3 | 3 | 0.76 s | 0.07 | 26 % | 0 % | 12 % |
| 3 → 1 | 1 | 0.58 s | 0.07 | 28 % | 43 % | 7 % |

- Lap std is 0.15 s (p10 3.52 s, p90 3.90 s). The slowest 20 % of laps lose 0.10–0.14 s each in
  legs 0→1, 2→3 and 3→1.
- **Gate randomization explains little of it.** A linear fit of the lap time on the signed gate offsets
  gives R² 0.16. For legs 0→1, 2→3 and 3→1 it is 0.24–0.29. The rest comes from the other noise
  sources (dynamics disturbance, start pose, mass) and how they propagate through the lap.
- The two tight turns still hit the 85° limit in 26 % and 43 % of their steps, so θ 90° may help a little.
- Crashes happen mostly in leg 2→3 (37 of 86).

**Reward variants**, each a 500M-step continuation of `night_cont_theta85` (critic kept, fine-tuning recipe,
`outputs/speed_chain.sh`), 100 official runs each:

| Run | Change | Success | Mean lap | Best |
| --- | --- | --- | --- | --- |
| `night_cont_theta85` (baseline) | – | 96/100 | 3.714 s | 3.32 s |
| `spd_time0.2_g995` | race time 0.2, γ 0.995 | 98/100 | 3.768 s | 3.52 s |
| `spd_crash5` | race crash 5 | 96/100 | 3.798 s | 3.52 s |
| `spd_lapbonus20_g995` | lap bonus 20/s under 4.0 s, γ 0.995 | 97/100 | 3.919 s | 3.64 s |

All three got *slower* and slightly more reliable. Success dropped to 58–77 % in the first
~50M steps while the kept critic adapted, then recovered more cautiously. Two likely reasons:
- **A crash forfeits all remaining gate and lap rewards** (up to 10 per gate + 50). That is the
  dominant penalty, not the crash term. γ 0.995 makes this future reward weigh even more.
- Changing the reward under a converged critic causes a disruption phase, and the policy
  relearns from there.

The lap bonus is implemented (`--race_lap_bonus`, `--race_lap_ref`; it pays only for laps from the
official start) and is 0 by default.

## Point-mass lower bound on the lap time (2026-11-09)

`min_time_bound.py` solves the minimum-time problem on the nominal level-2 track with CasADi/IPOPT.
- **Model:** a point mass with a thrust vector of 4 × the rotor limits (0.085–0.8 N, up to 1.88 g at 43.4 g).
- **Constraints:** tilt ≤ θ_max; each gate is crossed inside its opening shrunk by `margin`, moving
  forward; obstacles are cylinders; the safety box applies; start at rest.
- **Solve:** a relaxed problem (no minimum thrust, no obstacles) first, then the full problem warm-started from it.

It ignores rotational dynamics, the attitude controller, drag and disturbances, so it is
a true lower bound, but a loose one: the thrust direction can flip instantly.

| θ_max | Margin | Mass | Bound | Legs (s) |
| --- | --- | --- | --- | --- |
| 85° | 0.05 m | 43.4 g | **2.25 s** | 0.63, 0.25, 0.65, 0.46, 0.25 |
| 85° | 0 | 43.4 g | 2.19 s | 0.64, 0.22, 0.65, 0.44, 0.24 |
| 85° | 0.05 m | 38.4 g (lightest) | 2.08 s | 0.58, 0.23, 0.60, 0.42, 0.24 |
| 90° | 0.05 m | 43.4 g | 2.25 s | 0.63, 0.25, 0.65, 0.46, 0.25 |

For comparison, our policy's mean legs are 0.95, 0.65, 0.78, 0.76, 0.58 s (3.71 s) and the leaderboard top is 3.27 s.
- Thrust is at its maximum the whole lap (bang-bang), and θ 85° → 90° changes nothing in this model.
  The tilt limit is not what binds; the thrust magnitude and the time to turn the thrust vector are.
- The ±5 g mass randomization alone moves the bound by 0.17 s, so it is likely part of our lap-time
  spread. The earlier variance analysis did not include it.
- The gap to the bound is largest in legs 0→1 (0.40 s), 2→3 (0.30 s) and 3→1 (0.33 s), the same legs
  that dominate our variance. These are the tight turns, where the bound flips the thrust direction
  instantly. A tighter bound needs an attitude-response model, for example a rate limit on the thrust
  direction identified from the simulator.

**With the attitude response.** `outputs/debug/attitude_step.py` measured roll steps in the simulator:
- the thrust direction turns at most **~8 rad/s** (the same at 0.43–0.75 N);
- peak angular acceleration is 120–160 rad/s²;
- t90 is 0.32 s even for a 30° step;
- a 170° flip command does not finish in 0.6 s.

`min_time_bound.py` (default `--rate_max=8 --acc_max=160`) now models the thrust direction as a
unit-vector state with those limits. It solves a homotopy: relaxed → point mass → rate limit
30, 15, 12, 10, 9, 8.5, 8 rad/s.

| Rate limit | Lap |
| --- | --- |
| none | 2.25 s |
| 30 / 15 rad/s | 2.31 s |
| 12 rad/s | 2.42 s |
| 10 rad/s | 2.60 s |
| 8.5 rad/s | 2.59 s |
| **8 rad/s (measured)** | **3.00 s** (legs 0.70, 0.53, 0.93, 0.43, 0.41; v_max 5.7 m/s) |

The sequence is not monotonic (9 rad/s gave 3.04 s, worse than 8.5), so some stages ended in local
optima. Each solution is a feasible trajectory of the model, so 3.0 s is an upper estimate of the
model's optimum, and the model's optimum is probably between 2.6 and 3.0 s. The model is still optimistic:
small rotations in the simulator are slower than the rate limit (t90 0.32 s at 30°), and the motors
are assumed to change thrust instantly. **Practical limit: roughly 2.8–3.0 s.** The leaderboard top
(3.27 s) is about 10 % above it and our mean (3.71 s) about 25 %.

**Round 2** (`outputs/speed_chain2.sh`): critic re-initialized, smaller gate and lap rewards. Also
`cont_theta90` (θ 90°, critic kept). All are 500M-step continuations of `night_cont_theta85`, with 100 official runs each:

| Run | Success | Mean lap | Best |
| --- | --- | --- | --- |
| `r2_control` (critic reset, reward unchanged) | 95/100 | 3.709 s | 3.32 s |
| `r2_g5l10_t0.1` (gate 5, lap 10, time 0.1) | 94/100 | 3.726 s | 3.36 s |
| `r2_g5l10_t0.2` (gate 5, lap 10, time 0.2) | 91/100 | **3.671 s** | 3.36 s |
| `cont_theta90` | 97/100 | 3.692 s | 3.36 s |
| `r2_g2l5_t0.1b` (gate 2, lap 5, time 0.1) | 98/100 | 3.787 s | 3.48 s |

The standard error is about 0.018 s per run, so `r2_g5l10_t0.2` (−0.04 s) is at the edge of noise
and `cont_theta90` (−0.02 s) is within it. Resetting the critic by itself costs nothing (control
3.709 s). This is the first reward change that trades reliability for speed, as intended. A
longer run of it, or a stronger time penalty, is the next candidate. `r2_g2l5_t0.1` ran out of GPU memory at
start (5 runs on the GPU); its rerun is `r2_g2l5_t0.1b`.

The level-2 dynamics disturbance is resampled at every simulator step (white noise). Mass and inertia
are randomized once per episode.

**Reading of rounds 1–2.** Under discounting, collecting the future rewards one step earlier is
worth about (1 − γ)·V. So the gate and lap rewards themselves pay for speed. At the same time,
everything still ahead is lost in a crash. Shrinking them with time 0.1 removed speed incentive
(gate 2 / lap 5: slower, 98 %). Shrinking them with time 0.2 shifted the balance toward speed
(gate 5 / lap 10: faster, 91 %). The ratio of the time penalty to the value at risk is the lever.

**Exploration noise** of the leaderboard policy (mean pre-tanh std along official laps): roll/pitch
0.045/0.055 (about 4–5° of attitude noise), thrust 0.18 (about ±9 % of the thrust range per step).
The policy is trained to survive this noise, but flies without it at deployment.

**Start leg** (`outputs/debug/takeoff.py`, 64 official starts, `night_cont_theta85` and `r2_g5l10_t0.2`).
The drone starts 1–3 cm above the ground and first drops about 2 cm while the motors spin up. Even an
open-loop full-thrust command needs **0.20 s** to rise 1 cm. The policies need 0.23–0.25 s: they
command 0.69–0.79 N from the first step and already pitch to about 60° (tilt about 40°) on the ground.
So the takeoff costs at most about 0.05 s. The rest of the first leg (gate 0 at 0.94 s) is a deliberate
brake: at about 4 m/s the policy pitches back 40–58° and drops to about 0.5 N, then crosses gate 0 at
about 2.7 m/s (point-mass optimum: 4.2 m/s) to set up the tight turn into gate 1. The bound has
no motor spin-up, so about 0.2 s should be added to it: **realistic floor about 3.0–3.2 s.**

**Round 3 and noise runs (started 2026-11-09 14:51):**
- `outputs/speed_chain3.sh`: gate 5, lap 10 with time 0.3 / 0.5 (critic reset), and
  `r3_t0.2_long`, which continues `r2_g5l10_t0.2`; 1B steps each.
- `outputs/noise_chain.sh`: `--noise_scale` 0.5 then 0.25 with entropy 0, from `night_cont_theta85`,
  reward unchanged.
- `--noise_scale` multiplies the sampling std (`NormalTanhDistribution(var_scale=...)`); the learned
  std parameters and checkpoints are unchanged.

**Learning-rate shock on continuation.** A `--continue_race` run starts a fresh Adam at the full policy and
value learning rates (1e-4 / 3e-4). The parent run ended at 10 % of that (`--lr_decay_to=0.1`).
- **Effect:** the 10× jump caused the early collapse seen in every round-1 run (success down to 58–77 %).
  `noise0.5` fell to 14 % (halving the noise makes the same update relatively larger), and
  `r3_t0.2_long` fell to 51 %. Both were stopped.
- **Restarts** `noise0.5_lr` / `r3_t0.2_long_lr` use policy 2e-5, value 6e-5, decay to 0.5 (`outputs/lowlr_chain.sh`).
  They hold 93–98 % success from the first evaluation.
- **Rule:** start a continuation near the learning rate its parent ended with. Runs that re-initialize the
  critic (`--continue_race=False`) freeze the policy for 10 iterations and dip much less.
- **Consequence:** the round-1 results (all slower) were partly confounded by this shock.

## Trunk and anneal stage (2026-11-09)

The learning-rate decay is now a separate training stage, so a run's main checkpoint keeps the full
learning rate and can serve as a branch point.
- **Race stage:** with `--lr_decay_to=1.0` (the default) it trains at a constant learning rate and saves
  `race_final.pkl`, the trunk/branch point.
- **Anneal stage:** `--anneal_steps=N` (> 0) adds a stage `anneal`. It has the same reward and keeps the
  critic. It starts with a linear warmup over `--anneal_warmup_iterations` (default 5) PPO iterations,
  because the fresh Adam has no moment estimates. Then both learning rates decay linearly to
  `--anneal_to` (default 0.1). It saves `anneal_final.pkl` and `anneal_config.json`;
  export it with `export --stage=anneal`.
- **Anneal only:** `--start_stage=anneal --start_params=<full-LR checkpoint> --anneal_steps=N` anneals
  an existing checkpoint without further race training.
- **Warmup on branches:** `--warmup_iterations` (default 0) adds the same warmup to `--continue_race`
  branches.
- **Schedule:** `optim.lr_schedule` composes freeze → warmup → constant or decay; it is unit-tested.

**Workflow:**
1. Trunk: constant learning rate.
2. Experiment: branch from a trunk `race_final.pkl` at the trunk's learning rate (with a short warmup).
3. Every arm, including the control, gets the same anneal stage before evaluation.

Usable full-learning-rate checkpoints today: `cont_s3_theta85`, `race_v3_s3` (all later runs decayed).

## Round 3 and noise results (2026-11-09)

100 official runs each (standard error about 0.015 s):

| Run | Change | Success | Mean lap | Best |
| --- | --- | --- | --- | --- |
| `night_cont_theta85` (reference) | – | 96/100 | 3.714 s | 3.32 s |
| **`r3_t0.3`** | gate 5, lap 10, **time 0.3**, critic reset, 1B | **94/100** | **3.643 s** | 3.40 s |
| `r3_t0.5` | gate 5, lap 10, time 0.5, critic reset, 1B | 33/100 | 3.685 s | 3.40 s |
| `r3_t0.2_long_lr` | +1B on `r2_g5l10_t0.2` (low LR) | 95/100 | 3.663 s | 3.36 s |
| `noise0.5_lr` | sampling noise × 0.5, entropy 0, low LR, 500M | 97/100 | 3.706 s | 3.38 s |
| `noise0.25_lr` | sampling noise × 0.25, entropy 0, low LR, 500M | 98/100 | 3.728 s | 3.40 s |

- **`r3_t0.3` is the best so far:** −0.07 s at nearly the same reliability, about 3.5 standard errors of
  the difference.
- **Time 0.5 overshoots, and the arithmetic shows why.** An official lap is about 180 steps, so the time cost
  is 0.5 × 180 = 90 against 5 × 5 + 10 = 35 of gate and lap reward. Finishing (about −55) is worse than
  crashing (−20), so crashing early becomes the better option. At time 0.3 finishing is worth about −19,
  roughly the same as a crash. The time penalty can only rise further together with the crash penalty
  (or the lap reward), keeping finishing clearly better than crashing.
- **Longer training at time 0.2** adds nothing (3.663 vs 3.671 s).
- **Less exploration noise gives no speed gain** (3.706 / 3.728 vs 3.714 s); success rose slightly.
  Caveat: these runs used the low continuation LR (policy 2e-5). The eval lap stayed at about 3.70 s
  throughout, so the policy barely changed. A test at the trunk LR would be more conclusive.

## From-scratch trunk (2026-11-09, RTX 5080)

`outputs/trunk_chain.sh` / `trunk_chain_s23.sh`. Pre-training as `race_v3`, then gate 5 / lap 10 / time
0.3 / crash 20 at a constant learning rate for 2.5B race steps, θ 85°, 8192 envs, 50 % official
starts, entropy 5e-3. The first attempt (16:18) died when the previous PC shut down
(`trunk_t0.3_s1_aborted_pcdown`) and was restarted at 17:00. 100 official runs of each `race_final.pkl`
(full learning rate, no fine-tuning):

| Seed | Success | Mean lap | Median | Best |
| --- | --- | --- | --- | --- |
| 1 | 87/100 | 3.688 s | 3.64 s | **3.24 s** |
| 2 | 86/100 | **3.639 s** | 3.64 s | 3.28 s |
| 3 | 85/100 | 3.702 s | 3.70 s | 3.30 s |

- **The recipe holds across seeds:** 3.64–3.70 s, 85–87 % success.
- **No fine-tuning needed for the trunk's speed.** Without decay or official-only fine-tuning, the trunk
  already matches the best fine-tuned policy (`r3_t0.3`, 3.643 s, 94 %). It trades about 8 points of
  success, as intended by time 0.3.
- **Best single laps** (3.24–3.30 s) reach the leaderboard top (3.268 s).
- Seed 2 is annealed for 300M steps with official starts only (`outputs/anneal_best.sh`,
  run `trunk_t0.3_s2_anneal`).

**Anneal of seed 2** (`trunk_t0.3_s2_anneal`, 300M steps, official starts only, warmup 5 iterations,
LR decay to 10 %, otherwise the trunk's settings; 9 min of training): 86/100, **mean 3.616 s**
(± 0.019), median 3.62 s, best 3.22 s.
- **Gain over its trunk:** −0.023 s (3.639 → 3.616), about one standard error of the difference, at equal success.
- **During training:** the evaluation lap fell steadily (3.69 → 3.60 s) and success held at 84–89 %.
  The warmup prevented the learning-rate shock.
- **Smaller than the night run's −0.2 s.** That run also lowered the entropy to 1e-3, used 16384 envs,
  a batch of 2048 and 1B steps, and started from a slower policy (4.04 s).
- **Fastest policy so far:** 3.616 s mean, best lap 3.22 s, below the leaderboard top of 3.268 s.

**1B anneal of all seeds** (`outputs/anneal_long.sh`, `trunk_t0.3_s{1,2,3}_anneal1b`): 1B steps from each
full-LR `race_final.pkl`, official starts only, entropy 1e-3, warmup 5 iterations, LR 1e-4 / 3e-4 →
1e-5 / 3e-5. 100 official runs each:

| Seed | Trunk (full LR) | Anneal 300M | **Anneal 1B** | Median | Best |
| --- | --- | --- | --- | --- | --- |
| 1 | 87 %, 3.688 s | – | 81 %, **3.584 s** | 3.54 s | **3.16 s** |
| 2 | 86 %, 3.639 s | 86 %, 3.616 s | 87 %, **3.591 s** | 3.54 s | 3.24 s |
| 3 | 85 %, 3.702 s | – | 79 %, **3.558 s** | 3.54 s | 3.18 s |

- **Every seed got faster:** −0.05 to −0.14 s against its trunk. The seed spread shrank from
  3.64–3.70 s to 3.56–3.59 s, and all medians are 3.54 s.
- **Reliability:** seeds 1 and 3 traded 6 points of success, while seed 2 stayed at 87 %.
- **Exploration noise:** `policy_dist_min_std` falls from about 0.012 to about 0.001–0.002 during the anneal
  (Brax floor 0.001). With entropy 1e-3 the anneal turns exploration almost off by itself.
- **Seed 2, 1B vs 300M** (3.591 vs 3.616 s): the longer anneal with lower entropy gains a further 0.025 s.
- **Logging:** Brax's `training/learning_rate` logs the constant `--learning_rate` argument, not the
  schedule. `train.py` now logs the actual rates as `lr/policy` and `lr/value` (not yet in these runs).
  `test_separate_lr_applies_the_schedule_to_updates` confirms that the optimizer applies the schedule.
