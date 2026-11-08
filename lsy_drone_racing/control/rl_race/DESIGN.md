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
- **Compatibility:**
  - Brax imports with our pins (jax 0.11.0, flax 0.12.8, optax 0.2.8, orbax 0.12.2) once
    `jaxopt`, `tensorboardX` and `mujoco-mjx==3.10.0` are added.
  - jax 0.11 removed `jax.device_put_replicated`, which Brax still calls. A three-line shim in our
    launcher fixes it. A toy PPO run learned on the GPU with the shim (reward −94 → −11 in 2M steps,
    13 s).
  - The shim is covered by a test and gets revisited when Brax updates.
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

**Actions (proposed).** Absolute roll/pitch commands in ±θ_max, yaw, and collective thrust over the
full range, from Brax's `tanh_normal` distribution. This removes the rate cap and the double squashing
(`mean_bound`, `alpha_max_rad`/`alpha_gain_rad`). To isolate trainer bugs, the trainer is validated
first with the old incremental action (see Validation).

**Starts.**
- A fixed share of official ground starts (about 30–40 %).
- The rest start on a nominal path for leg k: a cubic Hermite curve from the exit side of gate k−1 to
  the entry side of gate k. Its tangents follow each gate's passing normal, built from the provided
  positions and quaternions.
- Sample a point on the curve with about ±0.1–0.2 m position noise, and set `n_gates_passed = k`.
- Set the velocity along the curve's tangent with a random speed (0–4 m/s) and small directional
  noise, so it always points toward the next gate's correct side. Yaw follows the tangent, attitude is
  near level.
- The collision check is kept; a start inside an obstacle falls back to the official start.

**Reward schedule.**
1. **Pre-training with `dense_path`:**
   - progress = arc length gained along the path above (not the distance to the gate center, which
     also pays for approaching from the wrong side);
   - +10 per gate, +50 per lap, −0.05 per step, −0.005·‖ω_measured‖, −20 per crash.
2. **Switch** once jitted evaluation from the official start reaches **≥ 30 % success**.
3. **Critic reset:** the value network is re-initialized and trained alone for a few iterations
   before policy updates resume.
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

- θ_max for absolute attitude (e.g. 60–85°) and whether yaw is absolute or held.
- Share of official starts vs path starts, and the speed range of path starts.
- Number of critic-only iterations after the reward switch.
