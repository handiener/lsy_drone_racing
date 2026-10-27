"""
This py file is used to wrap the environment.
The standard environment provided by the offical crazyfly is not suitable for the SBX/SB3 (we want JAX version with acceleration)
Therefore the wrapper need to wrap the offical env for the format of in/output.

Except this, we also want some functions wrapped to the env as well. (which helps for the RL PPO trainig.)

大致的结构如下，我们先完成我们的简化版的主要保证接口全部通顺，一些高级的操作咱们先不要动。

(it is hard coded to level 2, in future, change it to corresponding level & setting)
大概的结构：
class Level2SBXEnv(VecEnv):
    def __init__(...): ...          # official level 2 env
    def reset(self): ...
    def step_async(self, actions): ...
    def step_wait(self): ...
    def close(self): ...
    # SB’s relevant management methods……

def encode_obs(...): ...        #convert obsercation to SBX’s format
def decode_action(...): ...     #action map to official env’s input
def compute_reward(...): ...    #our own reward defined

负责：读取 Level 2 配置并创建官方 VecDroneRaceEnv，使用 attitude 控制接口；将官方观测字典编码成网络输入，包括无人机状态、目标门和障碍物的相对信息，并进行归一化；
沿用reference的动作表示，将网络输出的推力原始值和三维姿态旋转增量转换为官方要求的 [roll, pitch, yaw, thrust]；
调用官方环境推进仿真，再根据前后状态计算训练奖励 (reward 也沿用reference的，不过简化一点
r="向目标门推进的奖励"+"完成奖励"-"撞毁惩罚"-"角速度惩罚"-"时间惩罚" 
)；
将 terminated 和 truncated 转换为 SBX 所需的 done 和超时信息，保存重置前的最终观测，只重置已经结束的并行环境，同时清空对应的历史动作；
最后返回符合 SB3/SBX VecEnv 协议的数据，并实现关闭、随机种子和必要的管理方法。第一版先让 Actor、Critic 共用观测，暂不加入中段起飞、
成功状态缓存、多阶段课程和自定义 JIT 采样。

请注意，你写的代码，要把obs reward 还有action的相关的function都有一个注释，然后写明具体的项都是什么，然后最好写一下和reference的区别，
就是有些项暂时没用上的，可以注释写一下，方便我以后考虑。代码写完之后吧reference从文档里删除。

我们的简化版主要就是这四部分：
1. 创建环境：读取官方 Level 2 配置，创建并行 VecDroneRaceEnv，使用 attitude 控制接口。
2. 数据与接口转换：编码、归一化观测；将网络动作转成姿态和推力；适配 SBX 的返回格式。
3. 自定义奖励：根据前后状态计算进度、完成、撞毁、时间等训练奖励。
4. 重置逻辑：谁结束就单独重置谁，保留结束前的观测和终止／超时信息，其他环境继续飞。
先不加入中段起飞、成功状态缓存和多阶段课程。

物理仿真、碰撞和过门判定继续用官方实现；PPO 更新交给 SBX。 我们的 env_gym.py 负责把两边正确连接起来。

"""







"""
Here are 2 reference code files, please don't copy paste them, these 2 files are too redundant. Let's reconstruct the whole stuff.
These are ONLY reference!!!!

"""

# reference 1:
# """Vectorized racing-env adapter for the Song-2023 PPO pipeline."""

# from __future__ import annotations

# import copy
# from pathlib import Path
# from typing import Any

# import jax
# import jax.numpy as jnp
# import numpy as np
# from drone_models.core import load_params
# from jax import Array
# from jax.scipy.spatial.transform import Rotation as JaxRotation
# from ml_collections import ConfigDict
# from scipy.spatial.transform import Rotation

# from lsy_drone_racing.control.rl_song import obs as obs_encoding
# from lsy_drone_racing.control.rl_song.config import (
#     ACTOR_OBS_DIM,
#     ENV_ACTION_DIM,
#     RAW_ACTION_DIM,
#     CurriculumConfig,
#     CurriculumStage,
#     TrainConfig,
#     default_curriculum,
# )
# from lsy_drone_racing.control.rl_song.policy import raw_to_env_action, raw_to_physical_action
# from lsy_drone_racing.control.rl_song.reward import step_reward
# from lsy_drone_racing.control.rl_song.rollout import _refresh_aux_fields_after_respawn
# from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
# from lsy_drone_racing.envs.race_core import _reset_env_data
# from lsy_drone_racing.envs.race_core import obs as race_core_obs
# from lsy_drone_racing.utils import load_config

# REPO_ROOT: Path = Path(__file__).resolve().parents[3]
# LEVEL_CONFIG_NAME: str = "level{level}.toml"
# N_DRONES: int = 1
# TOTAL_THRUST_MULTIPLIER: float = 4.0


# class RLSongVecEnv:
#     """Thin vector-env wrapper exposing PPO-ready tensors.

#     Parameters
#     ----------
#     train_cfg : TrainConfig, optional
#         Training configuration bundle.
#     n_envs : int, optional
#         Number of vectorized environments. Defaults to ``train_cfg.ppo.n_envs``.
#     stage_idx : int, optional
#         Zero-indexed curriculum stage. Defaults to
#         ``train_cfg.initial_stage_index``.
#     seed : int, optional
#         Random seed for the env and reset perturbations.
#     device : {"cpu", "gpu"}, optional
#         JAX device string passed to ``VecDroneRaceEnv``.

#     Notes:
#     -----
#     The wrapped env still autoresets internally. This wrapper observes the
#     env's reset mask and applies curriculum reset perturbations to newly reset
#     worlds before building policy observations.
#     """

#     def __init__(
#         self,
#         train_cfg: TrainConfig | None = None,
#         *,
#         n_envs: int | None = None,
#         stage_idx: int | None = None,
#         seed: int | None = None,
#         device: str = "gpu",
#     ):
#         """Initialize wrapper state and select the configured curriculum stage."""
#         self.train_cfg = TrainConfig() if train_cfg is None else train_cfg
#         self.curriculum = self.train_cfg.curriculum
#         self.reward_cfg = self.train_cfg.reward
#         self.n_envs = self.train_cfg.ppo.n_envs if n_envs is None else n_envs
#         self.seed = self.train_cfg.seed if seed is None else seed
#         self.device = device
#         self.max_episode_steps = self.train_cfg.max_episode_steps
#         self.rng_key = jax.random.PRNGKey(self.seed)
#         self.normalizer = obs_encoding.init_normalizer(ACTOR_OBS_DIM)
#         self.prev_action_env_4vec = jnp.zeros((self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
#         self.prev_physical_action = jnp.zeros((self.n_envs, RAW_ACTION_DIM), dtype=jnp.float32)
#         self.env: VecDroneRaceEnv | None = None
#         self.stage_idx = -1
#         self.stage = self._stage_from_index(
#             self.train_cfg.initial_stage_index if stage_idx is None else stage_idx
#         )

#         initial_cfg = self._load_stage_config(self.stage)
#         drone_params = load_params(initial_cfg.sim.physics, initial_cfg.sim.drone_model)
#         self._thrust_bounds = (
#             float(drone_params["thrust_min"] * TOTAL_THRUST_MULTIPLIER),
#             float(drone_params["thrust_max"] * TOTAL_THRUST_MULTIPLIER),
#         )
#         self.current_env_obs: dict[str, Array] | None = None
#         self.prev_env_obs: dict[str, Array] | None = None
#         self.set_stage(self.curriculum.stages.index(self.stage))

#     def reset(self, seed: int | None = None) -> tuple[dict[str, Array], dict[str, Any]]:
#         """Reset every vectorized world and return actor/critic observations.

#         Parameters
#         ----------
#         seed : int, optional
#             Env seed. When provided, the wrapper's JAX reset-perturbation key is
#             also reset to this seed.

#         Returns:
#         -------
#         observations : dict[str, Array]
#             ``actor_obs`` and ``critic_obs`` arrays, each shaped
#             ``(n_envs, ACTOR_OBS_DIM)``.
#         info : dict[str, Any]
#             Info dictionary returned by the underlying env.
#         """
#         if self.env is None:
#             raise RuntimeError("RLSongVecEnv.reset called before env construction.")
#         if seed is not None:
#             self.rng_key = jax.random.PRNGKey(seed)

#         env_obs, info = self.env.reset(seed=seed)
#         self.current_env_obs = _to_jax_obs(env_obs)
#         reset_mask = jnp.ones((self.n_envs,), dtype=bool)
#         self._apply_reset_perturbation(reset_mask)
#         self.prev_action_env_4vec = jnp.zeros((self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
#         self.prev_physical_action = jnp.zeros((self.n_envs, RAW_ACTION_DIM), dtype=jnp.float32)
#         self.current_env_obs = self._read_env_obs()
#         self.prev_env_obs = self.current_env_obs
#         return self.build_observations(), info

#     def step(
#         self, raw_action: Array
#     ) -> tuple[dict[str, Array], Array, Array, Array, dict[str, Any]]:
#         """Step the env with raw policy actions.

#         Parameters
#         ----------
#         raw_action : Array, shape (n_envs, RAW_ACTION_DIM)
#             Raw 4-vector ``[T_raw, tau_x, tau_y, tau_z]`` sampled by the
#             policy. The log probability must have already been computed
#             before this call.

#         Returns:
#         -------
#         observations : dict[str, Array]
#             Next ``actor_obs`` and ``critic_obs`` tensors.
#         reward : Array, shape (n_envs,)
#             Song reward replacing the env's sparse reward.
#         terminated : Array, shape (n_envs,)
#             Termination flags.
#         truncated : Array, shape (n_envs,)
#             Timeout flags.
#         info : dict[str, Any]
#             Reward components and rollout metrics.
#         """
#         if self.env is None or self.current_env_obs is None:
#             raise RuntimeError("RLSongVecEnv.step called before reset.")
#         _validate_action_shape(raw_action, self.n_envs, RAW_ACTION_DIM, "raw_action")

#         prev_env_obs = self.current_env_obs
#         thrust_min, thrust_max = self.get_thrust_bounds()
#         env_action = raw_to_env_action(
#             jnp.asarray(raw_action), jnp.asarray(prev_env_obs["quat"]), thrust_min, thrust_max
#         )
#         physical_action = raw_to_physical_action(jnp.asarray(raw_action), thrust_min, thrust_max)
#         _validate_action_shape(env_action, self.n_envs, ENV_ACTION_DIM, "env_action")

#         env_obs, _, terminated, truncated, env_info = self.env.step(env_action)
#         self.current_env_obs = _to_jax_obs(env_obs)

#         terminated = jnp.asarray(terminated, dtype=bool)
#         truncated = jnp.asarray(truncated, dtype=bool)
#         current_target = self.current_env_obs["target_gate"]
#         prev_target = prev_env_obs["target_gate"]
#         finished = current_target < 0
#         terminated = terminated | finished
#         gate_just_passed = ((current_target > prev_target) & (prev_target >= 0)) | (
#             finished & (prev_target >= 0)
#         )

#         reward, components = step_reward(
#             self.current_env_obs,
#             prev_env_obs,
#             terminated,
#             truncated,
#             finished,
#             gate_just_passed,
#             self.reward_cfg,
#             true_gates_pos=self.true_gates_pos(),
#             true_gates_quat=self.true_gates_quat(),
#             true_obstacles_pos=self.true_obstacles_pos(),
#             physical_action=physical_action,
#             prev_physical_action=self.prev_physical_action,
#         )

#         done = terminated | truncated
#         if bool(np.asarray(jnp.any(done))):
#             self._reset_done_worlds(done)
#             self.current_env_obs = self._read_env_obs()

#         reset_prev_action = jnp.zeros_like(env_action)
#         self.prev_action_env_4vec = jnp.where(done[:, None], reset_prev_action, env_action)
#         reset_prev_physical_action = jnp.zeros_like(physical_action)
#         self.prev_physical_action = jnp.where(
#             done[:, None], reset_prev_physical_action, physical_action
#         )
#         self.prev_env_obs = self.current_env_obs
#         n_gates = prev_env_obs["gates_pos"].shape[1]
#         target_gate_progress = jnp.where(finished, n_gates, current_target)

#         info = dict(env_info)
#         info.update(
#             {
#                 "reward_components": components,
#                 "target_gate": current_target,
#                 "target_gate_progress": target_gate_progress,
#                 "finished": finished,
#                 "gate_just_passed": gate_just_passed,
#                 "crash": terminated & ~finished,
#                 "env_action": env_action,
#             }
#         )
#         return self.build_observations(), reward, terminated, truncated, info

#     def build_observations(self) -> dict[str, Array]:
#         """Build normalized actor and critic observations for current env state.

#         Returns:
#         -------
#         observations : dict[str, Array]
#             ``actor_obs`` and ``critic_obs`` arrays with shape
#             ``(n_envs, ACTOR_OBS_DIM)``.
#         """
#         if self.current_env_obs is None:
#             raise RuntimeError("No current env observation is available.")
#         actor_obs = obs_encoding.vmap_build_actor_obs(
#             self.current_env_obs, self.prev_action_env_4vec, self.normalizer
#         )
#         critic_obs = self._build_critic_obs(self.current_env_obs)
#         return {"actor_obs": actor_obs, "critic_obs": critic_obs}

#     def update_normalizer_from_batch(self, normalized_actor_obs: Array) -> None:
#         """Update the running observation normalizer from a rollout batch.

#         Parameters
#         ----------
#         normalized_actor_obs : Array, shape (n_samples, ACTOR_OBS_DIM)
#             Actor observations built with the previous normalizer state.

#         Notes:
#         -----
#         ``obs.py`` currently exposes only the normalized builder. The wrapper
#         inverts the affine normalization before the Welford update; clipped
#         features remain clipped and are therefore conservative outliers.
#         """
#         batch = jnp.asarray(normalized_actor_obs)
#         _validate_batch_shape(batch, ACTOR_OBS_DIM, "normalized_actor_obs")
#         std = jnp.sqrt(self.normalizer.var + obs_encoding.NORM_VAR_EPS)
#         raw_batch = batch * std + self.normalizer.mean
#         self.normalizer = obs_encoding.update_normalizer(self.normalizer, raw_batch)

#     def set_normalizer(self, normalizer: obs_encoding.NormalizerState) -> None:
#         """Replace the wrapper normalizer with a frozen/restored state."""
#         self.normalizer = normalizer

#     def set_stage(self, stage_idx: int) -> None:
#         """Switch curriculum stage and reinstantiate the vectorized env.

#         Parameters
#         ----------
#         stage_idx : int
#             Zero-indexed curriculum stage.
#         """
#         self.stage = self._stage_from_index(stage_idx)
#         self.stage_idx = stage_idx
#         if self.env is not None:
#             self.env.close()
#         self.env = self._make_env(self.stage)
#         self.prev_action_env_4vec = jnp.zeros((self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
#         self.prev_physical_action = jnp.zeros((self.n_envs, RAW_ACTION_DIM), dtype=jnp.float32)
#         self.reset(seed=self.seed + stage_idx)

#     def get_thrust_bounds(self) -> tuple[float, float]:
#         """Return cached total-thrust bounds ``(min, max)`` in newtons."""
#         return self._thrust_bounds

#     def true_gates_pos(self) -> Array:
#         """Return unmasked true gate positions, shape ``(n_envs, n_gates, 3)``."""
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         return jnp.asarray(self.env.data.gates_pos)

#     def true_gates_quat(self) -> Array:
#         """Return unmasked true gate orientations, shape ``(n_envs, n_gates, 4)``."""
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         return jnp.asarray(self.env.data.gates_quat)

#     def true_obstacles_pos(self) -> Array:
#         """Return true obstacle positions, shape ``(n_envs, n_obstacles, 3)``."""
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         return jnp.asarray(self.env.data.obstacles_pos)

#     def render(self) -> None:
#         """Render the underlying env through Crazyflow."""
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         self.env.render()

#     def close(self) -> None:
#         """Close the underlying env."""
#         if self.env is not None:
#             self.env.close()

#     def _make_env(self, stage: CurriculumStage) -> VecDroneRaceEnv:
#         """Create a ``VecDroneRaceEnv`` for one curriculum stage."""
#         race_cfg = self._load_stage_config(stage)
#         randomizations = self._stage_randomizations(race_cfg, stage)
#         disturbances = self._stage_disturbances(stage)
#         env = VecDroneRaceEnv(
#             num_envs=self.n_envs,
#             freq=race_cfg.env.freq,
#             sim_config=race_cfg.sim,
#             track=race_cfg.env.track,
#             sensor_range=race_cfg.env.sensor_range,
#             control_mode="attitude",
#             disturbances=disturbances,
#             randomizations=randomizations,
#             seed=self.seed,
#             max_episode_steps=self.max_episode_steps,
#             device=self.device,
#         )
#         # v69: optional per-env sensor_range domain randomization.
#         # Deploy sensor_range=0.7 is in-distribution by intent of the [min,max]
#         # range chosen at CLI; sampled once per stage construction and held
#         # constant across the stage's training (re-sampled on stage switch
#         # via :meth:`set_stage`). Broadcast shape (n_envs, 1, 1) matches the
#         # ``(n_envs, n_drones, n_gates)`` shape that ``data.sensor_range``
#         # is compared against in ``race_core._reset_env_data`` and
#         # ``_update_visited_objects``.
#         if stage.sensor_range_random_max > stage.sensor_range_random_min:
#             rng = np.random.default_rng(self.seed + self.stage_idx + 1)
#             per_env_sr = rng.uniform(
#                 low=float(stage.sensor_range_random_min),
#                 high=float(stage.sensor_range_random_max),
#                 size=(self.n_envs, 1, 1),
#             ).astype(np.float32)
#             env.data = env.data.replace(sensor_range=jnp.asarray(per_env_sr))
#         return env

#     def _load_stage_config(self, stage: CurriculumStage) -> ConfigDict:
#         """Load and patch the TOML config for a stage without touching disk."""
#         config_path = REPO_ROOT / "config" / LEVEL_CONFIG_NAME.format(level=stage.level)
#         race_cfg = copy.deepcopy(load_config(config_path))
#         race_cfg.env.control_mode = "attitude"
#         return race_cfg

#     def _stage_randomizations(
#         self, race_cfg: ConfigDict, stage: CurriculumStage
#     ) -> ConfigDict | None:
#         """Return per-stage track-side randomizations (gate_pos, gate_rpy, obstacle_pos).

#         Levels 2 / 3 use the toml's ``gate_pos`` / ``gate_rpy`` / ``obstacle_pos``
#         bounds scaled by ``stage.gate_rand_scale``. The framework's
#         randomize fns (post-upstream PR #91) populate ``nominal_*`` from
#         the placed layout and apply the wobble on top of ``gates_pos`` /
#         ``gates_quat`` / ``obstacles_pos`` so the pre-visit obs reports
#         the placed layout and the post-visit obs reports placement + wobble.

#         Drone-pose randomizations are excluded because
#         ``_apply_reset_perturbation`` applies stage-specific drone perturbations
#         itself. ``gate_rpy`` matches the eval-time distribution applied by
#         ``scripts/sim.py`` on level 3.
#         """
#         if stage.level not in (2, 3):
#             return None
#         scale = float(stage.gate_rand_scale)
#         if scale <= 0.0:
#             return None
#         cfg = ConfigDict()
#         for key in ("gate_pos", "gate_rpy", "obstacle_pos"):
#             if key not in race_cfg.env.randomizations:
#                 continue
#             src = race_cfg.env.randomizations[key]
#             entry = ConfigDict()
#             entry.fn = src.fn
#             kwargs = ConfigDict()
#             kwargs.minval = [float(v) * scale for v in src.kwargs.minval]
#             kwargs.maxval = [float(v) * scale for v in src.kwargs.maxval]
#             entry.kwargs = kwargs
#             cfg[key] = entry
#         return cfg if len(cfg) else None

#     def _stage_disturbances(self, stage: CurriculumStage) -> ConfigDict | None:
#         """Return disturbances for a curriculum stage."""
#         _ = stage
#         # TODO(stage4): inject DRSchedule action/dynamics disturbances here.
#         return None

#     def _build_critic_obs(self, env_obs: dict[str, Array]) -> Array:
#         """Build batched critic obs with privileged true gate/obstacle poses."""
#         return obs_encoding.vmap_build_critic_obs(
#             env_obs,
#             self.prev_action_env_4vec,
#             self.normalizer,
#             true_gates_pos=self.true_gates_pos(),
#             true_gates_quat=self.true_gates_quat(),
#             true_obstacles_pos=self.true_obstacles_pos(),
#         )

#     def _apply_reset_perturbation(self, mask: Array) -> None:
#         """Apply curriculum reset perturbations to selected env worlds."""
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         mask = jnp.asarray(mask, dtype=bool)
#         if not bool(np.asarray(jnp.any(mask))):
#             return

#         self.rng_key, pos_key, vel_key, yaw_key = jax.random.split(self.rng_key, 4)
#         states = self.env.data.sim_data.states
#         # v41: dropped ``start_pos`` snapshot — seg-init no longer needs the
#         # spawn position as a segment-0 anchor. Entry geometry is computed
#         # from the next gate's pose only.
#         pos_delta = jax.random.uniform(
#             pos_key,
#             shape=states.pos.shape,
#             minval=-self.stage.reset_pos_perturb_m,
#             maxval=self.stage.reset_pos_perturb_m,
#         )
#         vel = jax.random.uniform(
#             vel_key,
#             shape=states.vel.shape,
#             minval=-self.stage.reset_vel_perturb_mps,
#             maxval=self.stage.reset_vel_perturb_mps,
#         )
#         yaw_delta = jax.random.uniform(
#             yaw_key,
#             shape=states.quat.shape[:-1],
#             minval=-self.stage.reset_yaw_perturb_rad,
#             maxval=self.stage.reset_yaw_perturb_rad,
#         )

#         mask_broadcast = mask[:, None, None]
#         pos = jnp.clip(
#             states.pos + pos_delta, self.env.data.pos_limit_low, self.env.data.pos_limit_high
#         )
#         quat = _apply_yaw_delta(states.quat, yaw_delta, mask_broadcast)
#         states = states.replace(
#             pos=jnp.where(mask_broadcast, pos, states.pos),
#             vel=jnp.where(mask_broadcast, vel, states.vel),
#             quat=quat,
#         )
#         sim_data = self.env.data.sim_data.replace(states=states)
#         self.env.data = self.env.data.replace(sim_data=sim_data)
#         self.env.data = _reset_env_data(self.env.data, mask)
#         self._apply_segment_init(mask)

#     def _apply_segment_init(self, mask: Array) -> None:
#         """Re-spawn a Bernoulli-selected subset of envs at the target gate's entry waypoint.

#         Implements Phase 1 of Song et al. 2023 §III-B: with probability
#         ``stage.segment_init_prob``, override the just-reset drone state so
#         the drone arrives at the entry side of a uniformly-random target
#         gate with velocity in that gate's traversal direction, and
#         ``target_gate`` advanced to match.

#         v41: see ``rollout._apply_segment_init`` docstring for the diagnosis
#         that motivated dropping v29's midpoint+straight-line geometry.

#         Parameters
#         ----------
#         mask : Array, shape (n_envs,)
#             Boolean mask of envs that just reset (and are eligible for
#             segment-init). Envs outside this mask are untouched.
#         """
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         if self.stage.segment_init_prob <= 0.0:
#             return
#         mask = jnp.asarray(mask, dtype=bool)
#         if not bool(np.asarray(jnp.any(mask))):
#             return

#         self.rng_key, bern_key, seg_key, jit_key = jax.random.split(self.rng_key, 4)
#         data = self.env.data
#         states = data.sim_data.states
#         # Pre-wobble layout. ``nominal_gates_pos`` is set by the framework's
#         # ``build_full_track_randomization_fn`` to the just-placed layout
#         # (upstream PR #91) and is not mutated by the per-axis gate_pos
#         # randomization that follows, so segment anchors use the placed
#         # geometry rather than the post-wobble physics.
#         nominal_gates_pos = data.nominal_gates_pos
#         nominal_gates_quat = data.nominal_gates_quat
#         n_gates = nominal_gates_pos.shape[1]

#         do_seg = (
#             jax.random.bernoulli(bern_key, p=self.stage.segment_init_prob, shape=(self.n_envs,))
#             & mask
#         )
#         if not bool(np.asarray(jnp.any(do_seg))):
#             return

#         segment_idx = jax.random.randint(seg_key, shape=(self.n_envs,), minval=0, maxval=n_gates)
#         env_arange = jnp.arange(self.n_envs)
#         next_gate = nominal_gates_pos[env_arange, segment_idx]
#         next_gate_quat = nominal_gates_quat[env_arange, segment_idx]

#         # v41: spawn at the target gate's entry waypoint on its -x_local
#         # side with velocity in the gate's +x_local direction. Replaces
#         # v29's midpoint(prev_gate, next_gate) + unit(next-prev) geometry
#         # — which was ~90 deg off the U-turn gate's traversal axis and
#         # left Phase 2 buffer slot 2 essentially empty across the v38
#         # series. Mirrors ``rollout._apply_segment_init`` exactly so the
#         # eager and JIT branches share the same state distribution. See
#         # the full diagnosis in that function's inline comment.
#         gate_xaxis_world = JaxRotation.from_quat(next_gate_quat).apply(
#             jnp.array([1.0, 0.0, 0.0])
#         )  # (n_envs, 3); gate's traversal axis in world
#         entry_offset_m = self.reward_cfg.lookahead_entry_offset_m
#         entry_waypoint = next_gate - entry_offset_m * gate_xaxis_world

#         jitter = jax.random.uniform(
#             jit_key,
#             shape=(self.n_envs, 3),
#             minval=-self.stage.segment_init_perturb_m,
#             maxval=self.stage.segment_init_perturb_m,
#         )
#         new_pos = jnp.clip(entry_waypoint + jitter, data.pos_limit_low, data.pos_limit_high)

#         seg_vel = self.stage.segment_init_vel_mps * gate_xaxis_world  # (n_envs, 3)

#         mask_b3 = do_seg[:, None, None]
#         new_pos_b = new_pos[:, None, :]
#         new_vel_b = seg_vel[:, None, :]
#         # 2026-05-25 seg-init audit: identity quat caused nose/velocity
#         # mismatch (drone body +x pointed world +x while velocity pointed
#         # along gate's +x_local), and ang_vel was inherited from the
#         # previous terminated episode (typically a tumble from crash).
#         # Mirror the fix in rollout._apply_segment_init exactly so the
#         # eager and JIT-scan branches share the same state distribution.
#         yaw = jnp.arctan2(gate_xaxis_world[..., 1], gate_xaxis_world[..., 0])
#         half_yaw = yaw * 0.5
#         seg_quat = jnp.stack(
#             [jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(half_yaw), jnp.cos(half_yaw)],
#             axis=-1,
#         )
#         seg_quat_b = seg_quat[:, None, :]
#         new_ang_vel_b = jnp.zeros_like(states.ang_vel)
#         new_states = states.replace(
#             pos=jnp.where(mask_b3, new_pos_b, states.pos),
#             vel=jnp.where(mask_b3, new_vel_b, states.vel),
#             quat=jnp.where(mask_b3, seg_quat_b, states.quat),
#             ang_vel=jnp.where(mask_b3, new_ang_vel_b, states.ang_vel),
#         )

#         new_target = jnp.where(
#             do_seg[:, None], segment_idx[:, None].astype(data.target_gate.dtype), data.target_gate
#         )

#         sim_data = data.sim_data.replace(states=new_states)
#         self.env.data = data.replace(sim_data=sim_data, target_gate=new_target)
#         # Refresh ``last_drone_pos`` / ``takeoff_pos`` / ``gates_visited`` /
#         # ``obstacles_visited`` so they're consistent with the just-spawned
#         # state. Mirrors the in-scan call from ``rollout._apply_segment_init``.
#         self.env.data = _refresh_aux_fields_after_respawn(
#             self.env.data, do_seg, new_pos, segment_idx
#         )

#     def _reset_done_worlds(self, mask: Array) -> None:
#         """Reset completed worlds after terminal rewards have been computed."""
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         self.env.data, _ = self.env._reset(self.env.data, seed=None, mask=mask)
#         self._apply_reset_perturbation(mask)

#     def _read_env_obs(self) -> dict[str, Array]:
#         """Read squeezed env observations directly from ``RaceCoreEnv.data``."""
#         if self.env is None:
#             raise RuntimeError("Env is not constructed.")
#         env_obs = race_core_obs(self.env.data)
#         return {key: jnp.asarray(value[:, 0]) for key, value in env_obs.items()}

#     def _stage_from_index(self, stage_idx: int) -> CurriculumStage:
#         """Validate and return a curriculum stage."""
#         if stage_idx < 0 or stage_idx >= len(self.curriculum.stages):
#             raise ValueError(
#                 f"stage_idx must be in [0, {len(self.curriculum.stages) - 1}], got {stage_idx}"
#             )
#         return self.curriculum.stages[stage_idx]


# def make_env(
#     train_cfg: TrainConfig | None = None,
#     *,
#     n_envs: int | None = None,
#     stage_idx: int | None = None,
#     seed: int | None = None,
#     device: str = "gpu",
# ) -> RLSongVecEnv:
#     """Construct an :class:`RLSongVecEnv`.

#     Parameters
#     ----------
#     train_cfg : TrainConfig, optional
#         Training config. A default config is used when omitted.
#     n_envs, stage_idx, seed, device
#         Forwarded to :class:`RLSongVecEnv`.

#     Returns:
#     -------
#     RLSongVecEnv
#         Wrapped vectorized racing environment.
#     """
#     return RLSongVecEnv(train_cfg, n_envs=n_envs, stage_idx=stage_idx, seed=seed, device=device)


# def _to_jax_obs(env_obs: dict[str, Any]) -> dict[str, Array]:
#     """Convert a squeezed env observation dict to JAX arrays."""
#     return {key: jnp.asarray(value) for key, value in env_obs.items()}


# def _apply_yaw_delta(quat: Array, yaw_delta: Array, mask: Array) -> Array:
#     """Apply a yaw delta with SciPy's rotation implementation."""
#     flat_quat = np.asarray(quat).reshape(-1, quat.shape[-1])
#     flat_yaw = np.asarray(yaw_delta).reshape(-1)
#     rpy = Rotation.from_quat(flat_quat).as_euler("xyz")
#     rpy[:, 2] = rpy[:, 2] + flat_yaw
#     perturbed = Rotation.from_euler("xyz", rpy).as_quat().reshape(quat.shape)
#     return jnp.where(mask, jnp.asarray(perturbed, dtype=quat.dtype), quat)


# def _validate_action_shape(action: Array, n_envs: int, action_dim: int, name: str) -> None:
#     """Validate a batched action shape."""
#     if action.shape != (n_envs, action_dim):
#         raise ValueError(f"{name} must have shape {(n_envs, action_dim)}; got {action.shape}")


# def _validate_batch_shape(batch: Array, obs_dim: int, name: str) -> None:
#     """Validate a two-dimensional observation batch."""
#     if batch.ndim != 2 or batch.shape[-1] != obs_dim:
#         raise ValueError(f"{name} must have shape (n_samples, {obs_dim}); got {batch.shape}")


# def _default_curriculum() -> CurriculumConfig:
#     """Return the default curriculum for external callers."""
#     return default_curriculum()


# reference 2:
# """SB3 ``VecEnv`` adapter for the JAX crazyflow race env.

# Bridges :class:`lsy_drone_racing.envs.drone_race.VecDroneRaceEnv` to the
# Stable-Baselines3 / SBX vectorized-env protocol so the SBX PPO training loop
# in :mod:`sbx` can drive the same JAX simulation the Song-2023 prototype used.

# Three design choices distinguish this wrapper from
# :class:`lsy_drone_racing.control.rl_song.env_wrapper.RLSongVecEnv`.

# 1. **Flat-concat observation space.** A single ``Box(2*ACTOR_OBS_DIM,)`` whose
#    first half is the masked actor obs and second half is the privileged critic
#    obs. The Task 4 ``Actor`` / ``Critic`` flax modules slice their respective
#    half. ``sbx.PPO`` has no dict-obs support
#    (``sbx/ppo/ppo.py:297`` calls ``rollout_data.observations.numpy()``
#    unconditionally), so a single tensor is the only transport that works with
#    the stock training loop. See the 2026-05-24 addendum in
#    ``docs/specs/2026-05-24-sbx-migration-design.md``.
# 2. **Masked-geometry reward.** ``step_reward`` is called without ``true_*``
#    kwargs, so the reward gradients only see what the actor sees. This is
#    risk-3 mitigation from the migration design: the v85 line of attack
#    removes the masked-vs-true reward leak that hid the gate-3 graze.
# 3. **Two independent normalizers.** ``self.actor_normalizer`` is fed the
#    masked obs (what the actor will see at deploy); ``self.critic_normalizer``
#    is fed the privileged obs (what only the critic ever sees). Both are
#    updated externally by a SB3 callback after each rollout — this wrapper
#    only exposes setters and the current state.

# The wrapper depends on the JAX env's private ``_reset(data, seed=None,
# mask=mask)`` to autoreset only the worlds that finished on the last step.
# The public ``.reset()`` resets every world, which would discard half the
# rollout buffer in an off-policy or asynchronous setting.
# """

# from __future__ import annotations

# from typing import TYPE_CHECKING, Any

# import jax.numpy as jnp
# import numpy as np
# from gymnasium import spaces
# from jax import Array
# from stable_baselines3.common.vec_env import VecEnv

# from lsy_drone_racing.control.rl_song import obs as obs_encoding
# from lsy_drone_racing.control.rl_song.config import (
#     ACTOR_OBS_DIM,
#     ENV_ACTION_DIM,
#     RAW_ACTION_DIM,
#     TANGENT_ALPHA_MAX_RAD,
#     RewardConfig,
# )
# from lsy_drone_racing.control.rl_song.policy import raw_to_env_action, raw_to_physical_action
# from lsy_drone_racing.control.rl_song.reward import step_reward
# from lsy_drone_racing.control.rl_song.rollout import (
#     SRC_TRUE_START,
#     Phase2Buffer,
#     empty_phase2_buffer,
#     phase2_state_dim,
# )
# from lsy_drone_racing.envs.race_core import obs as race_core_obs

# if TYPE_CHECKING:
#     from collections.abc import Callable

#     from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

# # Bounds for the SB3 Box observation space. The normalizer clips every feature
# # into ``±NORM_CLIP``, so anything beyond this range is a normalizer bug.
# OBS_LOW: float = -obs_encoding.NORM_CLIP
# OBS_HIGH: float = +obs_encoding.NORM_CLIP

# # SB3's base_class.py:199 asserts finite bounds on continuous action spaces.
# # The policy emits a Gaussian raw action (mu near 0, sigma ≈ 0.6 from
# # log_std_init=-0.5); ``raw_to_env_action`` squashes its output regardless, so
# # the cap here is purely the SB3 sanity check. ±10 is many sigmas above the
# # sampling range without ever clipping a realistic action.
# RAW_ACTION_BOUND: float = 10.0


# class RLSBXVecEnv(VecEnv):
#     """SB3 ``VecEnv`` wrapping a JAX :class:`VecDroneRaceEnv`.

#     Parameters
#     ----------
#     jax_env : VecDroneRaceEnv
#         Already-constructed vectorized JAX env. The wrapper does not create or
#         configure the env — that's the caller's job (see Task 7 / the training
#         CLI). The wrapper does call the env's ``.step``, ``._reset``, and
#         reads ``.data``.
#     reward_cfg : RewardConfig
#         Reward shaping configuration consumed by
#         :func:`lsy_drone_racing.control.rl_song.reward.step_reward`.
#     alpha_max : float
#         Tangent-space scaling for the raw-to-env action projection (Schuck
#         et al. 2025).
#     thrust_min, thrust_max : float
#         Total-thrust bounds in newtons. Used by the raw-to-env projection to
#         rescale the policy's collective-thrust output.
#     n_envs : int
#         Vectorization width. Must equal ``jax_env``'s vec width.
#     seed : int
#         Initial JAX env seed.
#     reset_done_hook : Callable[[Array], None], optional
#         Optional callable invoked inside :meth:`step_wait` after the JAX env
#         autoreset, with the boolean ``done`` mask of shape ``(n_envs,)``.
#         Wired by the training entry to
#         :meth:`RLSongVecEnv._apply_reset_perturbation` so the curriculum
#         reset-perturbation (drone pos/vel/yaw noise) and Song 2023 §III-B
#         Phase-1 seg-init (segment_init_prob mid-track re-spawn) fire on the
#         same worlds that just autoreset. Without this hook the wrapper would
#         silently disable seg-init — the lever that took v77 from 0% to 19%
#         L2 finish-rate. The hook may mutate ``self.jax_env.data`` in place;
#         the subsequent ``race_core_obs`` rebuild reads the post-hook state.

#     Notes:
#     -----
#     SB3's modern (2.x) VecEnv contract is enforced:

#     * ``reset(self)`` takes no args and returns just the obs.
#     * ``step_wait()`` returns ``(obs, reward, done, infos)`` — a single ``done``
#       flag combining terminated and truncated, with ``TimeLimit.truncated``
#       stored on the per-env info dict so PPO can do correct value
#       bootstrapping on timeouts.
#     """

#     metadata = {"render_modes": []}

#     def __init__(
#         self,
#         jax_env: Any,
#         reward_cfg: RewardConfig,
#         alpha_max: float = TANGENT_ALPHA_MAX_RAD,
#         *,
#         thrust_min: float,
#         thrust_max: float,
#         n_envs: int,
#         seed: int,
#         reset_done_hook: Callable[[Array], None] | None = None,
#         seg_init_kwargs: dict[str, float | int] | None = None,
#         phase2_capacity_per_gate: int = 1,
#     ):
#         """Construct the wrapper. See the class docstring for parameter details.

#         Parameters
#         ----------
#         seg_init_kwargs : dict[str, float], optional
#             Seg-init / drone-state perturbation knobs forwarded verbatim
#             to :func:`rl_sbx.rollout.make_static_config` by
#             :class:`JitScanPPO`. Expected keys: ``reset_pos_perturb_m``,
#             ``reset_vel_perturb_mps``, ``reset_yaw_perturb_rad``,
#             ``segment_init_prob``, ``segment_init_perturb_m``,
#             ``segment_init_vel_mps``. ``None`` defaults to an empty dict
#             (seg-init disabled). Mirrors :attr:`RLSongVecEnv.stage`
#             fields one-for-one — the training entry is expected to
#             populate this from ``wrapper.stage``.
#         """
#         observation_space = spaces.Box(
#             low=OBS_LOW, high=OBS_HIGH, shape=(2 * ACTOR_OBS_DIM,), dtype=np.float32
#         )
#         action_space = spaces.Box(
#             low=-RAW_ACTION_BOUND, high=RAW_ACTION_BOUND, shape=(RAW_ACTION_DIM,), dtype=np.float32
#         )
#         super().__init__(
#             num_envs=n_envs, observation_space=observation_space, action_space=action_space
#         )

#         self.jax_env = jax_env
#         self.reward_cfg = reward_cfg
#         self.alpha_max = float(alpha_max)
#         self.thrust_min = float(thrust_min)
#         self.thrust_max = float(thrust_max)
#         self.seed_value = int(seed)
#         self.reset_done_hook = reset_done_hook
#         self.seg_init_kwargs: dict[str, float | int] = (
#             dict(seg_init_kwargs) if seg_init_kwargs is not None else {}
#         )
#         self.phase2_prob = float(self.seg_init_kwargs.get("phase2_prob", 0.0))
#         self.phase2_warmup_steps = int(self.seg_init_kwargs.get("phase2_warmup_steps", 0))
#         self.phase2_capacity_per_gate = int(phase2_capacity_per_gate)

#         n_obstacles = int(self.jax_env.data.obstacles_pos.shape[1])
#         n_gates = int(self.jax_env.data.gates_pos.shape[1])
#         state_dim = phase2_state_dim(n_obstacles, n_gates)
#         self.phase2_buffer = empty_phase2_buffer(
#             n_gates=n_gates, capacity=self.phase2_capacity_per_gate, state_dim=state_dim
#         )
#         self.episode_source = jnp.full((self.num_envs,), SRC_TRUE_START, dtype=jnp.int8)

#         # v126 (2026-05-26): single-normalizer ablation. ``critic_normalizer``
#         # is aliased to ``actor_normalizer`` so both halves of the flat-concat
#         # obs are normalized against the same Welford statistics — matches
#         # rl_song's reference behaviour (`rl_song/rollout.py` uses one
#         # ``normalizer`` for both ``build_actor_obs`` and ``build_critic_obs``).
#         # The SBX migration default had been two independent normalizers; the
#         # v56-step163M replay (2026-05-26) confirmed env stability and pointed
#         # to the framework diffs (two normalizers + halved PPO update geometry)
#         # as the regression source for v113h-v124. The alias is kept valid on
#         # every ``set_actor_normalizer`` call.
#         self.actor_normalizer = obs_encoding.init_normalizer(ACTOR_OBS_DIM)
#         self.critic_normalizer = self.actor_normalizer

#         self._prev_action: Array = jnp.zeros((n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
#         self._prev_physical_action: Array = jnp.zeros((n_envs, RAW_ACTION_DIM), dtype=jnp.float32)
#         self._prev_env_obs: dict[str, Array] | None = None
#         self._pending_actions: np.ndarray | None = None

#     # ------------------------------------------------------------------
#     # SB3 VecEnv interface
#     # ------------------------------------------------------------------
#     def reset(self) -> VecEnvObs:
#         """Reset every world and return the flat-concat observation.

#         Returns:
#         -------
#         observations : np.ndarray, shape (n_envs, 2*ACTOR_OBS_DIM)
#             First half is masked actor obs, second half is privileged critic
#             obs. See module docstring for the layout rationale.
#         """
#         env_obs, _info = self.jax_env.reset(seed=self.seed_value)
#         env_obs = _to_jax_obs(env_obs)
#         self._prev_env_obs = env_obs
#         self._prev_action = jnp.zeros((self.num_envs, ENV_ACTION_DIM), dtype=jnp.float32)
#         self._prev_physical_action = jnp.zeros((self.num_envs, RAW_ACTION_DIM), dtype=jnp.float32)
#         self.episode_source = jnp.full((self.num_envs,), SRC_TRUE_START, dtype=jnp.int8)
#         return self._build_obs(env_obs)

#     def step_async(self, actions: np.ndarray) -> None:
#         """Stash actions for the next :meth:`step_wait`."""
#         self._pending_actions = np.asarray(actions, dtype=np.float32)

#     def step_wait(self) -> VecEnvStepReturn:
#         """Apply the stashed actions and return ``(obs, reward, done, infos)``.

#         Returns:
#         -------
#         obs : np.ndarray, shape (n_envs, 2*ACTOR_OBS_DIM)
#             Flat-concat obs, see :meth:`reset` for layout.
#         reward : np.ndarray, shape (n_envs,)
#         done : np.ndarray, shape (n_envs,)
#             ``terminated | truncated``. SB3 PPO uses the per-env
#             ``TimeLimit.truncated`` flag on the ``infos`` list to decide
#             whether to bootstrap value on timeout.
#         infos : list[dict]
#         """
#         if self._pending_actions is None:
#             raise RuntimeError("step_wait called before step_async.")
#         if self._prev_env_obs is None:
#             raise RuntimeError("step_wait called before reset.")

#         raw_action = jnp.asarray(self._pending_actions, dtype=jnp.float32)
#         self._pending_actions = None
#         prev_env_obs = self._prev_env_obs

#         env_action = raw_to_env_action(
#             raw_action,
#             jnp.asarray(prev_env_obs["quat"]),
#             self.thrust_min,
#             self.thrust_max,
#             alpha_max=self.alpha_max,
#         )
#         physical_action = raw_to_physical_action(
#             raw_action, self.thrust_min, self.thrust_max, alpha_max=self.alpha_max
#         )

#         env_obs, _env_reward, terminated, truncated, _env_info = self.jax_env.step(env_action)
#         env_obs = _to_jax_obs(env_obs)

#         terminated = jnp.asarray(terminated, dtype=bool)
#         truncated = jnp.asarray(truncated, dtype=bool)
#         current_target = env_obs["target_gate"]
#         prev_target = prev_env_obs["target_gate"]
#         finished = current_target < 0
#         terminated = terminated | finished
#         gate_just_passed = ((current_target > prev_target) & (prev_target >= 0)) | (
#             finished & (prev_target >= 0)
#         )

#         # Masked-geometry reward: no ``true_*`` kwargs. The reward sees only
#         # what the actor sees, so gradients can't exploit information the
#         # policy is denied at deploy. The critic still consumes privileged
#         # values via the ``critic`` channel of the dict obs.
#         reward, _components = step_reward(
#             env_obs,
#             prev_env_obs,
#             terminated,
#             truncated,
#             finished,
#             gate_just_passed,
#             self.reward_cfg,
#             physical_action=physical_action,
#             prev_physical_action=self._prev_physical_action,
#         )

#         done = terminated | truncated
#         done_np = np.asarray(done)
#         truncated_np = np.asarray(truncated)
#         terminated_np = np.asarray(terminated)

#         # Autoreset only worlds that finished. The high-level ``.reset()``
#         # resets every world, which would discard fresh in-flight transitions.
#         # ``race_core_obs(data)`` is shape ``(n_envs, n_drones=1, ...)`` — the
#         # high-level ``VecDroneRaceEnv.step/reset`` squeezes the drone axis
#         # before returning. We replicate that squeeze here so the obs dict is
#         # consistent across the pre-step and post-autoreset paths.
#         if bool(done_np.any()):
#             self.jax_env.data, _ = self.jax_env._reset(self.jax_env.data, seed=None, mask=done)
#             # The hook may mutate ``self.jax_env.data`` (e.g. seg-init re-spawns
#             # a Bernoulli-selected subset of just-reset worlds at mid-track
#             # gate-entry waypoints). Must run BEFORE the ``race_core_obs``
#             # rebuild so the rebuilt obs reflects the post-hook state.
#             if self.reset_done_hook is not None:
#                 self.reset_done_hook(done)
#             raw_env_obs = race_core_obs(self.jax_env.data)
#             env_obs = {key: jnp.asarray(value[:, 0]) for key, value in raw_env_obs.items()}

#         reset_prev_action = jnp.zeros_like(env_action)
#         self._prev_action = jnp.where(done[:, None], reset_prev_action, env_action)
#         reset_prev_physical_action = jnp.zeros_like(physical_action)
#         self._prev_physical_action = jnp.where(
#             done[:, None], reset_prev_physical_action, physical_action
#         )
#         self._prev_env_obs = env_obs

#         obs_array = self._build_obs(env_obs)
#         # ``np.asarray`` on a JAX array returns a read-only view backed by the
#         # device buffer. SBX's ``OnPolicyAlgorithm.collect_rollouts`` mutates
#         # the returned reward in-place when bootstrapping value on timeout
#         # (``rewards[idx] += gamma * terminal_value`` at
#         # ``sbx/common/on_policy_algorithm.py:210``), so the reward must be a
#         # writable numpy array. ``np.array`` forces a copy.
#         reward_np = np.array(reward, dtype=np.float32)
#         infos = _build_infos(self.num_envs, terminated_np, truncated_np, obs_array)
#         return obs_array, reward_np, done_np, infos

#     def close(self) -> None:
#         """Close the underlying JAX env."""
#         if self.jax_env is not None:
#             self.jax_env.close()

#     def seed(self, seed: int | None = None) -> list[int | None]:
#         """Set the wrapper seed.

#         The JAX env is reseeded on the next :meth:`reset`. SB3's
#         :class:`VecNormalize` and friends call this once at construction.
#         """
#         if seed is not None:
#             self.seed_value = int(seed)
#         return [self.seed_value] * self.num_envs

#     def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
#         """Return ``getattr(self, attr_name)`` replicated per requested env.

#         The wrapper is a single Python object backing all ``num_envs``
#         simulated envs, so the attribute is the same across indices.
#         """
#         n = _resolve_indices_count(indices, self.num_envs)
#         return [getattr(self, attr_name)] * n

#     def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
#         """Set ``attr_name`` on the wrapper.

#         Per-env setting is not supported because the wrapper holds a single
#         JAX env shared across all vectorized worlds.
#         """
#         if indices is not None:
#             raise NotImplementedError(
#                 "RLSBXVecEnv.set_attr does not support per-env indices: all worlds "
#                 "share a single backing JAX env."
#             )
#         setattr(self, attr_name, value)

#     def env_method(
#         self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
#     ) -> list[Any]:
#         """Call ``method_name`` on the wrapper and replicate the result.

#         Mirrors :meth:`get_attr`'s single-backing-env semantics.
#         """
#         n = _resolve_indices_count(indices, self.num_envs)
#         result = getattr(self, method_name)(*method_args, **method_kwargs)
#         return [result] * n

#     def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
#         """Return ``False`` for every env — this wrapper is the only layer."""
#         n = _resolve_indices_count(indices, self.num_envs)
#         del wrapper_class
#         return [False] * n

#     # ------------------------------------------------------------------
#     # Normalizer accessors (consumed by the Task 6 callback)
#     # ------------------------------------------------------------------
#     def set_actor_normalizer(self, normalizer: obs_encoding.NormalizerState) -> None:
#         """Replace the running actor-obs normalizer.

#         v126: also rebinds ``critic_normalizer`` to maintain the
#         single-normalizer alias established at ``__init__``.
#         """
#         self.actor_normalizer = normalizer
#         self.critic_normalizer = normalizer

#     def set_critic_normalizer(self, normalizer: obs_encoding.NormalizerState) -> None:
#         """Forward to :meth:`set_actor_normalizer` (single-normalizer alias).

#         v126: in two-normalizer mode this would set ``self.critic_normalizer``
#         independently. Under the single-normalizer ablation, any update to
#         the critic stats must also update the actor stats (they are the same
#         object), so we route through :meth:`set_actor_normalizer`.
#         """
#         self.set_actor_normalizer(normalizer)

#     def set_phase2_buffer(self, buffer: Phase2Buffer) -> None:
#         """Replace the Phase-2 replay buffer carried between JIT rollouts."""
#         self.phase2_buffer = buffer

#     def set_episode_source(self, source: Array) -> None:
#         """Replace the per-env episode-source codes carried between rollouts."""
#         self.episode_source = jnp.asarray(source, dtype=jnp.int8)

#     # ------------------------------------------------------------------
#     # Internals
#     # ------------------------------------------------------------------
#     def _build_obs(self, env_obs: dict[str, Array]) -> np.ndarray:
#         """Encode the flat-concat ``[actor | critic]`` obs of shape ``(n_envs, 2D)``.

#         The actor half is built from masked geometry via
#         :func:`obs_encoding.vmap_build_actor_obs`; the critic half is built
#         from privileged geometry via :func:`obs_encoding.vmap_build_critic_obs`
#         with ``true_*`` kwargs read straight from the JAX env's ``data``. Each
#         half is normalized by its own ``NormalizerState`` before
#         concatenation.
#         """
#         actor = obs_encoding.vmap_build_actor_obs(env_obs, self._prev_action, self.actor_normalizer)
#         critic = obs_encoding.vmap_build_critic_obs(
#             env_obs,
#             self._prev_action,
#             self.critic_normalizer,
#             true_gates_pos=jnp.asarray(self.jax_env.data.gates_pos),
#             true_gates_quat=jnp.asarray(self.jax_env.data.gates_quat),
#             true_obstacles_pos=jnp.asarray(self.jax_env.data.obstacles_pos),
#         )
#         return np.concatenate(
#             [np.asarray(actor, dtype=np.float32), np.asarray(critic, dtype=np.float32)], axis=-1
#         )


# def _to_jax_obs(env_obs: dict[str, Any]) -> dict[str, Array]:
#     """Convert an env observation dict to JAX arrays."""
#     return {key: jnp.asarray(value) for key, value in env_obs.items()}


# def _resolve_indices_count(indices: Any, num_envs: int) -> int:
#     """Return the number of envs targeted by SB3's ``indices`` argument."""
#     if indices is None:
#         return num_envs
#     if isinstance(indices, int):
#         return 1
#     return len(list(indices))


# def _build_infos(
#     num_envs: int, terminated: np.ndarray, truncated: np.ndarray, obs: np.ndarray
# ) -> list[dict[str, Any]]:
#     """Build the per-env info list with SB3 timeout-bootstrap fields.

#     Parameters
#     ----------
#     num_envs : int
#     terminated, truncated : np.ndarray, shape (n_envs,)
#     obs : np.ndarray, shape (n_envs, 2*ACTOR_OBS_DIM)
#         Already-built post-step (post-autoreset) flat-concat obs. Stored under
#         ``terminal_observation`` for envs whose ``done`` flag is set, so
#         SB3's PPO can bootstrap correctly on timeouts. Note this is the
#         observation of the freshly-reset world, not of the terminal state
#         before reset — matching the SB3 ``DummyVecEnv`` convention where
#         ``terminal_observation`` is the *reset* obs that VecEnv consumers
#         already have in hand.
#     """
#     infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
#     done_mask = terminated | truncated
#     for env_idx in range(num_envs):
#         if not done_mask[env_idx]:
#             continue
#         infos[env_idx]["TimeLimit.truncated"] = bool(truncated[env_idx] and not terminated[env_idx])
#         infos[env_idx]["terminal_observation"] = obs[env_idx]
#     return infos

