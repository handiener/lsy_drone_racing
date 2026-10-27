"""
保存与恢复完整训练状态
希望方便 resume，就不要只保存 Actor 权重。需要区分两种用途：
类型	应保存什么
推理模型	Actor 参数、观测归一化统计、网络／观测／动作转换配置
训练 checkpoint	上述内容，加 Critic、双方优化器状态、累计步数、训练调度进度和随机数状态
此外，保存：
•	配置及格式版本。
•	当前代码提交、关键依赖版本。
•	网络输入维度、动作维度、标准差方案等兼容性信息。
•	归一化统计的均值、方差和计数。
具体参数和优化器的序列化，优先使用所选 SBX 版本支持的机制，我们的代码负责补齐环境统计、配置等信息，并验证恢复结果。
保存方式建议是：
先写临时 checkpoint → 写入完成 → 原子替换
保留最近几个版本 + 独立的最佳评估模型
恢复时提供两个明确模式：
•	resume：恢复权重、优化器、统计量、步数和相关调度，继续同一次训练。
•	warm_start：加载已有网络和统计量，开启一次新训练；优化器、计数按新实验初始化。
如果不保存并行环境的物理状态，可以在恢复时重新开始 episode，仍然能继续训练；但不能称为逐步完全一致的断点续跑。

"""






# reference code:
# """Checkpoint format for the SBX stack.

# Files written to ``<run_dir>/step_<global_step:012d>/``:

# - ``actor.params.msgpack``  — flax-serialized actor parameters.
# - ``critic.params.msgpack`` — flax-serialized critic parameters.
# - ``actor_normalizer.json``  — Welford running stats for the actor obs.
# - ``critic_normalizer.json`` — Welford running stats for the critic obs.
# - ``policy_config.json``     — ``tangent_alpha_max_rad`` and other deploy-time
#   scalars (mirrors the format ``rl_song.controller`` reads).

# Deploy reads ONLY the ``actor.*`` files plus ``policy_config.json``. The deploy
# loader (``load_actor_only``) explicitly fails if a caller tries to load the
# critic from a deploy path — risk-4 mitigation per the design doc.
# """

# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any

# import jax.numpy as jnp
# from flax import serialization

# from lsy_drone_racing.control.rl_song.obs import NormalizerState

# # Sub-directory name template for a single training step's checkpoint. The
# # zero-padded global step keeps lexical sort == temporal sort, matching the
# # rl_song stack so existing tooling (eval scripts, latest-checkpoint helpers)
# # can be reused without bespoke parsing.
# _STEP_DIR_TEMPLATE: str = "step_{global_step:012d}"

# # File names inside a step directory. Kept as module-level constants so the
# # deploy fence in ``load_actor_only`` and the training resume in ``load_all``
# # stay literally in sync.
# _ACTOR_PARAMS_FILE: str = "actor.params.msgpack"
# _CRITIC_PARAMS_FILE: str = "critic.params.msgpack"
# _ACTOR_NORMALIZER_FILE: str = "actor_normalizer.json"
# _CRITIC_NORMALIZER_FILE: str = "critic_normalizer.json"
# _POLICY_CONFIG_FILE: str = "policy_config.json"


# def save_step(
#     run_dir: Path,
#     global_step: int,
#     actor_params: Any,
#     critic_params: Any,
#     actor_normalizer: NormalizerState,
#     critic_normalizer: NormalizerState,
#     tangent_alpha_max_rad: float,
# ) -> Path:
#     """Write a full step checkpoint and return the step directory.

#     Parameters
#     ----------
#     run_dir : Path
#         Run-level directory; the step sub-directory is created beneath it.
#     global_step : int
#         Training step count, zero-padded to 12 digits in the directory name so
#         lexical sort matches temporal order.
#     actor_params, critic_params : Any
#         Flax parameter PyTrees from ``Actor().init(...)['params']`` and
#         ``Critic().init(...)['params']`` in ``rl_sbx.policy``.
#     actor_normalizer, critic_normalizer : NormalizerState
#         Welford running statistics from ``rl_song.obs.init_normalizer``.
#     tangent_alpha_max_rad : float
#         Deploy-time scalar mirrored in ``policy_config.json`` so the deploy
#         controller can recover the projection cone half-angle without
#         re-reading the training config.

#     Returns:
#     -------
#     Path
#         The created step directory (``run_dir / step_<global_step:012d>``).
#     """
#     step_dir = Path(run_dir) / _STEP_DIR_TEMPLATE.format(global_step=global_step)
#     step_dir.mkdir(parents=True, exist_ok=True)

#     (step_dir / _ACTOR_PARAMS_FILE).write_bytes(serialization.to_bytes(actor_params))
#     (step_dir / _CRITIC_PARAMS_FILE).write_bytes(serialization.to_bytes(critic_params))

#     _save_normalizer(step_dir / _ACTOR_NORMALIZER_FILE, actor_normalizer)
#     _save_normalizer(step_dir / _CRITIC_NORMALIZER_FILE, critic_normalizer)

#     policy_config: dict[str, Any] = {"tangent_alpha_max_rad": float(tangent_alpha_max_rad)}
#     (step_dir / _POLICY_CONFIG_FILE).write_text(
#         json.dumps(policy_config, indent=2), encoding="utf-8"
#     )

#     return step_dir


# def load_actor_only(step_dir: Path) -> dict[str, Any]:
#     """Load actor params + actor normalizer + policy config for deploy.

#     Parameters
#     ----------
#     step_dir : Path
#         Step directory previously written by :func:`save_step`.

#     Returns:
#     -------
#     dict[str, Any]
#         Mapping with keys ``"actor_params"`` (nested dict from
#         ``flax.serialization.msgpack_restore``), ``"actor_normalizer"``
#         (``NormalizerState``), and ``"tangent_alpha_max_rad"`` (float). The
#         ``actor_params`` value is applied at the call site via
#         ``actor.apply({"params": loaded_dict}, obs)`` — no flax template
#         required.

#     Raises:
#     ------
#     FileNotFoundError
#         If ``step_dir`` itself or any of the actor-side files
#         (``actor.params.msgpack``, ``actor_normalizer.json``,
#         ``policy_config.json``) is missing.

#     Notes:
#     -----
#     Risk-4 mitigation: the deploy controller cannot construct the critic
#     observation (no privileged ground-truth gate/obstacle poses on hardware),
#     so the loader is structurally fenced off from any ``critic.*`` file. The
#     fence is enforced by code (this function never references the critic
#     file-name constants) rather than by discipline alone.
#     """
#     step_dir = Path(step_dir)
#     if not step_dir.is_dir():
#         raise FileNotFoundError(f"Step directory does not exist: {step_dir}")

#     actor_params_path = step_dir / _ACTOR_PARAMS_FILE
#     actor_normalizer_path = step_dir / _ACTOR_NORMALIZER_FILE
#     policy_config_path = step_dir / _POLICY_CONFIG_FILE
#     for required in (actor_params_path, actor_normalizer_path, policy_config_path):
#         if not required.is_file():
#             raise FileNotFoundError(f"Required deploy file missing: {required}")

#     actor_params = serialization.msgpack_restore(actor_params_path.read_bytes())
#     actor_normalizer = _load_normalizer(actor_normalizer_path)
#     policy_config = json.loads(policy_config_path.read_text(encoding="utf-8"))

#     return {
#         "actor_params": actor_params,
#         "actor_normalizer": actor_normalizer,
#         "tangent_alpha_max_rad": float(policy_config["tangent_alpha_max_rad"]),
#     }


# def load_all(step_dir: Path, actor_template: Any, critic_template: Any) -> dict[str, Any]:
#     """Load actor + critic params + both normalizers for training resume.

#     Parameters
#     ----------
#     step_dir : Path
#         Step directory previously written by :func:`save_step`.
#     actor_template, critic_template : Any
#         Flax parameter PyTrees with the expected structure (typically
#         ``Actor().init(...)['params']`` and ``Critic().init(...)['params']``).
#         Passed to ``flax.serialization.from_bytes`` so the restored leaves
#         keep their concrete array types.

#     Returns:
#     -------
#     dict[str, Any]
#         Mapping with keys ``"actor_params"``, ``"critic_params"``,
#         ``"actor_normalizer"``, ``"critic_normalizer"``, and
#         ``"tangent_alpha_max_rad"``.

#     Raises:
#     ------
#     FileNotFoundError
#         If ``step_dir`` or any of the expected files is missing.
#     """
#     step_dir = Path(step_dir)
#     if not step_dir.is_dir():
#         raise FileNotFoundError(f"Step directory does not exist: {step_dir}")

#     paths = {
#         "actor_params": step_dir / _ACTOR_PARAMS_FILE,
#         "critic_params": step_dir / _CRITIC_PARAMS_FILE,
#         "actor_normalizer": step_dir / _ACTOR_NORMALIZER_FILE,
#         "critic_normalizer": step_dir / _CRITIC_NORMALIZER_FILE,
#         "policy_config": step_dir / _POLICY_CONFIG_FILE,
#     }
#     for name, path in paths.items():
#         if not path.is_file():
#             raise FileNotFoundError(f"Required file missing ({name}): {path}")

#     actor_params = serialization.from_bytes(actor_template, paths["actor_params"].read_bytes())
#     critic_params = serialization.from_bytes(critic_template, paths["critic_params"].read_bytes())
#     actor_normalizer = _load_normalizer(paths["actor_normalizer"])
#     critic_normalizer = _load_normalizer(paths["critic_normalizer"])
#     policy_config = json.loads(paths["policy_config"].read_text(encoding="utf-8"))

#     return {
#         "actor_params": actor_params,
#         "critic_params": critic_params,
#         "actor_normalizer": actor_normalizer,
#         "critic_normalizer": critic_normalizer,
#         "tangent_alpha_max_rad": float(policy_config["tangent_alpha_max_rad"]),
#     }


# def _save_normalizer(path: Path, state: NormalizerState) -> None:
#     """Serialize a :class:`NormalizerState` to JSON.

#     Parameters
#     ----------
#     path : Path
#         Output JSON file.
#     state : NormalizerState
#         Welford running statistics with jnp.array ``mean``/``var`` and a
#         scalar ``count``.
#     """
#     payload = {
#         "mean": jnp.asarray(state.mean).tolist(),
#         "var": jnp.asarray(state.var).tolist(),
#         "count": float(jnp.asarray(state.count)),
#     }
#     path.write_text(json.dumps(payload), encoding="utf-8")


# def _load_normalizer(path: Path) -> NormalizerState:
#     """Deserialize a :class:`NormalizerState` from JSON.

#     Parameters
#     ----------
#     path : Path
#         Input JSON file written by :func:`_save_normalizer`.

#     Returns:
#     -------
#     NormalizerState
#         Running statistics with float32 jnp.array fields, matching the
#         dtype convention of :func:`rl_song.obs.init_normalizer`.
#     """
#     payload = json.loads(path.read_text(encoding="utf-8"))
#     return NormalizerState(
#         mean=jnp.asarray(payload["mean"], dtype=jnp.float32),
#         var=jnp.asarray(payload["var"], dtype=jnp.float32),
#         count=jnp.asarray(payload["count"], dtype=jnp.float32),
#     )