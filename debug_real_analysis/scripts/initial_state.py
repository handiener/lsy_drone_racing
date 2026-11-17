"""The shared initial-state perturbations used by the recorded validation phases."""

import numpy as np
import jax.numpy as jp
from scipy.spatial.transform import Rotation as R
from harness import rc


def add_start_perturbations(runner):
    original_reset = runner.reset

    def reset_with_jitter(cfg, seed):
        data, raw = original_reset(cfg, seed)
        if seed > 0:
            rng = np.random.default_rng(900000 + seed)
            delta_pos = rng.normal(size=3) * np.array([0.005, 0.005, 0.001])
            delta_rotation = rng.normal(size=3) * np.deg2rad(0.3)
            states = data.sim_data.states.replace(
                pos=jp.asarray(raw["pos"] + delta_pos, dtype=jp.float32)[None, None],
                quat=jp.asarray(
                    (R.from_quat(raw["quat"]) * R.from_rotvec(delta_rotation)).as_quat(),
                    dtype=jp.float32,
                )[None, None],
            )
            data = data.replace(sim_data=data.sim_data.replace(states=states))
            data = rc._reset_env_data(data)
            data = data.replace(
                sim_data=data.sim_data.replace(
                    controls=data.sim_data.controls.replace(
                        body_rate=data.sim_data.controls.body_rate.replace(cmd=states.quat)
                    )
                )
            )
            raw = {k: np.asarray(v[0, 0]) for k, v in rc.obs(data).items()}
        return data, raw

    runner.reset = reset_with_jitter
