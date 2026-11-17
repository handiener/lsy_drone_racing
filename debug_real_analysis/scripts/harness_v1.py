"""Offline wrappers; scores physical terminal states before the official crash teleport."""

from __future__ import annotations
import os, sys, json, hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from paths import BASE, ROOT, OUT

sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation as R
import jax
import jax.numpy as jp
from crazyflow.sim.pipeline import insert_fn_after
from crazyflow.control.transform import motor_force2rotor_vel
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.envs import race_core as rc
from lsy_drone_racing.control.rl_race.controller import RaceController
from lsy_drone_racing.utils import load_config

jax.config.update("jax_compilation_cache_dir", str(OUT / "jax_cache"))


@dataclass(frozen=True)
class Noise:
    omega_tau: float = 0.0
    omega_delay: int = 0
    omega_average: int = 1
    state_delay: int = 0
    state_average: int = 1
    pose_tau: float = 0.0
    vel_tau: float = 0.0
    sample_hold: int = 1
    command_delay: int = 0
    thrust_cap: float = 0.8
    request_cap: float = 0.8
    thrust_scale: float = 1.0
    thrust_tau: float = 0.0
    rotor_rate: float = 1.0
    damping: float = 1.0
    stiffness: float = 1.0
    pose_bias_pitch: float = 0.0
    onset: str = "global"


class DisturbanceWrapper:
    def __init__(self, cfg, initial):
        self.cfg = cfg
        self.history = []
        self.commands = []
        self.fomega = np.array(initial["ang_vel"], float)
        self.fvel = np.array(initial["vel"], float)
        self.fpos = np.array(initial["pos"], float)
        self.frot = R.from_quat(initial["quat"])
        self.held = None
        self.thrust = 0.0854505226

    def observe(self, raw, step):
        c = self.cfg
        self.history.append({k: np.array(v, copy=True) for k, v in raw.items()})
        seen = {k: np.array(v, copy=True) for k, v in raw.items()}
        if c.onset != "global" and int(raw["n_gates_passed"]) < 2:
            return seen
        k = max(0, step - c.state_delay)
        old = self.history[k]
        for f in ["pos", "quat", "vel", "ang_vel"]:
            seen[f] = old[f].copy()
        if c.state_average > 1:
            hs = self.history[max(0, k - c.state_average + 1) : k + 1]
            for f in ["pos", "vel", "ang_vel"]:
                seen[f] = np.mean([h[f] for h in hs], axis=0)
            qs = np.array([h["quat"] for h in hs])
            qs *= np.where(qs @ qs[-1] < 0, -1, 1)[:, None]
            q = qs.mean(axis=0)
            seen["quat"] = q / np.linalg.norm(q)
        k = max(0, step - c.state_delay - c.omega_delay)
        hs = self.history[max(0, k - c.omega_average + 1) : k + 1]
        if c.omega_delay or c.omega_average > 1:
            seen["ang_vel"] = np.mean([h["ang_vel"] for h in hs], axis=0)
        if c.omega_tau > 0:
            self.fomega += (1 - np.exp(-0.02 / c.omega_tau)) * (seen["ang_vel"] - self.fomega)
            seen["ang_vel"] = self.fomega.copy()
        if c.vel_tau > 0:
            self.fvel += (1 - np.exp(-0.02 / c.vel_tau)) * (seen["vel"] - self.fvel)
            seen["vel"] = self.fvel.copy()
        if c.pose_tau > 0:
            alpha = 1 - np.exp(-0.02 / c.pose_tau)
            self.fpos += alpha * (seen["pos"] - self.fpos)
            self.frot = self.frot * R.from_rotvec(
                alpha * (self.frot.inv() * R.from_quat(seen["quat"])).as_rotvec()
            )
            seen["pos"] = self.fpos.copy()
            seen["quat"] = self.frot.as_quat()
        if c.pose_bias_pitch:
            seen["quat"] = (
                R.from_quat(seen["quat"]) * R.from_euler("y", c.pose_bias_pitch, degrees=True)
            ).as_quat()
        if self.held is None or step % c.sample_hold == 0:
            self.held = {f: seen[f].copy() for f in ["pos", "quat", "vel", "ang_vel"]}
        if c.sample_hold > 1:
            for f in self.held:
                seen[f] = self.held[f].copy()
        return {
            k: np.asarray(v, dtype=np.float32) if k in ["pos", "quat", "vel", "ang_vel"] else v
            for k, v in seen.items()
        }

    def command(self, a, raw):
        c = self.cfg
        self.commands.append(a.copy())
        if c.onset != "global" and int(raw["n_gates_passed"]) < 2:
            return a.copy()
        applied = self.commands[max(0, len(self.commands) - 1 - c.command_delay)].copy()
        applied[3] = min(float(applied[3]) * c.thrust_scale, c.request_cap)
        if c.thrust_tau:
            self.thrust += (1 - np.exp(-0.02 / c.thrust_tau)) * (applied[3] - self.thrust)
            applied[3] = self.thrust
        return applied


class Runner:
    def __init__(self, scene="113735", randomized=False):
        self.scene = scene
        self.randomized = randomized
        self.cfg = load_config(ROOT / "config/level2.toml")
        self.cfg.sim.render = False
        paths = sorted((ROOT / "runs").glob(f"deploy_*T{scene}.*Z"))
        self.real_events = (
            [json.loads(l) for l in (paths[0] / "events.jsonl").read_text().splitlines()]
            if paths
            else None
        )
        self.real_track = (
            next(e["track"] for e in self.real_events if e["kind"] == "reset") if paths else None
        )
        self.real_initial = (
            next(e["obs"] for e in self.real_events if e["kind"] == "observation")
            if paths
            else None
        )
        cfg = self.cfg
        self.env = DroneRaceEnv(
            freq=cfg.env.freq,
            sim_config=cfg.sim,
            track=cfg.env.track,
            sensor_range=cfg.env.sensor_range,
            control_mode=cfg.env.control_mode,
            disturbances=cfg.env.disturbances if randomized else None,
            randomizations=cfg.env.randomizations if randomized else None,
            seed=0,
            device="cpu",
            max_episode_steps=450,
        )

        def available_motor_thrust(data):
            rpmmax = motor_force2rotor_vel(
                data.controls.force_torque.params["thrust_max"], data.params.rpm2thrust
            )
            return data.replace(
                controls=data.controls.replace(
                    rotor_vel=jp.minimum(data.controls.rotor_vel, rpmmax)
                )
            )

        insert_fn_after(
            self.env.sim.step_pipeline, "force_torque_controller", available_motor_thrust
        )
        simstep = self.env.sim.build_step_fn()
        apply = self.env.build_apply_action_fn()
        contacts = self.env.build_contact_check_fn()

        @jax.jit
        def advance(data, action):
            data = apply(action, data)
            data = data.replace(sim_data=simstep(data.sim_data, 10))
            hit = contacts(data)
            data = rc._update_disabled_drones(data, hit)
            data = rc._update_visited_objects(data)
            data = rc._update_target_gates(data)
            data = rc._mark_drones_for_reset(data)
            data = data.replace(steps=data.steps + 1)
            return data, rc.obs(data), hit

        self.advance = advance
        obs, _ = self.env.reset(seed=0)
        self.controller = RaceController({k: np.asarray(v) for k, v in obs.items()}, {}, cfg)
        self.ctrl_default = self.env.sim.default_data.controls.attitude.params
        self.ft_default = self.env.sim.default_data.controls.force_torque.params
        print("RUNNER_READY", scene, randomized, flush=True)

    def reset(self, noise, seed):
        self.env.reset(seed=seed)
        data = self.env.data
        if self.real_track and not self.randomized:
            data = data.replace(
                gates_pos=jp.asarray(self.real_track["gates_pos"])[None],
                gates_quat=jp.asarray(self.real_track["gates_quat"])[None],
                obstacles_pos=jp.asarray(self.real_track["obstacles_pos"])[None],
            )
            fields = {
                k: jp.asarray(self.real_initial[k], dtype=jp.float32)[None, None]
                for k in ["pos", "quat", "vel", "ang_vel"]
            }
            data = data.replace(
                sim_data=data.sim_data.replace(states=data.sim_data.states.replace(**fields))
            )
            data = rc._reset_env_data(data)
        params = data.sim_data.params.replace(
            rotor_dyn_coef=data.sim_data.params.rotor_dyn_coef * noise.rotor_rate
        )
        cp = dict(self.ctrl_default)
        cp["kw"] = cp["kw"] * noise.damping
        cp["kR"] = cp["kR"] * noise.stiffness
        fp = dict(self.ft_default)
        fp["thrust_max"] = fp["thrust_max"] * (noise.thrust_cap / 0.8)
        controls = data.sim_data.controls.replace(
            attitude=data.sim_data.controls.attitude.replace(params=cp),
            force_torque=data.sim_data.controls.force_torque.replace(params=fp),
        )
        data = data.replace(sim_data=data.sim_data.replace(params=params, controls=controls))
        self.controller.episode_reset()
        raw = {k: np.asarray(v[0, 0]) for k, v in rc.obs(data).items()}
        return data, raw

    def run(self, noise, seed=0, save=None):
        data, raw = self.reset(noise, seed)
        w = DisturbanceWrapper(noise, raw)
        records = []
        done = False
        for step in range(450):
            seen = w.observe(raw, step)
            requested = self.controller.compute_control(seen, {})
            applied = w.command(requested, raw)
            rpm = np.asarray(data.sim_data.states.rotor_vel[0, 0])
            co = np.asarray(data.sim_data.params.rpm2thrust)
            f = float(np.sum(co[..., 0] + co[..., 1] * rpm + co[..., 2] * rpm**2))
            records.append(
                {
                    "t": step * 0.02,
                    "pos": raw["pos"].tolist(),
                    "quat": raw["quat"].tolist(),
                    "vel": raw["vel"].tolist(),
                    "ang_vel": raw["ang_vel"].tolist(),
                    "seen_pos": seen["pos"].tolist(),
                    "seen_quat": seen["quat"].tolist(),
                    "seen_vel": seen["vel"].tolist(),
                    "seen_ang_vel": seen["ang_vel"].tolist(),
                    "passed": int(raw["n_gates_passed"]),
                    "requested": requested.tolist(),
                    "applied": applied.tolist(),
                    "motor_thrust": f,
                    "rotor_rpm": rpm.tolist(),
                    "obstacles_pos": raw["obstacles_pos"].tolist(),
                }
            )
            data, obs, hit = self.advance(data, jp.asarray(applied))
            raw = {k: np.asarray(v[0, 0]) for k, v in obs.items()}
            term = bool(np.asarray(data.disabled_drones[0, 0]))
            finished = int(raw["n_gates_passed"]) >= 5
            if term or finished:
                records.append(
                    {
                        "t": (step + 1) * 0.02,
                        "pos": raw["pos"].tolist(),
                        "quat": raw["quat"].tolist(),
                        "vel": raw["vel"].tolist(),
                        "ang_vel": raw["ang_vel"].tolist(),
                        "passed": int(raw["n_gates_passed"]),
                        "terminal": True,
                        "collision": bool(np.asarray(hit).any()),
                    }
                )
                done = True
                break
        result = metrics(records, noise, self.scene, seed, done)
        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            Path(save).write_text(
                json.dumps(
                    {
                        "noise": asdict(noise),
                        "scene": self.scene,
                        "seed": seed,
                        "randomized": self.randomized,
                        "metrics": result,
                        "records": records,
                    }
                )
            )
        return result, records


def metrics(records, noise, scene, seed, done):
    t = np.array([r["t"] for r in records])
    p = np.array([r["pos"] for r in records])
    n = np.array([r["passed"] for r in records])
    v = np.array([r["vel"] for r in records])
    tilt = np.degrees(
        np.arccos(np.clip(R.from_quat([r["quat"] for r in records]).as_matrix()[:, 2, 2], -1, 1))
    )
    ids = np.flatnonzero(n == 2)
    i2 = next(iter(np.flatnonzero(n >= 2)), None)
    t2 = t[i2] if i2 is not None else None
    m = {
        "scene": scene,
        "seed": seed,
        "noise": asdict(noise),
        "finished": bool(n[-1] >= 5),
        "passed": int(n[-1]),
        "duration": float(t[-1]),
        "terminal_z": float(p[-1, 2]),
        "terminal_pos": p[-1].tolist(),
        "collision": records[-1].get("collision", False),
        "gate_times": {
            str(i): float(t[np.flatnonzero(n >= i)[0]]) for i in range(1, 6) if np.any(n >= i)
        },
    }
    if len(ids):
        idx = ids[p[ids, 2].argmin()]
        rel = t[ids] - t2
        mid = ids[(rel > 0.25) & (rel < 1.0)]
        near = ids[(p[ids, 0] < -0.5) & (p[ids, 0] > -1.55) & (np.abs(p[ids, 1] + 0.25) < 0.65)]
        m.update(
            g2g3_min_z=float(p[idx, 2]),
            g2g3_min_t=float(t[idx]),
            g2g3_min_pos=p[idx].tolist(),
            g2g3_duration=float(t[ids[-1]] - t2),
            g2g3_min_vz=float(v[ids, 2].min()),
            g2g3_peak_tilt=float(tilt[ids].max()),
            g2g3_second_tilt=float(tilt[mid].max()) if len(mid) else None,
            near_gate3_min_z=float(p[near, 2].min()) if len(near) else None,
            near_floor=bool(p[ids, 2].min() < 0.15),
            ground_height=bool(p[ids, 2].min() < 0.06),
        )
    else:
        m.update(g2g3_min_z=None, near_floor=False, ground_height=False, near_gate3_min_z=None)
    return m


if __name__ == "__main__":
    r = Runner()
    for c in [
        Noise(),
        Noise(omega_tau=0.07),
        Noise(thrust_cap=0.7),
        Noise(thrust_cap=0.68, omega_tau=0.07),
    ]:
        m, _ = r.run(c, save=OUT / f"smoke_{hashlib.sha256(str(c).encode()).hexdigest()[:8]}.json")
        print(json.dumps(m), flush=True)
