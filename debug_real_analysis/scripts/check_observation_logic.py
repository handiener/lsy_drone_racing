"""Sensitivity check: task input counters/reveals use the delayed estimated position."""

import harness as h
from harness import *

Base = h.DisturbanceWrapper
DEST = OUT / "consistent_observation_logic"
DEST.mkdir(exist_ok=True)
presets = {
    "baseline": Noise(),
    "paired": Noise(
        omega_tau=0.05,
        pose_tau=0.01,
        thrust_cap=0.76,
        damping=0.7,
        command_substeps=2,
        drag_scale=1.4,
    ),
    "input_match": Noise(
        omega_tau=0.1,
        pose_tau=0.02,
        thrust_cap=0.72,
        damping=0.7,
        command_substeps=3,
        drag_scale=1.4,
    ),
    "minimal": Noise(omega_tau=0.1, thrust_cap=0.72, damping=0.7),
}
results = []
for scene in ["113735", "115247"]:
    runner = Runner(scene)
    track = runner.real_track

    class Consistent(Base):
        def __init__(self, cfg, initial):
            super().__init__(cfg, initial)
            self.counter = 0
            self.last = np.array(initial["pos"])
            self.gv = np.array(initial["gates_visited"])
            self.ov = np.array(initial["obstacles_visited"])
            self.gp = np.array(initial["gates_pos"])
            self.gq = np.array(initial["gates_quat"])
            self.op = np.array(initial["obstacles_pos"])
            self.actualg = np.array(track["gates_pos"])
            self.actualq = np.array(track["gates_quat"])
            self.actualo = np.array(track["obstacles_pos"])

        def observe(self, raw, step):
            seen = super().observe(raw, step)
            p = seen["pos"]
            self.gv |= np.linalg.norm(self.actualg[:, :2] - p[:2], axis=-1) < 0.7
            self.ov |= np.linalg.norm(self.actualo[:, :2] - p[:2], axis=-1) < 0.7
            if self.counter < len(seen["gate_sequence"]):
                g = int(seen["gate_sequence"][self.counter])
                rot = R.from_quat(self.actualq[g])
                p0 = rot.apply(self.last - self.actualg[g], inverse=True)
                p1 = rot.apply(p - self.actualg[g], inverse=True)
                reverse = seen["gate_sequence_direction"][self.counter] < 0
                cross = (p0[0] > 0 and p1[0] < 0) if reverse else (p0[0] < 0 and p1[0] > 0)
                if cross:
                    a = -p0[0] / (p1[0] - p0[0])
                    inter = p0 * (1 - a) + p1 * a
                    if np.all(np.abs(inter[1:]) < 0.225):
                        self.counter += 1
            self.last = p.copy()
            seen["n_gates_passed"] = np.asarray(self.counter, np.int32)
            seen["gates_visited"] = self.gv.copy()
            seen["obstacles_visited"] = self.ov.copy()
            seen["gates_pos"] = np.where(self.gv[:, None], self.actualg, self.gp).astype(np.float32)
            seen["gates_quat"] = np.where(self.gv[:, None], self.actualq, self.gq).astype(
                np.float32
            )
            seen["obstacles_pos"] = np.where(self.ov[:, None], self.actualo, self.op).astype(
                np.float32
            )
            return seen

    h.DisturbanceWrapper = Consistent
    for name, cfg in presets.items():
        out = DEST / f"{scene}_{name}.json"
        m, rs = runner.run(cfg, save=out)
        m.update(preset=name, policy_logic="estimated", output=str(out))
        results.append(m)
        print(
            scene,
            name,
            "min_z",
            m["g2g3_min_z"],
            "passed",
            m["passed"],
            "times",
            m["gate_times"],
            flush=True,
        )
    runner.env.close()
h.DisturbanceWrapper = Base
(DEST / "metrics.json").write_text(json.dumps(results, indent=2))
