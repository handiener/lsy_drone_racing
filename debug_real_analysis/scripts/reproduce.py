"""Run the frozen policy with documented temporary disturbance presets. Offline only."""

from harness import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--preset", choices=["baseline", "paired", "input_match", "minimal"], default="paired"
)
parser.add_argument("--scene", choices=["113735", "115247"], default="113735")
parser.add_argument("--output", default=None)
parser.add_argument("--real-bounds", action="store_true")
args = parser.parse_args()
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
r = Runner(args.scene)
if args.real_bounds:
    original = r.reset

    def realbounds(cfg, seed):
        data, raw = original(cfg, seed)
        data = data.replace(
            pos_limit_low=jp.array([-2.5, -1.5, -0.001], jp.float32),
            pos_limit_high=jp.array([2.5, 1.5, 2.0], jp.float32),
        )
        return data, raw

    r.reset = realbounds
out = (
    Path(args.output)
    if args.output
    else OUT
    / f"reproduced_{args.scene}_{args.preset}{'_realbounds' if args.real_bounds else ''}.json"
)
m, records = r.run(presets[args.preset], save=out)
print(
    json.dumps(
        {"preset": args.preset, "real_bounds": args.real_bounds, "output": str(out), "metrics": m},
        indent=2,
    )
)
r.env.close()
