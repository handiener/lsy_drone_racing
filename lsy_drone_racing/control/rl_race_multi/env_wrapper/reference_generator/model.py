"""Original staged minimum-time optimization used for the bundled dynamic guide.

Position and velocity have constant-acceleration updates; thrust direction uses a normalized
Euler update with bounded angular rate and acceleration. This reduced model omits drag,
onboard-controller dynamics and full gate-frame collision constraints. It produces a soft
training guide, not a certified physical optimum or collision-free flight plan.
"""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np

from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[5]
G = 9.81


def track(config: str | Path = "level2.toml") -> dict:
    """Nominal gates, obstacles, start and drone limits of a level config."""
    from crazyflow.dynamics import load_params

    cfg = load_config(ROOT / "config" / config)
    t = cfg.env.track
    gates = np.array([g["pos"] for g in t.gates], float)
    yaws = np.array([g["rpy"][2] for g in t.gates], float)
    order = [int(o) for o in cfg.env.track.get("gate_order", [1, 2, 3, 4])]
    params = load_params(cfg.sim.dynamics, cfg.sim.drone)
    return {
        "gates": gates,
        "normals": np.stack([np.cos(yaws), np.sin(yaws), np.zeros_like(yaws)], 1),
        "order": [abs(o) - 1 for o in order],
        "sign": [1.0 if o > 0 else -1.0 for o in order],
        "obstacles": np.array([o["pos"] for o in t.obstacles], float),
        "start": np.array(t.drones[0]["pos"], float),
        "mass": float(params["mass"]),
        "f_max": 4 * float(params["thrust_max"]),
        "f_min": 4 * float(params["thrust_min"]),
        "low": np.array(t.safety_limits["pos_limit_low"], float),
        "high": np.array(t.safety_limits["pos_limit_high"], float),
    }


def solve_model(
    theta_max_deg: float = 80.0,
    margin: float = 0.05,
    opening: float = 0.4,
    obstacle_radius: float = 0.015,
    nodes: int = 24,
    mass: float | None = None,
    guess_speed: float = 3.5,
    rate_max: float | None = 8.0,
    acc_max: float = 160.0,
    config: str | Path = "level2.toml",
    initial_guess: list[np.ndarray] | None = None,
    max_iter: int = 1200,
    max_cpu_time: float | None = 600.0,
    verbose: bool = False,
) -> dict:
    """Solve the minimum-time problem and return the lap time and per-leg times.

    Args:
        theta_max_deg: Maximum tilt of the thrust vector.
        margin: Drone radius: the gate opening shrinks and obstacles grow by this much.
        opening: Side length of the square gate opening (m).
        obstacle_radius: Obstacle cylinder radius (m).
        nodes: Nodes per leg.
        mass: Override the nominal mass (kg); level 2 randomizes it by +-5 g.
        guess_speed: Speed of the straight-line initial guess (m/s).
        rate_max: Turn-rate limit of the thrust direction (rad/s); None = pure point mass (the
            thrust direction may jump). The original guide uses 8 rad/s.
        acc_max: Angular-acceleration limit of the thrust direction (rad/s^2); measured 120-160.
        config: Level config in config/.
        initial_guess: Previous stages resampled to the current node count.
        max_iter: IPOPT iteration limit.
        max_cpu_time: CPU seconds per solve, or None for no time limit.
        verbose: Print IPOPT output.
    """
    tr = track(config)
    m = mass or tr["mass"]
    half = opening / 2 - margin
    legs = len(tr["order"])
    cos_t = np.cos(np.deg2rad(theta_max_deg))

    def build(relaxed: bool, attitude: float | None, warm: list | None = None) -> tuple:
        """Build one stage of the constrained trajectory problem.

        Relaxed mode removes min thrust and obstacles; attitude limits turning rate.
        """
        opti = ca.Opti()
        dts = opti.variable(legs)
        P, V, F, B, W, U, T = [], [], [], [], [], [], []
        for _ in range(legs):
            P.append(opti.variable(3, nodes + 1))
            V.append(opti.variable(3, nodes + 1))
            F.append(opti.variable(3, nodes))
            if attitude:
                B.append(opti.variable(3, nodes + 1))  # thrust direction (unit)
                W.append(opti.variable(3, nodes + 1))  # its angular velocity
                U.append(opti.variable(3, nodes))  # angular acceleration
                T.append(opti.variable(1, nodes))  # thrust magnitude
        gvec = ca.DM([0, 0, -G])
        opti.subject_to(P[0][:, 0] == tr["start"])
        opti.subject_to(V[0][:, 0] == 0)
        if attitude:
            opti.subject_to(B[0][:, 0] == ca.DM([0, 0, 1]))
            opti.subject_to(W[0][:, 0] == 0)
        for k in range(legs):
            p, v, f, dt = P[k], V[k], F[k], dts[k]
            opti.subject_to(opti.bounded(0.005, dt, 0.2))
            if k > 0:  # continuity between legs
                opti.subject_to(p[:, 0] == P[k - 1][:, -1])
                opti.subject_to(v[:, 0] == V[k - 1][:, -1])
                if attitude:
                    opti.subject_to(B[k][:, 0] == B[k - 1][:, -1])
                    opti.subject_to(W[k][:, 0] == W[k - 1][:, -1])
            for i in range(nodes):
                if attitude:  # f along b, b turns with w, |w| and |dw/dt| bounded
                    b, w = B[k][:, i], W[k][:, i]
                    opti.subject_to(f[:, i] == T[k][i] * b)
                    nb = b + dt * ca.cross(w, b)
                    opti.subject_to(B[k][:, i + 1] == nb / ca.sqrt(ca.sumsqr(nb)))
                    opti.subject_to(W[k][:, i + 1] == w + dt * U[k][:, i])
                    opti.subject_to(ca.sumsqr(W[k][:, i + 1]) <= attitude**2)
                    opti.subject_to(ca.sumsqr(U[k][:, i]) <= acc_max**2)
                a = f[:, i] / m + gvec
                opti.subject_to(p[:, i + 1] == p[:, i] + v[:, i] * dt + 0.5 * a * dt**2)
                opti.subject_to(v[:, i + 1] == v[:, i] + a * dt)
                fn2 = ca.sumsqr(f[:, i])
                opti.subject_to(fn2 <= tr["f_max"] ** 2)
                if not relaxed:
                    opti.subject_to(fn2 >= tr["f_min"] ** 2)
                opti.subject_to(f[2, i] >= cos_t * ca.sqrt(fn2 + 1e-9))
                mid = p[:, i] + 0.5 * v[:, i] * dt + 0.125 * a * dt**2
                for q in () if relaxed else (p[:, i + 1], mid):
                    for o in tr["obstacles"]:
                        opti.subject_to(
                            (q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2
                            >= (obstacle_radius + margin) ** 2
                        )
            for i in range(1, nodes + 1):  # stay in the arena and above the start height
                opti.subject_to(opti.bounded(tr["low"][:2], p[:2, i], tr["high"][:2]))
                opti.subject_to(opti.bounded(tr["start"][2], p[2, i], tr["high"][2]))
            # Leg ends in the gate plane, inside the shrunk opening, moving forward through it.
            g = tr["order"][k]
            c, n = tr["gates"][g], tr["normals"][g] * tr["sign"][k]
            lat = np.array([-n[1], n[0], 0.0])
            d = p[:, -1] - c
            opti.subject_to(ca.dot(d, n) == 0)
            opti.subject_to(opti.bounded(-half, ca.dot(d, lat), half))
            opti.subject_to(opti.bounded(-half, d[2], half))
            opti.subject_to(ca.dot(v[:, -1], n) >= 0.5)
        opti.minimize(nodes * ca.sum1(dts))

        if warm is not None:  # previous solution
            for var, val in zip([dts, *P, *V, *F], warm):
                opti.set_initial(var, val)
            if attitude and len(warm) > 1 + 3 * legs:  # previous attitude stage
                for var, val in zip([*B, *W, *U, *T], warm[1 + 3 * legs :]):
                    opti.set_initial(var, val)
            elif attitude:  # thrust direction and its rates from the point-mass thrust
                dtv = np.asarray(warm[0]).ravel()
                for k in range(legs):
                    fk = np.asarray(warm[1 + 2 * legs + k]).reshape(3, nodes)
                    bk = fk / np.maximum(np.linalg.norm(fk, axis=0), 1e-6)
                    bk = np.concatenate([bk, bk[:, -1:]], axis=1)
                    wk = np.cross(bk[:, :-1].T, np.diff(bk, axis=1).T).T / dtv[k]
                    wk = np.concatenate([wk, wk[:, -1:]], axis=1)
                    wk *= np.minimum(1.0, attitude / np.maximum(np.linalg.norm(wk, axis=0), 1e-6))
                    opti.set_initial(B[k], bk)
                    opti.set_initial(W[k], wk)
                    opti.set_initial(U[k], np.diff(wk, axis=1) / dtv[k])
                    opti.set_initial(T[k], np.linalg.norm(fk, axis=0)[None])
            return opti, dts, P, V, F, B + W + U + T
        # Initial guess: straight lines through the gate centers at guess_speed, hover thrust.
        pts = [tr["start"]] + [tr["gates"][g] for g in tr["order"]]
        for k in range(legs):
            a, b = pts[k], pts[k + 1]
            dist = np.linalg.norm(b - a)
            opti.set_initial(dts[k], dist / guess_speed / nodes)
            s = np.linspace(0, 1, nodes + 1)
            opti.set_initial(P[k], (a[:, None] + (b - a)[:, None] * s[None]))
            opti.set_initial(V[k], np.tile(((b - a) / dist * guess_speed)[:, None], nodes + 1))
            opti.set_initial(F[k], np.tile(np.array([[0], [0], [m * G]]), nodes))
        return opti, dts, P, V, F, B + W + U + T

    dts, P, V, F = None, [], [], []
    sol = None
    # Homotopy: relaxed point mass -> point mass -> rate limit tightened step by step.
    stages = [(True, None), (False, None)]
    if rate_max:
        stages += [(False, r) for r in (30.0, 15.0, 12.0, 10.0, 9.0, 8.5) if r > rate_max] + [
            (False, rate_max)
        ]
    if initial_guess is not None:
        stages = [(False, rate_max)]
    extra: list = []
    for relaxed, attitude in stages:
        warm = (
            initial_guess
            if sol is None
            else [np.asarray(sol.value(x)) for x in [dts, *P, *V, *F, *extra]]
        )
        opti, dts, P, V, F, extra = build(relaxed, attitude, warm)
        sol, ok = _solve(opti, verbose, max_iter, max_cpu_time)
        stats = opti.stats()
        print(
            f"stage relaxed={relaxed} rate={attitude}: {stats['return_status']}, "
            f"lap {nodes * float(np.sum(sol.value(dts))):.3f} s",
            flush=True,
        )
        if not ok and not relaxed:
            raise RuntimeError(f"IPOPT did not converge: {opti.stats()['return_status']}")
    dt = np.asarray(sol.value(dts)).ravel()
    leg_t = dt * nodes
    p = np.concatenate(
        [np.asarray(sol.value(P[k]))[:, :-1] for k in range(legs)]
        + [np.asarray(sol.value(P[-1]))[:, -1:]],
        axis=1,
    )
    v = np.concatenate(
        [np.asarray(sol.value(V[k]))[:, :-1] for k in range(legs)]
        + [np.asarray(sol.value(V[-1]))[:, -1:]],
        axis=1,
    )
    f = np.concatenate([np.asarray(sol.value(F[k])) for k in range(legs)], axis=1)
    tilt = np.degrees(np.arccos(np.clip(f[2] / np.linalg.norm(f, axis=0), -1, 1)))
    result = {
        "converged": ok,
        "max_constraint_violation": float(
            max(
                0.0,
                np.max(np.asarray(sol.value(opti.lbg)) - np.asarray(sol.value(opti.g))),
                np.max(np.asarray(sol.value(opti.g)) - np.asarray(sol.value(opti.ubg))),
            )
        ),
        "lap_time": float(leg_t.sum()),
        "leg_times": leg_t.round(3).tolist(),
        "max_speed": float(np.linalg.norm(v, axis=0).max()),
        "tilt_at_limit": float(np.mean(tilt > theta_max_deg - 1)),
        "thrust_at_max": float(np.mean(np.linalg.norm(f, axis=0) > 0.99 * tr["f_max"])),
        "settings": {
            "theta_max_deg": theta_max_deg,
            "margin": margin,
            "mass": m,
            "thrust_max": tr["f_max"],
            "rate_max": rate_max,
            "acc_max": acc_max if rate_max else None,
            "a_max_g": tr["f_max"] / m / G,
        },
    }
    t = np.concatenate([[0.0], np.cumsum(np.repeat(dt, nodes))])
    return {
        **result,
        "t": t.tolist(),
        "pos": p.T.tolist(),
        "vel": v.T.tolist(),
        "thrust": f.T.tolist(),
        "warm": [np.asarray(sol.value(x)).tolist() for x in [dts, *P, *V, *F, *extra]],
    }


def _solve(opti: ca.Opti, verbose: bool, max_iter: int, max_cpu_time: float | None) -> tuple:
    """Run IPOPT; return (solution or debug values, converged)."""
    options = {"max_iter": max_iter, "print_level": 5 if verbose else 0, "tol": 1e-6}
    if max_cpu_time is not None:
        options["max_cpu_time"] = max_cpu_time
    opti.solver("ipopt", {"print_time": False}, options)
    try:
        return opti.solve(), True
    except RuntimeError:
        return opti.debug, False


def reduced_seed(
    point_solution: dict,
    config: str | Path = "level2.toml",
    theta: float = 80.0,
    verbose: bool = False,
    max_cpu_time: float = 600.0,
) -> dict:
    """Build the original 12-interval, loose-rate seed from a point-mass solution."""
    n = 12
    tr = track(config)
    legs = len(tr["order"])
    m = tr["mass"]
    margin = 0.05
    half = 0.15
    op = ca.Opti()
    dt = op.variable(legs)
    P = []
    V = []
    F = []
    for k in range(legs):
        P.append(op.variable(3, n + 1))
        V.append(op.variable(3, n + 1))
        F.append(op.variable(3, n))
    rate = op.parameter()
    acc = op.parameter()
    op.subject_to(P[0][:, 0] == tr["start"])
    op.subject_to(V[0][:, 0] == 0)
    op.subject_to(op.bounded(0.005, dt, 0.2))
    bs = []
    ds = []
    for k in range(legs):
        p, v, f = P[k], V[k], F[k]
        if k:
            op.subject_to(p[:, 0] == P[k - 1][:, -1])
            op.subject_to(v[:, 0] == V[k - 1][:, -1])
        for i in range(n):
            a = f[:, i] / m + ca.DM([0, 0, -9.81])
            op.subject_to(p[:, i + 1] == p[:, i] + v[:, i] * dt[k] + 0.5 * a * dt[k] ** 2)
            op.subject_to(v[:, i + 1] == v[:, i] + a * dt[k])
            f2 = ca.sumsqr(f[:, i])
            op.subject_to(op.bounded(tr["f_min"] ** 2, f2, tr["f_max"] ** 2))
            op.subject_to(f[2, i] >= np.cos(np.deg2rad(theta)) * ca.sqrt(f2))
            bs.append(f[:, i] / ca.sqrt(f2))
            ds.append(dt[k])
            mid = p[:, i] + 0.5 * v[:, i] * dt[k] + 0.125 * a * dt[k] ** 2
            for q in [p[:, i + 1], mid]:
                for o in tr["obstacles"]:
                    op.subject_to(ca.sumsqr(q[:2] - o[:2]) >= (0.015 + margin) ** 2)
        for i in range(1, n + 1):
            op.subject_to(op.bounded(tr["low"][:2], p[:2, i], tr["high"][:2]))
            op.subject_to(op.bounded(0, p[2, i], tr["high"][2]))
        g = tr["order"][k]
        c = tr["gates"][g]
        normal = tr["normals"][g] * tr["sign"][k]
        lat = np.array([-normal[1], normal[0], 0])
        d = p[:, -1] - c
        op.subject_to(ca.dot(d, normal) == 0)
        op.subject_to(op.bounded(-half, ca.dot(d, lat), half))
        op.subject_to(op.bounded(-half, d[2], half))
        op.subject_to(ca.dot(v[:, -1], normal) >= 0.5)
    # Angular velocity producing the normalized Euler direction update in the historical model.
    # Choose zero component along the thrust axis to remove redundant rotational degrees of freedom.
    op.subject_to(bs[0][:2] == 0)
    op.subject_to(bs[1][:2] == 0)
    ws = []
    for i in range(len(bs) - 1):
        dot = ca.dot(bs[i], bs[i + 1])
        op.subject_to(dot >= 0.5)
        w = ca.cross(bs[i], bs[i + 1]) / (dot * ds[i])
        ws.append(w)
        op.subject_to(ca.sumsqr(w) <= rate**2)
        if i:
            op.subject_to(ca.sumsqr(w - ws[i - 1]) <= (acc * ds[i - 1]) ** 2)
    op.minimize(n * ca.sum1(dt))
    r = point_solution
    oldn = (len(r["t"]) - 1) // legs
    oldt = np.diff(r["t"])[::oldn]
    for k in range(legs):
        s0 = np.linspace(0, 1, oldn + 1)
        s1 = np.linspace(0, 1, n + 1)

        def interp(a: np.ndarray) -> np.ndarray:
            return np.array([np.interp(s1, s0, x) for x in np.asarray(a)])

        p = interp(np.array(r["pos"])[k * oldn : (k + 1) * oldn + 1].T)
        v = interp(np.array(r["vel"])[k * oldn : (k + 1) * oldn + 1].T)
        f = np.array(r["thrust"])[k * oldn : (k + 1) * oldn].T
        f = interp(np.c_[f, f[:, -1]])[:, :-1]
        op.set_initial(P[k], p)
        op.set_initial(V[k], v)
        op.set_initial(F[k], f)
        op.set_initial(dt[k], oldt[k] * oldn / n)

    op.set_value(rate, 40.0)
    op.set_value(acc, 1000.0)
    sol, ok = _solve(op, verbose, 1500, max_cpu_time)
    if not ok:
        raise RuntimeError(f"Reduced seed did not converge: {op.stats()['return_status']}")
    x = sol.value(op.g)
    violation = float(max(0, np.max(sol.value(op.lbg) - x), np.max(x - sol.value(op.ubg))))
    return {
        "converged": True,
        "lap_time": float(n * np.sum(sol.value(dt))),
        "max_constraint_violation": violation,
        "leg_times": (n * np.asarray(sol.value(dt))).tolist(),
        "t": np.r_[0, np.cumsum(np.repeat(sol.value(dt), n))].tolist(),
        "pos": np.hstack([sol.value(p)[:, :-1] for p in P] + [sol.value(P[-1])[:, -1:]]).T.tolist(),
        "vel": np.hstack([sol.value(v)[:, :-1] for v in V] + [sol.value(V[-1])[:, -1:]]).T.tolist(),
        "thrust": np.hstack([sol.value(f) for f in F]).T.tolist(),
    }
