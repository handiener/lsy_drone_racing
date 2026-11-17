# Official sim.py obstacle-order check

## What was actually run

Executed `scripts.sim.simulate(config="level2.toml", n_runs=3, render=False)` using the
current exported `rl_race_controller.py` policy and current config. To make this diagnostic
repeatable, set the runtime environment seed to 0 (the config file still contains -1).
These are three consecutive randomized resets in one runner invocation, not three distinct
seed values. Used the CPU backend because this session cannot access the host GPU.

Temporary instrumentation subclasses the loaded controller to copy its input/output at
every step. It invokes the original controller and the official simulation loop; it does
not replace the dynamics or alter policy actions. Policy diagnostic capture was enabled.
The official sim/training/config source files were not edited.

All three completed 5/5 crossings, with runner lap times 3.42 s, 3.56 s, and 3.48 s.
The process emitted a JAX overflow-in-cast warning during setup but completed all three runs.
These results describe this local environment and three resets, not all possible seeds or
the unpreserved historical training environment.

## Answer: obstacle 1 and 3 DO exchange order in simulation

Before gate 3, all three simulation runs exhibit precisely the same ordering transition
as the latest hardware run, using one-based obstacle numbers:

**[4, 1, 3, 2] -> [4, 3, 1, 2]**

This exchanges feature slots 2 and 3. Both training and deployment use the same distance
sorting in `env_wrapper/observations.py`. Therefore absence of rank exchanges in simulation
is not the explanation for its successful flights.

| At this exchange | Sim 0 | Sim 1 | Sim 2 | Hardware 11:37:35 UTC |
| --- | --- | --- | --- | --- |
| Time from first controller observation | 2.02 s | 2.10 s | 2.16 s | 2.149 s |
| Measured height | 0.897 m | 0.851 m | 0.784 m | 0.847 m |
| Measured tilt from vertical | 41.5° | 41.8° | 46.5° | **74.1°** |
| Vertical velocity | -0.821 m/s | -0.764 m/s | -0.163 m/s | **-1.222 m/s** |
| Thrust before -> after | 0.782 -> 0.771 N | 0.789 -> 0.789 N | 0.727 -> 0.733 N | **0.356 -> 0.131 N** |
| Roll command change in one step | +6.5° | +5.9° | +4.3° | **+45.5°** |
| First observation after gate 3 crossing | 2.34 s | 2.38 s | 2.44 s | 3.575 s |

Gate crossing times above use the first controller observation with the updated counter,
so they need not be the exact geometric intersection time.

## Same-state counterfactual: rank sensitivity also differs

A before/after command difference mixes sorting with physical state changes. To isolate
sorting, evaluate the current policy input with exactly the same state and previous action,
but keep the previous obstacle feature-slot ordering for that one step:

| Same-state output at exchange | Actual order | Previous order | Difference caused by current ordering |
| --- | --- | --- | --- |
| Sim 0 | 0.77057 N | 0.77585 N | -0.00528 N |
| Sim 1 | 0.78854 N | 0.79021 N | -0.00167 N |
| Sim 2 | 0.73273 N | 0.70545 N | +0.02728 N |
| Hardware | 0.13115 N | 0.31458 N | **-0.18343 N** |

These are single-step comparisons, not closed-loop trials of a modified encoder. The
hardware previous-order comparison is also recorded in `../flight_113735_metrics.json`.

## Interpretation

The same policy can respond very differently to the same slot exchange when its other
inputs differ. Those inputs include attitude, velocity, angular velocity, gate geometry,
obstacle vectors, and the previous policy action. Similar xyz positions do not imply the
same policy input.

In the aligned plots, the hardware attitude already diverges during the roughly 0.2 s
before this exchange; the swap does not initiate all of the preceding divergence. The
hardware reaches the boundary at a much larger tilt and faster downward velocity, and the
policy is particularly sensitive to ordering at that state. The simulated controller
maintains high thrust through this exchange and subsequently arrests descent near the gate
height; hardware descends to near floor height before recovering.

This is consistent with sim-to-real differences bringing the policy into a more vulnerable
state, where sorting sensitivity exacerbates the command change. It does not identify
which difference initiated the divergence. We have not isolated actuator response, thrust
mapping, battery, offboard/onboard estimation, timing, or small track-geometry differences.
Neither the sorting exchange alone nor a generic "sim-to-real gap" is a complete proven
root cause. The simulated states used here also include the simulator's randomizations;
the hardware scene and initial state are not exactly matched.

The corrected hardware obstacle 1 height is about 1.584 m. Its old 0.43 m height mismatch
is absent in this comparison.

## Artifacts and reproduction

- `sim_episode_00.json`, `sim_episode_01.json`, `sim_episode_02.json`: complete per-step
  observations, commands, encoded policy features, previous actions, and raw policy actions.
- `*_ordering.csv`: per-step obstacle ordering, position, velocity, tilt, and commands for
  each sim episode and the comparison hardware flight.
- `comparison_metrics.json`: exact swap states, command changes, all obstacle 1/3 exchanges,
  and same-state simulation counterfactual outputs.
- `sim_vs_real_swap.png` / `.pdf`: all three simulations and the hardware flight aligned
  at the same ordering boundary.
- `config.json`, `runner_results.json`: runtime config snapshot and runner outcomes.

```bash
.pixi/envs/deploy/bin/python runs/deployment_analysis/sim_order_check/capture_sim.py
.pixi/envs/deploy/bin/python runs/deployment_analysis/sim_order_check/compare.py
```

The scripts are offline diagnostics and do not connect to hardware. The comparison script
regenerates tables, plots, and same-state simulation counterfactuals from saved features,
holding the previous action fixed and permuting only the final four obstacle xyz triples.
The recorder subclass computes its diagnostics locally and does not require deployment
recorder/controller changes to be committed.
