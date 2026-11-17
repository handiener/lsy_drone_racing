# Race-only diagnosis

This analysis excludes the finish and return phases. All times are relative to the first
policy observation. Flight A is `deploy_20261117T100017.801841Z`; flight B is
`deploy_20261117T100827.697140Z`.

## 1. Abrupt thrust cut before gate 1: obstacle 1 height reveal

Obstacle 1 is nominally represented at z = 1.55 m. When it becomes visible, its recorded
measured height is approximately 0.431 m: a 1.12 m discontinuity. Its ordering among
obstacle features does not change on this step, isolating this from a sorting event.

| Quantity | Flight A | Flight B |
| --- | --- | --- |
| Reveal time | 0.543 s | 0.562 s |
| Thrust on previous step | 0.744 N | 0.753 N |
| Thrust at reveal | 0.0867 N | 0.0948 N |
| Thrust if only obstacle 1 z is set to 1.55 m at the same state/history | 0.7339 N | 0.7531 N |
| Pitch command before → at reveal | -12.3° → +37.4° | -8.5° → +31.6° |
| Pitch with only obstacle 1 z changed to 1.55 m | -13.7° | -11.1° |

This is a reproducible explanation for the abrupt command change before the first gate.
It is not simply failure of the drone to follow a smooth policy command: the policy itself
issues the thrust cut and pitch reversal in response to that input.

The offline test uses the exact model identified by the recorded SHA256 hashes. Replaying
all factual recorded feature vectors reproduces the logged command components to a maximum
absolute error of 1.3e-6. Each counterfactual preserves the recorded physical state and
previous policy action; it changes only the indicated current input. These are single-step
policy evaluations, not simulated trajectories or proof that changing the input fixes a
complete flight.

Verify what obstacle 1's mocap frame means and what its physical height actually is. If
0.431 m is erroneous, correct the tracking/frame/configuration. If it is physically correct,
the policy needs consistent nominal geometry and training coverage for that geometry. Do
not blindly substitute a fictitious obstacle height during flight.

## 2. Another command discontinuity after gate 2: obstacle slots swap

`env_wrapper/observations.py:encode()` sorts obstacles by horizontal distance and flattens
their body-frame positions into fixed MLP feature slots. When distances exchange order,
the input vector is discontinuous even when the physical scene moves smoothly.

At 1.808 s (A) and 1.827 s (B), obstacle IDs [1, 0, 3, 2] switch to [0, 1, 3, 2]
(zero-based IDs). The gate counter stays at 2; this is not the gate-crossing step.

| Quantity | Flight A | Flight B |
| --- | --- | --- |
| Thrust before → at swap | 0.721 → 0.433 N | 0.684 → 0.327 N |
| Roll before → at swap | +51.1° → +5.2° | +40.4° → -11.8° |
| Thrust at same state with previous obstacle slot ordering | 0.695 N | 0.621 N |
| Roll at same state with previous obstacle slot ordering | +38.2° | +25.9° |

Preserving the previous order for this single input avoids most of this immediate command
jump. Setting only obstacle 1 height to 1.55 m also increases the command at the swap to
0.650 N / 0.550 N. These are two interacting sensitivities, not proof that sorting is the
only contributor to the subsequent descent. Subsequent states are still the factual
recorded trajectory in these comparisons.

A persistent fix for ordering sensitivity should be evaluated in training and deployment
together, e.g. stable obstacle identities or an encoder insensitive to input permutations.
Changing feature ordering only for the existing deployed model changes its learned input
semantics and is not validated by this one-step experiment.

## 3. What happens in front of gate 3

On the first approach, neither flight actually crosses the gate plane. The drone reverses
before it and descends below the opening, then loops back for a later attempt.

- A: closest initial approach to the plane is about 5.9 cm short at 2.571 s, with
  z = 0.411 m, approximately 0.248 m below the gate center in the gate's local frame.
- B: about 7.9 cm short at 2.611 s, with z = 0.346 m, approximately 0.312 m below the
  gate center in the gate's local frame.
- A then bottoms out around 0.310 m at 2.780 s; B around 0.093 m at 2.975 s (mocap).
- Actual forward crossings of gate 3 occur around 3.516 s / 3.677 s in transition records;
  the next policy observations report the incremented gate counter roughly one step later.

The second flight arrests its descent more slowly despite similarly high commanded thrust
in the recovery interval. A linear fit to estimated vertical velocity over 2.45–2.70 s gives
approximately +3.40 m/s² in A and +1.45 m/s² in B. This is a descriptive comparison of
filtered observations, not a direct force measurement. Battery voltage, actual motor
outputs, onboard estimates, and tracking latency are missing, so the reason for this
difference remains unresolved.

Measured attitude visibly lags the rapid commanded reversals. This is expected for a
physical drone and makes the policy input/command discontinuities more consequential;
the logs alone do not isolate an additional firmware fault or quantify a calibrated
actuator delay.

## Priority

First resolve obstacle 1's nominal/measured geometry mismatch, then assess the policy's
sensitivity to obstacle rank swaps and re-evaluate the approach to gate 3 with consistent
inputs. The observation discontinuities are directly supported by exact policy replay;
the complete cause of the low pass still requires flight-dynamics validation and onboard
telemetry. No controller, config, or deployment code was changed during this analysis.

See `race_only.png`, `race_only.pdf`, `race_counterfactuals.json`, and
`race_diagnostics.py` in this directory.
