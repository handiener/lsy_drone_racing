# Generate the dynamic reference

Run from the repository root. This is a CPU-only offline optimization; no training is started.

```bash
pixi run -e gpu-tests python -m pip install casadi==3.7.2

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run -e gpu-tests python -m lsy_drone_racing.control.rl_race_multi.env_wrapper.reference_generator \
  --output /tmp/dynamic_reference.json --check
```

This generates a **new file**, then `--check` compares it with the bundled
[`dynamic_reference.json`](../dynamic_reference.json). Existing output files are never overwritten.
The original settings reproduce approximately **11.050272528 s**, with 20 legs and 33 points per leg.
Use `--work-dir /tmp/reference_work` to retain intermediate solutions in an empty folder.

## Method and settings

CasADi + IPOPT minimize total flight time using direct multiple shooting. The original recipe is:
24-interval point mass → loose 12-interval seed → 12-interval attitude model → 24-interval refinement.
Positions/velocities use constant-acceleration integration; thrust direction uses normalized Euler
integration with bounded direction rate and angular acceleration. The final trajectory is sampled
at 33 times per leg and rounded to seven decimal places, as in the original export.

Defaults use the nominal `config/level2.toml` track, 43.4 g mass, 0.8 N maximum thrust,
**80° thrust-vector tilt**, 8 rad/s direction rate, 160 rad/s² angular acceleration and 0.05 m
point-clearance margin. The 80° limit is a tilt cone; this model does not optimize separate
roll/pitch/yaw states. `--tilt-limit-deg` changes it; the default reproduces the existing guide.

This is a local optimum of a simplified model. Gate openings and pole clearances are constrained,
but full gate frames, drag and onboard-controller dynamics are omitted. The original guide has
known simulator geometry contacts and is intended only as a soft training reference.
Training loads the JSON and adapts it to randomized gate poses; it does not run this optimizer.
