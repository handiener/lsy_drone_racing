**Current-reference package verification — 17 November 2026**

Checks used an independent relocated copy of `lsy_drone_racing/`, `config/`, the branch's existing hardware `runs/`, and the packaged analysis. The older local `evidence/` and `figures/` directories were excluded. Commands ran from outside that copy using the existing project interpreter, `python -B`, and an external `DRONE_DEBUG_OUTPUT_DIR`; no project installation or dependency upgrade occurred.

- All 27 Python scripts parse successfully and were formatted with Ruff.
- `validate_reference.py` regenerated all **50** exact-start, perturbed-start and factor-withdrawal episodes. Every configuration, gate time, duration, minimum height, crossing count, completion flag, terminal position and collision flag matches its archived record exactly. Every array in all 50 NPZ trajectories also matches, including physical and observed states, actions and motor outputs; intentional terminal NaNs compare equal.
- The default `reproduce.py` entry point uses `actuator_delay`. Its exact 12:37 result matches the archive: model-ground minimum -0.0007502614753320813 m, two crossings, duration 2.44 s. The 12:52 counterexample matches the archive: minimum 0.7530589699745178 m and three crossings.
- `reproduce.py --scene 113735 --seed 16` matches the archived selected example. Exact and seed-16 nominal controls also match their archive. The explicit historical `paired` preset retains its previous 0.00010247227328363806 m minimum.
- `plot_reference.py` regenerated bundled-data comparisons for both exact starts and seed 16. `plot_replays.py` regenerated fresh-replay comparisons with its new default and seed option. The selected-example figure is visibly marked as selected after inspection.
- The relocated default-deployment-filter smoke check imports the hash-verified locked wheel through `DRONE_ESTIMATOR_WHEEL`, without installation. Its physics recorder produces exactly identical physical states/actions/metrics when observations are unmodified, and its two default-filter cases reproduce the original smoke results. The package's downloader is pinned to the public lockfile URL and hash; no environment-wide install or JAX precision change occurs.
- All **92** bundled trajectory hashes match the archive manifest, and their physical arrays are finite. The **750** follow-up records have portable references and correct inclusion flags. Non-reference bulk trajectories are explicitly omitted.
- No original production module, policy weight or competition configuration changes. Changed files are confined to `debug_real_analysis/`. Generated outputs, caches and the older local bulk archive are ignored by Git.

The older complete sweeps were not rerun as a packaging check. Their preserved metrics retain source phase hashes; the prior historical verification remains available in Git commit `d6512f4`. Runtime JAX initialization emitted the previously observed numeric-cast warning; physical-array finiteness and the exact trajectory comparisons above passed.

Verify the committed package from `debug_real_analysis/`:

```bash
sha256sum --check SHA256SUMS
```
