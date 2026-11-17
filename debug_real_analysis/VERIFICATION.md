**Package verification — 17 November 2026**

The package was tested in a second independent directory, with the working directory outside the repository. Inputs were copied with the package, `config/`, `lsy_drone_racing/` and the existing hardware `runs/`. Tests used the project interpreter and the dependency versions recorded in `evidence/reproducibility.json`. Python ran with `-B`; caches and generated outputs were redirected outside the package.

- All 14 Python scripts parse successfully and were formatted with Ruff.
- `reproduce.py --preset baseline --scene 113735` completed all five crossings. Minimum reference height between gates 2 and 3: 0.7469068169593811 m.
- `reproduce.py --preset baseline --scene 115247` also completed all five crossings, with minimum reference height 0.7469025254249573 m.
- `reproduce.py --preset paired --scene 113735` completed all five crossings. Minimum reference height: 0.00010247227328363806 m.
- `reproduce.py --preset paired --scene 115247` completed all five crossings. Minimum reference height: 0.09086063504219055 m.
- For all three relocated replays, minimum height, duration, crossing count, completion flag and all gate times match the archived zero-seed validation metrics exactly. Their recorded physical time, position, orientation, velocity and angular velocity are finite.
- `build_summary.py` regenerated the summary PNG/PDF and metadata from the locally retained archive, with the correct model hashes and all 5,822 historical metric rows. Published copies require locally regenerated validation outputs or an optional local archive.
- `plot_candidates.py 113735 bc60e02f023f input_match_113735` regenerated the detailed comparison PNG/PDF from the local archive. It also supports a regenerated validation-only baseline.
- `rank.py coarse extended focused validation` read 5,822 rows and scored 164 locally retained trajectories that reach gate 2. It explicitly reported that omitted bulk traces were skipped.
- `trend_metrics.py` reproduced the archived rate-lag and peak comparisons.
- The retained local archive has 5,822 portable trajectory references. All 170 locally retained NPZ files match their original hash inventory. These additional data files are not published in this commit.
- Generated outputs and Python bytecode directories are ignored by Git. New files are confined to `debug_real_analysis/`.

A new `plot_replays.py` entry point regenerates comparison figures directly from fresh single-episode JSON outputs and the existing branch logs, without any bundled experiment archive. It was checked with both `paired` replay scenes.

The full parameter sweeps were not rerun as a packaging check. Their original metrics are preserved, with only trajectory paths normalized and inclusion flags added. The older coarse harness remains separate from the later extended/focused/validation harness.

JAX emitted a numeric-cast overflow warning during simulation initialization in this environment; the replay checks above verified finite physical state records. As explained in the report, terminal action/observation fields in historical NPZ traces may intentionally contain NaNs when no next command exists.

Verify the published report and scripts from inside `debug_real_analysis/`:

```bash
sha256sum --check SHA256SUMS
```
