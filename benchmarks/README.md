# Benchmark records

Append-only log of search runs. Later phases add lines; they do not rewrite old ones.

Each line of a JSONL file is one object matching [`run_record.schema.json`](run_record.schema.json) (schema version 1). `unsga3_extropic.results.validate_run_record` checks the same keys.

| Field | Meaning |
|-------|---------|
| `date` | UTC ISO-8601 (`2026-10-02T16:19:45Z`) |
| `git_sha` | `git rev-parse HEAD` at record time, or `unknown` |
| `issue`, `phase` | Issue this run is evidence for (`4`, `1` for the Potts smoke). A plumbing row uses `none` |
| `problem`, `backend` | `potts_chain` / `thrml_potts` for the categorical smoke. `stub` / `none` is not a measurement |
| `seeds` | Seeds passed to `sample_weight`. `WeightSweepLoop` uses `base + 1009 * weight_index` |
| `schedule` | Betas, warmup, samples per beta, steps between samples, weight count. `SamplingSchedule` has no beta; betas rescale Potts weights |
| `eval_budget` | Number of recorded objective vectors (`n_weights * len(betas) * n_samples` for a finished sweep). Not Gibbs micro-steps |
| `metrics.nd_count` | Unique non-dominated objective rows kept by the archive |
| `metrics.hypervolume_2d` | 2-D minimization hypervolume against `metrics.hv_ref`. Null when the front is not 2-D |
| `metrics.generational_distance` | Mean Euclidean distance from the run front to the reference front. Null when either front is empty |
| `metrics.coverage` | Fraction of the **reference** front weakly dominated by the run front (minimization). Null when either front is empty |
| `artifacts.front` | Path to the run's non-dominated front, `.npy` or `.npz` shape `(n, M)`. An `.npz` stores the array as `front`, or as its only array |
| `notes` | Free text. The Potts smoke says the reference is `PottsChainProblem.enumerate_front` |

Front files are local artifacts (gitignored under `artifacts/` next to the JSONL). Commit the JSONL when a number should stay in the history. The loader is `unsga3_extropic.fidelity.load_front`. It does not launch Bend, C#, or ZDT1.

```bash
python benchmarks/append_run.py --stub --out benchmarks/records/runs.jsonl
python benchmarks/append_run.py --smoke --out benchmarks/records/runs.jsonl
```

`--stub` writes a schema-valid row with null HV / GD / coverage and does not sample. `--smoke` runs the in-repo Potts chain (THRML, issue 4) and fills metrics with the harness from issue 6. Neither command searches for a beta schedule.

`demos/run_potts_thrml.py` writes the same kind of record under `demos/_out/`.
