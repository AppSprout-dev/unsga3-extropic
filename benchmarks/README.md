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
| `notes` | Free text. A measured run names its reference front. It also records how many closer occupants niche survival kept (quota 1). `metrics.nd_count` stays the non-dominated archive size |

Front files are local artifacts (gitignored under `artifacts/` next to the JSONL). Commit the JSONL when a number should stay in the history. The loader is `unsga3_extropic.fidelity.load_front`. It does not launch Bend, C#, or ZDT1.

```bash
python benchmarks/append_run.py --stub --out benchmarks/records/runs.jsonl
python benchmarks/append_run.py --smoke --out benchmarks/records/runs.jsonl
```

`--stub` writes a schema-valid row with null HV / GD / coverage and does not sample. `--smoke` runs the in-repo Potts chain (THRML, issue 4) and fills metrics with the harness from issue 6. Neither command searches for a beta schedule.

`demos/run_codon_thrml.py` and `demos/run_potts_thrml.py` append the same kind of record under `demos/_out/` and to `benchmarks/records/runs.jsonl`. The codon line is issue `5` / phase `2`. The Potts line stays issue `4` / phase `1` and mentions the niche count in `notes`.

`demos/run_domain_wall.py` appends a line for the domain-wall Ising image of the same Potts chain (issue `10`, phase `optional`, backend `thrml_domain_wall`). `notes` names the sampler (`thrml_ising` or `spin_ebm`) and how many invalid thermometers were left out of the archive. That run is a THRML simulation. It is not a Z1 measurement.

The optional Torx smoke (`demos/run_torx_pswap.py`, issue 7) reports a PSWAP stay/swap rate. That rate is not a minimization front, so the demo does not append a line here. The `torx` pytest is the record of that rate.

## Smoke, default, and deep

`notes` on new rows start with `profile=smoke`, `profile=default`, or `profile=deep`. Rows committed before that prefix are the CI smokes: codon `eval_budget` 24, Potts and domain wall `eval_budget` 16. The log is append-only, so those lines are not rewritten.

| Profile | Where | What it is |
|---------|--------|------------|
| `smoke` | `demos/run_*.py --smoke`, `benchmarks/append_run.py --smoke`, default CI | Two weights, two betas, one seed. Potts chain has 6 sites |
| `default` | the same demos without `--smoke` | Medium budget, one seed. Potts and domain wall use 8 sites |
| `deep` | `benchmarks/run_deep.py` | 11 simplex weights, six betas, 24 warmup, 24 samples, 2 steps between samples, seeds `7`, `11`, `19`. Same instances as smoke. Not in default CI |

Deep recorded rows per seed, before the domain-wall thermometer filter: `11 * 6 * 24 = 1584`. Across three seeds that is 4752. The codon smoke is 24. The Potts smoke is 16.

```bash
python benchmarks/run_deep.py
python benchmarks/run_deep.py --only codon
python benchmarks/run_deep.py --skip-torx
```

`benchmarks/DEEP_RESULTS.md` is rewritten from that invocation. Search rows append to `records/runs.jsonl`. The Torx deep check (100,000 samples, three seeds, same `p = 0.3` as the docs quickstart) is a stay/swap rate in `records/torx_pswap_deep.json`. It is not a front, so it is not a JSONL row. `.github/workflows/deep.yml` runs the full command on `workflow_dispatch` and uploads the files. It does not commit them.

Deep rows are THRML simulations, plus a NumPy `exact_ew` bit-flip Metropolis run on the codon chain. They are not Z1 measurements and they do not call Thermalizers.
