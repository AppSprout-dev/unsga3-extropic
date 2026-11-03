# Run records and deep

`benchmarks/append_run.py` appends schema-v1 JSONL run records (a non-measurement stub, or a measured THRML Potts smoke with HV/GD/coverage), and `benchmarks/run_deep.py` runs the deep profile (11 weights, six betas, 24 warmup, 24 samples, seeds 7/11/19), one process per seed, then writes a markdown summary.

## Sub-features

- `records-stub` `--stub` writes a schema-valid null-metric row without sampling.
- `records-smoke` `--smoke` samples the Potts chain and fills metrics and a front `.npz` next to `--out`.
- `records-schema` `validate_run_record` accepts both rows.
- `deep-codon` deep codon THRML + NumPy exact_ew rows, 1584 recorded rows per seed.
- `deep-resume` rerunning skips rows already in `--out` with `profile=deep` and the same schedule.

## How to get to it (user POV)

- `python benchmarks/append_run.py (--stub|--smoke) [--out PATH] [--front PATH]`
- `python benchmarks/run_deep.py [--only codon|potts|domain_wall|exact_ew|torx ...] [--skip-torx] [--out PATH] [--summary PATH] [--torx-out PATH]`
- GitHub: dispatch `.github/workflows/deep.yml` (uploads records as an artifact, commits nothing).

## Driving it with verify.sh

Preconditions:

- Baseline preconditions from the README hold.

- **Records.** `$V drive records`. Steps `records_stub`, `records_smoke`, `records_validate` end `rc=0`; `records_validate.log` says `validated 2 rows`; `records/artifacts/` holds `stub-*.npz` and `potts_chain-*.npz`; `records_smoke.log` shows `nd_count: 3`.
- **Deep codon.** `$V drive deep-codon` (~3 min). Step `deep_codon_exact_ew` ends `rc=0`; `deep/DEEP_RESULTS.md` lists thrml_ising seeds 7/11/19 with `eval_budget 1584`, `nd 3`, `hv 18.65`, `gd 0`, `coverage 1.0`, and exact_ew seed 11 with `hv 21.89`, `coverage 0.6667` (matching the committed `benchmarks/DEEP_RESULTS.md`).
- **Resume.** Rerun `$V drive deep-codon` in the same session; the log reports the rows as already present and finishes in seconds.

## Gotchas

- Always pass `--out`/`--summary` (the harness does); defaults append to tracked `records/runs.jsonl` and rewrite tracked `DEEP_RESULTS.md`.
- Each seed runs in its own subprocess because one process holding every seed ran out of host memory. Do not "optimize" this away while verifying.
- The full deep run (all backends + Torx 100k) is ~tens of minutes on CPU; it is the `deep.yml` dispatch, not part of default CI.
