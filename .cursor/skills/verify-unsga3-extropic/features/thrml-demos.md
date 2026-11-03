# THRML demos

The three demo CLIs run a `WeightSweepLoop` on a THRML sampler (Ising block Gibbs, categorical Potts, and the domain-wall Ising image of the Potts chain), print the non-dominated front and fidelity metrics against the exact enumerated front, write summaries under `demos/_out/`, and append one run record to `benchmarks/records/runs.jsonl`.

## Sub-features

- `demo-codon-smoke` codon Ising, 12 spins, `profile=smoke` (CI).
- `demo-potts-smoke` Potts chain, 6 sites K=3, `profile=smoke` (CI).
- `demo-domain-wall-smoke` domain-wall image of the same Potts chain, `profile=smoke` (CI).
- `demo-default` the same three without `--smoke` (`profile=default`, 8-site Potts/domain wall).
- `demo-record` each run appends exactly one schema-valid row to the tracked log.

## How to get to it (user POV)

- `python demos/run_codon_thrml.py [--smoke]`
- `python demos/run_potts_thrml.py [--smoke]`
- `python demos/run_domain_wall.py [--smoke]`

## Driving it with verify.sh

Preconditions:

- Baseline preconditions from the README hold.

- **Run the smokes.** `$V drive demos`. Each `demo_*_smoke` step ends `rc=0`.
- **Codon result.** In `demo_codon_thrml_smoke.log`: `eval_budget: 24`, `nd_count: 1`, `hypervolume_2d: 8.05`, `generational_distance: 0.0`, `exact_front_coverage: 0.333…` (seed 7, smoke budget; at a5e6f0b these match the committed smoke rows).
- **Potts result.** In `demo_potts_thrml_smoke.log`: `nd_count: 3`, `hypervolume_2d: 10.3`, `coverage: 0.4`.
- **Domain-wall result.** In `demo_domain_wall_smoke.log`: `nd_count: 4`, `hypervolume_2d: 10.8`, `coverage: 0.6`, notes say `Invalid thermometers excluded from the archive: 0` and `Sampler program is spin_ebm`.
- **Side effect.** `demos_runs_jsonl.numstat` reads `3	0	benchmarks/records/runs.jsonl` (three appended rows, nothing rewritten). `demos_out/` holds `*_summary.json` and front `.npy`/`.npz` files.
- **Default profile (optional, slower).** Run `.venv/bin/python demos/run_potts_thrml.py` by hand; `profile: default` and an 8-site instance appear in the summary.

## Gotchas

- The demos have no `--out` flag; they always append to the tracked `benchmarks/records/runs.jsonl`. Never commit those rows from a verification run; `$V cleanup` restores the file.
- Metric values are deterministic per seed and THRML/JAX version. A different JAX minor can legitimately move them; compare against the committed smoke rows and report the versions from doctor before calling it a regression.
- The domain-wall K>=3 image does not use `ThrmlIsingBackend` (its spin graph is not 2-colored), so `spin_ebm` in notes is expected.
- The first THRML call JIT-compiles; ~5–10 s per demo on CPU is normal.
