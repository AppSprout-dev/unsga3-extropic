# Deep benchmark results

Generated 2026-10-02T18:14:27Z by `benchmarks/run_deep.py`.
`git_sha` on each row is `HEAD` when that row was recorded (`69f1cb7fc141d4e2c8fa4b502d3a1cf87354835d` for this invocation).
The JSONL log is append-only; this file is the latest invocation, not a rewrite of older lines.

These search rows are THRML simulations on CPU, plus a NumPy exact-E_w Metropolis comparator on the codon chain. They are not Z1 measurements and they do not use Thermalizers.

Deep uses the smoke problem instances so the exact fronts match: codon `n_spins=12, J=1, field_seed=0, global_bias=0.05`; Potts and domain wall `n_sites=6, K=3`. `profile=default` is a different Potts chain (`n_sites=8, K=3`) and is not this matrix.

`eval_budget` is recorded objective rows (`n_weights * len(betas) * n_samples`), except domain wall, where invalid thermometers are left out of the archive and the budget counts scored categorical rows. `schedule_product` in the notes is the full sample count before that filter. Niche is quota-1 closer-in-niche survival. HV, GD, and coverage use `enumerate_front` as the reference (minimization). GD is mean Euclidean distance from the run front to that reference. Coverage is the fraction of the reference front weakly dominated by the run front.

## Budget

| | codon smoke | potts / domain-wall smoke | deep (each search backend) |
|---|---:|---:|---:|
| weights | 2 | 2 | 11 |
| betas | 2 | 2 | 6 |
| beta values | 1.0, 4.0 | 1.0, 4.0 | 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 |
| warmup | 4 | 2 | 24 |
| samples per beta | 6 | 4 | 24 |
| steps between samples | 1 | 1 | 2 |
| base seeds | 1 | 1 | 3 |
| recorded rows / seed | 24 | 16 | 1584 |
| recorded rows, all seeds | 24 | 16 | 4752 |

Deep weights are `simplex_weights(11, 2)` (the endpoints are included). Smoke weights are the two fixed vectors `fixed [[0.5, 0.5], [0.8, 0.2]]`. Base seeds for every deep search backend: [7, 11, 19]. `WeightSweepLoop` still passes `base_seed + 1009 * weight_index` into `sample_weight`; those expanded seeds are the record's `seeds` array.

## Deep matrix

| backend | problem | instance | base seed | replicate | weights | betas | warmup | samples | steps | eval_budget | ND | niche | HV | GD | coverage | exact ND | invalid | elapsed_s | git_sha |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 37.747 | 69f1cb7fc141 |
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 40.638 | 69f1cb7fc141 |
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 40.365 | 69f1cb7fc141 |
| thrml_potts | potts_chain | n_sites=6, K=3 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | — | 53.054 | 69f1cb7fc141 |
| thrml_potts | potts_chain | n_sites=6, K=3 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | — | 48.192 | 69f1cb7fc141 |
| thrml_potts | potts_chain | n_sites=6, K=3 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | — | 47.588 | 69f1cb7fc141 |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1374 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | 210 | 58.151 | 69f1cb7fc141 |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1359 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | 225 | 63.845 | 69f1cb7fc141 |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1365 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | 219 | 68.225 | 69f1cb7fc141 |
| exact_ew | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 1.356 | 69f1cb7fc141 |
| exact_ew | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 21.8900 | 0.6928 | 0.6667 | 3 | — | 1.119 | 69f1cb7fc141 |
| exact_ew | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 1.124 | 69f1cb7fc141 |

## Seed aggregates

Min / mean / max across the seeds in this invocation.

| backend | seeds | ND min/mean/max | HV min/mean/max | GD min/mean/max | coverage min/mean/max | invalid sum | elapsed sum s |
|---|---:|---|---|---|---|---:|---:|
| exact_ew | 3 | 3.00 / 3.00 / 3.00 | 18.6500 / 19.7300 / 21.8900 | 0.0000 / 0.2309 / 0.6928 | 0.6667 / 0.8889 / 1.0000 | — | 3.599 |
| thrml_domain_wall | 3 | 5.00 / 5.00 / 5.00 | 12.2000 / 12.2000 / 12.2000 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | 654 | 190.221 |
| thrml_ising | 3 | 3.00 / 3.00 / 3.00 | 18.6500 / 18.6500 / 18.6500 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | — | 118.750 |
| thrml_potts | 3 | 5.00 / 5.00 / 5.00 | 12.2000 / 12.2000 / 12.2000 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | — | 148.834 |

## Earlier log rows

Rows already in `benchmarks/records/runs.jsonl` before this invocation, excluding any previous `profile=deep` lines. Historical smoke rows were written before notes carried `profile=smoke`. Their budgets (24 for codon, 16 for Potts and domain wall) are the CI smokes.

| backend | problem | instance | base seed | replicate | weights | betas | warmup | samples | steps | eval_budget | ND | niche | HV | GD | coverage | exact ND | invalid | elapsed_s | git_sha |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 7 | — | 2 | 2 | 4 | 6 | 1 | 24 | 1 | 1/1 | 8.0500 | 0.0000 | 0.3333 | — | — | — | 34044ac48cb4 |
| thrml_potts | potts_chain | n_sites=6, K=3 | 7 | — | 2 | 2 | 2 | 4 | 1 | 16 | 3 | 2/3 | 10.3000 | 0.5774 | 0.4000 | — | — | 3.201 | 34044ac48cb4 |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 11 | — | 2 | 2 | 2 | 4 | 1 | 16 | 4 | 2/4 | 10.8000 | 0.5000 | 0.6000 | — | 0 | 3.183 | 4f97718e2e11 |

## Torx PSWAP rate check

Separate from the search log. These rates are not a minimization front, so they are not appended to `benchmarks/records/runs.jsonl`. The circuit is Torx `PSWAP` on `BranchingSimulator`. It is not a THRML program, not a Z1 run, and not Thermalizers.

- p_swap: 0.3
- n_samples: 100000 (smoke is 20_000)
- git_sha: `69f1cb7fc141d4e2c8fa4b502d3a1cf87354835d`
- sanity bound: absolute swap error < 0.02

| seed | stay | swap | |swap − p| | elapsed_s | within bound |
|---:|---:|---:|---:|---:|---|
| 0 | 0.6997 | 0.3003 | 0.0003 | 1.193 | yes |
| 1 | 0.6995 | 0.3005 | 0.0005 | 0.086 | yes |
| 2 | 0.6997 | 0.3003 | 0.0003 | 0.072 | yes |

Mean swap rate across 3 seeds: 0.3003.

## Re-run

From the repository root, with THRML, JAX, and (for the rate check) the `torx` extra:

```bash
pip install -e ".[dev,cpu,torx]"
export JAX_PLATFORMS=cpu
python benchmarks/run_deep.py
```

That appends one JSONL line per backend and seed to `benchmarks/records/runs.jsonl`, rewrites this file, and writes `benchmarks/records/torx_pswap_deep.json`. Each seed is its own process so compiled JAX programs from the previous seed are released. A deep row already in the log for the same backend, base seed, and schedule is skipped. Front arrays go to `benchmarks/records/artifacts/` and stay gitignored.

Default CI does not run this. `.github/workflows/deep.yml` is `workflow_dispatch` only. A dispatch run uploads the log and this summary as artifacts; it does not commit them.

Subset of backends, for a local check:

```bash
python benchmarks/run_deep.py --only codon
python benchmarks/run_deep.py --skip-torx
```

`--only` rewrites this summary as a partial invocation. Commit a summary only from a full run.
