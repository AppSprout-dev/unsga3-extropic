# Deep benchmark results

Generated 2026-11-26T23:15:16Z by `benchmarks/run_deep.py`.
`git_sha` on each row is `HEAD` when that row was recorded (`28caddaedb2c710f9e18c33c2a82da99aa6799cd` for this invocation).
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
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 6.003 | 28caddaedb2c |
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 6.475 | 28caddaedb2c |
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 6.948 | 28caddaedb2c |
| thrml_potts | potts_chain | n_sites=6, K=3 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | — | 9.276 | 28caddaedb2c |
| thrml_potts | potts_chain | n_sites=6, K=3 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | — | 9.796 | 28caddaedb2c |
| thrml_potts | potts_chain | n_sites=6, K=3 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | — | 9.650 | 28caddaedb2c |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1374 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | 210 | 9.998 | 28caddaedb2c |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1359 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | 225 | 11.073 | 28caddaedb2c |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1365 | 5 | 5/5 | 12.2000 | 0.0000 | 1.0000 | 5 | 219 | 8.659 | 28caddaedb2c |
| exact_ew | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 7 | 1/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 0.701 | 28caddaedb2c |
| exact_ew | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 11 | 2/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.4500 | 0.6928 | 0.6667 | 3 | — | 0.509 | 28caddaedb2c |
| exact_ew | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 19 | 3/3 | 11 | 6 | 24 | 24 | 2 | 1584 | 3 | 3/3 | 18.6500 | 0.0000 | 1.0000 | 3 | — | 0.553 | 28caddaedb2c |

## Seed aggregates

Min / mean / max across the seeds in this invocation.

| backend | seeds | ND min/mean/max | HV min/mean/max | GD min/mean/max | coverage min/mean/max | invalid sum | elapsed sum s |
|---|---:|---|---|---|---|---:|---:|
| exact_ew | 3 | 3.00 / 3.00 / 3.00 | 18.4500 / 18.5833 / 18.6500 | 0.0000 / 0.2309 / 0.6928 | 0.6667 / 0.8889 / 1.0000 | — | 1.763 |
| thrml_domain_wall | 3 | 5.00 / 5.00 / 5.00 | 12.2000 / 12.2000 / 12.2000 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | 654 | 29.730 |
| thrml_ising | 3 | 3.00 / 3.00 / 3.00 | 18.6500 / 18.6500 / 18.6500 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | — | 19.426 |
| thrml_potts | 3 | 5.00 / 5.00 / 5.00 | 12.2000 / 12.2000 / 12.2000 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | — | 28.722 |

## Earlier log rows

Rows already in `benchmarks/records/runs.jsonl` before this invocation, excluding any previous `profile=deep` lines. Historical smoke rows were written before notes carried `profile=smoke`. Their budgets (24 for codon, 16 for Potts and domain wall) are the CI smokes.

| backend | problem | instance | base seed | replicate | weights | betas | warmup | samples | steps | eval_budget | ND | niche | HV | GD | coverage | exact ND | invalid | elapsed_s | git_sha |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| thrml_ising | codon_ising | n_spins=12, J=1, field_seed=0, global_bias=0.05 | 7 | — | 2 | 2 | 4 | 6 | 1 | 24 | 1 | 1/1 | 8.0500 | 0.0000 | 0.3333 | — | — | — | 34044ac48cb4 |
| thrml_potts | potts_chain | n_sites=6, K=3 | 7 | — | 2 | 2 | 2 | 4 | 1 | 16 | 3 | 2/3 | 10.3000 | 0.5774 | 0.4000 | — | — | 3.201 | 34044ac48cb4 |
| thrml_domain_wall | potts_chain | n_sites=6, K=3 | 11 | — | 2 | 2 | 2 | 4 | 1 | 16 | 4 | 2/4 | 10.8000 | 0.5000 | 0.6000 | — | 0 | 3.183 | 4f97718e2e11 |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 1 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 20 | 10/20 | 1.7976 | — | — | — | — | 0.070 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 2 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 17 | 10/17 | 2.6868 | — | — | — | — | 0.067 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 3 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 15 | 12/15 | 2.3642 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 4 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 19 | 11/19 | 3.4880 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 5 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 27 | 11/27 | 2.6228 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 6 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 26 | 13/26 | 2.6867 | — | — | — | — | 0.067 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 7 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 12 | 10/12 | 0.9682 | — | — | — | — | 0.067 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 8 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 18 | 10/18 | 1.8277 | — | — | — | — | 0.067 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 9 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 13 | 11/13 | 1.5711 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 10 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 13 | 8/13 | 1.9583 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 11 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 14 | 10/14 | 2.7204 | — | — | — | — | 0.067 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 12 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 18 | 12/18 | 2.1057 | — | — | — | — | 0.067 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 13 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 19 | 13/19 | 2.5226 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 14 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 21 | 10/21 | 2.5247 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt1 | n_var=30 box=[0,1] partitions=12 pop=52 gens=100 | 15 | — | 13 | 3 | 13 | 120 | 1 | 4680 | 23 | 11/23 | 2.7203 | — | — | — | — | 0.068 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 1 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 10 | 7/10 | 1.8209 | — | — | — | — | 0.168 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 2 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 10 | 5/10 | 2.1400 | — | — | — | — | 0.168 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 3 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 11 | 9/11 | 1.8460 | — | — | — | — | 0.171 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 4 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 12 | 10/12 | 0.3613 | — | — | — | — | 0.167 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 5 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 8 | 6/8 | 0.7346 | — | — | — | — | 0.168 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 6 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 10 | 8/10 | 1.5016 | — | — | — | — | 0.169 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 7 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 11 | 7/11 | 1.3861 | — | — | — | — | 0.168 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 8 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 3 | 3/3 | 0.1497 | — | — | — | — | 0.172 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 9 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 10 | 4/10 | 1.3845 | — | — | — | — | 0.170 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 10 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 4 | 4/4 | 0.1263 | — | — | — | — | 0.172 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 11 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 13 | 9/13 | 1.2005 | — | — | — | — | 0.170 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 12 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 5 | 3/5 | 0.9076 | — | — | — | — | 0.170 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 13 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 10 | 5/10 | 1.7175 | — | — | — | — | 0.172 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 14 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 9 | 5/9 | 1.2591 | — | — | — | — | 0.173 | 3db013e7cc4a |
| exact_ew_continuous | zdt2 | n_var=30 box=[0,1] partitions=12 pop=52 gens=250 | 15 | — | 13 | 3 | 33 | 300 | 1 | 11700 | 4 | 4/4 | 0.1751 | — | — | — | — | 0.170 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 1 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 132 | 61/132 | — | — | — | — | — | 0.339 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 2 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 169 | 70/169 | — | — | — | — | — | 0.339 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 3 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 184 | 78/184 | — | — | — | — | — | 0.351 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 4 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 119 | 51/119 | — | — | — | — | — | 0.340 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 5 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 128 | 66/128 | — | — | — | — | — | 0.362 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 6 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 134 | 62/134 | — | — | — | — | — | 0.340 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 7 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 155 | 69/155 | — | — | — | — | — | 0.338 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 8 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 152 | 67/152 | — | — | — | — | — | 0.340 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 9 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 186 | 73/186 | — | — | — | — | — | 0.340 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 10 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 147 | 74/147 | — | — | — | — | — | 0.342 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 11 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 157 | 67/157 | — | — | — | — | — | 0.339 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 12 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 144 | 63/144 | — | — | — | — | — | 0.340 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 13 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 151 | 69/151 | — | — | — | — | — | 0.340 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 14 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 147 | 65/147 | — | — | — | — | — | 0.343 | 3db013e7cc4a |
| exact_ew_continuous | dtlz2 | n_var=12 box=[0,1] partitions=12 pop=92 gens=150 | 15 | — | 91 | 3 | 5 | 45 | 1 | 12285 | 155 | 65/155 | — | — | — | — | — | 0.350 | 3db013e7cc4a |

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
