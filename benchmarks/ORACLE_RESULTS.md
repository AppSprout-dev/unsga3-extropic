# Continuous ExactEw oracle results

Generated 2026-10-02T18:49:22Z. `git_sha` at record time: `3db013e7cc4a3cf1b9cdda29bcd2219a32b7ee2f`.

Full seeds 1–15 on ZDT1, ZDT2, and DTLZ2.

Sampler: **NumPy ExactEw** (`ExactEwContinuousBackend`). One-coordinate truncated-normal Metropolis (`step_scale=0.1`) on exact \(E_w = w \cdot f(x)\), then the weight-sweep archive. **Not THRML-native.** Not an Ising model, not a Potts factor, and not a fitted surrogate. Not U-NSGA-III: no SBX, no polynomial mutation, no generational population, tournament not applicable. The classical continuous U-NSGA-III column is a separate file, `benchmarks/ORACLE_UNSGA3_RESULTS.md`.

Bend and C# columns are the published PymooCompatible cells. They were not re-run here.
Source: AppSprout-dev/unsga3-bend docs/ORACLE-MULTISEED.md (2026-09-21, seeds 1–15, PymooCompatible).

Extropic IGD is only the `igd=` line from `unsga3-bend/ab/igd_vs_pymoo.py`. The in-repo generational distance is a different formula and is left null on these records.

Reference fronts required by that script: ZDT1 and ZDT2 analytic PF, 500 points; DTLZ2 Das–Dennis at partitions 12 (91 points). The pymoo default ~136-point DTLZ2 sample is not used.

Objective evaluations are initial state plus every proposal. `eval_budget` in the JSONL counts recorded archive rows (warmup excluded). The Bend yardstick these budgets track is `pop * gens` (ZDT1 5200, ZDT2 13000, DTLZ2 13800). DTLZ2 uses 91 Das–Dennis directions, so 13800 does not divide evenly; the ladder spends 13741 objective calls.

## How the Extropic column was scored

```bash
python3 ab/igd_vs_pymoo.py --front EXTROPIC.csv --problem zdt1 --pf-points 500
python3 ab/igd_vs_pymoo.py --front EXTROPIC.csv --problem zdt2 --pf-points 500 --partitions 12
python3 ab/igd_vs_pymoo.py --front EXTROPIC.csv --problem dtlz2 --partitions 12
```

Front CSVs are the unique non-dominated archive, one objective vector per line, no header.

## Medians

| Problem | Seeds scored | median Bend (published) | median C# (published) | median Extropic (`igd=`) | Extropic objective evals | yardstick pop×gens |
|---|---:|---:|---:|---:|---:|---:|
| zdt1 | 15 | 0.071475 | 0.082152 | 2.3144182108708304 | 5200 | 5200 |
| zdt2 | 15 | 0.025621 | 0.018684 | 3.3547378401332653 | 13000 | 13000 |
| dtlz2 | 15 | 0.004243 | 0.004512 | 0.27690988609911626 | 13741 | 13800 |

The published median column is the summary in the Bend document, not a recomputation. The Extropic median is the middle `igd=` cell after sorting the scored seeds for that problem.

## ZDT1 n=30, partitions=12, Bend pop=52 gens=100

| seed | Bend IGD | C# IGD | Extropic IGD | Extropic/Bend | Extropic n | objective evals | recorded samples | pf_rows | pf_source |
|-----:|---------:|-------:|-------------:|--------------:|-----------:|----------------:|-----------------:|--------:|-----------|
| 1 | 0.061603 | 0.097733 | 2.573954812980915 | 41.783 | 20 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 2 | 0.062526 | 0.156085 | 2.3181313832106145 | 37.075 | 17 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 3 | 0.114314 | 0.074783 | 2.11714920768485 | 18.520 | 15 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 4 | 0.084756 | 0.130272 | 2.550915646169431 | 30.097 | 19 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 5 | 0.083329 | 0.101950 | 2.4304560125805246 | 29.167 | 27 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 6 | 0.045916 | 0.082152 | 1.97493669221992 | 43.012 | 26 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 7 | 0.060659 | 0.060123 | 2.498340926086739 | 41.187 | 12 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 8 | 0.082262 | 0.112883 | 2.165663380358867 | 26.326 | 18 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 9 | 0.061749 | 0.072583 | 2.0264104444773463 | 32.817 | 13 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 10 | 0.084640 | 0.072246 | 2.3144182108708304 | 27.344 | 13 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 11 | 0.151236 | 0.066878 | 2.0128962973772593 | 13.310 | 14 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 12 | 0.126977 | 0.067323 | 2.514348818369275 | 19.802 | 18 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 13 | 0.061183 | 0.051285 | 2.1410554845354697 | 34.994 | 19 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 14 | 0.069462 | 0.103394 | 2.0921986003850757 | 30.120 | 21 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |
| 15 | 0.071475 | 0.112429 | 2.4669042240363375 | 34.514 | 23 | 5200 | 4680 | 500 | analytic-zdt1 n=500 |

Seeds scored: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15. Extropic median of those `igd=` cells: 2.3144182108708304.

## ZDT2 n=30, partitions=12, Bend pop=52 gens=250

| seed | Bend IGD | C# IGD | Extropic IGD | Extropic/Bend | Extropic n | objective evals | recorded samples | pf_rows | pf_source |
|-----:|---------:|-------:|-------------:|--------------:|-----------:|----------------:|-----------------:|--------:|-----------|
| 1 | 0.024550 | 0.019230 | 2.800481960915036 | 114.073 | 10 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 2 | 0.025621 | 0.017794 | 3.691944745280984 | 144.098 | 10 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 3 | 0.028093 | 0.018684 | 3.219632199880277 | 114.606 | 11 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 4 | 0.023679 | 0.033702 | 3.504236825930411 | 147.989 | 12 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 5 | 0.108520 | 0.029099 | 3.194275913660782 | 29.435 | 8 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 6 | 0.024795 | 0.021193 | 3.137348744717338 | 126.532 | 10 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 7 | 0.022320 | 0.016622 | 3.175246660902046 | 142.260 | 11 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 8 | 0.075328 | 0.021164 | 3.42828259753458 | 45.511 | 3 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 9 | 0.025847 | 0.017039 | 3.4768545918955267 | 134.517 | 10 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 10 | 0.023553 | 0.018429 | 3.2518478846481877 | 138.065 | 4 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 11 | 0.033346 | 0.042790 | 3.427559578691599 | 102.788 | 13 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 12 | 0.092069 | 0.016716 | 3.3547378401332653 | 36.437 | 5 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 13 | 0.024122 | 0.014222 | 3.4124045698649197 | 141.464 | 10 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 14 | 0.061379 | 0.017277 | 2.9869022713762194 | 48.663 | 9 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |
| 15 | 0.022635 | 0.019121 | 3.476804151222168 | 153.603 | 4 | 13000 | 11700 | 500 | analytic-zdt2 n=500 |

Seeds scored: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15. Extropic median of those `igd=` cells: 3.3547378401332653.

## DTLZ2 M=3 k=10 (n=12), partitions=12, Bend pop=92 gens=150

| seed | Bend IGD | C# IGD | Extropic IGD | Extropic/Bend | Extropic n | objective evals | recorded samples | pf_rows | pf_source |
|-----:|---------:|-------:|-------------:|--------------:|-----------:|----------------:|-----------------:|--------:|-----------|
| 1 | 0.003914 | 0.004032 | 0.22501888314606172 | 57.491 | 132 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 2 | 0.004250 | 0.005666 | 0.28223824174102874 | 66.409 | 169 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 3 | 0.003777 | 0.005130 | 0.29277339941332103 | 77.515 | 184 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 4 | 0.004369 | 0.004777 | 0.2731700136126506 | 62.525 | 119 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 5 | 0.004848 | 0.004663 | 0.26811522872347165 | 55.304 | 128 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 6 | 0.004026 | 0.005460 | 0.2954215858597582 | 73.378 | 134 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 7 | 0.003603 | 0.003610 | 0.26888841801777696 | 74.629 | 155 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 8 | 0.004243 | 0.005111 | 0.27690988609911626 | 65.263 | 152 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 9 | 0.004004 | 0.004670 | 0.29194714775010533 | 72.914 | 186 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 10 | 0.004882 | 0.003816 | 0.27243615629702717 | 55.804 | 147 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 11 | 0.003727 | 0.003974 | 0.3211235186284744 | 86.161 | 157 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 12 | 0.004901 | 0.004512 | 0.2667310235654121 | 54.424 | 144 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 13 | 0.004502 | 0.003984 | 0.2708174872960866 | 60.155 | 151 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 14 | 0.004066 | 0.004172 | 0.2833926365288975 | 69.698 | 147 | 13741 | 12285 | 91 | pymoo-das-dennis |
| 15 | 0.004357 | 0.003627 | 0.28115128161400366 | 64.529 | 155 | 13741 | 12285 | 91 | pymoo-das-dennis |

Seeds scored: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15. Extropic median of those `igd=` cells: 0.27690988609911626.

## Reading the Extropic column

A larger IGD is farther from the analytic front. This sampler does not share Bend's SBX, polynomial mutation, or population. Matching `pop * gens` spends a similar number of objective calls; it does not reproduce the U-NSGA-III trajectory. `Extropic n` is `front_rows` from the IGD script (unique non-dominated rows), which is not forced to equal the Bend population.
