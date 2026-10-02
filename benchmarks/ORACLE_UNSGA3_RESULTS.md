# Classical continuous U-NSGA-III oracle results

Generated 2026-10-02T19:11:00Z. `git_sha` at record time: `4bb4053e6851128ac120e4a66c41356875ead3d6`.

Full seeds 1–15 on ZDT1, ZDT2, and DTLZ2.

Sampler: **classical continuous U-NSGA-III** (`unsga3_extropic.classical`). Generational population, Das–Dennis reference directions, non-dominated ranking, perpendicular association, PymooCompatible tournament (same niche: better rank, then closer perpendicular distance; otherwise a coin), SBX (η=30, p_c=1), polynomial mutation (η=20, p_m=1/n). NumPy only. **Not THRML-native.** Not an Ising model, not a Potts factor, and not Extropic sampling. `WeightSweepLoop` does not call this path. `ExactEwContinuousBackend` is unchanged.

Bend and C# columns are the published PymooCompatible cells. They were not re-run here.
Source: AppSprout-dev/unsga3-bend docs/ORACLE-MULTISEED.md (2026-09-21, seeds 1–15, PymooCompatible).

ExactEw cells are the committed NumPy weight-sweep column. Source: benchmarks/ORACLE_RESULTS.md (NumPy ExactEwContinuousBackend, not U-NSGA-III).

IGD is the in-repo yardstick formula, the same quantity as the `igd=` line of `unsga3-bend/ab/igd_vs_pymoo.py`: the mean Euclidean distance from each reference-front point to the nearest obtained point. `fidelity.inverted_generational_distance` computes it. `fidelity.generational_distance` is a different formula (obtained toward reference, RMS at `p=2`) and is not this column.

Reference fronts: ZDT1 and ZDT2 analytic curves, 500 points (`f1 = i / 499`); DTLZ2 Das–Dennis at partitions 12, each direction scaled to unit L2 length (91 points, `pf_source=analytic-das-dennis-l2`). That is the Bend script's analytic fallback and the C# `ParetoFronts.Dtlz2` construction. pymoo is not imported, and pymoo's default ~136-point DTLZ2 sample is not used.

Objective evaluations are `pop * n_gen`, including the initial population. ZDT1 is 52×100 = 5200. ZDT2 is 52×250 = 13000. DTLZ2 is 92×150 = 13800. The population is at least the Das–Dennis count (ZDT H=13, DTLZ2 H=91).

## Medians

| Problem | Seeds scored | median Bend (published) | median C# (published) | median ExactEw (`igd=`) | median classical U-NSGA-III | classical objective evals | yardstick pop×gens |
|---|---:|---:|---:|---:|---:|---:|---:|
| zdt1 | 15 | 0.071475 | 0.082152 | 2.3144182108708304 | 0.06574897075266269 | 5200 | 5200 |
| zdt2 | 15 | 0.025621 | 0.018684 | 3.3547378401332653 | 0.01805821133378065 | 13000 | 13000 |
| dtlz2 | 15 | 0.004243 | 0.004512 | 0.27690988609911626 | 0.005050379531838151 | 13800 | 13800 |

The published median is the summary in the Bend document. The ExactEw median is the middle committed `igd=` cell. The classical median is the middle in-repo IGD after sorting the scored seeds.

## ZDT1 n=30, partitions=12, pop=52 gens=100

| seed | Bend IGD | C# IGD | ExactEw IGD | classical U-NSGA-III IGD | classical/Bend | classical n | objective evals | pf_rows | pf_source |
|-----:|---------:|-------:|------------:|-------------------------:|---------------:|------------:|----------------:|--------:|-----------|
| 1 | 0.061603 | 0.097733 | 2.573954812980915 | 0.05827076935009233 | 0.946 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 2 | 0.062526 | 0.156085 | 2.3181313832106145 | 0.05867980226057724 | 0.938 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 3 | 0.114314 | 0.074783 | 2.11714920768485 | 0.06274675841460689 | 0.549 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 4 | 0.084756 | 0.130272 | 2.550915646169431 | 0.07602520643571233 | 0.897 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 5 | 0.083329 | 0.101950 | 2.4304560125805246 | 0.06574897075266269 | 0.789 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 6 | 0.045916 | 0.082152 | 1.97493669221992 | 0.1104659497811228 | 2.406 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 7 | 0.060659 | 0.060123 | 2.498340926086739 | 0.0767755186948639 | 1.266 | 51 | 5200 | 500 | analytic-zdt1 n=500 |
| 8 | 0.082262 | 0.112883 | 2.165663380358867 | 0.05337442900737393 | 0.649 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 9 | 0.061749 | 0.072583 | 2.0264104444773463 | 0.08018258743349349 | 1.299 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 10 | 0.084640 | 0.072246 | 2.3144182108708304 | 0.099827756850001 | 1.179 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 11 | 0.151236 | 0.066878 | 2.0128962973772593 | 0.0939673900497076 | 0.621 | 50 | 5200 | 500 | analytic-zdt1 n=500 |
| 12 | 0.126977 | 0.067323 | 2.514348818369275 | 0.05270158905192945 | 0.415 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 13 | 0.061183 | 0.051285 | 2.1410554845354697 | 0.06333235537386869 | 1.035 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 14 | 0.069462 | 0.103394 | 2.0921986003850757 | 0.1276912182338529 | 1.838 | 52 | 5200 | 500 | analytic-zdt1 n=500 |
| 15 | 0.071475 | 0.112429 | 2.4669042240363375 | 0.06514572123010116 | 0.911 | 52 | 5200 | 500 | analytic-zdt1 n=500 |

Seeds scored: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15. Classical median of those IGD cells: 0.06574897075266269.

## ZDT2 n=30, partitions=12, pop=52 gens=250

| seed | Bend IGD | C# IGD | ExactEw IGD | classical U-NSGA-III IGD | classical/Bend | classical n | objective evals | pf_rows | pf_source |
|-----:|---------:|-------:|------------:|-------------------------:|---------------:|------------:|----------------:|--------:|-----------|
| 1 | 0.024550 | 0.019230 | 2.800481960915036 | 0.01703131322864674 | 0.694 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 2 | 0.025621 | 0.017794 | 3.691944745280984 | 0.01729995355739914 | 0.675 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 3 | 0.028093 | 0.018684 | 3.219632199880277 | 0.02881370993860632 | 1.026 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 4 | 0.023679 | 0.033702 | 3.504236825930411 | 0.05073060778037122 | 2.142 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 5 | 0.108520 | 0.029099 | 3.194275913660782 | 0.01857215127958672 | 0.171 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 6 | 0.024795 | 0.021193 | 3.137348744717338 | 0.0201585079855249 | 0.813 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 7 | 0.022320 | 0.016622 | 3.175246660902046 | 0.01720403070584484 | 0.771 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 8 | 0.075328 | 0.021164 | 3.42828259753458 | 0.01657166612370455 | 0.220 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 9 | 0.025847 | 0.017039 | 3.4768545918955267 | 0.03652659022313405 | 1.413 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 10 | 0.023553 | 0.018429 | 3.2518478846481877 | 0.0177622684716716 | 0.754 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 11 | 0.033346 | 0.042790 | 3.427559578691599 | 0.01865972631888592 | 0.560 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 12 | 0.092069 | 0.016716 | 3.3547378401332653 | 0.01805821133378065 | 0.196 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 13 | 0.024122 | 0.014222 | 3.4124045698649197 | 0.01794446206396589 | 0.744 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 14 | 0.061379 | 0.017277 | 2.9869022713762194 | 0.01861763297667171 | 0.303 | 52 | 13000 | 500 | analytic-zdt2 n=500 |
| 15 | 0.022635 | 0.019121 | 3.476804151222168 | 0.01602077359392442 | 0.708 | 52 | 13000 | 500 | analytic-zdt2 n=500 |

Seeds scored: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15. Classical median of those IGD cells: 0.01805821133378065.

## DTLZ2 M=3 k=10 (n=12), partitions=12, pop=92 gens=150

| seed | Bend IGD | C# IGD | ExactEw IGD | classical U-NSGA-III IGD | classical/Bend | classical n | objective evals | pf_rows | pf_source |
|-----:|---------:|-------:|------------:|-------------------------:|---------------:|------------:|----------------:|--------:|-----------|
| 1 | 0.003914 | 0.004032 | 0.22501888314606172 | 0.003679800984466357 | 0.940 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 2 | 0.004250 | 0.005666 | 0.28223824174102874 | 0.009812560306782498 | 2.309 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 3 | 0.003777 | 0.005130 | 0.29277339941332103 | 0.005850732075558946 | 1.549 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 4 | 0.004369 | 0.004777 | 0.2731700136126506 | 0.005813940466147796 | 1.331 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 5 | 0.004848 | 0.004663 | 0.26811522872347165 | 0.004779418681922744 | 0.986 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 6 | 0.004026 | 0.005460 | 0.2954215858597582 | 0.005677063505382737 | 1.410 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 7 | 0.003603 | 0.003610 | 0.26888841801777696 | 0.004747712990627939 | 1.318 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 8 | 0.004243 | 0.005111 | 0.27690988609911626 | 0.005289638791355924 | 1.247 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 9 | 0.004004 | 0.004670 | 0.29194714775010533 | 0.005050379531838151 | 1.261 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 10 | 0.004882 | 0.003816 | 0.27243615629702717 | 0.004170553778100818 | 0.854 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 11 | 0.003727 | 0.003974 | 0.3211235186284744 | 0.004186408854357564 | 1.123 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 12 | 0.004901 | 0.004512 | 0.2667310235654121 | 0.005176533981113274 | 1.056 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 13 | 0.004502 | 0.003984 | 0.2708174872960866 | 0.004433996817118117 | 0.985 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 14 | 0.004066 | 0.004172 | 0.2833926365288975 | 0.005448755434941923 | 1.340 | 92 | 13800 | 91 | analytic-das-dennis-l2 |
| 15 | 0.004357 | 0.003627 | 0.28115128161400366 | 0.004594146067271587 | 1.054 | 92 | 13800 | 91 | analytic-das-dennis-l2 |

Seeds scored: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15. Classical median of those IGD cells: 0.005050379531838151.

## Reading the classical column

A larger IGD is farther from the reference front. This column is the generational algorithm the ExactEw weight-sweep is not: a population, SBX, polynomial mutation, and reference-direction survival. The RNG is NumPy's Generator, not Bend's LCG, so fronts are not bit-identical to the published Bend runs. The median is the comparison. `classical n` is the number of unique non-dominated objective rows written to the CSV.
