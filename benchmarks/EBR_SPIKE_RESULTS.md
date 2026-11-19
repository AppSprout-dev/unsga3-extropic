# EBR spike results: lowest-energy selection and energy-plateau stop

Measured Nov 19, 2026 with `benchmarks/run_ebr_spike.py` on the branch working tree over main `da7e505`, with the tables rendered by `benchmarks/ebr_spike_report.py` from [records/ebr_spike.json](records/ebr_spike.json). The context is in [docs/ebr_note.md](../docs/ebr_note.md).

**Hardware:** the shared Linux box, an 8-vCPU Intel Xeon (x86_64) with 15 GB RAM and no GPU. **Software:** Python 3.13.5, NumPy 2.5.5, JAX 0.12.1 (CPU), THRML 0.1.4.

**Tasks:** the two deep instances whose fronts can be enumerated, both on the existing `DEEP_SCHEDULE`: 11 weights `simplex_weights(11, 2)`, betas 0.25–8, 24 warmup and 24 recorded samples per beta, 2 steps per sample, seeds 7, 11, 19.
- `codon`: the 12-spin chain (4,096 states). Its exact front has 3 points.
- `potts`: 6 sites with K = 3 (729 states). Its exact front has 5 points.

**Variants:** the default ExactEw Metropolis chain (`baseline`), `LowestEnergySelection` (one argmin row per weight), and `ExactEwMetropolisBackend(plateau=PlateauStop(window, tol=0))`. The `thrml` rows use the shipped THRML backend (Ising block Gibbs for codon, categorical block Gibbs for Potts) with the argmin applied afterward. The plateau setting (window 8, tol 0) was fixed before the run, and window 4 is a sensitivity row. On the Potts task the ExactEw chain uses a symmetric categorical proposal. THRML runs one compiled program per beta stage, so on THRML the plateau rule is only scored after the fact and saves no work.

**Columns:**
- **recorded rows:** samples added to the archive.
- **energy evals:** objective evaluations, including warmup. For THRML this is recorded samples, since Gibbs micro-steps aren't counted.
- **ND:** non-dominated archive size.
- **HV ratio:** 2-D hypervolume of the archive divided by that of the exact front, with a fixed reference point per task taken from the exact front.
- **GD:** `fidelity.generational_distance` to the exact front.
- **coverage:** fraction of the exact front weakly dominated.
- **argmin hits:** weights where the best sample's \(w \cdot f\), recomputed in float64, equals the exact enumerated minimum.

## Verdict

- **The plateau stop helps.** With window 8 it records 53% and 50% fewer samples and makes 35% and 34% fewer energy evaluations on codon and Potts. HV, GD, and coverage are equal to the baseline or better on all six task-seed pairs. Codon seed 11 rose from 0.989 to 1.000 HV, but that is a different random path, not a property of the rule. The cost shows in argmin hits: per-weight minima were found slightly less often on Potts (30/33 against 33/33, and 24/33 at window 4). The front quality the loop reports didn't change.
- **Lowest-energy selection doesn't help as an archive replacement.** It never improves HV or coverage and usually lowers them: Potts mean coverage fell from 1.000 to 0.533 on ExactEw and to 0.600 on THRML. Neighbouring weights share an argmin, so 11 weights collapse onto 2–4 front points. The archive keeps off-target samples that land on other front points. On THRML the argmin was already the exact per-weight minimum on 66/66 weights, so there was nothing for selection to fix.
- Neither tool is on by default.

## Tables

### codon

| variant | backend | seed | recorded rows | energy evals | ND | HV ratio | GD | coverage | argmin hits |
|---|---|---|---|---|---|---|---|---|---|
| baseline | exact_ew | 7 | 1584 | 4763 | 3 | 1.000 | 0.000 | 1.000 | 10/11 |
| baseline | exact_ew | 11 | 1584 | 4763 | 3 | 0.989 | 0.693 | 0.667 | 8/11 |
| baseline | exact_ew | 19 | 1584 | 4763 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| select | exact_ew | 7 | 11 | 4763 | 2 | 0.453 | 0.000 | 0.667 | 10/11 |
| select | exact_ew | 11 | 11 | 4763 | 3 | 0.989 | 0.693 | 0.667 | 8/11 |
| select | exact_ew | 19 | 11 | 4763 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=8) | exact_ew | 7 | 740 | 3075 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=8) | exact_ew | 11 | 755 | 3105 | 3 | 1.000 | 0.000 | 1.000 | 10/11 |
| plateau(w=8) | exact_ew | 19 | 752 | 3099 | 3 | 1.000 | 0.000 | 1.000 | 8/11 |
| plateau(w=8)+select | exact_ew | 7 | 11 | 3075 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=8)+select | exact_ew | 11 | 11 | 3105 | 2 | 0.453 | 0.000 | 0.667 | 10/11 |
| plateau(w=8)+select | exact_ew | 19 | 11 | 3099 | 3 | 0.989 | 0.693 | 0.667 | 8/11 |
| plateau(w=4) | exact_ew | 7 | 408 | 2411 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=4) | exact_ew | 11 | 439 | 2473 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=4) | exact_ew | 19 | 413 | 2421 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=4)+select | exact_ew | 7 | 11 | 2411 | 2 | 0.453 | 0.000 | 0.667 | 9/11 |
| plateau(w=4)+select | exact_ew | 11 | 11 | 2473 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=4)+select | exact_ew | 19 | 11 | 2421 | 3 | 1.000 | 0.000 | 1.000 | 9/11 |
| thrml | thrml_ising | 7 | 1584 | 1584 | 3 | 1.000 | 0.000 | 1.000 | 11/11 |
| thrml | thrml_ising | 11 | 1584 | 1584 | 3 | 1.000 | 0.000 | 1.000 | 11/11 |
| thrml | thrml_ising | 19 | 1584 | 1584 | 3 | 1.000 | 0.000 | 1.000 | 11/11 |
| thrml+select | thrml_ising | 7 | 11 | 1584 | 2 | 0.453 | 0.000 | 0.667 | 11/11 |
| thrml+select | thrml_ising | 11 | 11 | 1584 | 2 | 0.453 | 0.000 | 0.667 | 11/11 |
| thrml+select | thrml_ising | 19 | 11 | 1584 | 2 | 0.453 | 0.000 | 0.667 | 11/11 |

Means over seeds 7, 11, 19:

| variant | recorded rows | energy evals | HV ratio | GD | coverage | argmin hits |
|---|---|---|---|---|---|---|
| baseline | 1584.0 | 4763.0 | 0.996 | 0.231 | 0.889 | 27/33 |
| select | 11.0 | 4763.0 | 0.814 | 0.231 | 0.778 | 27/33 |
| plateau(w=8) | 749.0 | 3093.0 | 1.000 | 0.000 | 1.000 | 27/33 |
| plateau(w=8)+select | 11.0 | 3093.0 | 0.814 | 0.231 | 0.778 | 27/33 |
| plateau(w=4) | 420.0 | 2435.0 | 1.000 | 0.000 | 1.000 | 27/33 |
| plateau(w=4)+select | 11.0 | 2435.0 | 0.818 | 0.000 | 0.889 | 27/33 |
| thrml | 1584.0 | 1584.0 | 1.000 | 0.000 | 1.000 | 33/33 |
| thrml+select | 11.0 | 1584.0 | 0.453 | 0.000 | 0.667 | 33/33 |

THRML plateau rule scored after the fact (window 8, tol 0; no THRML work saved): seed 7: 745 of 1584 rows, seed 11: 771 of 1584 rows, seed 19: 795 of 1584 rows.

### potts

| variant | backend | seed | recorded rows | energy evals | ND | HV ratio | GD | coverage | argmin hits |
|---|---|---|---|---|---|---|---|---|---|
| baseline | exact_ew | 7 | 1584 | 4763 | 5 | 1.000 | 0.000 | 1.000 | 11/11 |
| baseline | exact_ew | 11 | 1584 | 4763 | 5 | 1.000 | 0.000 | 1.000 | 11/11 |
| baseline | exact_ew | 19 | 1584 | 4763 | 5 | 1.000 | 0.000 | 1.000 | 11/11 |
| select | exact_ew | 7 | 11 | 4763 | 2 | 0.344 | 0.000 | 0.400 | 11/11 |
| select | exact_ew | 11 | 11 | 4763 | 2 | 0.344 | 0.000 | 0.400 | 11/11 |
| select | exact_ew | 19 | 11 | 4763 | 4 | 0.918 | 0.000 | 0.800 | 11/11 |
| plateau(w=8) | exact_ew | 7 | 756 | 3107 | 5 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=8) | exact_ew | 11 | 799 | 3193 | 5 | 1.000 | 0.000 | 1.000 | 11/11 |
| plateau(w=8) | exact_ew | 19 | 799 | 3193 | 5 | 1.000 | 0.000 | 1.000 | 10/11 |
| plateau(w=8)+select | exact_ew | 7 | 11 | 3107 | 4 | 0.918 | 0.000 | 0.800 | 9/11 |
| plateau(w=8)+select | exact_ew | 11 | 11 | 3193 | 3 | 0.836 | 0.000 | 0.600 | 11/11 |
| plateau(w=8)+select | exact_ew | 19 | 11 | 3193 | 4 | 0.918 | 0.000 | 0.800 | 10/11 |
| plateau(w=4) | exact_ew | 7 | 419 | 2433 | 5 | 1.000 | 0.000 | 1.000 | 8/11 |
| plateau(w=4) | exact_ew | 11 | 415 | 2425 | 5 | 1.000 | 0.000 | 1.000 | 7/11 |
| plateau(w=4) | exact_ew | 19 | 425 | 2445 | 5 | 1.000 | 0.000 | 1.000 | 9/11 |
| plateau(w=4)+select | exact_ew | 7 | 11 | 2433 | 4 | 0.836 | 0.000 | 0.800 | 8/11 |
| plateau(w=4)+select | exact_ew | 11 | 11 | 2425 | 4 | 0.836 | 0.000 | 0.800 | 7/11 |
| plateau(w=4)+select | exact_ew | 19 | 11 | 2445 | 4 | 0.918 | 0.000 | 0.800 | 9/11 |
| thrml | thrml_potts | 7 | 1584 | 1584 | 5 | 1.000 | 0.000 | 1.000 | 11/11 |
| thrml | thrml_potts | 11 | 1584 | 1584 | 5 | 1.000 | 0.000 | 1.000 | 11/11 |
| thrml | thrml_potts | 19 | 1584 | 1584 | 5 | 1.000 | 0.000 | 1.000 | 11/11 |
| thrml+select | thrml_potts | 7 | 11 | 1584 | 2 | 0.344 | 0.000 | 0.400 | 11/11 |
| thrml+select | thrml_potts | 11 | 11 | 1584 | 4 | 0.918 | 0.000 | 0.800 | 11/11 |
| thrml+select | thrml_potts | 19 | 11 | 1584 | 3 | 0.836 | 0.000 | 0.600 | 11/11 |

Means over seeds 7, 11, 19:

| variant | recorded rows | energy evals | HV ratio | GD | coverage | argmin hits |
|---|---|---|---|---|---|---|
| baseline | 1584.0 | 4763.0 | 1.000 | 0.000 | 1.000 | 33/33 |
| select | 11.0 | 4763.0 | 0.536 | 0.000 | 0.533 | 33/33 |
| plateau(w=8) | 784.7 | 3164.3 | 1.000 | 0.000 | 1.000 | 30/33 |
| plateau(w=8)+select | 11.0 | 3164.3 | 0.891 | 0.000 | 0.733 | 30/33 |
| plateau(w=4) | 419.7 | 2434.3 | 1.000 | 0.000 | 1.000 | 24/33 |
| plateau(w=4)+select | 11.0 | 2434.3 | 0.863 | 0.000 | 0.800 | 24/33 |
| thrml | 1584.0 | 1584.0 | 1.000 | 0.000 | 1.000 | 33/33 |
| thrml+select | 11.0 | 1584.0 | 0.699 | 0.000 | 0.600 | 33/33 |

THRML plateau rule scored after the fact (window 8, tol 0; no THRML work saved): seed 7: 837 of 1584 rows, seed 11: 801 of 1584 rows, seed 19: 758 of 1584 rows.


## Reproduce

```bash
export JAX_PLATFORMS=cpu
python benchmarks/run_ebr_spike.py          # about 9 minutes on this box, nearly all THRML
python benchmarks/ebr_spike_report.py
```

The ExactEw rows are deterministic for a given seed. THRML wall time was 71–91 s per task-seed. ExactEw took under 0.2 s.
