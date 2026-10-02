# Fidelity hooks

This package centers a weight-sweep archive on three samplers:

- `ThrmlIsingBackend` when every scalarized energy is pairwise Ising (`CodonIsingProblem`).
- `ThrmlPottsBackend` when every scalarized energy is a categorical Potts factor model (`PottsChainProblem`: one unary term and one nearest-neighbor term on a chain). THRML-native. States are integers in `[0, K)`, and `K` is `CategoricalGibbsConditional.n_categories`.
- `ExactEwMetropolisBackend` when the objective is a general function of a bitstring. That backend is NumPy bit-flip Metropolis with exact \(E_w = w \cdot f(x)\). It does not call THRML.
- `ExactEwContinuousBackend` when the objective is a general function on a box (continuous ZDT1, ZDT2, DTLZ2). Same exact scalarization, NumPy only, truncated-normal steps of scale 0.1. Those problems are not Ising energies, and this package does not fit a surrogate into `IsingEBM`.

```text
weight niches          WeightSweepLoop + simplex_weights / custom w
niche survival         Archive.niche_survival (ideal–nadir, perpendicular distance, closer occupant)
Ising-native search    ThrmlIsingBackend + problems/codon_ising.py
Potts-native search    ThrmlPottsBackend + problems/potts_chain.py
Potts p-bit image      ThrmlDomainWallBackend (THRML simulation, not a device)
general objectives     ExactEwMetropolisBackend (bits) or ExactEwContinuousBackend (box)
front comparison       fidelity.py (hypervolume_2d, generational distance, coverage)
```

`demos/run_codon_thrml.py` and `demos/run_potts_thrml.py` check the Ising and categorical Potts paths. `demos/run_domain_wall.py` checks the domain-wall Ising image of the same Potts chain. On the categorical problem the archived objectives are the factor energies. On the domain-wall path they are those same Potts objectives after thermometer decoding. Invalid thermometers are counted and omitted, not entered as feasible rows. The image is a THRML simulation ([example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html)). It is not a Z1 run and it does not vendor [codon_opt](https://github.com/extropic-ai/codon_opt).

The harness does not retune betas to chase an external number. `benchmarks/run_deep.py` scores the same functions at the deep budget. That run is not a device measurement.

## In-repo metrics

`unsga3_extropic.fidelity` scores minimization arrays. It does not take a sampler.

| Function | Definition |
|----------|------------|
| `hypervolume_2d` (in `archive`) | 2-D hypervolume of the non-dominated rows that sit strictly below a reference point |
| `generational_distance` | \((\mathrm{mean}_i d_i^p)^{1/p}\), Euclidean distance from each obtained row to the nearest reference row. Default \(p = 2\) |
| `coverage(A, B)` | Fraction of rows of `B` weakly dominated by some row of `A`. For minimization, `a` weakly dominates `b` when `a <= b` on every objective |
| `load_front` | Read a `.npy` or `.npz` array of shape `(n, M)` |
| `compare_fronts` | `nd_count`, plus hypervolume (2-D), generational distance, and coverage when a reference front is passed |

`compare_fronts(obtained, reference)` sets `coverage` to the fraction of the **reference** front weakly dominated by the obtained front.

### Front file schema

- `.npy`: a numeric array, shape `(n, M)`, one objective vector per row. Minimization.
- `.npz`: the array named `front`, or the only array in the file, same shape.
- Any other suffix, rank, or objective count raises `ValueError`.

`benchmarks/README.md` describes the JSONL run record that points at these files. `eval_budget` in that record is the number of recorded samples; the anneal schedule is stored beside it.

## External classical references

Bend, C#, and any ZDT1 harness live outside this repository. They are not imported, not started as a subprocess, and not required for CI. A binary they produce is not a CI dependency. When you have a front from one of them:

1. Save it as a `.npy` or `.npz` file in the schema above. Encode the same discrete decisions the in-repo problem uses.
2. Load it with `load_front`. Score it against an in-repo front (`enumerate_front` for the Ising or Potts chain at a size the enumerator allows, or a `LoopResult` non-dominated front) using `compare_fronts`.
3. Do not retune betas inside this repo to match that file.

A THRML run of a non-factorized objective still needs an Ising or Potts expression of \(E_w\) first. This repo does not fit that surrogate. `ExactEwMetropolisBackend` remains the NumPy path for a general bitstring \(f\). `ExactEwContinuousBackend` is the NumPy path for a general box \(f\).

`fidelity.py` still does not start Bend, C#, or pymoo. The optional benchmark script `benchmarks/run_oracle_continuous.py` is separate from that harness. It shells out to `unsga3-bend/ab/igd_vs_pymoo.py` when that file is on disk, and it records only the printed `igd=` line. CI does not run it and does not install pymoo. The reference front for that script is the analytic 500-point ZDT curve, or the 91-point Das–Dennis DTLZ2 sphere at partitions 12. A missing script is an error, not a made-up IGD.

Thermalizers are not wired up here. The optional Torx circuit (`unsga3_extropic.torx_circuit`) samples a documented `PSWAP` and is not a search backend, so this harness does not score it.
