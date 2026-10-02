# Fidelity hooks

This package centers a weight-sweep archive on two samplers:

- `ThrmlIsingBackend` when every scalarized energy is pairwise Ising (see `CodonIsingProblem`).
- `ExactEwMetropolisBackend` when the objective is a general function of the decision vector. That backend is NumPy Metropolis with exact \(E_w = w \cdot f(x)\). It does not call THRML.

```text
weight niches          WeightSweepLoop + simplex_weights / custom w
Ising-native search    ThrmlIsingBackend + problems/codon_ising.py
general objectives     ExactEwMetropolisBackend
```

`demos/run_codon_thrml.py` checks the Ising path: both terms are exactly the THRML energy, so there is no surrogate between \(f\) and the sampler.

## External classical references

Bend, C#, and any ZDT1 harness live outside this repository. They are not imported and not required for CI. When you have one:

1. Encode the same discrete decisions.
2. Run `ExactEwMetropolisBackend` with a comparable evaluation budget.
3. Compare the non-dominated archive (hypervolume, generational distance, coverage) offline.
4. A THRML run of a non-Ising objective needs an Ising (or Potts) expression of \(E_w\) first. This repo does not fit that surrogate.

Potts / `CategoricalNode`, Thermalizers, and Torx are not wired up here. THRML's own codon-optimization walkthrough is documented at [docs.thrml.ai](https://docs.thrml.ai).
