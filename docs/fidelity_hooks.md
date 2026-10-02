# Fidelity hooks (Bend / C# / ZDT1 toys)

**Package center:** U-NSGA-III *algorithm intent* on Extropic substrate.  
Bend and C# U-NSGA-III are **reference implementations** for numerical fidelity — they do not own the idea.

## Existing toys (do not reimplement here)

| Path | What it is | Backend |
|------|------------|---------|
| `/workspace/extropic-first-job/toys/zdt1_thermo_unsga3.py` | ZDT1 binary toy; pairwise **Ising surrogate** of E_w via least-squares; THRML anneal; ND archive; vs tiny NSGA-II | `ThrmlIsingBackend`-like (surrogate) |
| `/workspace/extropic-first-job/toys/zdt1_true_ew/` | ZDT1 with **exact** E_w = w·f; annealed Metropolis (no Ising surrogate) | matches `ExactEwMetropolisBackend` |
| `/workspace/extropic-first-job/toys/REPORT.md` | Numbers / HV / GD for surrogate toy | — |
| `/workspace/extropic-first-job/toys/zdt1_true_ew/REPORT.md` | Numbers for exact-Ew MH | — |

Bend ZDT1 fidelity work (if any) is handled **elsewhere** — this package must not redo Bend.

## How this package plugs in

```text
U-NSGA-III intent
    │
    ├─ niches / diversity  →  WeightSweepLoop + simplex_weights / custom w
    ├─ Ising/Potts-native  →  ThrmlIsingBackend  (+ problems/codon_ising)
    └─ general objectives  →  ExactEwMetropolisBackend
         └─ ZDT1 fidelity   →  wire objective_fn to toys' zdt1(); compare
                               archive HV/GD to Bend/C# baselines offline
```

### Suggested fidelity check (when Bend/C# artifacts exist)

1. Encode the same discrete decision space as the reference.
2. Run `ExactEwMetropolisBackend` (honest E_w) with a comparable eval budget.
3. Compare ND archive: hypervolume, generational distance, coverage.
4. Optionally run a THRML surrogate path and document the surrogate gap (as the toys already do).

### Codon demo vs ZDT1

The in-package demo (`demos/run_codon_thrml.py`) is **THRML-native**: energies are exactly Ising, so there is no surrogate gap. It validates the outer loop + THRML backend, not ZDT1 fidelity.

## Reproduce toys (workspace venv)

```bash
VENV=/workspace/extropic-first-job/.venv/bin/python

# Ising-surrogate ZDT1
$VENV /workspace/extropic-first-job/toys/zdt1_thermo_unsga3.py

# Exact Ew MH ZDT1
$VENV /workspace/extropic-first-job/toys/zdt1_true_ew/zdt1_true_ew.py
```
