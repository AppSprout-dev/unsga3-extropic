# unsga3-extropic

**Center:** the **U-NSGA-III** many-objective algorithm, mapped onto an Extropic-compatible substrate (THRML energy sampling / exact E_w Metropolis).

Bend and C# U-NSGA-III exist as **fidelity references** only. They do not own the idea. ZDT1 MH fidelity toys already live under `../toys/` — this package does not redo Bend.

## Algorithm → Extropic mapping

| U-NSGA-III intent | This package |
|-------------------|--------------|
| Vector fitness \(f_1,\ldots,f_M\) | Problem objectives (Ising terms or general `objective_fn`) |
| Preference / niches | Weight / reference-direction sweep (`weights.simplex_weights`, custom `w`) |
| Variation / search | Annealed sampling under \(E_w=\sum_i w_i f_i\) |
| Ranking / survival | Offline non-dominated archive (`archive.Archive`) |

Outer loop shape: **`WeightSweepLoop`** — for each weight vector, sample → decode objectives → accumulate → filter ND.

## Layout

```
unsga3-extropic/
  pyproject.toml          # depends on thrml; jax is peer
  README.md
  src/unsga3_extropic/
    archive.py            # ND archive, HV-2D helper
    weights.py            # simplex / Das–Dennis directions
    loop.py               # WeightSweepLoop (U-NSGA-III-shaped)
    backends/
      thrml_ising.py      # REAL: THRML IsingEBM block Gibbs + anneal
      ew_metropolis.py    # REAL: exact Ew Metropolis for general f
    problems/
      codon_ising.py      # REAL: THRML-native multi-term Ising (codon-style)
  demos/run_codon_thrml.py
  docs/fidelity_hooks.md  # pointers to ../toys ZDT1 MH / Bend-C# compare
  tests/
```

### Stub vs real

| Piece | Status |
|-------|--------|
| Package layout + `pyproject.toml` | **real** |
| `WeightSweepLoop` / archive / weights | **real** |
| `ThrmlIsingBackend` | **real** (THRML 0.1.4 API) |
| `ExactEwMetropolisBackend` | **real** (NumPy MH; wire any `objective_fn`) |
| `CodonIsingProblem` + demo | **real** THRML-native smoke |
| Potts / `CategoricalNode` backend | **stub / not yet** (Ising path first; Potts via THRML categorical factors later) |
| Full classical U-NSGA-III (NSGA-III niching inside population) | **not here** — thermo mapping uses weight niches + offline ND |
| Bend / ZDT1 reimplementation | **out of scope** (see `docs/fidelity_hooks.md`) |
| GitHub remote | **not created** — local scaffold; push instructions below |

## Environment

Use the existing workspace venv (already has `thrml` + `jax`):

```bash
VENV=/workspace/extropic-first-job/.venv
export PYTHONPATH=/workspace/extropic-first-job/unsga3-extropic/src

# unit tests (no sampling)
$VENV/bin/python -m pytest /workspace/extropic-first-job/unsga3-extropic/tests -q

# THRML codon demo (smoke)
$VENV/bin/python /workspace/extropic-first-job/unsga3-extropic/demos/run_codon_thrml.py
```

Optional editable install into that venv:

```bash
$VENV/bin/python -m pip install -e /workspace/extropic-first-job/unsga3-extropic
```

**Peer dependency:** install a JAX build that matches your platform (`jax` / `jax[cpu]` / CUDA). THRML does not pin JAX in its own deps.

## Backends

### (a) THRML — Ising/Potts-expressible energies

`ThrmlIsingBackend`: problem supplies `build_ising(w) -> (biases, edges, J)` so \(E_w\) stays on the Ising substrate. Native path for codon-style multi-term energies.

### (b) Exact E_w Metropolis — general objectives

`ExactEwMetropolisBackend`: bit-flip MH with exact \(E_w=w\cdot f(x)\). Use for nonlinear toys (ZDT1, etc.). Fidelity path against Bend/C# — see `docs/fidelity_hooks.md` and:

- `/workspace/extropic-first-job/toys/zdt1_true_ew/`
- `/workspace/extropic-first-job/toys/zdt1_thermo_unsga3.py`

## Push to GitHub (optional)

No remote was created from this scaffold. To publish under **AppSprout-dev**:

```bash
cd /workspace/extropic-first-job/unsga3-extropic
git init
git add .
git commit -m "Scaffold unsga3-extropic: U-NSGA-III on THRML substrate"
# create empty repo AppSprout-dev/unsga3-extropic on GitHub, then:
git branch -M main
git remote add origin git@github.com:AppSprout-dev/unsga3-extropic.git
git push -u origin main
```

## License

Apache-2.0 (aligned with THRML / Extropic public stack).
