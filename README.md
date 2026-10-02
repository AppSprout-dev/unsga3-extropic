# unsga3-extropic

Weight-sweep multiobjective search on a **THRML** Ising model, plus a NumPy exact-\(E_w\) Metropolis fallback.

The outer loop follows U-NSGA-III *intent*: several objectives, diverse niches, search, then ranking. Classical U-NSGA-III (population niching, variation operators) is not reimplemented here. The phased plan, drift guards, and definition of done are in [ROADMAP.md](ROADMAP.md).

## Extropic alignment

[THRML](https://docs.thrml.ai) is Extropic's JAX library for block Gibbs sampling of energy-based models — the same factor-graph structure their hardware is built to accelerate. This repository uses that library for Ising problems. It does not ship Thermalizers, Torx, or a hardware runner.

| Piece | Role |
|-------|------|
| `ThrmlIsingBackend` | **THRML-native.** Block Gibbs on `IsingEBM` with the current public API: `SpinNode`, `Block`, `IsingEBM`, `IsingSamplingProgram`, `SamplingSchedule`, `sample_states`, `hinton_init`, and `jax.random.key`. Checked against THRML 0.1.4 and the [docs quickstart](https://docs.thrml.ai). Even/odd blocks match that two-color chain. Bool states follow THRML (`True → +1`, `False → -1`). Energy is \(\mathcal{E}(s)=-\beta(b\cdot s+\sum J_{ij}s_i s_j)\). `SamplingSchedule` has no beta; annealing rebuilds `IsingEBM` per temperature and carries the free-block state. |
| `CodonIsingProblem` | **THRML-native energies** on a nearest-neighbor chain: a unary field and a pairwise term. Scalarization stays inside `IsingEBM`. This is a small in-repo problem. The codon-optimization walkthrough (Potts model and an equivalent Ising model) is the one linked from [docs.thrml.ai](https://docs.thrml.ai). |
| `ExactEwMetropolisBackend` | **NumPy fallback.** Bit-flip Metropolis on exact \(E_w=w\cdot f(x)\) when \(f\) is not pairwise Ising. No THRML program and no Extropic device. |
| Not in this repo | Thermalizers, Torx, Potts / `CategoricalNode` sampling, and device execution. |

Source for the library: [extropic-ai/thrml](https://github.com/extropic-ai/thrml).

## Algorithm mapping

| U-NSGA-III intent | This package |
|-------------------|--------------|
| Vector fitness \(f_1,\ldots,f_M\) | Problem objectives |
| Preference / niches | Weight sweep (`weights.simplex_weights` or a custom `w`) |
| Variation / search | Annealed sampling under \(E_w=\sum_i w_i f_i\) |
| Ranking / survival | Offline non-dominated archive (`archive.Archive`) |

`WeightSweepLoop` runs each weight, samples, decodes objectives, and filters the non-dominated set.

## Layout

```
pyproject.toml
README.md
ROADMAP.md
src/unsga3_extropic/
  archive.py            # ND archive, 2-D hypervolume helper
  weights.py            # simplex / Das–Dennis directions
  loop.py               # WeightSweepLoop
  backends/
    thrml_ising.py      # THRML IsingEBM block Gibbs + anneal
    ew_metropolis.py    # exact E_w Metropolis (NumPy)
  problems/
    codon_ising.py      # two-term Ising chain
demos/run_codon_thrml.py
docs/fidelity_hooks.md
tests/
.github/workflows/ci.yml
```

| Piece | Status |
|-------|--------|
| `WeightSweepLoop`, archive, weights | implemented |
| `ThrmlIsingBackend` | implemented (THRML 0.1.4 API) |
| `ExactEwMetropolisBackend` | implemented (NumPy) |
| `CodonIsingProblem` + demo | implemented, THRML Ising smoke |
| Potts / `CategoricalNode` | not implemented |
| Thermalizers / Torx / hardware | not implemented |
| Full classical U-NSGA-III niching | not implemented (weight niches + offline ND) |
| CI | GitHub Actions: unit tests + `demos/run_codon_thrml.py --smoke` |

## Install and test

Python 3.10+. THRML does not pin JAX; install a build for your platform. CPU:

```bash
pip install -e ".[dev,cpu]"
pytest -q
python demos/run_codon_thrml.py --smoke
```

`pytest` skips tests marked `thrml` when JAX or THRML is missing (`pytest -m 'not thrml'` runs only the NumPy tests). CI installs `thrml` and `jax[cpu]` and runs the full suite plus the smoke demo.

Default demo schedule (no `--smoke`):

```bash
python demos/run_codon_thrml.py
```

## Backends

### THRML Ising

`ThrmlIsingBackend` asks the problem for `build_ising(w) -> (biases, edges, J)` so \(E_w\) stays on the Ising substrate. Edges must connect an even index to an odd index, because free blocks are `Block(nodes[::2])` and `Block(nodes[1::2])`. A path meets that rule. An edge inside one parity is rejected before sampling.

### Exact \(E_w\) Metropolis

`ExactEwMetropolisBackend` evaluates \(E_w=w\cdot f(x)\) directly. Wire any `objective_fn`. See `docs/fidelity_hooks.md` for how that archive can be compared with an external classical reference.

## License

Apache-2.0 (aligned with the THRML / Extropic public stack).
