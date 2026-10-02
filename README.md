# unsga3-extropic

Weight-sweep multiobjective search on **THRML** Ising and Potts models, plus a NumPy exact-\(E_w\) Metropolis fallback. The Potts chain can also be sampled as a domain-wall Ising model inside THRML. That image is not a hardware run. An optional extra also runs one Torx circuit. That circuit is not the search step.

The outer loop samples one reference direction at a time, pools the candidates, and ranks them. Survival keeps the closer occupant of each direction: ideal–nadir normalization, then perpendicular distance. Candidates still come only from that sampling step. Recombination and mutation are not part of this package. The phased plan, drift guards, and definition of done are in [ROADMAP.md](ROADMAP.md).

## Extropic alignment

[THRML](https://docs.thrml.ai) is Extropic's JAX library for block Gibbs sampling of energy-based models — the same factor-graph structure their hardware is built to accelerate. This repository uses that library for Ising problems, for a categorical Potts chain, and for a domain-wall Ising image of that chain. It also runs an optional Torx circuit when the `torx` extra is installed. That circuit is not the search step and is not a hardware runner. This repository does not ship Thermalizers.

| Piece | Role |
|-------|------|
| `ThrmlIsingBackend` | **THRML-native.** Block Gibbs on `IsingEBM` with the current public API: `SpinNode`, `Block`, `IsingEBM`, `IsingSamplingProgram`, `SamplingSchedule`, `sample_states`, `hinton_init`, and `jax.random.key`. Checked against THRML 0.1.4 and the [docs quickstart](https://docs.thrml.ai). Even/odd blocks match that two-color chain. Bool states follow THRML (`True → +1`, `False → -1`). Energy is \(\mathcal{E}(s)=-\beta(b\cdot s+\sum J_{ij}s_i s_j)\). `SamplingSchedule` has no beta; annealing rebuilds `IsingEBM` per temperature and carries the free-block state. |
| `CodonIsingProblem` | **THRML-native energies** on a nearest-neighbor spin chain: a unary field and a pairwise term. Scalarization stays inside `IsingEBM`. This is a small in-repo problem, not the codon-optimization walkthrough. |
| `ThrmlPottsBackend` | **THRML-native.** Block Gibbs on `CategoricalNode` with `CategoricalEBMFactor`, `CategoricalGibbsConditional`, `FactorSamplingProgram`, `FactorizedEBM`, `BlockGibbsSpec`, `SamplingSchedule`, and `sample_states` ([example 00](https://docs.thrml.ai/en/latest/00_probabilistic_computing.html), [discrete EBM](https://docs.thrml.ai/en/latest/api-discrete-ebm.html), [EBM](https://docs.thrml.ai/en/latest/api-ebm.html)). Checked against THRML 0.1.4. `CategoricalNode()` takes no \(K\); \(K\) is `CategoricalGibbsConditional(n_categories)`. States are integers in \([0, K)\). Factor energy is \(-\sum W[\mathrm{state}]\). `SamplingSchedule` has no beta; annealing rescales factor weights (example 03) and carries the free-block state. Free blocks are an explicit coloring (default even/odd). An edge inside one block is rejected before sampling. |
| `PottsChainProblem` | **THRML-native energies** on a short categorical chain: one unary cost and one nearest-neighbor Potts cost. Those two costs are the archived objectives, and \(w\cdot f\) is the `FactorizedEBM` energy. Not the codon walkthrough. |
| `ThrmlDomainWallBackend` | **THRML simulation of the p-bit image** of `PottsChainProblem`. Domain-wall (thermometer) encoding from [example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html): \(K\) categories become \(K-1\) spins, Potts weights become Ising biases and couplings by the documented first and second differences, and a ferromagnetic penalty \(P\) prices broken thermometers. A \(K=2\) chain is 2-colored and is sampled with `ThrmlIsingBackend`. A \(K\ge 3\) chain is not; it uses `SpinEBMFactor`, `SpinGibbsConditional`, and `FactorSamplingProgram` with the example's `(site parity, spin-index parity)` blocks. Decoded categorical states are the archived objectives. Patterns that are not thermometers are counted and left out of that archive. This is not a Z1 run and not a copy of [codon_opt](https://github.com/extropic-ai/codon_opt). |
| `ExactEwMetropolisBackend` | **NumPy fallback.** Bit-flip Metropolis on exact \(E_w=w\cdot f(x)\) when \(f\) is not an Ising or Potts factor energy. No THRML program and no Extropic device. |
| `TorxPswapCircuit` | **Torx, optional, default off.** Two-pbit `PSWAP` on `DiscretePCircuit`, sampled with `BranchingSimulator`, as in the [Torx quickstart](https://docs.torx.ai/en/latest/) (`extro-torx` 0.0.2, Python ≥ 3.11). `theta` is the logit of the swap probability. Not a `SamplingBackend`, not a THRML program, and not a hardware runner. |
| Not in this repo | Thermalizers and device execution. The domain-wall path above is a THRML simulation of the in-repo Potts chain. |

Source for the library: [extropic-ai/thrml](https://github.com/extropic-ai/thrml). Pinned floor: THRML 0.1.4.

## Algorithm mapping

| U-NSGA-III intent | This package |
|-------------------|--------------|
| Vector fitness \(f_1,\ldots,f_M\) | Problem objectives |
| Preference / niches | The sweep's weight vectors (`weights.simplex_weights` or a custom `w`). Those same vectors are the reference directions |
| Variation / search | Annealed sampling under \(E_w=\sum_i w_i f_i\) via `sample_weight` only. No crossover or mutation |
| Ranking | Non-dominated archive (`Archive.nondominated`) |
| Niche survival | `Archive.niche_survival` / `LoopResult.niche_front`: pooled non-dominated rows are ideal–nadir normalized and associated to the sweep directions by perpendicular distance. Each direction keeps its closer occupant. A lone occupant of an empty direction is kept over a duplicate in a crowded direction. Dominated rows stay out. Both sets are returned |

`WeightSweepLoop` runs each weight and samples. `nondominated_front` is the non-dominated archive. `niche_front` is the closer-in-niche survivor set on that archive. The directions used for association are the weights the loop just sampled; a caller-supplied matrix is not replaced by a new simplex draw.

## Layout

```
pyproject.toml
README.md
ROADMAP.md
src/unsga3_extropic/
  archive.py            # ND archive, ideal–nadir niching, 2-D hypervolume helper
  fidelity.py           # generational distance, coverage, front loader
  results.py            # JSONL benchmark run records
  weights.py            # simplex / Das–Dennis directions
  loop.py               # WeightSweepLoop
  backends/
    thrml_ising.py      # THRML IsingEBM block Gibbs + anneal
    thrml_potts.py      # THRML CategoricalEBMFactor block Gibbs + weight rescale
    thrml_domain_wall.py # Potts chain as domain-wall Ising (THRML simulation)
    ew_metropolis.py    # exact E_w Metropolis (NumPy)
  domain_wall.py        # thermometer compile / decode (no JAX)
  torx_circuit.py       # optional Torx PSWAP circuit (not the search loop)
  problems/
    codon_ising.py      # two-term Ising chain
    potts_chain.py      # two-term Potts chain
benchmarks/             # run-record schema and append CLI
demos/run_codon_thrml.py
demos/run_potts_thrml.py
demos/run_domain_wall.py
demos/run_torx_pswap.py # optional; needs the torx extra
docs/fidelity_hooks.md
tests/
.github/workflows/ci.yml
```

| Piece | Status |
|-------|--------|
| `WeightSweepLoop`, archive, weights | implemented |
| `ThrmlIsingBackend` | implemented (THRML 0.1.4 API) |
| `ThrmlPottsBackend` + `PottsChainProblem` | implemented (THRML 0.1.4 categorical API) |
| Domain-wall Ising image of that Potts chain | THRML simulation (`ThrmlDomainWallBackend`, issue #10). Not Z1, not `codon_opt` |
| `ExactEwMetropolisBackend` | implemented (NumPy, not THRML) |
| `CodonIsingProblem` + demo | implemented, THRML Ising smoke |
| Front harness (HV, GD, coverage, `.npy`/`.npz`) | implemented |
| Benchmark JSONL records | implemented (`benchmarks/`) |
| Torx `PSWAP` circuit | optional extra `torx` (`extro-torx`, Python ≥ 3.11), default off. Not the search loop |
| Thermalizers / hardware | not implemented |
| Reference-direction niching | implemented (ideal–nadir normalization, perpendicular association, closer occupant). No crossover or mutation |
| CI | GitHub Actions: core tests without the torx extra; a separate job runs the Torx smoke |

## Install and test

Python 3.10+ for the core package. THRML does not pin JAX; install a build for your platform. The `torx` extra needs Python 3.11+ because that is what `extro-torx` requires. CPU:

```bash
pip install -e ".[dev,cpu]"
pytest -q -m 'not torx'
python demos/run_codon_thrml.py --smoke
python demos/run_potts_thrml.py --smoke
python demos/run_domain_wall.py --smoke
```

`pytest` skips tests marked `thrml` when JAX or THRML is missing (`pytest -m 'not thrml and not torx'` runs only the NumPy tests). Tests marked `torx` skip when `extro-torx` is missing. CI's core job does not install that extra.

Optional Torx circuit (Python ≥ 3.11):

```bash
pip install -e ".[dev,cpu,torx]"
pytest -q -m torx
python demos/run_torx_pswap.py
```

Default demos (no `--smoke`):

```bash
python demos/run_codon_thrml.py
python demos/run_potts_thrml.py
python demos/run_domain_wall.py
```

A measured Potts run can also be appended to the benchmark log:

```bash
python benchmarks/append_run.py --smoke --out benchmarks/records/runs.jsonl
```

See [benchmarks/README.md](benchmarks/README.md) for the record schema.

## Backends

### THRML Ising

`ThrmlIsingBackend` asks the problem for `build_ising(w) -> (biases, edges, J)` so \(E_w\) stays on the Ising substrate. Edges must connect an even index to an odd index, because free blocks are `Block(nodes[::2])` and `Block(nodes[1::2])`. A path meets that rule. An edge inside one parity is rejected before sampling.

### THRML Potts

`ThrmlPottsBackend` asks the problem for `build_potts(w) -> (unary W, edges, pairwise W)`. Those tensors are THRML weights: `FactorizedEBM` energy is \(-\sum W[\mathrm{state}]\), so a cost table is stored with a minus sign (see `PottsChainProblem.build_potts`). Annealing multiplies both tensors by beta and rebuilds `FactorSamplingProgram`. The schedule object itself has no temperature field.

Pass an explicit `coloring` (a partition of the sites into independent sets), or leave it as the even/odd chain coloring. An edge with both ends in one block raises `ValueError` before JAX is imported.

### Domain-wall image of the Potts chain

`PottsChainProblem.make_domain_wall_backend` compiles that same Potts energy into Ising spins (example 03). Feasible samples are thermometers. Their decoded categories carry the Potts objectives, so the archive is still \((f_\mathrm{unary}, f_\mathrm{pairwise})\). A row that is not a thermometer increments `n_invalid` and is not inserted. `ThrmlIsingBackend` still rejects a same-parity edge; the \(K\ge 3\) image does not use that backend, because the spin graph is not 2-colored. Nothing in this path submits a program to hardware.

### Exact \(E_w\) Metropolis

`ExactEwMetropolisBackend` evaluates \(E_w=w\cdot f(x)\) directly in NumPy. Wire any `objective_fn`. See `docs/fidelity_hooks.md` for how that archive, or a THRML front, can be compared with an external classical reference. The comparison functions do not call Bend or C#.

### Torx PSWAP (optional)

`TorxPswapCircuit` in `unsga3_extropic.torx_circuit` builds the [Torx docs](https://docs.torx.ai/en/latest/) quickstart: `DiscretePCircuit([PSWAP([0, 1])])`, a logit parameter `log(p / (1 - p))` with `p = 0.3`, and `BranchingSimulator`. Sampling starts at `|10)`. The smoke checks the empirical stay and swap rates against `1 - p` and `p` at 20,000 samples. `WeightSweepLoop` does not call this circuit. Nothing here turns the circuit into a THRML energy model.

## License

Apache-2.0 (aligned with the THRML / Extropic public stack).
