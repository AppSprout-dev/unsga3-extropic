# ROADMAP

Phased plan for U-NSGA-III-shaped search on the public Extropic sampling stack. The work order is the algorithm: scalarized search, reference-direction diversity, and an honest energy identity between the objective and the sampler. Bend and C# stay outside this repository as fidelity references.

Checked **2026-10-02** against `main` (`7946bc1`, package `0.1.1`, [README](README.md)) and the public sources in [Sources](#sources). Re-check versions and URLs at implementation time. This file is the drift guard. A phase is done when its acceptance criteria hold in the tree, not when an issue is opened.

## Glossary

**THRML.** The Thermodynamic Hypergraphical Model Library, a JAX library for block Gibbs sampling of probabilistic graphical models. Install with `pip install thrml` (Python ≥ 3.10). PyPI version on this check: **0.1.4**. Nodes, blocks, factors, and sampling programs are the public surface. Docs: [docs.thrml.ai](https://docs.thrml.ai/en/latest/), [getting started](https://docs.thrml.ai/en/latest/getting-started.html). Code: [extropic-ai/thrml](https://github.com/extropic-ai/thrml).

**EBM.** An energy-based model: a distribution defined by a scalar energy of a state. In THRML that is [`AbstractEBM`](https://docs.thrml.ai/en/latest/api-ebm.html) / [`FactorizedEBM`](https://docs.thrml.ai/en/latest/api-ebm.html), with energy \(\mathcal{E}(x)=\sum_i \mathcal{E}^i(x)\) over factors. The Ising specialization used here today is `IsingEBM`, documented in the [getting started](https://docs.thrml.ai/en/latest/getting-started.html) chain and [Ising API](https://docs.thrml.ai/en/latest/api-ising.html): \(\mathcal{E}(s)=-\beta(b\cdot s+\sum J_{ij}s_i s_j)\) with bool states `True → +1`, `False → -1`.

**THM.** A thermodynamic hypergraphical model: the factor-graph object THRML samples (nodes, many-body factors, a graph coloring into blocks, block Gibbs). "THM" is that model class, named by the library title. There is no separate `THM` type in the public API. Hardware acceleration, where it exists, targets this block-Gibbs structure ([getting started](https://docs.thrml.ai/en/latest/getting-started.html), [example 00](https://docs.thrml.ai/en/latest/00_probabilistic_computing.html)).

**Potts / `CategoricalNode`.** A Potts model is an EBM on categorical variables. [`CategoricalNode`](https://docs.thrml.ai/en/latest/api-pgm.html) is a public node type: one of \(K\) states, an integer in \([0, K)\), default dtype `uint8`. The constructor takes no \(K\). \(K\) is an argument of [`CategoricalGibbsConditional(n_categories)`](https://docs.thrml.ai/en/latest/api-discrete-ebm.html). Interactions are [`CategoricalEBMFactor`](https://docs.thrml.ai/en/latest/api-discrete-ebm.html) (and `SquareCategoricalEBMFactor` when the weight tensor is square). A worked Potts model is [example 00](https://docs.thrml.ai/en/latest/00_probabilistic_computing.html). The codon tutorial builds the same factor on a chain, then optionally re-encodes it as Ising with domain-wall encoding: [example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html).

**Torx.** A separate JAX library of parametrised stochastic circuits (PSCs) and directed factor graphs (DFGs). Install name **`extro-torx`** (Python ≥ 3.11). PyPI version on this check: **0.0.2**. Docs: [docs.torx.ai](https://docs.torx.ai/en/latest/), [getting started](https://docs.torx.ai/en/latest/getting-started.html). Code: [extropic-ai/torx](https://github.com/extropic-ai/torx). Whitepaper: [arXiv:2608.01612](https://arxiv.org/abs/2608.01612). Torx is not THRML and does not compile programs onto THRML.

**pbit / pdit / pmode.** Torx site types, from the [Torx docs home](https://docs.torx.ai/en/latest/) and [example 01](https://docs.torx.ai/en/latest/01_introduction_to_parametrised_stochastic_circuits.html):

| Primitive | State | Torx role |
|-----------|--------|-----------|
| pbit | \(\{0,1\}\) | Binary site. Discrete gates such as `PSWAP`, `PNOT`, `PISING`. |
| pdit | \(\{0,\ldots,d-1\}\) | \(d\)-state site. Permute and cycle gates. |
| pmode | \(\mathbb{R}^N\) | Continuous site. Gaussian gates. |

The same family is how the codon tutorial talks about hardware primitives: a **p-bit** samples an Ising spin, a **p-dit** would sample a categorical Potts state directly ([example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html)). A fourth primitive, p-MoG (mixture of Gaussians), appears in Extropic's stack description and is not a target of this repository.

**TSU / Z1.** A thermodynamic sampling unit (TSU) is the chip that stores a sparse probabilistic model and runs Gibbs sampling in place. Extropic's public Z1 description ([From One to One Billion](https://extropic.ai/writing/from-one-to-one-billion/)): an Ising machine of **269,568 pbits**, each coupled to **16** neighbors, chromatic Gibbs, **>50 MHz**, **<1 W**. The codon tutorial states that near-term hardware is p-bit / Ising, and that native Potts execution wants future p-dit hardware. No public device SDK is linked from [docs.thrml.ai](https://docs.thrml.ai/en/latest/) or [docs.torx.ai](https://docs.torx.ai/en/latest/).

**Thermalizers.** A planned compiler from Torx programs to thermodynamic hardware, described only in the paper [Thermalizing Stochastic Programs, arXiv:2608.01615](https://arxiv.org/abs/2608.01615) and the [Extropic post](https://extropic.ai/writing/from-one-to-one-billion/) ("watch our GitHub for the open-source release"). On 2026-10-02 the `extropic-ai` GitHub org lists `thrml`, `torx`, `codon_opt`, `thrml-skill`, and `sparse-transformers`. There is no `thermalizers` repository and no PyPI package under that name (also checked: `extro-thermalizers`, `thermalizer`). The paper's intent, quoted at the level of the abstract only: take a Torx directed factor graph / parametrised stochastic circuit and replace factors with energy-based kernels sampled with THRML. Context matching and trajectory-level REINFORCE are paper methods. They are not functions, flags, or modules in any library this repo can import. Do not invent call signatures for them.

**Weight-sweep archive vs classical U-NSGA-III.** Classical U-NSGA-III (Seada and Deb, [U-NSGA-III](https://www.egr.msu.edu/~kdeb/papers/c2014022.pdf)) is a generational evolutionary algorithm:

- a population of size \(N \ge H\), with \(H\) reference directions on the simplex (Das–Dennis);
- non-dominated ranking;
- association of points to directions by perpendicular distance in normalized objective space;
- niching that keeps diversity across directions;
- the unified selection rule: when two candidates share a niche, prefer the one **closer** to that reference direction;
- variation by recombination and mutation.

This repository's product is a **weight-sweep archive** with reference-direction niching on the pooled sample. `weights.simplex_weights` builds the directions. `WeightSweepLoop` runs each direction as its own annealed sampler under \(E_w=\sum_i w_i f_i\). `archive.Archive` keeps the union. `Archive.nondominated` is the non-dominated filter. `Archive.niche_survival` ideal–nadir normalizes those non-dominated rows, associates each to a sweep direction by perpendicular distance, and keeps the closer occupant of each direction. Dominated rows do not re-enter. The loop does not recombine or mutate a population. `ThrmlIsingBackend` and `ThrmlPottsBackend` are the Extropic-shaped search steps. `ExactEwMetropolisBackend` is a NumPy stand-in for the same outer loop when \(E_w\) is not an Ising or Potts energy ([docs/fidelity_hooks.md](docs/fidelity_hooks.md)).

`unsga3_extropic.classical` is a **separate fidelity column**, not that product. It is a NumPy generational U-NSGA-III (population, Das–Dennis directions, non-dominated ranking, perpendicular association, closer-in-niche tournament, SBX, polynomial mutation) so ZDT1, ZDT2, and DTLZ2 IGD can be compared with the published Bend/C# PymooCompatible cells. It does not sample THRML, Ising, or Potts, and `WeightSweepLoop` does not call it. Numbers live in [benchmarks/ORACLE_UNSGA3_RESULTS.md](benchmarks/ORACLE_UNSGA3_RESULTS.md). The ExactEw weight-sweep numbers stay in [benchmarks/ORACLE_RESULTS.md](benchmarks/ORACLE_RESULTS.md).

## Current state vs target

State was checked at `7946bc1` / v0.1.1, then updated as phases landed. Phase 1, phase 2, phase 3, and phase 4 boxes below are checked against this tree (package 0.7.0). Appendix item 7, the domain-wall image, is also in this tree as a THRML simulation. Continuous ZDT1/ZDT2/DTLZ2 have two NumPy columns, neither of them THRML: the exact-\(E_w\) weight-sweep, and a classical generational U-NSGA-III fidelity column beside it.

| Piece | Now | Target |
|-------|-----|--------|
| Outer loop | `WeightSweepLoop`: one independent anneal per weight (`sample_weight` only), then ideal–nadir association and closer-in-niche survival on the pooled non-dominated rows. `Archive.nondominated` remains beside that set | Same sampler contract, plus reference-direction association and closer-in-niche survival on the **pooled** candidates |
| Directions | `simplex_weights`: uniform 2-objective grid; Das–Dennis for \(M>2\), with Dirichlet fill if the grid is short | Keep this generator as \(H\). Niching consumes it. Do not fork a second, undocumented direction set |
| Ising search | `ThrmlIsingBackend` on THRML 0.1.4: `SpinNode`, `Block`, `IsingEBM`, `IsingSamplingProgram`, `SamplingSchedule`, `sample_states`, `hinton_init`. Even/odd blocks only. `SamplingSchedule` has no beta, so each temperature rebuilds `IsingEBM`. The Potts domain-wall image uses this backend only when its spin graph is 2-colored; otherwise it uses a separate 4-color `SpinEBMFactor` program | Remains the p-bit / Ising path. Coloring stays explicit. Same-parity edges stay rejected on `ThrmlIsingBackend` |
| In-repo problem | `CodonIsingProblem` (Ising chain) and `PottsChainProblem` (categorical chain). Neither is the codon walkthrough. The Potts chain has a domain-wall Ising image in `domain_wall.py`; that image is not a copy of codon_opt | Add a small Potts problem whose factors **are** the categorical energy. Leave the full spike-protein study in [extropic-ai/codon_opt](https://github.com/extropic-ai/codon_opt) |
| Categorical / Potts | `ThrmlPottsBackend` + `PottsChainProblem`: public `CategoricalNode`, `CategoricalEBMFactor`, `CategoricalGibbsConditional`, `FactorSamplingProgram`, `FactorizedEBM`. Even/odd coloring. Anneal rescales weights. The same energy also has a domain-wall Ising image (`ThrmlDomainWallBackend`), still sampled in THRML | `CategoricalNode` + `CategoricalEBMFactor` + `CategoricalGibbsConditional` on a 2-colored chain, public API only |
| General objectives | `ExactEwMetropolisBackend`: bit-flip Metropolis on exact \(E_w=w\cdot f(x)\). `ExactEwContinuousBackend`: the same scalarization on a box (continuous ZDT1/ZDT2/DTLZ2). NumPy. No THRML program | Stays the non-ecosystem fallback and a fidelity workhorse. A THRML label requires an EBM whose energy matches \(f\). Continuous ZDT/DTLZ stay on the NumPy label |
| Fidelity | `fidelity.py`: `hypervolume_2d`, generational distance, inverted generational distance (the yardstick `igd=` mean), coverage, `.npy`/`.npz` loader. Bend and C# are not imported. The classical continuous column reimplements their operators in NumPy and writes `benchmarks/ORACLE_UNSGA3_RESULTS.md` | An in-repo harness over array fronts. External oracles remain published numbers. The classical column is that comparison, not a new product phase |
| Torx | Optional extra `torx` (`extro-torx` 0.0.2, Python ≥ 3.11): `TorxPswapCircuit` samples the docs quickstart (`DiscretePCircuit`, `PSWAP`, `BranchingSimulator`). Default off. Not a search backend and not a THRML program | Optional extra only, default off, Python ≥ 3.11, package `extro-torx` |
| Thermalizers | Not imported. No public library | Watch. Zero code until a public package and docs exist |
| Hardware | Not imported. No device runner | After the algorithm phases, and only against a public device API. Z1 facts above are citations, not a backend |

`CodonIsingProblem.enumerate_front` (exact front for \(n \le 16\)) and the CI smoke (`demos/run_codon_thrml.py --smoke`, THRML tests skipped when JAX/THRML are absent) are the regression floor. New phases keep that floor. A larger multi-seed budget lives in `benchmarks/run_deep.py` and `.github/workflows/deep.yml` (`workflow_dispatch` only). Those rows are THRML simulations, plus a NumPy exact-\(E_w\) comparator. They are not Z1 measurements and they do not call Thermalizers.

## Drift guards / NON-goals

These are permanent, including inside a phase that sounds adjacent.

1. **Algorithm first.** The product is multiobjective search whose variation step is sampling under a scalarized energy. A language port of a classical generational GA is not a phase of that product. `unsga3_extropic.classical` is a labeled fidelity measurement beside the product, not a replacement of the weight-sweep.
2. **Bend and C# are fidelity references only.** They are not imported, not vendored, not built in CI, and not the spec. A front they emit may be checked in later as numbers. Their operators (SBX, polynomial mutation, generational survivor loops) are not ported into `WeightSweepLoop`.

   **Exception, this column only.** `unsga3_extropic.classical` may implement those operators in NumPy so continuous ZDT1/ZDT2/DTLZ2 IGD can be compared with the published Bend/C# cells. The exception does not extend to `WeightSweepLoop`, `ExactEwContinuousBackend`, or any THRML backend. That module must not be described as Extropic sampling, Ising, Potts, or a device run.
3. **Do not claim a library the import graph does not import.** The core search import is THRML. `ExactEwMetropolisBackend` stays described as NumPy. Torx is imported only inside `TorxPswapCircuit.sample`, and only when the `torx` extra is installed.
4. **Do not invent a Thermalizers API.** No module, class, or function is added because the paper sketches a compiler. No pseudocode from arXiv:2608.01615 is transcribed into `src/`. Context matching and trajectory-level REINFORCE are not implemented from the paper.
5. **Do not call Torx a compiler onto THRML.** The optional Torx path is a stochastic-circuit sampler (`TorxPswapCircuit`). It is not Thermalizers and not a TSU.
6. **Do not treat `CodonIsingProblem` as example 03.** The docs codon model is a \(K\)-state Potts model and, separately, a domain-wall Ising model ([example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html)). This repo's chain is a two-term Ising smoke. Renaming it does not make it that tutorial.
7. **Energy identity.** A backend marked THRML-native samples an EBM whose energy is the scalarization of the archived objectives. Host-side \(E_w=w\cdot f(x)\) for a non-factorized \(f\) stays on `ExactEwMetropolisBackend`. No learned surrogate is introduced to force a non-Ising \(f\) into `IsingEBM`.
8. **Coloring is part of the sampler.** Even/odd blocks are valid for a path. They are not a general graph colorer. An edge inside one block is a bug, not a silent recolor.
9. **Hardware claims follow a public API.** Z1's published size, degree, and Ising/p-bit nature may be quoted with the blog URL. This repo does not flash weights, estimate joules, or speak for a cloud simulator unless a documented public endpoint is actually called.
10. **p-mode / p-MoG / continuous EBMs** are out of scope until a concrete objective needs them. The open gap is categorical Potts, which the docs already sample.
11. **Reproducing [extropic-ai/codon_opt](https://github.com/extropic-ai/codon_opt)** (SARS-CoV-2 spike, domain-wall compilation at protein scale) is not required for any phase below. Link it. Do not absorb it.

## Phased work

Phases 1–3 are the algorithm. Phase 2 can start on NumPy archives before phase 1 merges. Phase 4 is optional and after phase 1's backend protocol is stable. Phase 5 is a watch, not a coding phase. Phase 6 waits on phases 1–3 and on a public device API.

### Phase 1 — Potts / categorical THRML backend

**Outcome.** `WeightSweepLoop` can sample a categorical Potts energy through public THRML, with the same result type as `ThrmlIsingBackend`.

**Public symbols to use, and no others for the sampler core.**

- [`CategoricalNode`](https://docs.thrml.ai/en/latest/api-pgm.html), [`Block`](https://docs.thrml.ai/en/latest/getting-started.html)
- [`CategoricalEBMFactor`](https://docs.thrml.ai/en/latest/api-discrete-ebm.html), [`CategoricalGibbsConditional`](https://docs.thrml.ai/en/latest/api-discrete-ebm.html)
- [`FactorSamplingProgram`](https://docs.thrml.ai/en/latest/api-factors.html), which [example 00](https://docs.thrml.ai/en/latest/00_probabilistic_computing.html) uses to sample a Potts model. Energy checks can call the factor `energy` methods or [`FactorizedEBM.energy`](https://docs.thrml.ai/en/latest/api-ebm.html) on the same factors
- `SamplingSchedule`, `sample_states`

`CategoricalNode()` carries no category count. `CategoricalGibbsConditional(n_categories)` does. States are integers in \([0, K)\). Unary bias is a factor on one block; a pairwise Potts term is a factor on two blocks, matching the shape rules on the discrete-EBM page (`weights` has leading batch dimension and one axis per categorical group).

`SamplingSchedule` still has no temperature. Annealing scales the factor weights by \(\beta\) and rebuilds the program, which is what [example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html) does for Potts and what `ThrmlIsingBackend` already does for Ising. Carry the free-block state across temperatures.

**Acceptance criteria.**

- [x] A backend, named in the spirit of `ThrmlPottsBackend`, implements `SamplingBackend.sample_weight` and returns `BackendResult`.
- [x] The in-repo problem is a short chain (or 2-colorable graph) of categorical variables with a unary term and a pairwise term. Archived objectives equal those two energies. The scalarization passed to THRML is exactly \(w\cdot f\), checked by evaluating `FactorizedEBM.energy` (or the factor `energy` methods) on the samples and comparing to \(w\cdot f\) within a tight tolerance.
- [x] Neighbors do not share a block. A same-block edge raises before any sample. The test uses a chain, which is 2-colorable, as in example 00 and example 03.
- [x] Decisions lie in \(\{0,\ldots,K-1\}\). \(K\) is the conditional's `n_categories`, not a field invented on the node.
- [x] A unit test with THRML installed runs a tiny schedule (`n_samples` small, few betas). CI keeps the existing skip when `thrml` or `jax` is missing.
- [x] `pytest -m 'not thrml'` still passes without JAX.
- [x] README status row and [docs/fidelity_hooks.md](docs/fidelity_hooks.md) name the new backend as THRML-native. Phase 1 left domain-wall Ising unclaimed.
- [x] Domain-wall encoding was not part of phase 1. Appendix item 7 (issue #10) is that later p-bit compilation, and it is a THRML simulation rather than a device run.

### Phase 2 — Closer reference-direction niching

**Outcome.** Survival on the pooled sample uses the U-NSGA-III niching rule: associate each point with a reference direction, and inside a niche prefer the point closer to that direction. Search stays annealed sampling.

**What landed.** `simplex_weights` still builds \(H\) directions, and a caller-supplied matrix is used unchanged. Those vectors scalarize each anneal and are the reference rays for association. `Archive.niche_survival` drops dominated rows, ideal–nadir normalizes the remaining pooled rows (`normalize_objectives`: coordinate-wise min as ideal, max as nadir, constant objectives left unscaled), and associates each row by perpendicular distance. Niche survival then:

- keeps a non-dominated point that is the only occupant of a direction in preference to a second point in an already occupied direction;
- when two non-dominated points associate to the same direction, keeps the one closer to the direction (quota 1);
- does not let dominated points re-enter.

Multiple samples from one weight compete with samples from other weights. Niching is a function of the pooled `Archive`, not an inner operator of Metropolis or Gibbs. `LoopResult.nondominated_front` remains; `LoopResult.niche_front` is the additional survivor set.

**Acceptance criteria.**

- [x] NumPy-only tests, no THRML. A hand-built 2-objective front where pure non-dominated filtering keeps a cluster on one direction, and niching also keeps a worse-scalarization point that is the nearest occupant of an empty direction.
- [x] A second fixture: two non-dominated points on one direction, the closer one kept, the farther one dropped when the niche quota is one.
- [x] `WeightSweepLoop` (or a successor with the same backend call) still obtains candidates only from `SamplingBackend.sample_weight`. No recombination, no mutation, no Bend types.
- [x] Directions passed in by the caller are the ones used for association. The loop does not secretly resample a new simplex.
- [x] The README algorithm-mapping row is updated in the same PR so "weight niches + offline ND" is not still described as the whole method after the behavior lands.
- [x] Existing archive tests still pass. The non-dominated filter remains available; niching is an additional survivor set, returned explicitly, so callers can see both.

### Phase 3 — In-repo fidelity harness

**Outcome.** Compare two minimization fronts inside this repo. Bend, C#, and any ZDT1 program stay outside. Their role is to produce a front file a human can drop in.

[docs/fidelity_hooks.md](docs/fidelity_hooks.md) already names the comparison (hypervolume, generational distance, coverage) and already forbids importing those oracles. This phase turns that note into tested functions.

**Acceptance criteria.**

- [x] A small module computes, for minimization: 2-D hypervolume (reuse `hypervolume_2d`), generational distance to a reference front, and coverage (fraction of one front weakly dominated by the other). Inputs are `ndarray` fronts, not sampler objects.
- [x] Tests are marked so `pytest -m 'not thrml'` runs them. One fixture uses `CodonIsingProblem.enumerate_front` or a static array with a known hypervolume. One fixture checks coverage on nested fronts (a front that dominates another covers it; the converse does not).
- [x] A loader accepts a `.npy` or `.npz` array of shape `(n, M)` and rejects a shape mismatch with a clear error. No `subprocess`, no `bend`, no `dotnet`.
- [x] `docs/fidelity_hooks.md` documents the file schema and states that a Bend or C# binary is not a CI dependency. The harness does not retune betas to chase an external number.
- [x] The harness can score a `LoopResult` non-dominated front against `enumerate_front` for the existing Ising chain at \(n \le 16\). That test may be marked `thrml` if it samples; the metric tests themselves must not be.

### Phase 4 — Optional Torx

**Outcome.** A default-off extra that can draw pbit samples from public Torx, labeled as Torx. Core install stays Python ≥ 3.10 and does not import `torx`.

Torx on this check is `extro-torx` 0.0.2, Python ≥ 3.11 ([getting started](https://docs.torx.ai/en/latest/getting-started.html), [PyPI](https://pypi.org/project/extro-torx/)). Re-checked 2026-10-02: that version and the quickstart are unchanged. The public quickstart constructs a `DiscretePCircuit`, holds parameters separately, and samples with `BranchingSimulator` ([docs home](https://docs.torx.ai/en/latest/)). This tree uses that entry point (`TorxPswapCircuit`). Directed factor graphs are a different public entry ([example 15 / factors](https://docs.torx.ai/en/latest/15_intro_to_factors.html), [example 16](https://docs.torx.ai/en/latest/16_gibbs_sampling_factor_graph.html)) and are not wrapped here.

**Acceptance criteria.**

- [x] Dependency lives in an optional extra (for example `torx`). `pip install -e .` on Python 3.10 still imports `unsga3_extropic` and runs `pytest -m 'not thrml'`.
- [x] The Torx import sits inside the optional backend, same pattern as the JAX import in `ThrmlIsingBackend`, so a missing `torx` package skips rather than breaks collection of the NumPy tests.
- [x] One smoke: build a two-site circuit from a documented gate (`PSWAP` or `PNOT` / `PCNOT` as in the Torx README), sample, and check the empirical stay/swap or bit-flip rate against the gate parameter within a loose tolerance at a documented sample count.
- [x] README calls this path Torx, not THRML, not Thermalizers, and not hardware.
- [x] No code path turns a Torx circuit into an `IsingEBM` or a `CategoricalEBMFactor`. That mapping is Thermalizers' unpublished job.

### Phase 5 — Thermalizers watch

**Outcome.** A documented trigger, and no implementation.

The paper ([arXiv:2608.01615](https://arxiv.org/abs/2608.01615), abstract dated with revisions through August 2026) says the thermalizers framework takes a Torx program and replaces factors with thermodynamic kernels implemented and sampled with THRML. The [Extropic post](https://extropic.ai/writing/from-one-to-one-billion/) calls the library upcoming and points at GitHub. That release is not on GitHub or PyPI as of 2026-10-02.

**Status (2026-10-02).** A check found no public package or API: public `extropic-ai` repos are `thrml`, `codon_opt`, `thrml-skill`, `torx`, and `sparse-transformers` (no `thermalizers` repo), and PyPI has no `thermalizers`, `extro-thermalizers`, or `thermalizer`. The trigger is not met. [arXiv:2608.01615](https://arxiv.org/abs/2608.01615) alone does not count.

**Acceptance criteria.**

- [ ] This phase adds no production code and no dependency.
- [ ] The watch, repeated when someone proposes a Thermalizers PR, is: a new public repository under `extropic-ai` (or a renamed successor announced from that org) **and** an install line in its README or docs **and** a documented function that accepts a Torx program and returns THRML factors or a THRML sampling program. All three. A new paper version is not the trigger.
- [ ] Until that trigger, PRs that add `import thermalizers`, a guessed `compile(...)`, context-matching training, or REINFORCE post-training are out of scope. Decline them by pointing at this section.
- [ ] When the trigger hits, open a new design note before coding. The first integration, if any, is a single documented example from the upstream README, run behind an optional extra, with a test that skips when the package is absent. Scope does not jump to "compile the whole U-NSGA-III loop."

### Phase 6 — Hardware later

**Outcome.** Device execution only after phases 1–3 have landed and a public programming interface exists. Nothing in phases 1–5 is blocked on a chip.

Published constraints to respect, once a backend is even discussable:

- Z1 samples a **sparse Ising** model, degree 16, chromatic Gibbs ([blog](https://extropic.ai/writing/from-one-to-one-billion/)). An all-to-all coupling is not a Z1 program.
- Native Potts is the categorical model in phase 1. The p-bit image is the domain-wall Ising encoding in [example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html). Appendix item 7 implements that encoding in THRML: `ThrmlIsingBackend` when the spin graph is 2-colored, and the example's 4-coloring otherwise. It is still a simulation. A device API is phase 6.
- The blog also mentions an early-access GPU simulator API. It is not documented on docs.thrml.ai or docs.torx.ai. It is not a dependency and not this phase's API.

**Acceptance criteria.**

- [ ] No device client is merged before the upstream documents install or authentication, a call that submits a program, and a call that returns samples. Quote those docs in the PR.
- [ ] The first hardware-backed test uses a graph that satisfies the published degree and coloring, and checks that returned states match the energy of the same THRML program in simulation on at least one fixed seed's worth of structure (the energy identity in phase 1, applied to device samples).
- [ ] The README does not say the package runs on Z1, X0, or XTR-0 until that test is real.
- [ ] Energy-per-sample numbers from the codon paper or the blog are not copied into this repo as measurements of our runs.

## Definition of done: "in the Extropic ecosystem"

Use this phrase only when every item below is true. It describes the **public software** surface. It does not mean on Z1, compiled by Thermalizers, or written in Torx.

1. **The search distribution is a public THRML program.** For every problem labeled THRML-native, `sample_weight` builds nodes, blocks, and factors from the installed `thrml` package and draws with `sample_states` (or another function the current [docs](https://docs.thrml.ai/en/latest/) present as the sampling entry point). Pinned floor: THRML **0.1.4** APIs already used by `ThrmlIsingBackend`. A newer THRML is allowed when the PR re-checks the docs and CI installs that version.
2. **Energy identity.** For those problems, the scalarized energy the EBM evaluates equals \(w \cdot f\) of the archived objectives, within a tested tolerance. `CodonIsingProblem` already has this shape for Ising. Phase 1 extends the same standard to Potts. `ExactEwMetropolisBackend` is outside the phrase.
3. **Coloring matches the model.** No two adjacent nodes share a free block. The Ising path's even/odd rule and the Potts path's tested coloring are both acceptable. An unchecked coloring is not.
4. **The README inventory matches the imports.** If Torx is not imported, the README does not say the package uses Torx. The same for Thermalizers and for hardware.
5. **Optional layers use their own names.** A Torx extra, once it meets phase 4, is "also runs an optional Torx circuit." It does not upgrade the sentence to "compiled onto thermodynamic hardware."
6. **Absent public APIs stay absent.** As of 2026-10-02 the honest stack is THRML 0.1.4 plus, optionally, extro-torx 0.0.2. Thermalizers is [arXiv:2608.01615](https://arxiv.org/abs/2608.01615) only. Z1 is a published chip description, not a client library.

Meeting items 1–4 on the current Ising path is already how this repo should describe **today's** THRML slice (`ThrmlIsingBackend` + `CodonIsingProblem`). Phases 1–3 are in this tree: categorical Potts, closer-in-niche survival, and the in-repo fidelity harness. Torx, Thermalizers, and hardware stay outside that description (phases 4–6).

## Appendix: issues to file after merge

File these on GitHub after this roadmap is on `main`. Do not file them from the roadmap PR. Titles are the issue titles. Acceptance criteria match the phases above; the issue body can link to the phase anchor.

### Epic

**Title:** U-NSGA-III-shaped search on the public Extropic stack

**Body.** Track [ROADMAP.md](ROADMAP.md). Order is algorithm, then optional Torx, then a Thermalizers watch, then hardware. Bend and C# are external oracles only. Thermalizers has no public API until phase 5's trigger trips. Close a child only when its boxes are checked in the repository.

### Child issues

**1. Add a Potts `CategoricalNode` THRML backend**

Phase 1. Filed as issue #4. Acceptance criteria (checked in the phase 1 section above):

- [x] Implement `SamplingBackend` with `CategoricalNode`, `CategoricalEBMFactor`, `CategoricalGibbsConditional`, and `FactorSamplingProgram` / `FactorizedEBM` as documented at [api-pgm](https://docs.thrml.ai/en/latest/api-pgm.html), [api-discrete-ebm](https://docs.thrml.ai/en/latest/api-discrete-ebm.html), [api-ebm](https://docs.thrml.ai/en/latest/api-ebm.html), and [example 00](https://docs.thrml.ai/en/latest/00_probabilistic_computing.html).
- [x] In-repo unary-plus-pairwise chain; tested energy identity between EBM energy and \(w\cdot f\); states in \([0, K)\); \(K\) lives on `CategoricalGibbsConditional`.
- [x] Reject an edge that falls inside one block. Anneal by rescaling weights, because `SamplingSchedule` has no beta ([example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html)).
- [x] THRML smoke in CI; NumPy tests still run with `-m 'not thrml'`.
- [x] README updated. Domain-wall Ising is a different issue.

**2. Prefer the closer occupant of each reference direction**

Phase 2. Acceptance criteria:

- Associate pooled objective rows to `simplex_weights` directions by perpendicular distance in a documented normalization.
- Keep a lone occupant of an empty direction over a duplicate in a crowded direction; keep the closer of two occupants of one direction.
- Candidates still come only from `sample_weight`. No crossover or mutation.
- NumPy fixtures. README mapping updated in the same PR. Non-dominated archive API remains.

**3. Add an in-repo front-comparison harness**

Phase 3. Filed as issue #6. Acceptance criteria (checked in the phase 3 section above):

- [x] Hypervolume (2-D), generational distance, and coverage on minimization arrays.
- [x] `.npy` / `.npz` loader. No Bend, C#, or ZDT1 process.
- [x] Tests under `pytest -m 'not thrml'`. Update [docs/fidelity_hooks.md](docs/fidelity_hooks.md).

**4. Optional `extro-torx` extra, default off**

Phase 4. Filed as issue #7. Acceptance criteria (checked in the phase 4 section above):

- [x] Extra dependency, Python ≥ 3.11 for that extra only. Core stays importable on 3.10 without `torx`.
- [x] One documented circuit (`DiscretePCircuit` + a documented simulator, or a documented DFG) with a statistical smoke test that skips if `extro-torx` is absent.
- [x] Docs call it Torx. No THRML lowering and no Thermalizers naming. Spec: [docs.torx.ai](https://docs.torx.ai/en/latest/).

**5. Watch for a public Thermalizers release**

Phase 5. Acceptance criteria:

- No code in the issue's resolution other than, if desired, a one-line status date in ROADMAP.md.
- Trigger is a public repo plus install docs plus a documented Torx-to-THRML entry point. Paper updates do not count. Reference: [arXiv:2608.01615](https://arxiv.org/abs/2608.01615).
- On trigger, write a design note before any compile wrapper. First patch, if any, is one upstream example behind an optional extra.

**6. Device runner only after a public hardware API**

Phase 6. Acceptance criteria:

- Blocked on children 1–3 and on upstream docs for submit-program and read-samples.
- First graph respects Z1's published sparse Ising limits (degree 16, chromatic Gibbs) from [the Z1 post](https://extropic.ai/writing/from-one-to-one-billion/).
- Energy of device samples matches the simulated THRML energy. README claims follow the test, not the chip announcement.

**7. Domain-wall Ising image of the in-repo Potts problem (not phase 1)**

Optional, after child 1. This is the p-bit compilation path in [example 03](https://docs.thrml.ai/en/latest/03_codon_optimization.html), not a new algorithm. Acceptance criteria:

- [x] Encode the phase-1 Potts energy as an Ising model by domain-wall encoding, sampled with `ThrmlIsingBackend` or a coloring-general Ising program if the constraint graph is not 2-colored.
- [x] A test shows decoded categorical states reproduce the Potts objectives on a toy chain. Invalid thermometers are counted, not silently scored as feasible.
- [x] Does not vendor `codon_opt` and does not claim Z1 execution (that is child 6).

## Sources

Checked 2026-10-02.

| Topic | URL |
|-------|-----|
| THRML home and quickstart | https://docs.thrml.ai/en/latest/ |
| THRML install and first Ising chain | https://docs.thrml.ai/en/latest/getting-started.html |
| Potts model, `CategoricalEBMFactor` | https://docs.thrml.ai/en/latest/00_probabilistic_computing.html |
| Codon Potts and domain-wall Ising | https://docs.thrml.ai/en/latest/03_codon_optimization.html |
| `CategoricalNode`, `SpinNode` | https://docs.thrml.ai/en/latest/api-pgm.html |
| Discrete EBM factors and Gibbs conditionals | https://docs.thrml.ai/en/latest/api-discrete-ebm.html |
| EBM / factorized energy | https://docs.thrml.ai/en/latest/api-ebm.html |
| Factors / `FactorSamplingProgram` | https://docs.thrml.ai/en/latest/api-factors.html |
| Ising model API | https://docs.thrml.ai/en/latest/api-ising.html |
| THRML paper | https://arxiv.org/abs/2510.23972 |
| THRML source | https://github.com/extropic-ai/thrml |
| Codon reproduction | https://github.com/extropic-ai/codon_opt |
| Torx home (pbit, pdit, pmode) | https://docs.torx.ai/en/latest/ |
| Torx getting started | https://docs.torx.ai/en/latest/getting-started.html |
| Torx primitives notebook | https://docs.torx.ai/en/latest/01_introduction_to_parametrised_stochastic_circuits.html |
| Torx directed factor graphs | https://docs.torx.ai/en/latest/15_intro_to_factors.html |
| Torx block Gibbs on a DFG | https://docs.torx.ai/en/latest/16_gibbs_sampling_factor_graph.html |
| Torx whitepaper | https://arxiv.org/abs/2608.01612 |
| Torx source | https://github.com/extropic-ai/torx |
| Thermalizers paper (no library) | https://arxiv.org/abs/2608.01615 |
| Torx, Thermalizers, Z1 announcement | https://extropic.ai/writing/from-one-to-one-billion/ |
| U-NSGA-III (Seada and Deb) | https://www.egr.msu.edu/~kdeb/papers/c2014022.pdf |
| This repo's current behavior | [README.md](README.md), [docs/fidelity_hooks.md](docs/fidelity_hooks.md) |
