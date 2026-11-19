# Energy-Based Reasoners: what transfers to this repo

Research note, Nov 19, 2026. Source: Chen, Gladstone, Li, Ji, Du, *Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based Model*, [arXiv:2611.08977](https://arxiv.org/abs/2611.08977) (v1, Nov 6, 2026), code at [wwillchen/ebr](https://github.com/wwillchen/ebr) (PyTorch: `pretrain.py`, `evaluate.py`, `models/`). Everything below that is attributed to the paper was read in v1 of the PDF. Section numbers refer to that version. The spike numbers are in [benchmarks/EBR_SPIKE_RESULTS.md](../benchmarks/EBR_SPIKE_RESULTS.md).

Scope: this repository covers the Extropic samplers (ExactEw Metropolis, THRML Ising, THRML Potts, domain-wall). The classical U-NSGA-III side (energy-based niching tie-breaks, a stopping rule, and best-of-K on ZDT/DTLZ by IGD) is covered by Bend in unsga3-bend/Unsga3. It is not touched here.

## What EBR is

An EBR is a small looped reasoning model (1.8M parameters in Table 1, against 5–27M for the baselines) that learns a scalar energy \(E_\theta(z; x)\) over a latent reasoning state \(z\). Each loop step is a gradient step on that energy, \(z_{t+1} = z_t - \alpha \nabla_z E_\theta(z_t; x)\) (eq. 2). In practice it is projected Langevin dynamics: noise is added for exploration and the state is projected onto a sphere (Algorithm 2). Training uses segment-wise deep supervision with truncated backprop through the descent. That needs second-order derivatives of \(E_\theta\) (§3.1). The tasks are puzzles only: Sudoku-Extreme, Maze-Hard, N-Queens, and Graph Coloring.

## Learned energy versus designed energy

| | EBR | This repo (energy-from-objective) |
|---|---|---|
| Where the energy comes from | Learned by a neural network from labelled puzzles | Written down from the objectives: \(E_w(x) = w \cdot f(x)\), compiled into `IsingEBM` biases and couplings, Potts factors, or a domain-wall Ising image |
| What it means | A score that the training made low on correct answers | Exactly the scalarized objective. A lower \(E_w\) is a better candidate for that weight by definition |
| How it is minimized | Gradient descent with noise on a continuous latent, through backprop | Gibbs or Metropolis sampling of discrete states at inverse temperature \(\beta\) (THRML block Gibbs or the NumPy ExactEw chain), the form Extropic hardware is built for |
| What it costs | A training run (second-order gradients, SXM rollouts) | No training. The energy is exact on day one |

**Why EBR doesn't replace energy-from-objective.** EBR answers the question "how do I get an energy when I can't write one down?" by learning it from solved examples. Here the energy already exists and is exact, so there is nothing to learn. A learned energy would only approximate \(w \cdot f\). It would also need differentiable continuous states, which a THRML spin or categorical program doesn't have, and labelled optima, which a weight sweep doesn't have. Porting the PyTorch trainer would add cost and error with no gain, so this repo doesn't do it.

## One energy, three jobs

The idea that transfers is that EBR uses its single energy for three jobs:

1. **Refine:** each step descends the energy (eq. 2).
2. **Stop:** inference stops when the tangent gradient norm drops below \(\tau\), or after S segments. The paper notes that the stepwise residual works as well (§3.3, Fig. 6). With \(\tau = 0.2\), stopping keeps Sudoku-Extreme at 99.9% with 10.8× fewer inference FLOPs than a fixed iteration budget (Table C.2, Fig. 5b).
3. **Select:** among independent trajectories, keep the lowest-energy endpoint (§3.3). In Fig. 7 that comes close to oracle selection and beats picking the most converged trajectory.

How each job maps onto our samplers:

| EBR job | Here | Status |
|---|---|---|
| Refine | Annealed sampling under \(E_w\) (`sample_weight`). Already the search step | Unchanged |
| Stop | The energy plateau rule: stop recording at a beta stage once the running best \(E_w\) has not improved by more than `tol` over the last `window` samples (`unsga3_extropic.ebr.PlateauStop`). Our chains are discrete with no gradient, so the energy plateau stands in for the gradient or residual signal | Opt-in on `ExactEwMetropolisBackend(plateau=...)`. THRML runs one compiled program per beta stage, so on THRML the rule was only scored after the fact |
| Select | Best of the samples by our own objective-derived \(E_w\) (`select_lowest_energy`, `LowestEnergySelection`). Unlike EBR's learned score, ours is exact, so the argmin is the best sample for that weight by definition | Opt-in wrapper on any `SamplingBackend` |

## Mode coverage: SXM versus Boltzmann sampling

**What SXM is in the paper (§3.2, App. A.4, E).** During training only, K descent trajectories are rolled out per input. The trajectory–target pair with the lowest loss is supervised: forward SXM picks the trajectory, then reverse SXM picks the closest valid solution. The unselected trajectories keep evolving and stay eligible in later segments. The paper notes that reverse SXM alone can pull every trajectory of an input toward the same solution ("causing collapse", App. A.4). It avoids that by not reusing a solution within a batch. In Table 4, adding F-SXM (K = 2) and then R-SXM raises 8×8 N-Queens coverage from 75.9 to 87.5 to 97.9. Its purpose is to shape a *learned* landscape so that it has a basin for each valid solution.

**Why we don't need it.** A Gibbs or Metropolis sampler at inverse temperature \(\beta\) targets the Boltzmann distribution \(p(x) \propto e^{-\beta E_w(x)}\). That distribution already puts mass on every low-energy state, in proportion to its energy. Our energy is designed rather than learned, so no basin can be missing, and no training-time exploration is needed to create one. What *can* go wrong for us is mixing: a finite chain may not leave one basin. The tools for that are temperature, the anneal schedule \(\beta_1 < \dots < \beta_n\), restarts or seeds, and THRML's parallel block updates. SXM-style exploration is a training fix for a learned energy. Temperature and annealing are the sampling fix for an exact one. The two are not interchangeable, and this repo only needs the second.

**Where mode collapse does show up here.** In the weight sweep itself. A linear scalarization \(w \cdot f\) only has minima on the convex hull of the front, and nearby weights share the same minimizer. If each weight kept only its argmin, 11 weights would collapse onto a few front points. The spike shows this: lowest-energy selection alone cut Potts coverage from 1.0 to 0.4 on two of three seeds. The current loop avoids that collapse because it archives *every* sample, including higher-temperature samples that land on other front points, and ranks them offline. So the EBR selection rule is right for "the best point for this weight" and wrong as a replacement for the archive. If it is used, it belongs next to the archive, not in place of it.

## Claim status (verified in the paper text)

| Claim | What the paper says |
|---|---|
| 345× less inference compute | Holds only on Sudoku-Extreme against EqR, "at comparable accuracy" (§4.2, Fig. 5b). EqR's 99.8% is its paper number at 128 samples (App. B). |
| 43× fewer samples | Holds: on Sudoku-Extreme, EBR passes EqR's 99.8% with 3 samples against 128 (§4.1, Fig. 4b). The EqR curve in Fig. 4b was run on Sudoku-Lite, the first 2,048 puzzles (App. B). |
| 7× | Chief of Staff found this only in the tweet. The PDF figure text does carry a "7× fewer iterations" label on Fig. 4a (one sample, more iterations, Sudoku-Extreme), but the body text never states or defines it. Treat it as unverified. |
| 2.6× training savings | Not supported as a general claim. Fig. 5a carries a "2.6× fewer FLOPs" label for Sudoku-Extreme training compute, with no definition in the body text. Table B.1 gives 15% fewer total training FLOPs than GRAM on Sudoku-Extreme and **1.2–3.7× more** on N-Queens and Graph Coloring, "primarily due to SXM candidate rollouts". The abstract says "comparable training compute". |
| Scale and domain | 1.8M parameters (Table 1). Up to 7M in the multi-solution table. Algorithmic puzzles only. |

## What the spike found

Seeds 7, 11, and 19 on the two deep tasks with enumerable fronts (codon, 12 spins; Potts, 6 sites with K = 3), on the existing `DEEP_SCHEDULE`. Full table in [benchmarks/EBR_SPIKE_RESULTS.md](../benchmarks/EBR_SPIKE_RESULTS.md).

- **Plateau stop (window 8, tol 0, set before the run):** about 50% fewer recorded samples and 34–35% fewer energy evaluations on ExactEw. HV, GD, and coverage of the non-dominated archive were equal to the baseline or better on all six task-seed pairs. Codon seed 11 rose from 0.989 to 1.000 HV, but that is a different random path, not a property of the rule. The per-weight minimum of \(E_w\) was found slightly less often on Potts (30/33 against 33/33). The savings come from samples that the default schedule spends at a stage after its energy has gone flat.
- **Lowest-energy selection:** no quality gain, and usually a loss in front quality, for the mode-collapse reason above. On THRML the argmin row was already the exact per-weight minimum on all 66 weights, so selection adds nothing there.
- **Verdict:** the stop rule helps (same fronts with about half the samples on ExactEw). Selection as an archive replacement doesn't help. Neither is turned on by default.
