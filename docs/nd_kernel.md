# Optional non-dominated ranking kernel

Jason asked (via CoS, 2026-11-10) for the largest hot spot in the sampling or ranking loops where an AI-written GPU kernel could win, then a prototype and a before/after benchmark. The inspiration was a custom CUDA kernel that cut an OCR training loop from hours to minutes at a fraction of GPU RAM.

This note is that prototype. It does not change the default ranking path, does not put SBX into `WeightSweepLoop`, and does not invent a Thermalizers API or replace THRML sampling.

## What dominates, and why

Profiled on 2026-11-10 at about 12:38 AM ET, on this box: Linux 6.12.113+, 8-core Xeon, 15 GiB RAM, no GPU, Python 3.13.5, NumPy 2.5.4. Warm-up excluded. Several seeds.

### Isolated ranking kernels

| Work | Size | Mean wall | Share of ranking |
|------|------|----------:|------------------|
| `nondominated_mask` (Python + per-row NumPy) | n=240, M=2 | 3.7 ms | baseline |
| `nondominated_mask` | n=1584, M=2 (deep archive width) | 104 ms | **ranking #1** |
| `nondominated_mask` | n=5000, M=2 | 951 ms | scales as n² |
| `nondominated_mask` | n=10000, M=2 | 3751 ms | scales as n² |
| `perpendicular_distances` | n=1584, H=11, M=2 | 0.63 ms | already vectorized |
| `niche_survival_rows` (includes ND) | n=1584, H=11 | 169 ms | almost all ND |
| `hypervolume_2d` | 80-point front | 1.5 ms | cheap |
| IGD | 80 vs 500 | 1.7 ms | cheap |
| IGD | 5000 vs 500 | 304 ms | materializes (R, N, M) |
| `_fronts` (classical full ranking) | n=104 (2N at pop 52) | 1.3 ms | small at published size |
| `_fronts` | n=4000 | 2606 ms | pairwise n×n×M array |

`nondominated_mask` is a Python loop over *n* rows. Each row compares the whole cloud to one point (`np.all` / `np.any`). That is O(*n*² *M*) arithmetic with one NumPy launch per row. At the deep-profile archive width (1,584 recorded rows) it is already ~100 ms. Niche survival calls the same filter again, so a `summary()` / `niche_front()` pair spends ~160–180 ms in ranking on a default or deep-like archive. Association itself is a few milliseconds.

### Sampling vs ranking on the product loop

`WeightSweepLoop` plus `ExactEwMetropolisBackend` on a cheap 12-bit linear objective, default anneal (5 weights, 6 betas, 40 warmup, 48 samples, 2 steps; 4,085 evals), seeds 7 / 11 / 19:

| Piece | Mean wall |
|-------|----------:|
| Sampling (`sample_weight` over all weights) | 35 ms |
| `nondominated_mask` on the 1,440-row archive | 75 ms |
| `Archive.nondominated` (mask + unique) | 79 ms |

On this cheap objective, ranking already costs more than sampling. Deep-like (11 weights, 24/24, 1,584 rows): sampling 39 ms, ND 90 ms.

That split is honest only for a cheap host-side objective. THRML / JAX block Gibbs was not timed here (this box has no GPU and the measurement venv did not install JAX). A compiled Ising or Potts sample would spend more time in `sample_weight`; ranking would stay the same function of archive size. Sequential Metropolis itself is a poor GPU-kernel target: each step depends on the previous accept/reject, and a single 12-spin or 30-variable chain does too little work per step to beat kernel-launch overhead.

### Classical U-NSGA-III (fidelity column, not the weight-sweep)

ZDT1, pop 52, 100 generations, seeds 1–3: mean 0.205 s. cProfile share: `_fronts` about 31% of a profiled run, `_niching` about 17%, duplicate repair about 25%. Scaled to pop 200, 80 generations: 1.49 s wall, `_fronts` about 62% (the n×n pairwise domination). `_fronts` is already a broadcasted NumPy comparison, so a custom kernel has less to gain at published sizes (each call is ~1 ms) than `nondominated_mask`, which is still a Python loop on the product path.

## Chosen target

**`archive.nondominated_mask`.** It is the ranking hot spot on the weight-sweep archive, it is called again from niche survival and 2-D hypervolume, and it is the loop that a compiled pairwise kernel can actually change. The CUDA/C kernel in `ND_MASK_CUDA_SOURCE` (and the matching Numba CUDA kernel) does one candidate per thread, stops when any other point dominates that candidate, and uses O(*n M*) memory instead of an *n*×*n*×*M* temporary.

## How to opt in

The default call is unchanged (`backend=None` reads `UNSGA3_ND_BACKEND` and falls back to `loop`).

```python
from unsga3_extropic.archive import nondominated_mask

mask = nondominated_mask(objectives)                  # loop
mask = nondominated_mask(objectives, backend="jit")   # Numba, optional extra
# or: export UNSGA3_ND_BACKEND=jit
```

| Backend | Needs | Role |
|---------|-------|------|
| `loop` | nothing | original per-row NumPy scan |
| `tiled` | nothing | chunked broadcast; same answers, not a real win at archive scale |
| `jit` | `pip install 'unsga3-extropic[kernels]'` (Numba) | compiled pairwise test, early exit; honest CPU win |
| `jit_parallel` | Numba | same test, `prange` over candidates; helps dense fronts, can add overhead on typical mixed clouds |
| `cuda` | a CUDA device and Numba CUDA or CuPy | same test as `ND_MASK_CUDA_SOURCE`; tests skip when no GPU is present |

Correctness is exact: every backend must produce the same boolean mask as `loop`, including the `1e-12` slack, maximization, and IEEE NaN/inf behavior of the NumPy comparisons (a NaN row does not dominate a finite row). GPU tests are marked `gpu` and skipped without a device. CI installs the `kernels` extra so the jit equivalence tests run; existing checks are not disabled or loosened.

## Before / after

Numbers: [benchmarks/ND_KERNEL_RESULTS.md](../benchmarks/ND_KERNEL_RESULTS.md). Summary on this box (no GPU):

- A broadcasted NumPy rewrite (`tiled`) is about 2.6× at n=240 and ~1.1× at n≥1584. It uses much more memory (12 MB peak at n=10,000 vs 0.1 MB for `loop`). Vectorizing this filter is not the win.
- `jit` is the win. Isolated random clouds: ~700× to ~3700×, because most points find a dominator quickly. Isolated full fronts (every point non-dominated, no early exit): ~11–12×. A realistic ExactEw archive of 1,440 rows: 75 ms → 2.4 ms (**31×**).
- `jit_parallel` is the better worst-case CPU path on a full front (55× at n=10,000 vs 11× serial) and can be slower than serial `jit` on already-easy random clouds.
- `cuda` was not timed. This box has no NVIDIA device (`nvidia-smi` missing, `jax.devices()` not applicable, Numba reports no CUDA driver). The Kaggle connector is installed but quota and session calls returned `Unauthenticated`, so there was no free GPU path that could run arbitrary code.

## What this is not

- Not a replacement for THRML block Gibbs, and not a Thermalizers compiler.
- Not a change to `WeightSweepLoop` variation (still `sample_weight` only).
- Not a claim that end-to-end THRML demos get a 1000× wall-clock cut. On a cheap ExactEw objective, ranking was the larger slice; on a compiled sampler, ranking is the part this kernel shrinks.
