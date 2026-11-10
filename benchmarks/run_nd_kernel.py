#!/usr/bin/env python3
"""Before/after timings for optional nondominated backends.

Warm-up is excluded. The default ``loop`` path is the BEFORE column.
This script does not write tracked run records unless ``--out`` is set.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unsga3_extropic.archive import nondominated_mask
from unsga3_extropic.kernels.nondominated import cuda_available, jit_available, list_backends


def _hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        count = str(np.__config__.get_info("config") or "")
    except Exception:
        count = ""
    cores = None
    try:
        import os

        cores = os.cpu_count()
    except Exception:
        pass
    gpu = "no CUDA device"
    if cuda_available():
        try:
            from numba import cuda

            gpu = str(cuda.get_current_device().name)
            if isinstance(gpu, bytes):
                gpu = gpu.decode()
        except Exception:
            try:
                import cupy as cp

                gpu = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            except Exception:
                gpu = "CUDA device present"
    return f"{platform.system()} {platform.release()}; Python {platform.python_version()}; NumPy {np.__version__}; CPU={cpu}; cores={cores}; GPU={gpu}{count and ''}"


def _objs(n: int, m: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.random((n, m))
    t = rng.random(n)
    if m == 2:
        return np.column_stack([t, 1.0 - np.sqrt(np.clip(t, 0.0, 1.0))])
    extra = rng.random((n, m - 1))
    extra = extra / np.maximum(extra.sum(axis=1, keepdims=True), 1e-12)
    return np.column_stack([t, extra * (1.0 - t)[:, None]])[:, :m]


def _time(fn, *, n_warm: int, n_run: int) -> tuple[float, float, float, int]:
    for _ in range(n_warm):
        fn()
    tracemalloc.start()
    times = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.min()), float(arr.max()), int(peak)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="optional markdown file to write")
    parser.add_argument("--backends", default="loop,tiled,jit,jit_parallel,cuda")
    args = parser.parse_args(argv)

    requested = [name.strip() for name in args.backends.split(",") if name.strip()]
    available = list_backends()
    backends = []
    skipped = []
    for name in requested:
        if name not in available:
            skipped.append((name, "unknown"))
        elif not available[name]:
            reason = "numba not installed" if name.startswith("jit") else "no CUDA device"
            skipped.append((name, reason))
        else:
            backends.append(name)
    if "loop" not in backends:
        backends.insert(0, "loop")

    sizes = [
        (240, 2, "random", 7),
        (1584, 2, "random", 5),
        (1584, 2, "front", 5),
        (5000, 2, "random", 3),
        (5000, 3, "random", 3),
        (10000, 2, "random", 3),
        (10000, 2, "front", 3),
    ]
    seeds = (7, 11, 19)
    lines = [
        f"Hardware: {_hardware()}",
        f"Backends timed: {', '.join(backends)}",
        f"Skipped: {', '.join(f'{n} ({why})' for n, why in skipped) or 'none'}",
        f"numba: {'yes' if jit_available() else 'no'}; cuda: {'yes' if cuda_available() else 'no'}",
        "",
        "| n | m | kind | backend | mean ms | min ms | max ms | peak alloc (KB) | vs loop | match |",
        "|---|---:|------|---------|--------:|-------:|-------:|----------------:|--------:|:-----:|",
    ]
    print("\n".join(lines[:5]))
    print(lines[5])
    print(lines[6])

    for n, m, kind, n_run in sizes:
        base_mean = None
        for backend in backends:
            means = []
            mins = []
            maxs = []
            peaks = []
            matched = True
            for seed in seeds:
                objs = _objs(n, m, kind, seed)
                expected = nondominated_mask(objs, backend="loop")
                got = nondominated_mask(objs, backend=backend)
                if not np.array_equal(got, expected):
                    matched = False
                mean, lo, hi, peak = _time(
                    lambda o=objs, b=backend: nondominated_mask(o, backend=b),
                    n_warm=1,
                    n_run=n_run,
                )
                means.append(mean)
                mins.append(lo)
                maxs.append(hi)
                peaks.append(peak)
            mean_ms = float(np.mean(means) * 1e3)
            min_ms = float(np.min(mins) * 1e3)
            max_ms = float(np.max(maxs) * 1e3)
            peak_kb = float(np.mean(peaks) / 1024.0)
            if backend == "loop":
                base_mean = mean_ms
                speed = 1.0
            else:
                speed = (base_mean / mean_ms) if mean_ms > 0 else float("inf")
            row = (
                f"| {n} | {m} | {kind} | `{backend}` | {mean_ms:.3f} | {min_ms:.3f} | "
                f"{max_ms:.3f} | {peak_kb:.1f} | {speed:.2f}x | {'yes' if matched else 'NO'} |"
            )
            lines.append(row)
            print(row)
            if not matched:
                raise SystemExit(f"correctness failure: {backend} n={n} kind={kind}")

    text = "\n".join(lines) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
