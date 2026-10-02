#!/usr/bin/env python3
"""Sample the documented Torx PSWAP circuit.

Python >= 3.11 and the optional extra::

    pip install -e ".[cpu,torx]"
    python demos/run_torx_pswap.py

Prints the empirical stay and swap rates for the docs quickstart
(``DiscretePCircuit`` + ``PSWAP`` + ``BranchingSimulator``). This is not
a search run and it does not append a benchmark front.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unsga3_extropic.torx_circuit import (  # noqa: E402
    DOCUMENTED_INITIAL_STATE,
    DOCUMENTED_N_SAMPLES,
    DOCUMENTED_P_SWAP,
    DOCUMENTED_SEED,
    SMOKE_ABS_TOLERANCE,
    TorxPswapCircuit,
)


def main() -> int:
    draw = TorxPswapCircuit().sample(
        p_swap=DOCUMENTED_P_SWAP,
        n_samples=DOCUMENTED_N_SAMPLES,
        seed=DOCUMENTED_SEED,
        initial_state=DOCUMENTED_INITIAL_STATE,
    )
    print(
        f"stay |10): {draw.stay_rate:.3f}, swap |01): {draw.swap_rate:.3f} "
        f"(p={draw.p_swap}, n={draw.n_samples}, seed={draw.seed})"
    )
    print(
        "Torx PSWAP smoke only. Not a search front, so it is not appended "
        "to benchmarks/records/runs.jsonl."
    )
    if abs(draw.swap_rate - DOCUMENTED_P_SWAP) >= SMOKE_ABS_TOLERANCE:
        print(
            f"swap rate {draw.swap_rate:.3f} is outside ±{SMOKE_ABS_TOLERANCE} "
            f"of {DOCUMENTED_P_SWAP}",
            file=sys.stderr,
        )
        return 1
    if abs(draw.stay_rate - (1.0 - DOCUMENTED_P_SWAP)) >= SMOKE_ABS_TOLERANCE:
        print(
            f"stay rate {draw.stay_rate:.3f} is outside ±{SMOKE_ABS_TOLERANCE} "
            f"of {1.0 - DOCUMENTED_P_SWAP}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
