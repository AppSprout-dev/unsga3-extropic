"""Default ranking path must not import numba or the kernels package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


_PROBE = r"""
import sys
import numpy as np
from unsga3_extropic.archive import nondominated_mask

mask = nondominated_mask(np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.6, 0.6]]))
assert mask.tolist() == [True, True, True, False]
assert "numba" not in sys.modules
leaked = [
    name
    for name in sys.modules
    if name == "unsga3_extropic.kernels" or name.startswith("unsga3_extropic.kernels.")
]
assert not leaked, leaked
print("default-path import-free: ok")
"""


def test_default_path_does_not_import_numba_or_kernels():
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "default-path import-free: ok" in result.stdout
