"""Optional compiled backends for ranking kernels.

Importing this package does not change any default algorithm. Callers
opt in through ``nondominated_mask(..., backend=...)`` or the
``UNSGA3_ND_BACKEND`` environment variable.
"""

from unsga3_extropic.kernels.nondominated import (
    BACKENDS,
    ND_MASK_CUDA_SOURCE,
    cuda_available,
    jit_available,
    list_backends,
    resolve_backend,
)

__all__ = [
    "BACKENDS",
    "ND_MASK_CUDA_SOURCE",
    "cuda_available",
    "jit_available",
    "list_backends",
    "resolve_backend",
]
