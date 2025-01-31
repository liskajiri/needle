"""
Logic for backend selection
"""

import enum
import os


class BACKENDS(enum.StrEnum):
    NEEDLE = "nd"
    NUMPY = "np"
    CUDA = "cuda"


DEFAULT_BACKEND = BACKENDS.NEEDLE

BACKEND = os.getenv("NEEDLE_BACKEND", DEFAULT_BACKEND)
if BACKEND not in BACKENDS:
    raise RuntimeError(f"Unknown needle array backend {BACKEND}")

BACKEND = BACKENDS.NEEDLE

if BACKENDS.NEEDLE == BACKEND:
    print("Using needle backend")
    from needle import backend_ndarray as array_api
    from needle.backend_ndarray import (
        BackendDevice as Device,
    )
    from needle.backend_ndarray import (
        NDArray,
    )
    from needle.backend_ndarray.ndarray import (
        all_devices,
        cpu,
        cuda,
    )

    default_device = cpu()

elif BACKENDS.NUMPY == BACKEND:
    print("Using numpy backend")
    import numpy as array_api  # noqa: F401, ICN001

    from needle.backend_numpy import (
        NDArray,  # noqa: F401
        all_devices,  # noqa: F401
        cpu,  # noqa: F401
        cuda,  # noqa: F401
    )

    default_device = cpu()
    from needle.backend_numpy import (
        NumpyBackend as Device,  # noqa: F401
    )
