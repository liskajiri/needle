"""
Logic for backend selection
"""

import enum
import logging
import os


class BACKENDS(enum.StrEnum):
    NEEDLE = "nd"
    NUMPY = "np"
    CUDA = "cuda"


DEFAULT_BACKEND = BACKENDS.NEEDLE

BACKEND = os.getenv("NEEDLE_BACKEND", DEFAULT_BACKEND)
if BACKEND not in BACKENDS:
    raise RuntimeError(f"Unknown needle array backend {BACKEND}")

if BACKENDS.NEEDLE == BACKEND:
    logging.info("Using needle backend")
    from needle import backend_ndarray as array_api
    from needle.backend_ndarray.ndarray import (
        BackendDevice as Device,
    )
    from needle.backend_ndarray.ndarray import (
        NDArray,
        all_devices,
        cpu,
        cuda,
    )

    default_device = cpu()

elif BACKENDS.NUMPY == BACKEND:
    logging.info("Using numpy backend")
    import numpy as array_api  # noqa: F401, ICN001

    from needle.backend_numpy import (  # noqa: F401
        NDArray,
        all_devices,
        cpu,  # noqa: F401
        cuda,
    )

    default_device = cpu()
    from needle.backend_numpy import (
        NumpyBackend as Device,  # noqa: F401
    )
