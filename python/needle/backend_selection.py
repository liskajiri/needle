"""
Logic for backend selection
"""

import enum
import os


class BACKENDS(enum.StrEnum):
    NEEDLE = "nd"
    NUMPY = "np"


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


if BACKENDS.NEEDLE == BACKEND:
    print("Using needle backend")
    from needle import backend_ndarray as array_api  # noqa: I001
    from needle.backend_ndarray import (
        BackendDevice as Device,
    )
    from needle.backend_ndarray import (
        NDArray,
        all_devices,
        cpu,
        cpu_numpy,  # noqa: F401
        cuda,
        default_device,
        DType,  # noqa: F401
    )

elif BACKENDS.NUMPY == BACKEND:
    print("Using numpy backend")
    import numpy as array_api  # noqa: F401
    from numpy import ndarray as NDArray  # noqa: F401

    from needle.backend_ndarray.backend_numpy import (
        Device,  # noqa: F401
        all_devices,  # noqa: F401
        cpu,  # noqa: F401
        cuda,  # noqa: F401
        default_device,  # noqa: F401
    )
else:
    raise RuntimeError(f"Unknown needle array backend {BACKEND}")
