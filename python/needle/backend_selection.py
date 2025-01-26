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
        default_device,
    )

elif BACKENDS.NUMPY == BACKEND:
    print("Using numpy backend")
    import numpy as array_api  # noqa: F401

    from needle.backend_ndarray.backend_numpy import (
        BackendDevice as Device,  # noqa: F401
    )
    from needle.backend_ndarray.backend_numpy import (
        NDArray,  # noqa: F401
        all_devices,  # noqa: F401
        cpu,  # noqa: F401
        cuda,  # noqa: F401
        default_device,  # noqa: F401
    )
else:
    raise RuntimeError(f"Unknown needle array backend {BACKEND}")
