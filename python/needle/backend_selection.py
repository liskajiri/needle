"""Logic for backend selection"""

import os
from typing import NewType

BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


if BACKEND == "nd":
    print("Using needle backend")
    from . import backend_ndarray as array_api  # noqa: I001
    from .backend_ndarray import (
        all_devices,
        cuda,  # noqa: F401
        cpu,
        cpu_numpy,  # noqa: F401
        default_device,
        BackendDevice as Device,
    )

    # NDArray = array_api.NDArray
    NDArray = NewType("NDArray", array_api.NDArray)
elif BACKEND == "np":
    print("Using numpy backend")
    import numpy as array_api

    from .backend_numpy import Device, all_devices, cpu, default_device  # noqa: F401
    # TODO: 2024 version
    # from .backend_ndarray import cuda

    # NDArray = array_api.ndarray
    # type annotation for NDArray
    NDArray = NewType("NDArray", array_api.ndarray)
else:
    raise RuntimeError(f"Unknown needle array backend {BACKEND}")
