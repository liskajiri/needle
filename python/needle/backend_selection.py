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

# Store loaded backend modules
_loaded_backend = None
default_device = None


def set_backend(backend_name: str) -> BACKENDS:
    """
    Set the backend for needle to use.

    Parameters:
    -----------
    backend_name : str
        Name of the backend to use.
        Options: "needle", "numpy", "cuda"

    Returns:
    --------
    BACKENDS
        The backend that was set.
    """
    global \
        BACKEND, \
        _loaded_backend, \
        default_device, \
        array_api, \
        Device, \
        NDArray, \
        all_devices, \
        cpu, \
        cuda

    if backend_name == "needle" or backend_name == BACKENDS.NEEDLE:
        BACKEND = BACKENDS.NEEDLE
    elif backend_name == "numpy" or backend_name == BACKENDS.NUMPY:
        BACKEND = BACKENDS.NUMPY
    else:
        raise ValueError(
            f"Unknown backend: {backend_name}.\
            Supported backends: 'needle', 'numpy', 'cuda'"
        )

    # Only reload if backend has changed
    if _loaded_backend != BACKEND:
        # TODO: this is temporary to avoid type checking issues
        if True:
            logging.info("Using needle backend")
            from needle import backend_ndarray as array_api
            from needle.backend_ndarray.ndarray import BackendDevice as Device
            from needle.backend_ndarray.ndarray import (
                NDArray,
                all_devices,
                cpu,
                cuda,
            )

            default_device = cpu()

        elif BACKEND == BACKENDS.NUMPY:
            logging.info("Using numpy backend")
            import numpy as array_api

            from needle.backend_numpy import NDArray, all_devices, cpu, cuda
            from needle.backend_numpy import NumpyBackend as Device

            default_device = cpu()

        _loaded_backend = BACKEND

    return BACKEND


set_backend(BACKEND)
