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
backend = os.getenv("NEEDLE_BACKEND", DEFAULT_BACKEND)

# Store loaded backend modules
_loaded_backend = None
default_device = None


def set_backend(backend_name: str) -> None:
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
        backend, \
        _loaded_backend, \
        default_device, \
        array_api, \
        Device, \
        NDArray, \
        all_devices, \
        cpu, \
        cuda

    if backend_name == "needle" or backend_name == BACKENDS.NEEDLE:
        backend = BACKENDS.NEEDLE
    elif backend_name == "numpy" or backend_name == BACKENDS.NUMPY:
        backend = BACKENDS.NUMPY
    else:
        raise ValueError(
            f"Unknown backend: {backend_name}.\
            Supported backends: 'needle', 'numpy', 'cuda'"
        )

    # Only reload if backend has changed
    if _loaded_backend != backend:
        if backend == BACKENDS.NEEDLE:
            logging.info("Using needle backend")
            import needle.backend_ndarray.array_api as array_api
            from needle.backend_ndarray.backend import BackendDevice as Device
            from needle.backend_ndarray.backend import all_devices, cpu, cuda
            from needle.backend_ndarray.ndarray import NDArray

            default_device = cpu()

        elif backend == BACKENDS.NUMPY:
            logging.info("Using numpy backend")

            from numpy_backend import NDArray, all_devices, array_api, cpu, cuda
            from numpy_backend import NumpyBackend as Device

            default_device = cpu()
        print(backend)

        _loaded_backend = backend


set_backend(backend)
