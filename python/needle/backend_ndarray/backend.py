from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, override

from needle.needle_typing import AbstractBackend

if TYPE_CHECKING:
    from needle.backend_ndarray.ndarray import NDArray
    from needle.needle_typing import (
        DType,
        IndexType,
        Shape,
        Strides,
    )
    from needle.needle_typing.device import ModuleProtocol, NDArrayBackendProtocol


class BackendDevice(AbstractBackend):
    def __init__(
        self, name: str, module: ModuleProtocol[NDArray] | None = None
    ) -> None:
        if module is None:
            super().__init__(name, module, -1, -1)
        else:
            super().__init__(
                name,
                module,
                tile_size=module.__tile_size__,
                itemsize=module.itemsize,
            )

    def randn(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        """
        Generate random array from standard normal distribution
        Uses native backend implementation when available, falls back to Python RNG.
        """
        if isinstance(shape, int):
            shape = (shape,)

        size = (math.prod(shape),)
        arr = self.empty(size, dtype=dtype)

        # Use native backend RNG if available
        if self.enabled():
            # backend.randn expects an AlignedArray/handle and fills it in-place
            self.module.randn(arr._handle)
        else:
            # deterministic fallback for tests / platforms without native backend
            random.seed(0)
            for i in range(arr.size):
                arr[i] = random.gauss(0.0, 1.0)

        return arr.reshape(shape)

    def rand(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        """
        Generate random samples from uniform distribution [0,1).
        Uses native backend implementation when available, falls back to Python RNG.
        """
        if isinstance(shape, int):
            shape = (shape,)

        size = (math.prod(shape),)
        arr = self.empty(size, dtype=dtype)

        if self.enabled():
            self.module.rand(arr._handle)
        else:
            random.seed(0)
            for i in range(arr.size):
                arr[i] = random.uniform(0.0, 1.0)

        return arr.reshape(shape)

    @override
    def one_hot(self, n: int, i: IndexType, dtype: DType) -> NDArray:
        if self.enabled():
            from needle.backend_ndarray.ndarray import NDArray

            # allocate output on device with shape (*idx.shape, n)
            i = NDArray(i, device=self)
            i_shape = (*i.shape, n)
            out = make(i_shape, device=self)

            assert self.module is not None
            self.module.one_hot(out._handle, i._handle, n)
            return out
        else:
            raise NotImplementedError()

    @override
    def empty(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        return make(shape, device=self)

    @override
    def set_seed(self, seed: int | None = None) -> None:
        # Set Python RNGs
        random.seed(seed)

        # Propagate seed to native backend if available
        if seed is not None and self.enabled():
            # backend.set_seed expects an unsigned int
            self.module.set_seed(int(seed))

    @staticmethod
    def _tiled_matmul(arr: NDArray, other: NDArray, m: int, n: int, p: int) -> NDArray:
        def _tile(a: NDArray, tile: int) -> NDArray:
            """
            Transforms a matrix [k, n] into a
            matrix [k // tile, n // tile, tile, tile].
            """
            return a._as_strided(
                (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                (a.shape[1] * tile, tile, a.shape[1], 1),
            ).compact()

        t = arr.device.__tile_size__
        a = _tile(arr, t)
        b = _tile(other, t)
        out = make((a.shape[0], b.shape[1], t, t), device=arr.device)
        arr.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

        return (
            out.permute((0, 2, 1, 3)).compact().reshape((arr.shape[0], other.shape[1]))
        )


def cuda() -> AbstractBackend:
    """Return cuda device."""
    try:
        from needle.backend_ndarray import ndarray_backend_cuda  # type: ignore

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy() -> AbstractBackend:
    """Return numpy device."""
    try:
        import numpy_backend
        from numpy_backend import NumpyBackend

        return NumpyBackend("cpu_numpy", numpy_backend)  # type: ignore
    except ImportError:
        raise ImportError("Numpy backend not available")


def cpu() -> AbstractBackend:
    """Return cpu device."""
    try:
        import ndarray_backend_cpu

        return BackendDevice("cpu", ndarray_backend_cpu)  # type: ignore
    except ImportError:
        raise ImportError("CPU backend not available")


def all_devices() -> list[AbstractBackend]:
    """Return a list of all available devices."""
    return [cpu(), cuda(), cpu_numpy()]


default_device = cpu()


def make(
    shape: Shape,
    strides: Strides | None = None,
    device: AbstractBackend = default_device,
    handle: NDArrayBackendProtocol | None = None,
    offset: int = 0,
) -> NDArray:
    """
    Create a new NDArray with the given properties.
    Allocates a new array if handle is not provided.

    Args:
        shape: Tuple specifying dimensions of the array
        strides: Optional tuple specifying stride for each dimension
        device: Device backend for the array (defaults to CPU)
        handle: Existing handle to use for memory (allocates new if None)
        offset: Memory offset for the array (default: 0)

    Returns:
        NDArray: New array with requested properties

    Raises:
        ValueError: If shape contains invalid dimensions

    Examples:
        >>> make((2, 3)).shape
        (2, 3)
        >>> make((2, 3), strides=(1, 2)).strides
        (1, 2)
        >>> make((2, 3), strides=(1, 2))._offset
        0
        >>> make((2, 3), strides=(1, 2), offset=5)._offset
        5
        >>> make((2, 3), strides=(1, 2), device=cpu()).device
        cpu()
    """
    from needle.backend_ndarray.ndarray import NDArray

    def prod(shape: Shape) -> int:
        """Calculate product of shape tuple, handling nested tuples."""
        result = 1
        for dim in shape:
            if isinstance(dim, tuple):
                result *= prod(dim)
            else:
                result *= dim
        return result

    array = NDArray.__new__(NDArray)
    array._shape = shape
    array._strides = NDArray._compact_strides(shape) if strides is None else strides
    array._offset = offset
    array._device = device

    array_size = prod(shape)
    if handle is None:
        if array_size < 0:
            raise ValueError(f"Array size cannot be negative, Invalid shape: {shape}")
        array._handle = array.device.Array(array_size)
    else:
        array._handle = handle
    return array
