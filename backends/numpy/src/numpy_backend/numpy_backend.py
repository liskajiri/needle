"""
This file defies specific implementations of devices
when using numpy as NDArray backend.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

# TODO:
from numpy_backend.device import AbstractBackend

if TYPE_CHECKING:
    from needle.needle_typing import (
        DType,
        IndexType,
        Scalar,
        Shape,
        Strides,
    )

type NDArrayType = np.ndarray

__device_name__ = "numpy"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


class NDArray:
    def __init__(self, elems, device=None, dtype=None) -> None:
        self.array = np.array(elems, dtype=dtype if dtype else _datatype)
        self._device = device

    @property
    def size(self) -> int:
        return self.array.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def strides(self) -> Strides:
        return self.array.strides

    @property
    def device(self) -> AbstractBackend:
        return self._device

    @property
    def dtype(self) -> DType:
        # only support float32 for now
        return "float32"

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.array.shape)

    def __len__(self) -> int:
        return self.array.shape[0]

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __str__(self) -> str:
        # TODO: overwrite for doctests
        return self.array.__str__()

    def compact(self):
        return self

    def reshape(self, new_shape: Shape) -> NDArray:
        return NDArray(self.array.reshape(new_shape), device=self._device)

    def permute(self, axes: tuple[int, ...]) -> NDArray:
        return NDArray(np.permute_dims(self.array, axes), device=self._device)

    def __neg__(self) -> NDArray:
        return NDArray(-self.array, device=self._device)

    def __add__(self, other: NDArray) -> NDArray:
        if not isinstance(other, NDArray):
            return NDArray(self.array + other, device=self._device)

        return NDArray(self.array + other.array, device=self._device)

    def __sub__(self, other: NDArray) -> NDArray:
        if not isinstance(other, NDArray):
            return NDArray(self.array - other, device=self._device)

        return NDArray(self.array - other.array, device=self._device)

    def __mul__(self, other: NDArray) -> NDArray:
        return NDArray(self.array * other.array, device=self._device)

    def __matmul__(self, other: NDArray) -> NDArray:
        return NDArray(self.array @ other.array, device=self._device)

    def maximum(self, other: NDArray) -> NDArray:
        return NDArray(np.maximum(self.array, other), device=self._device)

    def tanh(self) -> NDArray:
        return NDArray(np.tanh(self.array), device=self._device)

    def exp(self) -> NDArray:
        return NDArray(np.exp(self.array), device=self._device)

    def log(self) -> NDArray:
        return NDArray(np.log(self.array), device=self._device)

    def numpy(self) -> NDArrayType:
        return self.array

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying numpy array when the
        attribute is not found on NDArray.

        - If the underlying attribute is callable, return a wrapper that:
          * unwraps NDArray args to numpy arrays
          * calls the numpy function
          * wraps numpy.ndarray results back to NDArray preserving device
        - Non-callable attributes are returned directly.
        """
        # Called only when normal lookup fails; get attribute from ndarray.
        attr = getattr(self.array, name)

        if callable(attr):

            def _wrapped(*args, **kwargs):
                # Unwrap NDArray arguments to numpy arrays
                unwrapped_args = [
                    a.array if isinstance(a, NDArray) else a for a in args
                ]
                unwrapped_kwargs = {
                    k: (v.array if isinstance(v, NDArray) else v)
                    for k, v in kwargs.items()
                }
                result = attr(*unwrapped_args, **unwrapped_kwargs)
                # Wrap numpy.ndarray results back to NDArray with device info
                if isinstance(result, np.ndarray):
                    return NDArray(result, device=self._device)
                return result

            return _wrapped

        return attr


def to_numpy(a, shape, strides, offset) -> NDArrayType:
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
    )


class NumpyBackend(AbstractBackend):
    def randn(self, shape: Shape, dtype: DType = "float32") -> NDArrayType:
        return np.random.randn(*shape)

    def rand(self, shape: Shape, dtype: DType = "float32") -> NDArrayType:
        return np.random.rand(*shape)

    def one_hot(self, n: int, i: IndexType, dtype: DType) -> NDArrayType:
        return np.eye(n, dtype=dtype)[i]

    def zeros(self, shape: Shape, dtype: DType) -> NDArrayType:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Shape, dtype: DType) -> NDArrayType:
        return np.ones(shape, dtype=dtype)

    def empty(self, shape: Shape, dtype: DType) -> NDArrayType:
        return np.empty(shape, dtype=dtype)

    def full(self, shape: Shape, fill_value: Scalar, dtype: DType) -> NDArrayType:
        return np.full(shape, fill_value, dtype=dtype)


# Devices


def cpu() -> AbstractBackend:
    """Return cpu device."""
    return NumpyBackend("numpy", sys.modules[__name__])  # type: ignore


def all_devices() -> list[AbstractBackend]:
    """Return a list of all available devices."""
    return [cpu()]


class NumpyCUDADevice:
    def __init__(self, name: str) -> None:
        self.name = name

    def enabled(self) -> bool:
        return False


def cuda() -> NumpyCUDADevice:
    # raise NotImplementedError("CUDA is not supported with numpy backend")
    return NumpyCUDADevice("np_dummy-cuda")


default_device = cpu()


# API functions


def from_numpy(a, out):
    out.array[:] = a.flatten()


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    out.array[:] = a.array + b.array


def scalar_add(a, val, out):
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    out.array[:] = np.power(a.array, val)


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    out.array[:] = np.tanh(a.array)


def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)
