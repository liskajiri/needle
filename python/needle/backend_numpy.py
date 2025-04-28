"""
This file defies specific implementations of devices
when using numpy as NDArray backend.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from needle.typing.device import AbstractBackend

type NDArray = np.ndarray

if TYPE_CHECKING:
    from needle.typing import (
        DType,
        IndexType,
        Scalar,
        Shape,
    )


__device_name__ = "numpy"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


class Array:
    def __init__(self, size) -> None:
        self.array = np.empty(size, dtype=np.float32)

    @property
    def size(self) -> int:
        return self.array.size


def to_numpy(a, shape, strides, offset) -> NDArray:
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
    )


class NumpyBackend(AbstractBackend):
    def randn(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        return np.random.randn(*shape)

    def rand(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        return np.random.rand(*shape)

    def one_hot(self, n: int, i: IndexType, dtype: DType) -> NDArray:
        return np.eye(n, dtype=dtype)[i]

    def zeros(self, shape: Shape, dtype: DType) -> NDArray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Shape, dtype: DType) -> NDArray:
        return np.ones(shape, dtype=dtype)

    def empty(self, shape: Shape, dtype: DType) -> NDArray:
        return np.empty(shape, dtype=dtype)

    def full(self, shape: Shape, fill_value: Scalar, dtype: DType) -> NDArray:
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
