"""
This file defies specific implementations of devices
when using numpy as NDArray backend.
"""

import numpy
import numpy as np

from .backend_ndarray.ndarray_backend_numpy import Array, _datetype_size


class Device:
    """Baseclass of all device."""


class CPUDevice(Device):
    """Represents data that sits in CPU."""

    def __repr__(self):
        return "needle.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.randn(*shape)

    def rand(self, *shape):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]

    def empty(self, shape, dtype="float32"):
        return numpy.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return numpy.full(shape, fill_value, dtype=dtype)

    # MOVED from Array
    @property
    def Array(self):
        return Array

    def to_numpy(self, a, shape, strides, offset):
        return np.lib.stride_tricks.as_strided(
            a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
        )

    @staticmethod
    def from_numpy(a, out):
        out.array[:] = a.flatten()

    def fill(self, out, val):
        out.array.fill(val)

    def compact(self, a, out, shape, strides, offset):
        out.array[:] = self.to_numpy(a, shape, strides, offset).flatten()

    def ewise_setitem(self, a, out, shape, strides, offset):
        self.to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)

    def scalar_setitem(self, size, val, out, shape, strides, offset):
        self.to_numpy(out, shape, strides, offset)[:] = val

    def ewise_add(self, a, b, out):
        out.array[:] = a.array + b.array

    def scalar_add(self, a, val, out):
        out.array[:] = a.array + val

    def ewise_mul(self, a, b, out):
        out.array[:] = a.array * b.array

    def scalar_mul(self, a, val, out):
        out.array[:] = a.array * val

    def ewise_div(self, a, b, out):
        out.array[:] = a.array / b.array

    def scalar_div(self, a, val, out):
        out.array[:] = a.array / val

    def scalar_power(self, a, val, out):
        out.array[:] = np.power(a.array, val)

    def ewise_maximum(self, a, b, out):
        out.array[:] = np.maximum(a.array, b.array)

    def scalar_maximum(self, a, val, out):
        out.array[:] = np.maximum(a.array, val)

    def ewise_eq(self, a, b, out):
        out.array[:] = (a.array == b.array).astype(np.float32)

    def scalar_eq(self, a, val, out):
        out.array[:] = (a.array == val).astype(np.float32)

    def ewise_ge(self, a, b, out):
        out.array[:] = (a.array >= b.array).astype(np.float32)

    def scalar_ge(self, a, val, out):
        out.array[:] = (a.array >= val).astype(np.float32)

    def ewise_log(self, a, out):
        out.array[:] = np.log(a.array)

    def ewise_exp(self, a, out):
        out.array[:] = np.exp(a.array)

    def ewise_tanh(self, a, out):
        out.array[:] = np.tanh(a.array)

    def matmul(self, a, b, out, m, n, p):
        out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)

    def reduce_max(self, a, out, reduce_size):
        out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)

    def reduce_sum(self, a, out, reduce_size):
        out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)

    def broadcast_to(self, a, shape, out):
        out.array[:] = np.broadcast_to(a.array, shape)


def cpu():
    """Return cpu device."""
    return CPUDevice()


def default_device():
    return cpu()


def all_devices():
    """Return a list of all available devices."""
    return [cpu()]


class CUDANotSupportedError(NotImplementedError):
    """CUDA is not supported in numpy backend"""

    pass


def cuda():
    raise CUDANotSupportedError("CUDA is not supported in numpy backend")
