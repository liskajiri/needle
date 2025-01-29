"""Core data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import needle as ndl
from needle.autograd.value import Value
from needle.backend_ndarray.device import AbstractBackend
from needle.backend_selection import NDArray, array_api, default_device

if TYPE_CHECKING:
    from typing import Self

    from needle.backend_ndarray.device import AbstractBackend as Device
    from needle.typing.utils import DType


# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

type Scalar = int | float


class Tensor(Value):
    grad: Self

    def __init__(
        self,
        array: NDArray | Tensor,
        *,
        device: Device = default_device(),
        dtype: DType = "float32",
        requires_grad: bool = True,
        **kwargs,
    ) -> None:
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array: np.ndarray, device: Device, dtype: DType):
        if array_api is np:
            return np.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @classmethod
    def make_const(cls, data, *, requires_grad: bool = False) -> Self:
        if isinstance(data, Tensor):
            data = data.realize_cached_data()
        return super().make_const(data, requires_grad=requires_grad)

    @property
    def data(self) -> Tensor:
        return self.detach()

    @data.setter
    def data(self, value) -> None:
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, f"{value.dtype} {self.dtype}"
        self.cached_data = value.realize_cached_data()

    def detach(self) -> Tensor:
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self) -> AbstractBackend:
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        return data.device

    def backward(self, out_grad: Tensor | None = None) -> None:
        out_grad = (
            out_grad
            if out_grad
            else ndl.init.init_basic.ones(
                *self.shape, dtype=self.dtype, device=self.device
            )
        )
        ndl.autograd.compute_gradient_of_variables(self, out_grad)

    def __repr__(self) -> str:
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self) -> str:
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        # TODO: no need for this? numpy_api is patched
        if array_api is np:
            return data
        return data.numpy()

    def __add__(self, other: Tensor | Scalar) -> Tensor:
        if isinstance(other, Tensor):
            return ndl.ops.EWiseAdd()(self, other)
        return ndl.ops.AddScalar(other)(self)

    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        if isinstance(other, Tensor):
            return ndl.ops.EWiseMul()(self, other)
        return ndl.ops.MulScalar(other)(self)

    def __pow__(self, other):
        return ndl.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return ndl.ops.EWiseAdd()(self, ndl.ops.Negate()(other))
        return ndl.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return ndl.ops.EWiseDiv()(self, other)
        return ndl.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return ndl.ops.MatMul()(self, other)

    def matmul(self, other):
        return ndl.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return ndl.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return ndl.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return ndl.ops.Reshape(shape)(self)

    def __neg__(self):
        return ndl.ops.Negate()(self)

    def transpose(self, axes=None):
        return ndl.ops.Transpose(axes)(self)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    __radd__ = __add__
    __rmul__ = __mul__


class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self) -> int:
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return ndl.ops.tuple_get_item(self, index)

    def tuple(self) -> tuple[Self]:
        return tuple([x for x in self])
        # return tuple(self)

    def __repr__(self) -> str:
        return "TensorTuple" + str(self.tuple())

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: TensorTuple) -> TensorTuple:
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return ndl.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self) -> TensorTuple:
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())
