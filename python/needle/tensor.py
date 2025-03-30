"""Core data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import needle as ndl
from needle.autograd.value import Value
from needle.backend_selection import NDArray, array_api, default_device

if TYPE_CHECKING:
    from typing import Self

    from needle.typing import AbstractBackend as Device
    from needle.typing import DType, IndexType, Scalar, Shape, np_ndarray


class Tensor(Value):
    grad: Tensor

    def __init__(
        self,
        array: np_ndarray | NDArray | Tensor,
        device: Device = default_device,
        dtype: DType = "float32",
        requires_grad: bool = True,
    ) -> None:
        if isinstance(array, Tensor):
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        elif isinstance(array, NDArray):
            cached_data = array
        else:
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        super()._init(cached_data=cached_data, requires_grad=requires_grad)

    @classmethod
    def make_const(
        cls: type[Self], data: NDArray, requires_grad: bool = False
    ) -> Tensor:
        return super().make_const(data, requires_grad=requires_grad)

    @staticmethod
    def _array_from_numpy(
        numpy_array: np_ndarray, device: Device, dtype: DType
    ) -> NDArray:
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @property
    def data(self) -> Tensor:
        return self.detach()

    @data.setter
    def data(self, value: Tensor) -> None:
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, f"{value.dtype} {self.dtype}"
        self.cached_data = value.realize_cached_data()

    def detach(self) -> Tensor:
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self) -> Shape:
        return self.realize_cached_data().shape

    @property
    def dtype(self) -> DType:
        return self.realize_cached_data().dtype

    @property
    def device(self) -> Device:
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        return data.device

    def backward(self, out_grad: Tensor | None = None) -> None:
        out_grad = (
            out_grad
            if out_grad
            else ndl.init.init_basic.ones(
                self.shape, dtype=self.dtype, device=self.device
            )
        )
        ndl.autograd.compute_gradient_of_variables(self, out_grad)

    def __repr__(self) -> str:
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self) -> str:
        return self.realize_cached_data().__str__()

    def numpy(self) -> np_ndarray:
        return self.realize_cached_data().numpy()

    def __getitem__(self, index: IndexType) -> Tensor:
        if isinstance(index, Tensor):
            return ndl.ops.GetItem(index)(self)
        else:
            sliced_data = self.realize_cached_data()[index]

            # Create a new tensor with the sliced data, detached from the AD graph
            return Tensor(
                sliced_data,
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )

    def __setitem__(self, index: IndexType, value: NDArray | Scalar) -> None:
        self.realize_cached_data()[index] = value

    # Arithmetic operations

    # TODO: remove class instances, call functions directly
    def __add__(self, other: Tensor | Scalar) -> Tensor:
        if isinstance(other, Tensor):
            return ndl.ops.EWiseAdd()(self, other)
        return ndl.ops.AddScalar(other)(self)

    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        if isinstance(other, Tensor):
            return ndl.ops.EWiseMul()(self, other)
        return ndl.ops.MulScalar(other)(self)

    def __pow__(self, other: Scalar) -> Tensor:
        return ndl.ops.PowerScalar(other)(self)

    def __sub__(self, other: Tensor | Scalar) -> Tensor:
        if isinstance(other, Tensor):
            return ndl.ops.EWiseAdd()(self, ndl.ops.Negate()(other))
        return ndl.ops.AddScalar(-other)(self)

    def __truediv__(self, other: Tensor | Scalar) -> Tensor:
        if isinstance(other, Tensor):
            return ndl.ops.EWiseDiv()(self, other)
        return ndl.ops.DivScalar(other)(self)

    def __matmul__(self, other: Tensor) -> Tensor:
        return ndl.ops.MatMul()(self, other)

    def matmul(self, other: Tensor) -> Tensor:
        return ndl.ops.MatMul()(self, other)

    def sum(self, axes=None) -> Tensor:
        return ndl.ops.Summation(axes)(self)

    def broadcast_to(self, shape: Shape) -> Tensor:
        return ndl.ops.BroadcastTo(shape)(self)

    def reshape(self, shape: Shape) -> Tensor:
        return ndl.ops.Reshape(shape)(self)

    def __neg__(self) -> Tensor:
        return ndl.ops.Negate()(self)

    def transpose(self, axes: tuple[int, ...] = ()) -> Tensor:
        return ndl.ops.Transpose(axes)(self)

    def flip(self, axes: tuple[int, ...]) -> Tensor:
        return ndl.ops.flip(self, axes)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    __radd__ = __add__
    __rmul__ = __mul__


class TensorTuple(Value):
    """
    Represent a tuple of tensors.

    To keep things simple, do not support nested tuples.
    """

    def __len__(self) -> int:
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int) -> Value:
        return ndl.ops.tuple_get_item(self, index)

    def tuple(self) -> tuple[Value | type[Value], ...]:
        return tuple([x for x in self])

    def __repr__(self) -> str:
        return "TensorTuple" + str(self.tuple())

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: TensorTuple) -> TensorTuple:
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        assert all(
            isinstance(self[i], Tensor) and isinstance(other[i], Tensor)
            for i in range(len(self))
        )
        return ndl.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])  # type: ignore

    def detach(self) -> TensorTuple:
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())
