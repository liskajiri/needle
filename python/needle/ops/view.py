from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_selection import NDArray, array_api
from needle.ops.op import TensorOp, TensorTupleOp
from needle.ops.ops_tuple import make_tuple
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import Tensor, TensorTuple


class Stack(TensorOp):
    def __init__(self, axis: int) -> None:
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, arr: tuple[NDArray]) -> NDArray:
        return array_api.stack(arr, self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor) -> TensorTuple:
        return Split(self.axis)(out_grad)


def stack(arr: list[Tensor], axis: int) -> Tensor:
    return Stack(axis)(make_tuple(*arr))


class Split(TensorTupleOp):
    def __init__(self, axis: int) -> None:
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, arr: NDArray) -> tuple[NDArray]:
        return tuple(array_api.split(arr, self.axis))

    def gradient(self, out_grad, node: Tensor) -> Tensor:
        return Stack(self.axis)(out_grad)


def split(arr: Tensor, axis: int) -> TensorTuple:
    return Split(axis)(arr)


# TODO: not an op
def array_split(
    arr: NDArray, indices_or_sections: int | list[int], axis: int = 0
) -> list[NDArray]:
    return array_api.array_split(arr, indices_or_sections, axis)


class Flip(TensorOp):
    def __init__(self, axes: tuple[int] | int) -> None:
        self.axes = axes

    def compute(self, arr: NDArray) -> NDArray:
        return array_api.flip(arr, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return flip(out_grad, self.axes)


def flip(arr: Tensor, axes: tuple[int] | int) -> Tensor:
    return Flip(axes)(arr)


class Dilate(TensorOp):
    """
    Dilates a tensor along the given axes by inserting zeros between each element.

    math::
        \\begin{bmatrix}
        1 & 2 \\
        3 & 4
        \\end{bmatrix}
        \\Longrightarrow
        \\begin{bmatrix}
        1 & 0 & 2 & 0 \\
        0 & 0 & 0 & 0 \\
        3 & 0 & 4 & 0 \\
        0 & 0 & 0 & 0
        \\end{bmatrix}
    """

    def __init__(self, axes: tuple[int], dilation: int) -> None:
        self.axes = axes
        self.dilation = dilation

    def compute(self, arr: NDArray) -> NDArray:
        new_shape = list(arr.shape)
        for axis in self.axes:
            new_shape[axis] += new_shape[axis] * self.dilation

        new_arr = array_api.zeros(tuple(new_shape), arr.dtype, arr.device)

        indices = [slice(None)] * len(new_shape)
        # slice in the given axis
        for axis in self.axes:
            indices[axis] = slice(0, None, self.dilation + 1)
        indices = tuple(indices)

        new_arr[indices] = arr
        return new_arr

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return undilate(out_grad, self.axes, self.dilation)


def dilate(arr: Tensor, axes: tuple[int], dilation: int) -> Tensor:
    dilate.__doc__ = Dilate.__doc__
    return Dilate(axes, dilation)(arr)


class UnDilate(TensorOp):
    """
    Reverse operation to Dilate.
    """

    def __init__(self, axes: tuple[int], dilation: int) -> None:
        self.axes = axes
        self.dilation = dilation

    def compute(self, arr: NDArray) -> NDArray:
        new_shap = list(arr.shape)
        for axis in self.axes:
            new_shap[axis] = new_shap[axis] // (self.dilation + 1)

        # TODO: does not have to reallocate memory - probably enough to shrink size and
        # move around values
        new_arr = array_api.zeros(tuple(new_shap), arr.dtype, arr.device)

        indices = [slice(None)] * len(new_shap)
        for axis in self.axes:
            indices[axis] = slice(0, None, self.dilation + 1)
        indices = tuple(indices)

        new_arr = arr[indices]
        return new_arr

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a: Tensor, axes: tuple[int], dilation: int) -> Tensor:
    return UnDilate(axes, dilation)(a)
