from __future__ import annotations

from typing import TYPE_CHECKING

import needle.init as init
from needle.backend_selection import NDArray, array_api
from needle.ops.op import TensorOp, TensorTupleOp
from needle.ops.ops_tuple import make_tuple
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import TensorTuple


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


def stack(arr: tuple[Tensor, ...] | list[Tensor], axis: int) -> Tensor:
    return Stack(axis)(make_tuple(*arr))


class Split(TensorTupleOp):
    def __init__(self, axis: int, sections: int | list[int] | None = None) -> None:
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        sections - number of sections to split into
        """
        self.axis = axis
        self.sections = sections

    def compute(self, arr: NDArray) -> list[NDArray]:
        return array_api.split(arr, self.sections, self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # return concatenate(out_grad, self.axis)
        return Stack(self.axis)(out_grad)


def split(
    arr: Tensor, axis: int = 0, sections: int | list[int] | None = None
) -> TensorTuple:
    return Split(axis, sections)(arr)


class Concatenate(TensorOp):
    def __init__(self, axis: int) -> None:
        """
        Concatenates a sequence of arrays along an existing dimension.
        Parameters:
        axis - dimension to concatenate along
        Arrays need to have matching shapes except in the concat dimension.
        """
        self.axis = axis

    def compute(self, arr: tuple[NDArray]) -> NDArray:
        return array_api.concatenate(arr, self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor) -> TensorTuple:
        input_sizes = [array.shape[self.axis] for array in node.inputs]
        return Split(self.axis, sections=input_sizes)(out_grad)


def concatenate(arr: tuple[Tensor, ...] | list[Tensor], axis: int) -> Tensor:
    return Concatenate(axis)(make_tuple(*arr))


class Flip(TensorOp):
    def __init__(self, axes: tuple[int, ...] | int) -> None:
        self.axes = axes

    def compute(self, arr: NDArray) -> NDArray:
        return array_api.flip(arr, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return flip(out_grad, self.axes)


def flip(arr: Tensor, axes: tuple[int, ...] | int) -> Tensor:
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

    def __init__(self, axes: tuple[int, ...], dilation: int = 0) -> None:
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


def dilate(arr: Tensor, axes: tuple[int, ...], dilation: int = 0) -> Tensor:
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
    return Dilate(axes, dilation)(arr)


class UnDilate(TensorOp):
    """
    Reverse operation to Dilate.
    """

    def __init__(self, axes: tuple[int, ...], dilation: int) -> None:
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


def undilate(a: Tensor, axes: tuple[int, ...], dilation: int) -> Tensor:
    return UnDilate(axes, dilation)(a)


class GetItem(TensorOp):
    def __init__(self, index) -> None:
        self.index = index

    def _convert_to_numpy_index(self, index) -> NDArray:
        """Convert tensor indices to numpy arrays for indexing"""
        if isinstance(index, (Tensor | NDArray)):
            return index.numpy()
        elif isinstance(index, tuple):
            return tuple(self._convert_to_numpy_index(idx) for idx in index)
        return index

    def compute(self, a: NDArray) -> NDArray:
        # Convert any tensor indices to numpy arrays
        numpy_index = self._convert_to_numpy_index(self.index)
        return a[numpy_index]

    def gradient(self, out_grad, node) -> Tensor:
        input_shape = node.inputs[0].shape

        # Create a zero gradient with the shape of input
        grad = init.zeros(input_shape, device=out_grad.device, dtype=out_grad.dtype)

        grad_tensor = Tensor(grad, device=out_grad.device, requires_grad=False)
        grad_tensor[self.index] = out_grad

        return grad_tensor


def get_item(a, index) -> Tensor:
    return GetItem(index)(a)
