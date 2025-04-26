"""Shape-changing operators implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_selection import array_api

# from needle.ops.mathematic import summation
from needle.ops.op import TensorOp

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import Tensor
    from needle.typing import Shape


class Transpose(TensorOp):
    def __init__(self, axes: Shape = (-1, -2)) -> None:
        self.axes = axes if axes else (-1, -2)

    def compute(self, a: NDArray) -> NDArray:
        axes = tuple(ax if ax >= 0 else a.ndim + ax for ax in self.axes)
        if not all(0 <= ax < a.ndim for ax in axes):
            raise ValueError(f"Axes out of range for array of dimension {a.ndim}")
        if len(self.axes) == a.ndim:
            return array_api.transpose(a, self.axes)
        if len(self.axes) == 2:
            # swap two axes
            permutation = list(range(len(a.shape)))
            lhs, rhs = self.axes
            permutation[lhs], permutation[rhs] = (
                rhs,
                lhs,
            )
            return array_api.transpose(a, tuple(permutation))
        raise ValueError(f"Invalid axes: {self.axes}")

    def gradient(self, out_grad: Tensor, _node: Tensor) -> Tensor:
        """
        Apply the inverse transpose to the gradient.
        """
        if len(self.axes) == 2 or not self.axes:
            return transpose(out_grad, self.axes)

        # Compute the inverse permutation
        inverse_axes = tuple(self.axes.index(i) for i in range(len(self.axes)))
        return transpose(out_grad, inverse_axes)


def transpose(a: Tensor, axes: Shape = ()) -> Tensor:
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape: Shape) -> None:
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a: Tensor, shape: Shape) -> Tensor:
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape: Shape) -> None:
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # [7, 7, 7] -> [1, 7]
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape

        if len(input_shape) != out_grad.ndim:
            # expand curr out_grad
            # [1, 7] -> [1, 1, 7]
            new_axes = (1,) * (out_grad.ndim - len(input_shape)) + input_shape
        else:
            new_axes = input_shape

        axes_to_reduce = tuple(i for i, ax in enumerate(new_axes) if ax == 1)
        if axes_to_reduce:
            out_grad = out_grad.sum(axes=axes_to_reduce, keepdims=True)
            # out_grad = summation(out_grad, axes=axes_to_reduce, keepdims=True)

        if output_shape != input_shape:
            return reshape(out_grad, input_shape)
        return out_grad


def broadcast_to(a: Tensor, shape: Shape) -> Tensor:
    return BroadcastTo(shape)(a)


def broadcast_to_new_axis(x: Tensor, new_axis: Shape, new_shape: Shape) -> Tensor:
    new_axes = tuple(1 if i in new_axis else ax for i, ax in enumerate(new_shape))
    return broadcast_to(reshape(x, new_axes), new_shape)
