"""Operator implementations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from needle.backend_ndarray.ndarray import NDArray
from needle.backend_selection import array_api
from needle.needle_typing.types import Axis
from needle.ops.op import TensorOp
from needle.ops.shape import broadcast_to, broadcast_to_new_axis
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.backend_selection import NDArray


class Summation(TensorOp):
    def __init__(self, axes: Axis | None = None, keepdims: bool = False) -> None:
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a: NDArray) -> NDArray:
        return array_api.sum(a, axis=self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # Function from (m, ) -> (m, 1) -> (m, n)
        target_shape = node.inputs[0].shape
        if self.axes is None:
            return broadcast_to(out_grad, target_shape)
        # adds new axes: (m, ) -> (m, 1)
        return broadcast_to_new_axis(out_grad, self.axes, target_shape)
        # for axis in sorted(self.axes):
        #     out_grad = reshape(
        #         out_grad,
        #         list(out_grad.shape[:axis]) + [1] + list(out_grad.shape[axis:]),
        #     )
        # return broadcast_to(out_grad, target_shape)


def summation(a: Tensor, axes: Axis | None = None, keepdims: bool = False) -> Tensor:
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        grad_lhs = matmul(out_grad, rhs.T)
        grad_rhs = matmul(lhs.T, out_grad)
        # Broadcasting and extra dimensions
        if len(grad_lhs.shape) > len(lhs.shape):
            n_axes = grad_lhs.ndim - lhs.ndim
            grad_lhs = summation(grad_lhs, tuple(range(n_axes)))
        if len(grad_rhs.shape) > len(rhs.shape):
            n_axes = grad_rhs.ndim - rhs.ndim
            grad_rhs = summation(grad_rhs, tuple(range(n_axes)))
        return grad_lhs, grad_rhs


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return negate(out_grad)


def negate(a: Tensor) -> Tensor:
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / node.inputs[0]


def log(a: Tensor) -> Tensor:
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * exp(node.inputs[0])


def exp(a: Tensor) -> Tensor:
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # cannot be differentiated twice, so calling cached_data is ok
        return Tensor(
            out_grad.realize_cached_data() * (node.inputs[0].realize_cached_data() > 0)
        )


def relu(a: Tensor) -> Tensor:
    return ReLU()(a)


class SquareRoot(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        min = array_api.min(a).item()
        if min < 0.0:
            raise ValueError(
                f"Square root of negative number is not supported., got {a} with {min=}"
            )

        return a**0.5

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / (2 * node.inputs[0] ** 0.5)


def sqrt(x: Tensor) -> Tensor:
    return SquareRoot()(x)


class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.tanh(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        tanh = array_api.tanh(node.inputs[0].realize_cached_data())
        return Tensor(out_grad.realize_cached_data() * (1 - tanh**2))


def tanh(a: Tensor) -> Tensor:
    return Tanh()(a)


class Sigmoid(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return 1.0 / (1.0 + array_api.exp(-a))

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        sigmoid = 1.0 / (1.0 + array_api.exp(-node.inputs[0].realize_cached_data()))
        return Tensor(out_grad.realize_cached_data() * sigmoid * (1 - sigmoid))


def sigmoid(a: Tensor) -> Tensor:
    return Sigmoid()(a)


def mean(a: Tensor, axes: Axis | None = None) -> Tensor:
    if axes is None:
        axes = tuple(range(a.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    axes_size = math.prod(a.shape[axis] for axis in axes)
    return summation(a, axes=axes) / axes_size
