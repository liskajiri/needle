"""Operator implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from needle.backend_selection import array_api
from needle.ops.op import TensorOp

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import Tensor
    from needle.typing import Scalar, Shape

logger = logging.getLogger(__name__)

# TODO: split ops:
# - ewise / scalar
# - transformations
# pure math
# specialized


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        return out_grad, out_grad


def add(a: Tensor, b: Tensor) -> Tensor:
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar: Scalar) -> None:
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad


def add_scalar(a: Tensor, scalar: Scalar) -> Tensor:
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar: Scalar) -> None:
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * self.scalar


def mul_scalar(a: Tensor, scalar: Scalar) -> Tensor:
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * a.log()
        return grad_a, grad_b


def power(a: Tensor, b: Tensor) -> Tensor:
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: Scalar) -> None:
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        child = node.inputs[0]
        return self.scalar * out_grad * child ** (self.scalar - 1)


def power_scalar(a: Tensor, scalar: Scalar) -> Tensor:
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return (
            divide(out_grad, rhs),
            divide(-out_grad * lhs, rhs**2),
        )


def divide(a: Tensor, b: Tensor) -> Tensor:
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    """
    Divide a tensor by a scalar.
    """

    def __init__(self, scalar: float) -> None:
        if scalar == 0:
            raise ValueError("Cannot divide by 0")
        if isinstance(scalar, int):
            scalar = float(scalar)

        self.scalar = float(scalar)

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / self.scalar


def divide_scalar(a: Tensor, scalar: float) -> Tensor:
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: tuple = (-1, -2)) -> None:
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


def transpose(a: Tensor, axes: tuple = ()) -> Tensor:
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
            out_grad = summation(out_grad, axes=axes_to_reduce, keepdims=True)

        if output_shape != input_shape:
            return reshape(out_grad, input_shape)
        return out_grad


def broadcast_to(a: Tensor, shape: Shape) -> Tensor:
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: tuple | int | None = None, keepdims: bool = False) -> None:
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
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


def summation(
    a: Tensor, axes: tuple | int | None = None, keepdims: bool = False
) -> Tensor:
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
    def compute(self, a: NDArray):
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return negate(out_grad)


def negate(a: Tensor) -> Tensor:
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray) -> array_api.NDArray:
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
        return out_grad * (node.inputs[0].realize_cached_data() > 0)


def relu(a: Tensor) -> Tensor:
    return ReLU()(a)


def broadcast_to_new_axis(x: Tensor, new_axis: tuple, new_shape: tuple) -> Tensor:
    new_axes = tuple(1 if i in new_axis else ax for i, ax in enumerate(new_shape))
    return broadcast_to(reshape(x, new_axes), new_shape)


class SquareRoot(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a**0.5

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / (2 * node.inputs[0])


def sqrt(x: Tensor) -> Tensor:
    return x**0.5


def mean(a: Tensor, axes: int = 0) -> Tensor:
    return summation(a, axes=axes) / a.shape[axes]


class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.tanh(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        tanh = array_api.tanh(node.inputs[0].realize_cached_data())
        return out_grad * (1 - tanh**2)


def tanh(a: Tensor) -> Tensor:
    return Tanh()(a)


class Sigmoid(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return 1.0 / (1.0 + array_api.exp(-a))

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        sigmoid = 1.0 / (1.0 + array_api.exp(-node.inputs[0].realize_cached_data()))
        return out_grad * sigmoid * (1 - sigmoid)


def sigmoid(a: Tensor) -> Tensor:
    return Sigmoid()(a)
