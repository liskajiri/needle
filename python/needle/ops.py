"""Operator table."""

# Global operator table.
from typing import Optional

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api

from .autograd import NDArray, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value

# fixes imports in tests
___all__ = [
    "make_tuple",
    "summation" "tuple_get_item",
    "fused_add_scalars",
    "add",
    "add_scalar",
    "multiply",
    "mul_scalar",
    "power_scalar",
    "divide",
    "divide_scalar",
    "transpose",
    "reshape",
    "broadcast_to",
    "matmul",
    "negate",
    "log",
    "exp",
    "relu",
    "logsumexp",
    "softmax",
    "sqrt",
    "mean",
    "broadcast_to_new_axis",
]


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(array_api.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        child = node.inputs[0]
        return self.scalar * out_grad * child ** (self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (
            divide(out_grad, rhs),
            divide(-out_grad * lhs, rhs**2),
        )


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return divide_scalar(out_grad, self.scalar)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if not axes:
            axes = (-1, -2)
        self.axes = axes

    def compute(self, a):
        permutation = [i for i in range(len(a.shape))]
        lhs, rhs = self.axes
        permutation[lhs], permutation[rhs] = (
            rhs,
            lhs,
        )
        return array_api.transpose(a, permutation)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # [7, 7, 7] -> [1, 7]
        in_shape = node.inputs[0].shape

        if len(in_shape) != out_grad.ndim:
            # expand curr out_grad
            # [1, 7] -> [1, 1, 7]
            new_axes = tuple([1] * (out_grad.ndim - len(in_shape)) + list(in_shape))
        else:
            new_axes = in_shape

        different_axes = tuple(i for i, ax in enumerate(new_axes) if ax == 1)
        out_grad = summation(out_grad, axes=different_axes).reshape(in_shape)
        return out_grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Function from (m, ) -> (m, 1) -> (m, n)
        target_shape = node.inputs[0].shape
        if self.axes is None:
            return broadcast_to(out_grad, target_shape)
        # adds new axes: (m, ) -> (m, 1)
        return broadcast_to_new_axis(out_grad, self.axes, target_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        grad_lhs = matmul(out_grad, rhs.T)
        grad_rhs = matmul(lhs.T, out_grad)
        if grad_lhs.shape != lhs.shape:
            n_axes = grad_lhs.ndim - lhs.ndim
            grad_lhs = summation(grad_lhs, tuple(range(n_axes)))
        if grad_rhs.shape != rhs.shape:
            n_axes = grad_rhs.ndim - rhs.ndim
            grad_rhs = summation(grad_rhs, tuple(range(n_axes)))
        return grad_lhs, grad_rhs


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: Tensor):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        # cannot be differentiated twice, so calling cached_data is ok
        return out_grad * (node.inputs[0].realize_cached_data() > 0)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z: NDArray):
        max_Z = array_api.max(Z, axis=self.axes)

        if self.axes is None:
            max_Z_expanded = array_api.broadcast_to(max_Z, Z.shape)
        else:
            # tensor cannot be broadcasted without proper dimensions,
            # so we need to add axis to max_Z
            new_axes = tuple(
                [1 if i in self.axes else ax for i, ax in enumerate(Z.shape)]
            )
            out_grad = array_api.reshape(max_Z, new_axes)
            max_Z_expanded = array_api.broadcast_to(out_grad, Z.shape)

        e = array_api.exp(Z - max_Z_expanded).sum(axis=self.axes)
        return array_api.log(e) + max_Z

    def gradient(self, out_grad: Tensor, node: Value):
        # gradient of LogSumExp is softmax
        Z = node.inputs[0]
        max_Z = array_api.max(Z.cached_data, axis=self.axes, keepdims=True)
        numerator = exp(Z - max_Z)
        denominator = summation(numerator, axes=self.axes)
        # denominator has a different shape than numerator
        # so we need to add axes to denominator

        target_shape = Z.shape
        if self.axes:
            out_grad = broadcast_to_new_axis(out_grad, self.axes, target_shape)
            denominator = broadcast_to_new_axis(denominator, self.axes, target_shape)

        return out_grad * (numerator / denominator)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


def softmax(Z: Tensor, axes=None) -> Tensor:
    # Numerically stable softmax
    # ! Uses Z.cached_data
    max_Z = array_api.max(Z.cached_data, axes, keepdims=True)
    numerator = exp(Z - max_Z)
    denominator = summation(numerator, axes)
    return numerator / denominator


def broadcast_to_new_axis(x: Tensor, new_axis: tuple, new_shape: tuple) -> Tensor:
    new_axes = tuple(1 if i in new_axis else ax for i, ax in enumerate(new_shape))
    return broadcast_to(reshape(x, new_axes), new_shape)


def sqrt(x: Tensor) -> Tensor:
    return x**0.5


def mean(a: Tensor, axes=0) -> Tensor:
    return summation(a, axes=axes) / a.shape[axes]
