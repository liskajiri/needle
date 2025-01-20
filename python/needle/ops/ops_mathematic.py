"""Operator implementations."""

import needle
from needle.autograd import NDArray, Tensor, TensorOp, TensorTupleOp
from needle.backend_selection import array_api
from needle.ops.ops_tuple import make_tuple


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
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
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


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * a.log()
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


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
    def __init__(self, axes: tuple | None = None):
        if not axes:
            axes = (-1, -2)
        self.axes = axes

    def compute(self, a: NDArray):
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

    def compute(self, a: NDArray):
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
    def __init__(self, axes: tuple | None = None):
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
    def compute(self, a: NDArray):
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


def broadcast_to_new_axis(x: Tensor, new_axis: tuple, new_shape: tuple) -> Tensor:
    new_axes = tuple(1 if i in new_axis else ax for i, ax in enumerate(new_shape))
    return broadcast_to(reshape(x, new_axes), new_shape)


class SquareRoot(TensorOp):
    def compute(self, a: NDArray):
        return array_api.sqrt(a)

    def gradient(self, out_grad, node):
        return out_grad / (2 * node.inputs[0])


def sqrt(x: Tensor) -> Tensor:
    return x**0.5


def mean(a: Tensor, axes=0) -> Tensor:
    return summation(a, axes=axes) / a.shape[axes]


class Tanh(TensorOp):
    def compute(self, a: NDArray):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return out_grad.cached_data * (1 - node.cached_data**2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: list[NDArray]) -> Tensor:
        # # Convert tuple of tensors to list of arrays
        # arrays = [array for array in args]
        # # Stack arrays along specified axis
        # stacked_array = array_api.stack(arrays, axis=self.axis)
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))

        new_arr = needle.init.zeros(new_shape)
        for i, array in enumerate(args):
            slices = [slice(None)] * len(new_shape)
            slices[self.axis] = i
            new_arr[tuple(slices)] = array
        return new_arr

    def gradient(self, out_grad, node):
        # Split gradient along stacking axis to get individual gradients
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        split_size = A.shape[self.axis]
        return array_api.split(A, split_size, self.axis)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: int | None = 1, padding: int | None = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
