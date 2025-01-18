from typing import Optional

from .ops_mathematic import broadcast_to_new_axis, exp, summation
from ..autograd import NDArray, Tensor, Value, TensorOp

import numpy as array_api
# # TODO: 2024 version
# from ..backend_selection import array_api


# TODO: not covered by current tests - Convert to 2024 edition
class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray):
        # 2d input array
        ### BEGIN YOUR SOLUTION
        # Subtract the maximum value along axis=1 for numerical stability
        print("Shape: ", Z.shape)
        assert Z.ndim == 2, "Input must be a 2D array"
        max_Z = array_api.max(Z, axis=1, keepdims=True)
        shifted_Z = Z - max_Z

        # Compute log-softmax
        log_sum_exp = array_api.log(
            array_api.sum(array_api.exp(shifted_Z), axis=1, keepdims=True)
        )
        return shifted_Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        print(node, node.inputs[0])
        # Z = Z.numpy()
        log_softmax_Z = self.compute(Z)

        # Compute softmax values from log-softmax
        softmax_Z = array_api.exp(log_softmax_Z)

        # Gradient calculation
        sum_out_grad = array_api.sum(out_grad, axis=1, keepdims=True)
        return out_grad - sum_out_grad * softmax_Z
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


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
