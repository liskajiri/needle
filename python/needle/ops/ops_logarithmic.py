from typing import TYPE_CHECKING

from needle.backend_selection import NDArray, array_api
from needle.ops.op import TensorOp
from needle.ops.ops_mathematic import broadcast_to_new_axis, exp, summation
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.autograd.value import Value


# TODO: not covered by current tests - Convert to 2024 edition
class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # 2d input array
        # Subtract the maximum value along axis=1 for numerical stability
        assert Z.ndim == 2, "Input must be a 2D array"
        max_Z = array_api.max(Z, axis=1, keepdims=True)
        shifted_Z = Z - max_Z

        # Compute log-softmax
        log_sum_exp = array_api.log(
            array_api.sum(array_api.exp(shifted_Z), axis=1, keepdims=True)
        )
        return shifted_Z - log_sum_exp

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        log_softmax_Z = self.compute(Z)

        # Compute softmax values from log-softmax
        softmax_Z = array_api.exp(log_softmax_Z)

        # Gradient calculation
        sum_out_grad = array_api.sum(out_grad, axis=1, keepdims=True)
        return out_grad - sum_out_grad * softmax_Z


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: tuple | None = None) -> None:
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        """
        From the definition of LogSumExp:
        log(sum(exp(Z))) = log(exp(Z - max(Z)) * sum(exp(max(Z)))
        = log(exp(Z - max(Z))) + log(sum(exp(max(Z)))
        = Z - max(Z) + log(sum(exp(max(Z)))
        """
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        Z = Z - array_api.broadcast_to(max_Z, Z.shape)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z), axis=self.axes))
        return log_sum_exp + array_api.reshape(max_Z, log_sum_exp.shape)

    def gradient(self, out_grad: Tensor, node: "Value"):
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
