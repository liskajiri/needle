from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_selection import array_api
from needle.ops.mathematic import (
    broadcast_to_new_axis,
    exp,
    summation,
)
from needle.ops.op import TensorOp

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import Tensor
    from needle.typing.types import Shape


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # 2d input array
        # Subtract the maximum value along axis=1 for numerical stability
        assert Z.ndim == 2, (
            f"Input must be a 2D array, but got array with shape: {Z.shape}"
        )
        max_Z = array_api.broadcast_to(array_api.max(Z, axis=1, keepdims=True), Z.shape)
        shifted_Z = Z - max_Z
        # Compute log-softmax
        sum_exp = array_api.sum(array_api.exp(shifted_Z), axis=1, keepdims=True)
        return shifted_Z - array_api.broadcast_to(
            array_api.log(sum_exp), shifted_Z.shape
        )  # .broadcast_to(shifted_Z.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        Z = node.inputs[0].realize_cached_data()

        # Compute softmax values from log-softmax
        softmax_Z = array_api.exp(self.compute(Z))
        # Gradient calculation
        sum_out_grad = array_api.sum(
            out_grad.realize_cached_data(), axis=(1,), keepdims=True
        )
        return out_grad - sum_out_grad * softmax_Z


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Shape | None = None) -> None:
        self.axes = (axes,) if isinstance(axes, int) else axes

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

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # gradient of LogSumExp is softmax
        Z = node.inputs[0]
        max_Z = array_api.max(Z.cached_data, axis=self.axes, keepdims=True)
        exp_Z = exp(Z - max_Z)
        denominator = summation(exp_Z, axes=self.axes)
        # denominator has a different shape than numerator
        # so we need to add axes to denominator

        target_shape = Z.shape
        if self.axes:
            out_grad = broadcast_to_new_axis(out_grad, self.axes, target_shape)
            denominator = broadcast_to_new_axis(denominator, self.axes, target_shape)

        return out_grad * (exp_Z / denominator)


def logsumexp(a: Tensor, axes: Shape | None = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
