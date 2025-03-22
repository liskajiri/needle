"""The module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from needle import init, ops
from needle.backend_selection import default_device
from needle.nn.core import Module, Parameter

if TYPE_CHECKING:
    from needle.tensor import Tensor
    from needle.typing import DType
    from needle.typing.device import AbstractBackend


class BatchNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        config = {
            "device": device,
            "dtype": dtype,
        }

        self.weight = Parameter(init.ones(self.dim, requires_grad=True, **config))
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True, **config))
        self.running_mean = init.zeros(self.dim, **config)
        self.running_var = init.ones(self.dim, **config)

    def forward(self, x: Tensor) -> Tensor:
        def _update_running_variables(var: Tensor, running_var: Tensor) -> Tensor:
            return (1 - self.momentum) * var + self.momentum * running_var

        mean_x = ops.mean(x)
        x_less_mean = x - mean_x.broadcast_to(x.shape)
        var_x = ops.mean(x_less_mean**2)

        weights = self.weight.broadcast_to(x.shape)
        biases = self.bias.broadcast_to(x.shape)

        if self.training:
            self.running_mean = _update_running_variables(self.running_mean, mean_x)
            self.running_var = _update_running_variables(self.running_var, var_x)
        else:
            # we don't update the running vars at inference time
            return (
                weights
                * ((x - self.running_mean) / ops.sqrt(self.running_var + self.eps))
                + biases
            )

        var_plus_eps = ops.sqrt(var_x + self.eps).broadcast_to(x.shape)
        return weights * (x_less_mean / var_plus_eps) + biases


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        # format: NCHW -> NHCW -> NHWC
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: AbstractBackend = default_device,
        dtype="float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

        config = {
            "device": device,
            "dtype": dtype,
            "requires_grad": True,
        }

        self.weight = Parameter(init.ones(self.dim, **config))
        self.bias = Parameter(init.zeros(self.dim, **config))

    def forward(self, x: Tensor) -> Tensor:
        # We can assume 2D tensor: (batch, features)
        assert x.ndim == 2

        # mean_x: (batch, )
        mean_x = ops.mean(x, axes=1)
        # cast to x.shape
        mean_x = ops.broadcast_to_new_axis(
            mean_x, new_axis=(x.shape[0], 1), new_shape=x.shape
        )

        # var_x: (batch, )
        var_x = ops.mean((x - mean_x) ** 2, axes=1)
        # cast to x.shape
        var_x = ops.broadcast_to_new_axis(
            var_x, new_axis=(x.shape[0], 1), new_shape=x.shape
        )

        weights = self.weight.broadcast_to(x.shape)
        biases = self.bias.broadcast_to(x.shape)
        return weights * ((x - mean_x) / ops.sqrt(var_x + self.eps)) + biases
