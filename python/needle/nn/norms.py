"""The module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from needle import init, ops
from needle.backend_selection import default_device
from needle.nn.core import Module, Parameter
from needle.typing import TensorKwargs

if TYPE_CHECKING:
    from needle.tensor import Tensor
    from needle.typing import AbstractBackend, DType


class BatchNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        """
        Initialize BatchNorm1d module.

        Args:
            dim: Number of features/channels to normalize
            eps: Small constant for numerical stability
            momentum: Factor for running statistics update
            device: Device to place tensors on
            dtype: Data type for parameters
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        trainable_config = TensorKwargs(device=device, dtype=dtype, requires_grad=True)
        shape = (self.dim,)
        self.weight = Parameter(init.ones(shape, **trainable_config))
        self.bias = Parameter(init.zeros(shape, **trainable_config))

        non_trainable_config = TensorKwargs(
            device=device, dtype=dtype, requires_grad=False
        )
        self.running_mean = init.zeros(shape, **non_trainable_config)
        self.running_var = init.ones(shape, **non_trainable_config)

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

            std = ops.sqrt(var_x + self.eps).broadcast_to(x.shape)
            normalized = x_less_mean / std
        else:
            # we don't update the running vars at inference time
            running_mean = self.running_mean.broadcast_to(x.shape)
            running_std = ops.sqrt(self.running_var + self.eps).broadcast_to(x.shape)
            normalized = (x - running_mean) / running_std
        return weights * normalized + biases


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        # format: NCHW -> NHCW -> NHWC
        batch_size, channels, height, width = x.shape
        x = ops.transpose(x, (0, 2, 3, 1)).reshape(
            (
                batch_size * height * width,
                channels,
            )
        )
        y = super().forward(x).reshape((batch_size, height, width, channels))
        return y.transpose((0, 3, 1, 2))


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

        config = TensorKwargs(device=device, dtype=dtype, requires_grad=True)

        self.weight = Parameter(init.ones((self.dim,), **config))
        self.bias = Parameter(init.zeros((self.dim,), **config))

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
