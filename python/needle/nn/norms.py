"""The module."""

from needle import init, ops
from needle.nn.nn_basic import Module, Parameter
from needle.tensor import Tensor

__all__ = [
    "BatchNorm1d",
    "LayerNorm1d",
]


class BatchNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype="float32",
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


class LayerNorm1d(Module):
    def __init__(
        self, dim: tuple, eps: float = 1e-5, device=None, dtype="float32"
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
