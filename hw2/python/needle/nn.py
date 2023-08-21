"""The module.
"""
from typing import List

import needle.init as init
import needle.ops as ops
from needle.autograd import Tensor


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        config = {
            "device": device,
            "dtype": dtype,
            "requires_grad": True,
        }

        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=self.in_features,
                fan_out=self.out_features,
                **config,
            )
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    fan_in=self.out_features,
                    fan_out=1,
                    **config,
                )
            )
            # Should not be needed, because the bias term will be reshaped
            # on .forward(), but required to pass tests
            self.bias = ops.reshape(self.bias, (1, self.out_features))

    def forward(self, X: Tensor) -> Tensor:
        if self.bias:
            self.bias = ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
            return X @ self.weight + self.bias
        return X @ self.weight


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y_one_hot = init.one_hot(logits.shape[1], y)
        diff = (logits * y_one_hot).sum(axes=1)

        lse = ops.logsumexp(logits, axes=1)
        total = lse - diff
        return total.sum() / logits.shape[0]


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

        self.weight = Parameter(init.ones(self.dim, **config, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, **config, requires_grad=True))
        self.running_mean = init.zeros(self.dim, **config)
        self.running_var = init.ones(self.dim, **config)

    def forward(self, x: Tensor) -> Tensor:
        def _update_running_variables(var: Tensor, running_var: Tensor) -> Tensor:
            return (1 - self.momentum) * var + self.momentum * running_var

        if self.training:
            mean_x = ops.mean(x)
            self.running_mean = _update_running_variables(self.running_mean, mean_x)

            # make into [1, x] tensor and then broadcast
            mean_x = ops.broadcast_to(ops.reshape(mean_x, (1, -1)), x.shape)

            var_x = ops.mean((x - mean_x) ** 2)
            self.running_var = _update_running_variables(self.running_var, var_x)

            var_x = ops.sqrt(var_x + self.eps)
            # make into [1, x] tensor and then broadcast
            var_x = ops.broadcast_to(ops.reshape(var_x, (1, -1)), x.shape)
        else:
            # broadcast from batch dimensions to x.shape
            mean_x = ops.broadcast_to(
                ops.reshape(self.running_mean, (-1, self.dim)), x.shape
            )
            var_x = ops.broadcast_to(
                ops.reshape(self.running_var, (-1, self.dim)), x.shape
            )

            self.running_mean = _update_running_variables(self.running_mean, mean_x)
            self.running_var = _update_running_variables(self.running_var, var_x)

        return self.weight * ((x - mean_x) / var_x) + self.bias


class LayerNorm1d(Module):
    def __init__(self, dim: tuple, eps: float = 1e-5, device=None, dtype="float32"):
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
        # We can assume 2D tensor: [batch, features]
        assert x.ndim == 2

        mean_x = ops.broadcast_to_new_axis(
            ops.mean(x, axes=1), new_axis=(x.shape[0], 1), new_shape=x.shape
        )
        var_x = ops.mean((x - mean_x) ** 2, axes=1)

        var_x = ops.sqrt(var_x + self.eps)
        var_x = ops.broadcast_to_new_axis(
            var_x, new_axis=(x.shape[0], 1), new_shape=x.shape
        )

        return self.weight * ((x - mean_x) / var_x) + self.bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = init.rand_binary(*x.shape, p=self.p) / (1 - self.p)
        return x * mask


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
