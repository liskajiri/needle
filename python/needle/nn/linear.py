from typing import TYPE_CHECKING, TypedDict

from needle import init
from needle.backend_ndarray.backend_numpy import default_device
from needle.backend_ndarray.device import AbstractBackend
from needle.nn.nn_basic import Module, Parameter
from needle.ops.ops_mathematic import broadcast_to
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.backend_ndarray.utils import DType


class Config(TypedDict):
    device: AbstractBackend
    dtype: "DType"
    requires_grad: bool


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: AbstractBackend = default_device(),
        dtype: "DType" = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        config = Config(device=device, dtype=dtype, requires_grad=True)

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
                ).reshape((1, self.out_features))
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        X_weights = X @ self.weight
        if self.bias:
            return X_weights + broadcast_to(self.bias, X_weights.shape)
        return X_weights
