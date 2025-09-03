from __future__ import annotations

from typing import TYPE_CHECKING

from needle import init
from needle.backend_selection import default_device
from needle.needle_typing import TensorKwargs
from needle.nn.core import Module, Parameter

if TYPE_CHECKING:
    from needle.needle_typing import AbstractBackend, DType
    from needle.tensor import Tensor


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        config = TensorKwargs(device=device, dtype=dtype, requires_grad=True)

        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=self.in_features,
                fan_out=self.out_features,
                **config,
            )
        )
        self.bias = (
            Parameter(
                init.kaiming_uniform(
                    fan_in=self.out_features,
                    fan_out=1,
                    **config,
                ).reshape((1, self.out_features))
            )
            if bias
            else None
        )

    def forward(self, X: Tensor) -> Tensor:
        X_weights = X @ self.weight
        if self.bias:
            return X_weights + self.bias.broadcast_to(X_weights.shape)
        return X_weights
