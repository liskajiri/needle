from typing import TypedDict

import needle as ndl
import needle.init as init
from needle import nn
from needle.backend_selection import default_device
from needle.nn.core import Parameter
from needle.tensor import Tensor
from needle.typing.device import AbstractBackend
from needle.typing.types import DType


def ResidualBlock(
    dim: int,
    hidden_dim: int,
    norm: nn.Module = nn.norms.BatchNorm1d,
    drop_prob: float = 0.1,
) -> nn.Module:
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )


def MLPResNet(
    dim: int,
    hidden_dim: int = 100,
    num_blocks: int = 3,
    num_classes: int = 10,
    norm: nn.Module = nn.norms.BatchNorm1d,
    drop_prob: float = 0.1,
) -> nn.Module:
    # important to use tuples and unpacking - won't work with lists
    residual_blocks = (
        ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
        for _ in range(num_blocks)
    )
    return nn.Sequential(*(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *residual_blocks,
        nn.Linear(hidden_dim, num_classes),
    ))


class Config(TypedDict):
    device: AbstractBackend
    dtype: DType


class ResNet9(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 10,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = Config(device=device, dtype=dtype)

        self.layers1 = nn.Sequential(
            self._make_conv_layer(in_features, 16, 7, 4),
            self._make_conv_layer(16, 32, 3, 2),
            # Residual out
        )
        self.layers2 = nn.Sequential(
            self._make_conv_layer(32, 32, 3, 1),
            self._make_conv_layer(32, 32, 3, 1),
            # Residual + in
        )
        self.layers3 = nn.Sequential(
            self._make_conv_layer(32, 64, 3, 2),
            self._make_conv_layer(64, 128, 3, 2),
            # Residual out
        )

        self.layers4 = nn.Sequential(
            self._make_conv_layer(128, 128, 3, 1),
            self._make_conv_layer(128, 128, 3, 1),
            # Residual + in
        )

        self.layers_final = nn.Sequential(
            nn.Linear(128, 128, **self.config),
            nn.ReLU(),
            nn.Linear(128, out_features, **self.config),
        )

    def _make_conv_layer(
        self, in_channels, out_channels, kernel_size, stride
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                **self.config,
            ),
            nn.BatchNorm2d(out_channels, **self.config),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers1(x)
        x = self.layers2(x) + x
        x = self.layers3(x)
        x = self.layers4(x) + x
        x = nn.Flatten()(x)
        x = self.layers_final(x)
        return x
