from __future__ import annotations

from typing import TYPE_CHECKING

import needle.ops as ops
from needle.nn.core import Module

if TYPE_CHECKING:
    from needle.tensor import Tensor

__all__ = ["ReLU", "Sigmoid", "Tanh"]


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)
