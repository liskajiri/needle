"""
Important: some values need to be converted to float32,
otherwise they will overflow the division, thus making it a float64 result,
which will cause type errors downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from needle import init
from needle.nn.core import Module

if TYPE_CHECKING:
    from needle.tensor import Tensor


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        """
        Dropout layer.

        Args:
            p: probability of dropping out a unit.
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = init.rand_binary(x.shape, p=1 - self.p)
        return (x * mask) / (1 - self.p)
