"""
Important: some values need to be converted to float32,
otherwise they will overflow the division, thus making it a float64 result,
which will cause type errors downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from needle import init, ops
from needle.nn.core import Module

if TYPE_CHECKING:
    from needle.tensor import Tensor


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        y_one_hot = init.one_hot(logits.shape[1], y)
        diff = (logits * y_one_hot).sum(axes=(1,))

        lse = ops.logsumexp(logits, axes=(1,))
        return (lse - diff).sum() / logits.shape[0]
