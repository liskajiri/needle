from needle import ops
from needle.nn.core import Module
from needle.tensor import Tensor

__all__ = ["ReLU"]


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
