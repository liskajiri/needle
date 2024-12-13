from needle import ops
from needle.autograd import Tensor
from needle.nn.nn_basic import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
