"""
Important: some values need to be converted to float32,
otherwise they will overflow the division, thus making it a float64 result,
which will cause type errors downstream
"""

from numpy import float32

from needle import init
from needle.autograd import Tensor
from needle.nn.nn_basic import Module


class Dropout(Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        # division causes result to be float64
        mask = init.randb(*x.shape, p=1 - self.p) / float32(1 - self.p)
        return x * mask
