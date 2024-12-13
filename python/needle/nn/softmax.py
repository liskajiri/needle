"""
Important: some values need to be converted to float32,
otherwise they will overflow the division, thus making it a float64 result,
which will cause type errors downstream
"""

from numpy import float32

from needle import init, ops
from needle.autograd import Tensor
from needle.nn.nn_basic import Module


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        y_one_hot = init.one_hot(logits.shape[1], y)
        diff = (logits * y_one_hot).sum(axes=1)

        lse = ops.logsumexp(logits, axes=1)
        # division causes result to be float64
        result = (lse - diff).sum() / float32(logits.shape[0])
        return result
