"""
Convolutional layers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from needle import init
from needle.backend_selection import default_device
from needle.nn.core import Module
from needle.ops.conv import conv as conv_op

if TYPE_CHECKING:
    from needle.tensor import Tensor
    from needle.typing import DType
    from needle.typing.device import AbstractBackend


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    - Only supports padding=same
    - No grouped convolution or dilation
    - Only supports square kernels
    - Pads the input tensor to ensure output has the same shape as the input.
    """

    # TODO: supporting non-square kernels shouldn't be hard
    # TODO: supporting padding should be easy

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.dtype = dtype
        # ensure output has the same shape as the input
        self.padding = kernel_size // 2

        self.weight = init.kaiming_uniform(
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device,
            dtype=dtype,
        )
        bias_range = 1 / math.sqrt(in_channels * kernel_size**2)
        self.bias = (
            init.rand(
                out_channels,
                low=-bias_range,
                high=bias_range,
                device=device,
                dtype=dtype,
            )
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Runs the forward pass of the convolutional layer.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape NCHW

        Returns
        -------
        Tensor
            Output tensor with shape NCHW
        """
        # Conv accepts NHWC, so we need to transpose
        x = x.transpose((0, 2, 3, 1))
        conv_x = conv_op(x, self.weight, self.stride, self.padding)
        # transpose back to NCHW
        conv_x = conv_x.transpose((0, 3, 1, 2))
        if self.bias:
            conv_x += self.bias.reshape((1, self.out_channels, 1, 1))
        return conv_x
