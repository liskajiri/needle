"""
Convolutional layers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import needle as ndl
from needle import init
from needle.backend_selection import default_device
from needle.nn.core import Module, Parameter

if TYPE_CHECKING:
    from needle.tensor import Tensor
    from needle.typing import AbstractBackend, DType


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    - Only supports padding=same
    - No grouped convolution or dilation
    - Only supports square kernels
    - Pads the input tensor to ensure output has the same shape as the input.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        padding (int): Padding of the input
        bias (bool): Whether to use a bias term
    """

    # TODO: supporting non-square kernels shouldn't be hard

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
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
        self.padding = padding if padding else (kernel_size - 1) // 2

        self.weight = Parameter(
            init.kaiming_uniform(
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            bias_range = 1 / math.sqrt(in_channels * kernel_size**2)
            self.bias = Parameter(
                init.rand(
                    (out_channels,),
                    low=-bias_range,
                    high=bias_range,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            self.bias = None

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
        # Convert from NCHW to NHWC for the conv op
        x_nhwc = x.transpose((0, 2, 3, 1))

        conv_x = ndl.ops.conv(
            x_nhwc, self.weight, stride=self.stride, padding=self.padding
        )
        # NHWC -> NCHW
        if self.bias:
            bias_broadcasted = self.bias.broadcast_to(conv_x.shape)
            conv_x = conv_x + bias_broadcasted
        return conv_x.transpose((0, 3, 1, 2))
