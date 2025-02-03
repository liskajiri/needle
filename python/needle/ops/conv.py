from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_selection import array_api
from needle.ops.op import TensorOp

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import Tensor


class Conv(TensorOp):
    def __init__(self, stride: int = 1, padding: int = 0) -> None:
        self.stride = stride
        self.padding = padding

    def compute(self, img: NDArray, kernel: NDArray) -> NDArray:
        # Input format: NHWC (batch-height-width-channel)
        # Kernel format: HWIO (height-width-in_channel-out_channel)
        n, height, width, in_channels = img.shape
        kernel_height, kernel_width, _W, out_channels = kernel.shape

        if self.padding > 0:
            img = img.pad(
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                )
            )
            height += 2 * self.padding
            width += 2 * self.padding

        # kernel reduces size
        final_H = (height - kernel_height) // self.stride + 1
        final_W = (width - kernel_width) // self.stride + 1
        out_shape = (n, final_H, final_W, out_channels)

        # # Strided convolution (im2col) - reshape to 6d tensor
        # # to perform only single matmul with the kernel
        # Ns, Hs, Ws, Cs = img.strides
        # out = (
        #     img.as_strided(
        #     shape=(n, final_H, final_W, kernel_height, kernel_height, in_channels),
        #         strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
        #     )
        #     .compact()
        #     .reshape((-1, kernel_height * kernel_width * in_channels))
        # )
        # out._strides = tuple((Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs))

        # A = np.lib.stride_tricks.as_strided(
        #     img,
        #     shape=(n, final_H, final_W, kernel_height, kernel_height, in_channels),
        #     strides=(Ns, Hs, Ws, Hs, Ws, Cs),
        # ).reshape(-1, kernel_height * kernel_width * in_channels)

        # np.testing.assert_allclose(out.numpy(), A, rtol=1e-5, atol=1e-5)
        # out = out @ kernel.reshape((-1, out_channels))
        # return out.reshape(out_shape)

        out = array_api.zeros(out_shape)
        for i in range(kernel_height):
            for j in range(kernel_height):
                out += (
                    img[
                        :,
                        i : i + final_H * self.stride : self.stride,
                        j : j + final_W * self.stride : self.stride,
                        :,
                    ]
                    @ kernel[i, j]
                )
        return out

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        raise NotImplementedError()


def conv(img: Tensor, kernel: Tensor, stride: int = 1, padding: int = 0) -> Tensor:
    """
    Apply 2D convolution to input tensor.

    Args:
        img: Tensor
            Input tensor with shape (N, H, W, C_in)
        kernel: Tensor
            Kernel tensor with shape (kH, kW, C_in, C_out)
        stride: int, optional
            Stride of the convolution. Defaults to 1.
        padding: int, optional
            Zero-padding size. Defaults to 0.

    Returns:
        Tensor: Output tensor with shape (N, H_out, W_out, C_out)
    """

    assert img.ndim == 4, f"Expected 4D input tensor, got shape {img.shape}"
    assert kernel.ndim == 4, f"Expected 4D kernel tensor, got shape {kernel.shape}"
    return Conv(stride, padding)(img, kernel)
