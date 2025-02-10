from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_ndarray.ndarray import transpose
from needle.backend_selection import array_api
from needle.ops.op import TensorOp
from needle.ops.view import dilate
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.backend_selection import NDArray


class Conv(TensorOp):
    def __init__(self, stride: int = 1, padding: int = 0) -> None:
        self.stride = stride
        self.padding = padding

    def compute(self, *arr: tuple[NDArray, ...]) -> NDArray:
        # Input format: NHWC (batch-height-width-channel)
        # Kernel format: HWIO (height-width-in_channel-out_channel)
        img, kernel = arr
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
            for j in range(kernel_width):
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

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        """Compute gradients for the convolution operation.

        Args:
            out_grad: The gradient with respect to the output
            node: The node containing inputs (X: img, W: kernel)
                - shapes:
                    - X: (N, H, W, C_in)
                    - W: (K_H, K_W, C_in, C_out)

        Returns:
            tuple[Value]: Gradients with respect to inputs (dX, dW)
        """
        X, W = node.inputs
        K_H, K_W, C_in, C_out = W.shape
        _, out_H, out_W, _ = out_grad.shape

        W_flipped = W.flip(axes=(0, 1))  # Rotate kernel spatially
        W_flipped = W_flipped.transpose((0, 1, 3, 2))  # Swap C_in and C_out

        out_grad_orig = out_grad.realize_cached_data().compact()

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        X_grad = conv(out_grad, W_flipped, stride=1, padding=K_H - 1 - self.padding)

        W_grad = array_api.zeros((K_H, K_W, C_in, C_out))
        X_padded = X.realize_cached_data().pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        )

        # TODO: Convert this to a convolution operation
        for i in range(K_H):
            for j in range(K_W):
                # Extract windows from X with proper stride
                X_windows = X_padded[
                    :,  # N
                    i : i + out_H * self.stride : self.stride,  # H
                    j : j + out_W * self.stride : self.stride,  # W
                    :,  # C_in
                ]

                # Reshape X_windows to (C_in, N*H'*W')
                X_windows = (
                    transpose(X_windows, (3, 0, 1, 2)).compact().reshape((C_in, -1))
                )

                # Reshape out_grad to (N*H'*W', C_out)
                out_grad_ = out_grad_orig.reshape((-1, C_out))

                W_grad[i, j] = X_windows @ out_grad_

        return X_grad, W_grad


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
    n, height, width, in_channels = img.shape
    kernel_height, kernel_width, _W, out_channels = kernel.shape
    if in_channels != _W:
        raise ValueError(
            f"Input channels ({in_channels}) must match kernel channels ({_W})\n"
            f"In shape {img.shape} and {kernel.shape}"
        )
    if kernel_height > height or kernel_width > width:
        raise ValueError(
            f"Kernel dimensions ({kernel.shape}) must be less than input dimensions ({img.shape})"
        )
    return Conv(stride, padding)(img, kernel)
