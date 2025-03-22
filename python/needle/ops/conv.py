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

    def compute(self, img: NDArray, kernel: NDArray) -> NDArray:
        # Input format: NHWC (batch-height-width-channel)
        # Kernel format: HWIO (height-width-in_channel-out_channel)
        n, height, width, _in_channels = img.shape
        kernel_height, kernel_width, _W, out_channels = kernel.shape

        if self.padding > 0:
            img = img.pad((
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ))
            height += 2 * self.padding
            width += 2 * self.padding

        # kernel reduces size
        final_H = (height - kernel_height) // self.stride + 1
        final_W = (width - kernel_width) // self.stride + 1

        out = array_api.zeros((n, final_H, final_W, out_channels))
        for i in range(kernel_height):
            for j in range(kernel_width):
                curr_img = img[
                    :,
                    i : i + final_H * self.stride : self.stride,
                    j : j + final_W * self.stride : self.stride,
                    :,
                ]
                out += curr_img @ kernel[i, j]

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

        # Flip kernel spatially and swap input/output channels for transpose convolution
        W_flipped = W.flip(axes=(0, 1))
        W_flipped = W_flipped.transpose((3, 2))

        out_grad_orig = out_grad.realize_cached_data().compact()

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        X_grad = conv(out_grad, W_flipped, padding=K_H - 1 - self.padding)

        if X_grad.shape != X.shape:
            # In some cases with odd dimensions and strides, we might need to crop
            slices = tuple(slice(0, X.shape[i]) for i in range(4))
            X_grad = X_grad.realize_cached_data()[slices]

        X_padded = X.realize_cached_data()
        if self.padding > 0:
            X_padded = X_padded.pad((
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ))

        # TODO: Convert this to a convolution operation
        W_grad = array_api.zeros((K_H, K_W, C_in, C_out))
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

        return X_grad, Tensor(W_grad)


def conv(img: Tensor, kernel: Tensor, stride: int = 1, padding: int = 1) -> Tensor:
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

    _n, height, width, in_channels = img.shape
    kernel_height, kernel_width, kernel_in_channels, _out_channels = kernel.shape
    if in_channels != kernel_in_channels:
        raise ValueError(
            f"C_in={in_channels} does not match kernel C_in={kernel_in_channels}\n"
            f"In shape {img.shape} and {kernel.shape}"
        )

    if kernel_height > height or kernel_width > width:
        raise ValueError(
            f"Kernel size {kernel_height}x{kernel_width}\n"
            f"must be smaller than input size {height}x{width}"
        )
    return Conv(stride, padding)(img, kernel)
