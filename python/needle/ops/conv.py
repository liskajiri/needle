from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from needle.backend_selection import array_api
from needle.ops.op import TensorOp
from needle.ops.view import dilate
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.typing.types import Shape


class Conv(TensorOp):
    def __init__(self, stride: int = 1, padding: int = 0) -> None:
        self.stride = stride
        self.padding = padding

    @staticmethod
    def _im2col(
        img: NDArray,
        kernel_shape: Shape,
        out_height: int,
        out_width: int,
        stride: int = 1,
    ) -> NDArray:
        """
        Convert image to column format for convolution.

        Args:
            img: Input tensor of shape (N, H, W, C_in)
            kernel: Weight tensor of shape (kH, kW, C_in, C_out)

        Returns:
            NDArray: Column format of the input tensor
        """
        n, _in_height, _in_width, in_channels = img.shape
        kernel_height, kernel_width, _in_c, _out_channels = kernel_shape

        patches = array_api.empty((
            n * out_height * out_width,
            kernel_height * kernel_width * in_channels,
        ))

        # im2col transformation
        for y in range(out_height):
            y_pos = y * stride
            for x in range(out_width):
                x_pos = x * stride
                # Extract patches and flatten
                patch = (
                    img[
                        :,
                        y_pos : y_pos + kernel_height,
                        x_pos : x_pos + kernel_width,
                        :,
                    ]
                    .compact()
                    .flatten()
                )
                idx = y * out_width + x
                patches[idx :: out_height * out_width] = patch

        return patches

    def compute(self, img: NDArray, kernel: NDArray) -> NDArray:
        """
        Args:
            img: Input tensor of shape (N, H, W, C_in)
            kernel: Weight tensor of shape (kH, kW, C_in, C_out)
        """
        n, in_height, in_width, _in_channels = img.shape
        kernel_height, kernel_width, _in_c, out_channels = kernel.shape

        out_height = (in_height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_width) // self.stride + 1

        if self.padding > 0:
            img = array_api.pad(
                img,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )

        patches = self._im2col(img, kernel.shape, out_height, out_width, self.stride)
        reshaped_kernel = kernel.compact().reshape((-1, out_channels))

        # Compute convolution as matrix multiplication
        output = patches @ reshaped_kernel

        # Reshape output to proper dimensions (N, H', W', C_out)
        return output.reshape((n, out_height, out_width, out_channels))

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
        K_H, _K_W, _C_in, C_out = W.shape
        _, out_H, out_W, _ = out_grad.shape

        # Flip kernel spatially and swap input/output channels for transpose convolution
        W_flipped = W.flip(axes=(0, 1)).transpose((3, 2))

        # Reshape output gradients to (N*H'*W', C_out)
        out_grad_reshaped = (
            out_grad.realize_cached_data().compact().reshape((-1, C_out))
        )

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        X_grad = conv(out_grad, W_flipped, padding=K_H - 1 - self.padding)

        if X_grad.shape != X.shape:
            # In some cases with odd dimensions and strides, we might need to crop
            slices = tuple(slice(0, X.shape[i]) for i in range(4))
            X_grad = X_grad[slices]

        X_padded = X.realize_cached_data()
        if self.padding > 0:
            X_padded = array_api.pad(
                X_padded,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )
        patches = self._im2col(X_padded, W.shape, out_H, out_W, self.stride)

        # Compute weight gradients
        # patches: (N*H'*W', K_H*K_W*C_in)
        # out_grad_reshaped: (N*H'*W', C_out)
        # W_grad needs to be: (K_H, K_W, C_in, C_out)
        W_grad = (array_api.transpose(patches, (1, 0)) @ out_grad_reshaped).reshape(
            W.shape
        )

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

    _n, _img_height, _img_width, in_channels = img.shape
    kernel_height, kernel_width, kernel_in_channels, _out_channels = kernel.shape
    if in_channels != kernel_in_channels:
        raise ValueError(
            f"C_in={in_channels} does not match kernel C_in={kernel_in_channels}\n"
            f"In shape {img.shape} and {kernel.shape}"
        )
    # if kernel_height > img_height or kernel_width > img_width:
    #     raise ValueError(
    #         f"Kernel size {kernel.shape} is larger than input size {img.shape}"
    #     )

    logging.debug(
        f"Convolution with kernel size {kernel_height}x{kernel_width}, "
        f"stride {stride}, padding {padding}"
    )
    return Conv(stride, padding)(img, kernel)
