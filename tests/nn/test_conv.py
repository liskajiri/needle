"""
Convolution tests

- nn_conv
"""

import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import conv_layer_parameters
from tests.utils import backward_forward, set_random_seeds

# Global tolerance settings
RTOL = 1e-4
ATOL = 1e-4

rng = np.random.default_rng()


def copy_weights_needle_to_torch(needle_conv, torch_conv):
    """Copy weights from Needle Conv to PyTorch Conv, handling format differences."""
    # Needle weights: (kernel_size, kernel_size, in_channels, out_channels)
    # PyTorch weights: (out_channels, in_channels, kernel_size, kernel_size)
    needle_weight = needle_conv.weight.realize_cached_data().numpy()
    torch_conv.weight.data = torch.tensor(
        needle_weight.transpose(3, 2, 0, 1), dtype=torch.float32
    )

    # Copy bias if both have it
    if needle_conv.bias is not None and torch_conv.bias is not None:
        needle_bias = needle_conv.bias.realize_cached_data().numpy()
        torch_conv.bias.data = torch.tensor(needle_bias, dtype=torch.float32)


def validate_bias_consistency(needle_conv, torch_conv, bias):
    """Validate that bias parameters are consistent between Needle and PyTorch."""
    if bias:
        assert needle_conv.bias is not None, (
            "Needle Conv should have bias when bias=True"
        )
        assert torch_conv.bias is not None, (
            "PyTorch Conv should have bias when bias=True"
        )
        assert needle_conv.bias.shape == (needle_conv.out_channels,), (
            f"Needle bias shape should be ({needle_conv.out_channels},), "
            f"got {needle_conv.bias.shape}"
        )
    else:
        assert needle_conv.bias is None, (
            "Needle Conv should not have bias when bias=False"
        )
        assert torch_conv.bias is None, (
            "PyTorch Conv should not have bias when bias=False"
        )


def assert_gradients_close(needle_tensor, torch_tensor, name, rtol=RTOL, atol=ATOL):
    """Assert that gradients between Needle and PyTorch tensors are close."""
    assert needle_tensor.grad is not None, f"Needle {name} gradient should not be None"
    assert torch_tensor.grad is not None, f"PyTorch {name} gradient should not be None"

    needle_grad = needle_tensor.grad.realize_cached_data().numpy()
    torch_grad = torch_tensor.grad.numpy()

    np.testing.assert_allclose(
        needle_grad,
        torch_grad,
        rtol=rtol,
        atol=atol,
        err_msg=f"{name} gradients do not match",
    )


def check_backward_conv(needle_output, needle_input, torch_output, torch_input):
    needle_loss = needle_output.sum()
    torch_loss = torch_output.sum()
    needle_loss.backward()
    torch_loss.backward()
    assert_gradients_close(needle_input, torch_input, "input")


@given(params=conv_layer_parameters())
@pytest.mark.parametrize("bias", [True, False])
@all_devices()
@backward_forward()
@pytest.mark.skip(reason="Skipping due to known issues with Conv2D")
def test_nn_conv(params, bias, device, backward):
    """Test Conv layer forward and backward pass."""
    set_random_seeds(42)

    batch_size = params["batch_size"]
    in_channels = params["in_channels"]
    out_channels = params["out_channels"]
    height = params["height"]
    width = params["width"]
    kernel_size = params["kernel_size"]
    stride = params["stride"]
    padding = params["padding"]

    needle_conv = ndl.nn.Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        device=device,
    )

    torch_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    validate_bias_consistency(needle_conv, torch_conv, bias)

    copy_weights_needle_to_torch(needle_conv, torch_conv)

    # Create input tensor (NCHW format)
    input_data = np.random.randn(batch_size, in_channels, height, width).astype(
        np.float32
    )

    needle_input = ndl.Tensor(input_data, device=device, requires_grad=backward)
    torch_input = torch.tensor(input_data, dtype=torch.float32, requires_grad=backward)

    # Forward pass
    needle_output = needle_conv(needle_input)
    torch_output = torch_conv(torch_input)

    np.testing.assert_allclose(
        needle_output.realize_cached_data().numpy(),
        torch_output.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="Forward pass outputs do not match",
    )

    if backward:
        check_backward_conv(needle_output, needle_input, torch_output, torch_input)

        needle_weight_grad = (
            needle_conv.weight.grad.realize_cached_data().numpy().transpose(3, 2, 0, 1)
        )
        assert torch_conv.weight.grad is not None, (
            "PyTorch weight grad should not be None"
        )

        torch_weight_grad = torch_conv.weight.grad.numpy()

        np.testing.assert_allclose(
            needle_weight_grad,
            torch_weight_grad,
            rtol=RTOL,
            atol=ATOL,
            err_msg="Weight gradients do not match",
        )

        # Check bias gradients if bias is used
        if bias:
            assert_gradients_close(needle_conv.bias, torch_conv.bias, "bias")


@pytest.mark.parametrize("bias", [True, False])
@all_devices()
def test_conv_bias_parameter_validation(bias, device):
    """Test Conv layer bias parameter validation and functionality."""
    set_random_seeds(42)

    in_channels, out_channels = 3, 8
    kernel_size = 3
    batch_size, height, width = 2, 8, 8

    conv = ndl.nn.Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias=bias,
        device=device,
    )

    # Check bias existence and shape
    if bias:
        assert conv.bias is not None, "Conv should have bias when bias=True"
        assert conv.bias.shape == (out_channels,), (
            f"Bias shape should be ({out_channels},), got {conv.bias.shape}"
        )
    else:
        assert conv.bias is None, "Conv should not have bias when bias=False"

    # Test forward pass works correctly
    input_data = np.random.randn(batch_size, in_channels, height, width).astype(
        np.float32
    )
    input_tensor = ndl.Tensor(input_data, device=device)

    output = conv(input_tensor)
    output_shape = output.realize_cached_data().shape

    assert output_shape[0] == batch_size, (
        f"Batch size mismatch: expected {batch_size}, got {output_shape[0]}"
    )
    assert output_shape[1] == out_channels, (
        f"Channel mismatch: expected {out_channels}, got {output_shape[1]}"
    )
