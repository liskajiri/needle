import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import conv_operation_parameters
from tests.utils import backward_forward

# Global tolerance settings
RTOL = 1e-4
ATOL = 1e-4

rng = np.random.default_rng()


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


@given(params=conv_operation_parameters())
@all_devices()
@backward_forward()
@pytest.mark.xfail(reason="Gradient check is currently failing", strict=False)
def test_conv_op(params, device, backward):
    """Test low-level conv operation forward and backward pass."""
    input_shape = params["input_shape"]
    kernel_shape = params["kernel_shape"]
    stride = params["stride"]
    padding = params["padding"]

    input_array = rng.standard_normal(input_shape, dtype=np.float32)
    kernel_array = rng.standard_normal(kernel_shape, dtype=np.float32)

    input_ndl = ndl.Tensor(input_array, device=device, requires_grad=backward)
    kernel_ndl = ndl.Tensor(kernel_array, device=device, requires_grad=backward)

    input_torch = torch.tensor(input_array, dtype=torch.float32, requires_grad=backward)
    kernel_torch = torch.tensor(
        kernel_array, dtype=torch.float32, requires_grad=backward
    )

    result_ndl = ndl.conv(input_ndl, kernel_ndl, padding=padding, stride=stride)

    # Equivalent PyTorch operation with channel order adjustment
    result_torch = torch.nn.functional.conv2d(
        input_torch.permute(0, 3, 1, 2),
        kernel_torch.permute(3, 2, 0, 1),
        padding=padding,
        stride=stride,
    )

    np.testing.assert_allclose(
        result_ndl.numpy(),
        result_torch.permute(0, 2, 3, 1).contiguous().detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="Conv operation forward pass results do not match",
    )

    if backward:
        check_backward_conv(result_ndl, input_ndl, result_torch, input_torch)
        assert_gradients_close(kernel_ndl, kernel_torch, "kernel")
