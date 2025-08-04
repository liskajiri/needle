"""
Tests for the activations.
"""

from __future__ import annotations

import needle as ndl
import numpy as np
import torch
from hypothesis import assume, example, given
from needle.nn import ReLU, Sigmoid, Tanh

from tests.hypothesis_strategies import torch_tensors
from tests.utils import DTYPE_FLOAT, backward_forward

# Global tolerance settings for all activation tests
RTOL = 1e-5
ATOL = 1e-6

# === The tests are not more generic, because then logging would be less readable.


def check_backward_activation(needle_result, needle_x, torch_result, torch_x):
    torch_loss = torch_result.sum()
    needle_loss = needle_result.sum()
    torch_loss.backward()
    needle_loss.backward()
    assert torch_x.grad is not None, "PyTorch gradient should not be None"
    assert needle_x.grad is not None, "Needle gradient should not be None"
    np.testing.assert_allclose(
        needle_x.grad.numpy(),
        torch_x.grad.numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


@given(torch_tensors_list=torch_tensors(n=1, dtype=DTYPE_FLOAT))
@example(
    torch_tensors_list=[torch.tensor([float("nan"), 1.0, -1.0], dtype=torch.float32)]
)
@example(
    torch_tensors_list=[
        torch.tensor([float("inf"), float("-inf"), 0.0], dtype=torch.float32)
    ]
)
@backward_forward()
def test_tanh(torch_tensors_list, backward) -> None:
    """Test Tanh forward pass matches PyTorch."""
    torch_x = torch_tensors_list[0]
    assume(torch_x.dim() > 0)
    torch_x.requires_grad_(True)

    torch_result = torch.tanh(torch_x)
    needle_x = ndl.Tensor(torch_x.detach().numpy(), requires_grad=backward)

    tanh_layer = Tanh()
    needle_result = tanh_layer(needle_x)

    np.testing.assert_allclose(
        needle_result.numpy(),
        torch_result.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )

    if backward:
        check_backward_activation(needle_result, needle_x, torch_result, torch_x)


@given(torch_tensors_list=torch_tensors(n=1))
@example(
    torch_tensors_list=[torch.tensor([float("nan"), 1.0, -1.0], dtype=torch.float32)]
)
@example(
    torch_tensors_list=[
        torch.tensor([float("inf"), float("-inf"), 0.0], dtype=torch.float32)
    ]
)
@backward_forward()
def test_sigmoid(torch_tensors_list, backward):
    """Test Sigmoid forward pass matches PyTorch."""
    torch_x = torch_tensors_list[0]
    assume(torch_x.dim() > 0)

    torch_x.requires_grad_(True)
    torch_result = torch.sigmoid(torch_x)

    needle_x = ndl.Tensor(torch_x.detach().numpy(), requires_grad=backward)

    sigmoid_layer = Sigmoid()
    needle_result = sigmoid_layer(needle_x)

    np.testing.assert_allclose(
        needle_result.numpy(),
        torch_result.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    if backward:
        check_backward_activation(needle_result, needle_x, torch_result, torch_x)


@given(torch_tensors_list=torch_tensors(n=1))
@example(
    torch_tensors_list=[torch.tensor([-0.0, 0.0, -1e-10, 1e-10], dtype=torch.float32)]
)
@backward_forward()
def test_relu(torch_tensors_list, backward):
    """Test ReLU forward pass matches PyTorch."""
    torch_x = torch_tensors_list[0]
    assume(torch_x.dim() > 0)
    torch_x.requires_grad_(True)

    torch_result = torch.relu(torch_x)
    needle_x = ndl.Tensor(torch_x.detach().numpy(), requires_grad=backward)

    relu_layer = ReLU()
    needle_result = relu_layer(needle_x)

    np.testing.assert_allclose(
        needle_result.numpy(),
        torch_result.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )

    if backward:
        check_backward_activation(needle_result, needle_x, torch_result, torch_x)
