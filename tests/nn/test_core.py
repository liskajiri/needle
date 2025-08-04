"""Tests for needle.nn.core module."""

from __future__ import annotations

import needle as ndl
import numpy as np
import torch
from hypothesis import assume, given
from needle.nn.core import Flatten, Identity, Residual, Sequential

from tests.hypothesis_strategies import torch_tensors
from tests.utils import backward_forward

RTOL = 1e-6
ATOL = 1e-6


# TODO: unify with utils.py
def check_backward(needle_result, needle_x, torch_result, torch_x):
    needle_loss = needle_result.sum()
    torch_loss = torch_result.sum()
    torch_loss.backward()
    needle_loss.backward()
    assert torch_x.grad is not None
    assert needle_x.grad is not None
    np.testing.assert_allclose(
        needle_x.grad.numpy(), torch_x.grad.numpy(), rtol=RTOL, atol=ATOL
    )


@given(torch_tensors_list=torch_tensors(n=1))
@backward_forward()
def test_identity(torch_tensors_list, backward):
    """Test Identity module matches PyTorch's nn.Identity."""
    torch_x = torch_tensors_list[0]
    assume(torch_x.dim() > 0)

    torch_x.requires_grad_(backward)
    needle_x = ndl.Tensor(torch_x.detach().numpy(), requires_grad=backward)

    torch_layer = torch.nn.Identity()
    needle_layer = Identity()

    torch_result = torch_layer(torch_x)
    needle_result = needle_layer(needle_x)

    np.testing.assert_allclose(
        needle_result.numpy(), torch_result.detach().numpy(), rtol=RTOL, atol=ATOL
    )

    if backward:
        check_backward(needle_result, needle_x, torch_result, torch_x)


@given(torch_tensors_list=torch_tensors(n=1))
@backward_forward()
def test_flatten(torch_tensors_list, backward):
    """Test Flatten module matches PyTorch's nn.Flatten."""
    torch_x = torch_tensors_list[0]
    assume(torch_x.dim() > 1)

    torch_x.requires_grad_(backward)
    needle_x = ndl.Tensor(torch_x.detach().numpy(), requires_grad=backward)

    torch_layer = torch.nn.Flatten()
    needle_layer = Flatten()

    torch_result = torch_layer(torch_x)
    needle_result = needle_layer(needle_x)

    np.testing.assert_allclose(
        needle_result.numpy(), torch_result.detach().numpy(), rtol=RTOL, atol=ATOL
    )

    if backward:
        check_backward(needle_result, needle_x, torch_result, torch_x)


@given(torch_tensors_list=torch_tensors(n=1))
@backward_forward()
def test_sequential(torch_tensors_list, backward):
    """Test Sequential module matches PyTorch's nn.Sequential."""
    torch_x = torch_tensors_list[0]
    assume(torch_x.dim() > 1)

    torch_x.requires_grad_(backward)
    needle_x = ndl.Tensor(torch_x.detach().numpy(), requires_grad=backward)

    torch_layer = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Identity())
    needle_layer = Sequential(Flatten(), Identity())

    torch_result = torch_layer(torch_x)
    needle_result = needle_layer(needle_x)

    np.testing.assert_allclose(
        needle_result.numpy(), torch_result.detach().numpy(), rtol=RTOL, atol=ATOL
    )

    if backward:
        check_backward(needle_result, needle_x, torch_result, torch_x)


@given(torch_tensors_list=torch_tensors(n=1))
@backward_forward()
def test_residual(torch_tensors_list, backward):
    """Test Residual module output shape and value, and backward if possible."""
    torch_x = torch_tensors_list[0]
    assume(torch_x.dim() > 0)

    torch_x.requires_grad_(backward)
    needle_x = ndl.Tensor(torch_x.detach().numpy(), requires_grad=backward)

    needle_layer = Residual(Identity())
    needle_result = needle_layer(needle_x)
    expected = torch_x.detach().numpy() * 2

    np.testing.assert_allclose(needle_result.numpy(), expected, rtol=RTOL, atol=ATOL)

    if backward:
        needle_loss = needle_result.sum()
        needle_loss.backward()
        assert needle_x.grad is not None
        # For PyTorch, output = x + x, so grad should be 2
        assert np.allclose(needle_x.grad.numpy(), 2, rtol=RTOL, atol=ATOL)
