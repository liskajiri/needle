"""Tests for needle.nn.norms module."""

from __future__ import annotations

import needle as nd
import needle.nn as nn
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.hypothesis_strategies import single_array
from tests.utils import backward_forward, generic_op_test


@given(arr=single_array(shape=(3, 3)))
@all_devices()
def test_batchnorm_shape(arr, device) -> None:
    """
    Test that BatchNorm1d output shape matches input shape.
    """
    arr = arr[0]
    arr = nd.Tensor(arr, device=device)

    # BatchNorm1d expects num_features = arr.shape[1] (feature dimension)
    layer = nn.BatchNorm1d(arr.shape[1])
    y = layer.forward(arr)

    assert y.shape == arr.shape


@given(input=st.floats(-10, 10))
@all_devices()
def test_batchnorm_constant_input(input: float, device) -> None:
    """
    Test that BatchNorm1d output is zeros for constant input.
    """
    arr = np.full((2, 3), input, dtype=np.float32)
    x = nd.Tensor(arr, device=device)
    layer = nn.BatchNorm1d(x.shape[1])

    y = layer.forward(x)
    # Output should be zeros after normalization
    assert np.allclose(y.numpy(), 0.0, atol=1e-5)


@pytest.mark.parametrize("norm", [nn.BatchNorm1d, nn.LayerNorm1d])
@given(arr=single_array(shape=(4, 5)))
@all_devices()
@backward_forward()
def test_norms(norm, arr, device, backward) -> None:
    inputs = arr[0]
    features = inputs.shape[1]

    ndl_mod = norm(features, device=device)

    if norm == nn.BatchNorm1d:
        torch_mod = torch.nn.BatchNorm1d(
            features,
            affine=True,
            track_running_stats=True,
            eps=ndl_mod.eps,
            momentum=ndl_mod.momentum,
        )
        # sync running stats
        torch_mod.running_mean = torch.tensor(
            ndl_mod.running_mean.realize_cached_data().numpy(), dtype=torch.float32
        )
        torch_mod.running_var = torch.tensor(
            ndl_mod.running_var.realize_cached_data().numpy(), dtype=torch.float32
        )

    else:
        torch_mod = torch.nn.LayerNorm(
            normalized_shape=features, eps=ndl_mod.eps, elementwise_affine=True
        )

    ndl_mod.train()
    torch_mod.train()

    torch_mod.weight.data = torch.tensor(
        ndl_mod.weight.realize_cached_data().numpy(), dtype=torch.float32
    )
    torch_mod.bias.data = torch.tensor(
        ndl_mod.bias.realize_cached_data().numpy(), dtype=torch.float32
    )

    def ndl_op(x):
        return ndl_mod(x)

    def torch_op(x):
        return torch_mod(x)

    generic_op_test(ndl_op, torch_op, [inputs], backward, device, sum=True)
