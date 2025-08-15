"""Tests for needle.nn.norms module."""

from __future__ import annotations

import needle as nd
import needle.nn as nn
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.hypothesis_strategies import single_array


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
