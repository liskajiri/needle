"""
Property-based tests for Dropout layer in needle.nn.dropout.

Tests:
- Output shape matches input shape.
- For p=0, output equals input.
- For p=1, output is all zeros.
"""

import needle as nd
import needle.nn as nn
import numpy as np
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import array_strategy


@given(arr=array_strategy)
@all_devices()
def test_dropout_shape(arr, device) -> None:
    """
    Test that Dropout preserves input shape.
    """
    arr = nd.Tensor(arr, device=device)

    layer = nn.Dropout(p=0.5)
    y = layer.forward(arr)

    assert y.shape == arr.shape


@given(arr=array_strategy)
@all_devices()
def test_dropout_p0_identity(arr, device) -> None:
    """
    Test that Dropout with p=0 returns input unchanged.
    """
    arr = nd.Tensor(arr, device=device)

    layer = nn.Dropout(p=0.0)
    y = layer.forward(arr)

    assert np.allclose(y.numpy(), arr.numpy())


@given(arr=array_strategy)
@all_devices()
def test_dropout_p1_zeros(arr, device) -> None:
    """
    Test that Dropout with p=1 returns all zeros.
    """
    arr = nd.Tensor(arr, device=device)

    layer = nn.Dropout(p=1.0)
    y = layer.forward(arr)
    assert np.allclose(y.numpy(), 0.0)
