"""Property-based tests for Linear layer in needle.nn.linear.

Tests:
- Output shape matches expected shape.
- For identity weights and zero bias, output equals input.
"""

from __future__ import annotations

import needle as nd
import needle.nn as nn
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import (
    arrays,
)

from tests.devices import all_devices
from tests.hypothesis_strategies import float_strategy
from tests.utils import DTYPE_FLOAT


@given(data=st.data())
@all_devices()
def test_linear_shape(data, device) -> None:
    """
    Test that Linear layer output shape matches (batch, out_features).
    """
    input_shape = (3, 2)
    output_shape = (3, 3)

    arr = data.draw(
        arrays(dtype=DTYPE_FLOAT, shape=input_shape, elements=float_strategy)
    )

    arr = nd.Tensor(arr, device=device)
    layer = nn.Linear(input_shape[1], output_shape[1])
    y = layer.forward(arr)
    assert y.shape == (arr.shape[0], output_shape[1])


@given(data=st.data())
@all_devices()
def test_linear_identity(data, device) -> None:
    """
    Test that Linear layer with identity weights and zero bias returns input.
    """
    input_shape = (3, 2)
    arr = data.draw(
        arrays(dtype=DTYPE_FLOAT, shape=input_shape, elements=float_strategy)
    )

    arr = nd.Tensor(arr, device=device)
    layer = nn.Linear(input_shape[1], input_shape[1])
    # Set weights to identity and bias to zero
    layer.weight.data = nd.Tensor(
        np.eye(input_shape[1], dtype=np.float32), device=device
    )

    # satisfy type checker
    assert layer.bias is not None, "Bias should not be None"

    layer.bias.data = nd.Tensor(
        np.zeros(input_shape[1], dtype=np.float32), device=device
    )
    y = layer.forward(arr)
    assert np.allclose(y.numpy(), arr.numpy())
