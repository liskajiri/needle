import math

import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

rng = np.random.default_rng()

RTOL = 1e-0
ATOL = 1e-2
NUM_SAMPLES = 100_000


@pytest.mark.parametrize(
    "gain", [1.0, 2**0.5, 0.5, 0.0], ids=["1.0", "sqrt2", "0.5", "0.0"]
)
@pytest.mark.parametrize(
    "init_fn_pair",
    [
        (ndl.init.xavier_normal, torch.nn.init.xavier_normal_),
        (ndl.init.xavier_uniform, torch.nn.init.xavier_uniform_),
    ],
    ids=["xavier_normal", "xavier_uniform"],
)
@given(fan_in=st.integers(1, 10), fan_out=st.integers(1, 10))
def test_xavier_distributions(gain, init_fn_pair, fan_in: int, fan_out: int):
    ndl_init_fn, torch_init_fn = init_fn_pair

    # Make the tensor large enough for good statistics
    large_fan_out = fan_out * NUM_SAMPLES

    ndl_tensor = ndl_init_fn(fan_in, large_fan_out, gain=gain)
    torch_tensor = torch_init_fn(torch.empty(fan_in, large_fan_out), gain=gain)

    ndl_samples = ndl_tensor.numpy().flatten()
    torch_samples = torch_tensor.numpy().flatten()

    np.testing.assert_allclose(
        ndl_samples.std(), torch_samples.std(), atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        ndl_samples.mean(), torch_samples.mean(), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize(
    "init_fn_pair",
    [
        (ndl.init.kaiming_normal, torch.nn.init.kaiming_normal_),
        (ndl.init.kaiming_uniform, torch.nn.init.kaiming_uniform_),
    ],
    ids=["kaiming_normal", "kaiming_uniform"],
)
@given(fan_in=st.integers(1, 10), fan_out=st.integers(1, 10))
def test_kaiming_distributions(init_fn_pair, fan_in: int, fan_out: int) -> None:
    ndl_init_fn, torch_init_fn = init_fn_pair
    gain = math.sqrt(2)

    # Make the tensor large enough for good statistics
    large_fan_out = fan_out * NUM_SAMPLES

    ndl_tensor = ndl_init_fn(fan_in, large_fan_out, gain=gain, mode="fan_in")
    torch_tensor = torch_init_fn(
        torch.empty(large_fan_out, fan_in), a=gain, mode="fan_in", nonlinearity="relu"
    )

    ndl_samples = ndl_tensor.numpy().flatten()
    torch_samples = torch_tensor.numpy().flatten()

    np.testing.assert_allclose(
        ndl_samples.std(), torch_samples.std(), atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        ndl_samples.mean(), torch_samples.mean(), atol=ATOL, rtol=RTOL
    )
