import math

import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.utils import set_random_seeds

rng = np.random.default_rng()


def test_init_kaiming_uniform():
    set_random_seeds(42)
    np.testing.assert_allclose(
        ndl.init.kaiming_uniform(3, 5).numpy(),
        np.array(
            [
                [-0.35485414, 1.2748126, 0.65617794, 0.27904832, -0.9729262],
                [-0.97299445, -1.2499284, 1.0357026, 0.28599644, 0.58851814],
                [-1.3559918, 1.3291057, 0.9402898, -0.81362784, -0.8999349],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


@all_devices()
def test_init_kaiming_uniform_2(device):
    a = rng.standard_normal((3, 3, 16, 8))
    A = ndl.Tensor(a, device=device)
    set_random_seeds(0)
    A = ndl.init.kaiming_uniform(16 * 9, 8 * 9, shape=A.shape)
    np.testing.assert_allclose(A.sum().numpy(), -2.5719218, rtol=1e-4, atol=1e-4)


def test_init_kaiming_normal():
    set_random_seeds(42)
    np.testing.assert_allclose(
        ndl.init.kaiming_normal(3, 5).numpy(),
        np.array(
            [
                [0.4055654, -0.11289233, 0.5288355, 1.2435486, -0.19118543],
                [-0.19117202, 1.2894219, 0.62660784, -0.38332424, 0.4429984],
                [-0.37837896, -0.38026676, 0.19756137, -1.5621868, -1.4083896],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_init_xavier_uniform():
    set_random_seeds(42)
    np.testing.assert_allclose(
        ndl.init.xavier_uniform(3, 5, gain=1.5).numpy(),
        np.array(
            [
                [-0.32595432, 1.1709901, 0.60273796, 0.25632226, -0.8936898],
                [-0.89375246, -1.1481324, 0.95135355, 0.26270452, 0.54058844],
                [-1.245558, 1.2208616, 0.8637113, -0.74736494, -0.826643],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_init_xavier_normal():
    set_random_seeds(42)
    np.testing.assert_allclose(
        ndl.init.xavier_normal(3, 5, gain=0.33).numpy(),
        np.array(
            [
                [0.08195783, -0.022813609, 0.10686861, 0.25129992, -0.038635306],
                [-0.038632598, 0.2605701, 0.12662673, -0.07746328, 0.08952241],
                [-0.07646392, -0.07684541, 0.039923776, -0.31569123, -0.28461143],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


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
    large_fan_out = fan_out * 10_000

    ndl_tensor = ndl_init_fn(fan_in, large_fan_out, gain=gain)
    torch_tensor = torch_init_fn(torch.empty(fan_in, large_fan_out), gain=gain)

    ndl_samples = ndl_tensor.numpy().flatten()
    torch_samples = torch_tensor.numpy().flatten()

    atol, rtol = 1e-1, 1e-3
    np.testing.assert_allclose(
        ndl_samples.std(), torch_samples.std(), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        ndl_samples.mean(), torch_samples.mean(), rtol=rtol, atol=atol
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
    large_fan_out = fan_out * 10_000

    ndl_tensor = ndl_init_fn(fan_in, large_fan_out, gain=gain, mode="fan_in")
    torch_tensor = torch_init_fn(
        torch.empty(large_fan_out, fan_in), a=gain, mode="fan_in", nonlinearity="relu"
    )

    ndl_samples = ndl_tensor.numpy().flatten()
    torch_samples = torch_tensor.numpy().flatten()

    atol, rtol = 1e-1, 1e-3
    np.testing.assert_allclose(
        ndl_samples.std(), torch_samples.std(), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        ndl_samples.mean(), torch_samples.mean(), rtol=rtol, atol=atol
    )
