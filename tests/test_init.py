import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

rng = np.random.default_rng()


def test_init_kaiming_uniform():
    np.random.seed(42)
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


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_init_kaiming_uniform_2(device):
    a = rng.standard_normal((3, 3, 16, 8))
    A = ndl.Tensor(a, device=device)
    np.random.seed(0)
    A = ndl.init.kaiming_uniform(16 * 9, 8 * 9, shape=A.shape)
    assert abs(A.sum().numpy() - -2.5719218) < 1e-4


def test_init_kaiming_normal():
    np.random.seed(42)
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
    np.random.seed(42)
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
    np.random.seed(42)
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
        # TODO: Kaiming stuff does not work
        # (
        #     ndl.init.kaiming_normal,
        #     lambda t, **kwargs: torch.nn.init.kaiming_normal_(
        #         t, mode="fan_in", nonlinearity="relu"
        #     ),
        # ),
        # (
        #     ndl.init.kaiming_uniform,
        #     lambda t, **kwargs: torch.nn.init.kaiming_uniform_(
        #         t, mode="fan_in", nonlinearity="relu"
        #     ),
        # ),
    ],
    ids=[
        "xavier_normal",
        "xavier_uniform",
        # "kaiming_normal",
        # "kaiming_uniform",
    ],
)
@given(fan_in=st.integers(1, 100), fan_out=st.integers(1, 100))
def test_init_distributions_proptest(gain, init_fn_pair, fan_in: int, fan_out: int):
    ndl_init_fn, torch_init_fn = init_fn_pair

    # Test with appropriate gain
    # gain = 2**0.5 if "kaiming" in init_fn_pair[0].__name__ else 1.0

    # Make the tensor large enough for good statistics
    scale_factor = max(1, 10000 // (fan_in * fan_out))
    large_fan_out = fan_out * scale_factor

    # Generate one large tensor from each implementation
    ndl_tensor = ndl_init_fn(fan_in, large_fan_out, gain=gain, requires_grad=False)

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
