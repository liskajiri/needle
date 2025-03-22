import needle as ndl
import numpy as np
import pytest

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


# TODO: Add tests
# import pytest
# from hypothesis import given
# from hypothesis import strategies as st
# from hypothesis.extra.numpy import array_shapes, arrays
# import torch


# @given(fan_in=st.integers(1, 5), fan_out=st.integers(1, 5))
# def test_init_xavier_normal_proptest(fan_in: int, fan_out: int):
#     gain: float = 2**0.5
#     _ndl = ndl.init.xavier_normal(fan_in, fan_out, gain)
#     _pytorch = torch.nn.init.xavier_normal_(torch.empty(fan_in, fan_out), gain=gain)
#     np.testing.assert_allclose(_ndl.numpy(), _pytorch.numpy(), rtol=1e-4, atol=1e-4)


# @given(fan_in=st.integers(1, 5), fan_out=st.integers(1, 5))
# def test_init_kaiming_normal_proptest(fan_in: int, fan_out: int):
#     gain: float = 2**0.5
#     _ndl = ndl.init.kaiming_normal(fan_in, fan_out, gain)
#     _pytorch = torch.nn.init.kaiming_normal_(
#         torch.empty(fan_in, fan_out), nonlinearity="relu"
#     )
#     np.testing.assert_allclose(_ndl.numpy(), _pytorch.numpy(), rtol=1e-4, atol=1e-4)


# @given(fan_in=st.integers(1, 5), fan_out=st.integers(1, 5))
# def test_init_xavier_uniform_proptest(fan_in: int, fan_out: int):
#     gain: float = 2**0.5
#     ndl_xavier = ndl.init.xavier_normal(fan_in, fan_out, gain)
#     pytorch_xavier = torch.nn.init.xavier_normal_(
#         torch.empty(fan_in, fan_out), gain=gain
#     )
#     np.testing.assert_allclose(
#         ndl_xavier.numpy(), pytorch_xavier.numpy(), rtol=1e-4, atol=1e-4
#     )


# @given(fan_in=st.integers(1, 5), fan_out=st.integers(1, 5))
# def test_init_xavier_normal_proptest(fan_in: int, fan_out: int):
#     gain: float = 2**0.5
#     ndl_xavier = ndl.init.xavier_normal(fan_in, fan_out, gain)
#     pytorch_xavier = torch.nn.init.xavier_normal_(
#         torch.empty(fan_in, fan_out), gain=gain
#     )
#     np.testing.assert_allclose(
#         ndl_xavier.numpy(), pytorch_xavier.numpy(), rtol=1e-4, atol=1e-4
#     )
