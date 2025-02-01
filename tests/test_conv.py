import needle as ndl
import numpy as np
import pytest

from tests.gradient_check import backward_check

_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

rng = np.random.default_rng()

pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ((0, 0), (2, 2), (2, 2), (0, 0))},
    {"shape": (10, 32, 32, 8), "padding": ((0, 0), (0, 0), (0, 0), (0, 0))},
    # non-square padding
    {"shape": (10, 32, 32, 8), "padding": ((0, 1), (2, 0), (2, 1), (0, 0))},
]


@pytest.mark.parametrize("device", [ndl.cpu()], ids=["cpu"])
@pytest.mark.parametrize(
    "params",
    pad_params,
    ids=[f"{p['shape']}-{p['padding']}" for p in pad_params],
)
def test_pad_forward(params, device):
    shape, padding = params["shape"], params["padding"]
    _a = rng.standard_normal(shape)
    a = ndl.NDArray(_a, device=device)

    _b = np.pad(_a, padding)
    b = a.pad(padding)

    np.testing.assert_allclose(b.numpy(), _b, rtol=1e-6)


flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0, 1)},
    {"shape": (10, 32, 32, 8), "axes": (0, 1)},
    {"shape": (3, 3, 6, 8), "axes": (0, 1)},
    {"shape": (10, 32, 32, 8), "axes": (1, 2)},
    {"shape": (3, 3, 6, 8), "axes": (1, 2)},
    {"shape": (10, 32, 32, 8), "axes": (2, 3)},
    {"shape": (3, 3, 6, 8), "axes": (2, 3)},
    {"shape": (10, 32, 32, 8), "axes": (0, 1, 2, 3)},
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(
    "params",
    flip_forward_params,
    ids=[f"{p['shape']}-{p['axes']}" for p in flip_forward_params],
)
def test_flip_forward(params, device):
    shape, axes = params["shape"], params["axes"]
    _A = rng.standard_normal(shape)
    A = ndl.Tensor(_A, device=device)

    _B = np.flip(_A, axes)
    B = ndl.flip(A, axes=axes)

    np.testing.assert_allclose(B.numpy(), _B, rtol=1e-6)


flip_backward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0, 1)},
    {"shape": (2, 3, 3, 8), "axes": (0, 1)},
    {"shape": (3, 3, 6, 4), "axes": (0, 1)},
    {"shape": (2, 3, 3, 4), "axes": (1, 2)},
    {"shape": (3, 3, 6, 4), "axes": (1, 2)},
    {"shape": (2, 3, 3, 4), "axes": (2, 3)},
    {"shape": (3, 3, 6, 4), "axes": (2, 3)},
    {"shape": (2, 3, 3, 4), "axes": (0, 1, 2, 3)},
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(
    "params",
    flip_backward_params,
    ids=[f"{p['shape']}-{p['axes']}" for p in flip_backward_params],
)
def test_flip_backward(params, device):
    shape, axes = params["shape"], params["axes"]
    backward_check(
        ndl.flip, ndl.Tensor(rng.standard_normal(shape), device=device), axes=axes
    )
