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


DILATE_PARAMS = [
    pytest.param(
        np.array([[6.0, 1.0, 4.0, 4.0, 8.0], [4.0, 6.0, 3.0, 5.0, 8.0]]),
        0,
        (0,),
        np.array([[6.0, 1.0, 4.0, 4.0, 8.0], [4.0, 6.0, 3.0, 5.0, 8.0]]),
        id="2d_no_dilation_axis0",
    ),
    pytest.param(
        np.array([[7.0, 9.0, 9.0, 2.0, 7.0], [8.0, 8.0, 9.0, 2.0, 6.0]]),
        1,
        (0,),
        np.array(
            [
                [7.0, 9.0, 9.0, 2.0, 7.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 8.0, 9.0, 2.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        id="2d_dilation1_axis0",
    ),
    pytest.param(
        np.array([[9.0, 5.0, 4.0, 1.0, 4.0], [6.0, 1.0, 3.0, 4.0, 9.0]]),
        1,
        (1,),
        np.array(
            [
                [9.0, 0.0, 5.0, 0.0, 4.0, 0.0, 1.0, 0.0, 4.0, 0.0],
                [6.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 9.0, 0.0],
            ]
        ),
        id="2d_dilation1_axis1",
    ),
    pytest.param(
        np.array([[2.0, 4.0, 4.0, 4.0, 8.0], [1.0, 2.0, 1.0, 5.0, 8.0]]),
        1,
        (0, 1),
        np.array(
            [
                [2.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 5.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        id="2d_dilation1_axis01",
    ),
    pytest.param(
        np.array([[4.0, 3.0], [8.0, 3.0]]),
        2,
        (0, 1),
        np.array(
            [
                [4.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        id="2d_dilation2_axis01",
    ),
    pytest.param(
        np.array(
            [
                [[[1.0, 1.0], [5.0, 6.0]], [[6.0, 7.0], [9.0, 5.0]]],
                [[[2.0, 5.0], [9.0, 2.0]], [[2.0, 8.0], [4.0, 7.0]]],
            ]
        ),
        1,
        (1, 2),
        np.array(
            [
                [
                    [[1.0, 1.0], [0.0, 0.0], [5.0, 6.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[6.0, 7.0], [0.0, 0.0], [9.0, 5.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[2.0, 5.0], [0.0, 0.0], [9.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[2.0, 8.0], [0.0, 0.0], [4.0, 7.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
            ]
        ),
        id="4d_dilation1_axis12",
    ),
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("input,dilation,axes,expected", DILATE_PARAMS)
def test_dilate_forward(device, input, dilation, axes, expected):
    _A = input
    A = ndl.Tensor(_A, device=device)

    result = ndl.dilate(A, dilation=dilation, axes=axes).numpy()
    # Values are not changed, so tolerance=0
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


dilate_backward_params = [
    {"shape": (2, 5), "d": 1, "axes": (0,)},
    {"shape": (2, 5), "d": 2, "axes": (1,)},
    {"shape": (2, 5), "d": 1, "axes": (0, 1)},
    {"shape": (2, 5), "d": 0, "axes": (0, 1)},
    {"shape": (2, 3, 3, 4), "d": 2, "axes": (0, 1)},
    {"shape": (3, 3, 6, 4), "d": 3, "axes": (0, 1)},
    {"shape": (2, 3, 3, 4), "d": 0, "axes": (1, 2)},
    {"shape": (2, 3, 3, 4), "d": 1, "axes": (1, 2)},
    {"shape": (3, 3, 6, 4), "d": 1, "axes": (1, 2)},
    {"shape": (2, 3, 3, 4), "d": 1, "axes": (2, 3)},
    {"shape": (3, 3, 6, 4), "d": 1, "axes": (2, 3)},
    {"shape": (2, 3, 3, 4), "d": 1, "axes": (0, 1, 2, 3)},
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(
    "params",
    dilate_backward_params,
    ids=[f"{p['shape']}-{p['d']}-{p['axes']}" for p in dilate_backward_params],
)
def test_dilate_backward(params, device):
    shape, d, axes = params["shape"], params["d"], params["axes"]
    # Values are not changed, so tolerance=0
    backward_check(
        ndl.dilate,
        ndl.Tensor(rng.standard_normal(shape), device=device),
        dilation=d,
        axes=axes,
        tol=0,
    )
