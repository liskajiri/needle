import needle as ndl
import numpy as np
import pytest
import torch

from tests.gradient_check import backward_check

_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

rng = np.random.default_rng()

stack_back_params = [
    ((3, 4), 3, 0),
    ((3, 4), 3, 1),
    ((3, 4), 3, 2),
    ((3, 4), 5, 2),
    ((3, 4), 1, 2),
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("shape, n, axis", stack_back_params)
def test_stack_backward(shape, n, axis, device):
    def get_tensor(shape):
        return ndl.Tensor(rng.standard_normal(shape) * 5, device=device)

    backward_check(
        ndl.stack,
        [get_tensor(shape) for _ in range(n)],
        axis=axis,
    )


stack_params = [
    {"shape": (10, 3), "n": 4, "axis": 0},
    {"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2},
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("params", stack_params)
def test_stack_forward(params, device):
    np.random.seed(0)
    shape, n, axis = params["shape"], params["n"], params["axis"]
    to_stack_ndl = []
    to_stack_npy = []
    for i in range(n):
        _A = np.random.randn(*shape)
        to_stack_ndl += [ndl.Tensor(_A, device=device)]
        to_stack_npy += [_A]

    lhs = np.stack(to_stack_npy, axis=axis)
    rhs = ndl.stack(to_stack_ndl, axis=axis)

    np.testing.assert_allclose(rhs.numpy(), lhs, rtol=1e-6)


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


def test_stack_vs_pytorch():
    np.random.seed(0)
    import torch

    A = np.random.randn(5, 5)
    B = np.random.randn(5, 5)
    C = np.random.randn(5, 5)
    D = np.random.randn(15, 5)

    Andl = ndl.Tensor(A, requires_grad=True)
    Bndl = ndl.Tensor(B, requires_grad=True)
    Cndl = ndl.Tensor(C, requires_grad=True)
    Dndl = ndl.Tensor(D, requires_grad=True)

    Atch = torch.tensor(A, requires_grad=True)
    Btch = torch.tensor(B, requires_grad=True)
    Ctch = torch.tensor(C, requires_grad=True)
    Dtch = torch.tensor(D, requires_grad=True)

    Xndl = ndl.stack([Andl, Cndl @ Bndl, Cndl], axis=1)
    Xtch = torch.stack([Atch, Ctch @ Btch, Ctch], dim=1)

    assert Xndl.shape == Xtch.shape
    assert np.linalg.norm(Xndl.numpy() - Xtch.detach().numpy()) < 1e-3

    Yndl = (Dndl @ Xndl.reshape((5, 15)) @ Dndl).sum()
    Ytch = (Dtch @ Xtch.reshape(5, 15) @ Dtch).sum()

    assert np.linalg.norm(Yndl.numpy() - Ytch.detach().numpy()) < 1e-3

    Yndl.backward()
    Ytch.backward()

    assert (
        np.linalg.norm(Andl.grad.cached_data.numpy() - Atch.grad.detach().numpy())
        < 1e-3
    )
    assert (
        np.linalg.norm(Bndl.grad.cached_data.numpy() - Btch.grad.detach().numpy())
        < 1e-3
    )
    assert (
        np.linalg.norm(Cndl.grad.cached_data.numpy() - Ctch.grad.detach().numpy())
        < 1e-3
    )


conv_forward_params = [
    (4, 8, 16, 3, 1),
    (32, 8, 16, 3, 2),
    (32, 8, 8, 3, 2),
    (32, 16, 8, 3, 1),
    (32, 16, 8, 3, 2),
]


@pytest.mark.parametrize(
    "s,in_channels,out_channels,k,stride", conv_forward_params, ids=str
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_nn_conv_forward(s, in_channels, out_channels, k, stride, device):
    f = ndl.nn.Conv(in_channels, out_channels, k, stride=stride, device=device)
    x = ndl.init.rand((10, in_channels, s, s), device=device)

    g = torch.nn.Conv2d(in_channels, out_channels, k, stride=stride, padding=k // 2)
    g.weight.data = torch.tensor(
        f.weight.realize_cached_data().numpy().transpose(3, 2, 0, 1)
    )
    g.bias.data = torch.tensor(f.bias.realize_cached_data().numpy())
    z = torch.tensor(x.cached_data.numpy())

    my_out = f(x).realize_cached_data().numpy()
    torch_out = g(z).data.numpy()

    np.testing.assert_allclose(
        my_out,
        torch_out,
        rtol=1e-4,
        atol=1e-4,
    )
    assert np.linalg.norm(my_out - torch_out) < 1e-3


conv_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]


@pytest.mark.parametrize("s,in_channels,out_channels,k,stride", conv_back_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_nn_conv_backward(s, in_channels, out_channels, k, stride, device):
    f = ndl.nn.Conv(in_channels, out_channels, k, stride=stride, device=device)
    x = ndl.init.rand((1, in_channels, s, s), device=device, requires_grad=True)

    g = torch.nn.Conv2d(in_channels, out_channels, k, stride=stride, padding=k // 2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    needle_out = f(x)
    torch_out = g(z)

    needle_y = needle_out.sum()
    torch_y = torch_out.sum()

    torch_y.backward()
    needle_y.backward()

    np.testing.assert_allclose(
        g.weight.grad.data.numpy(),
        f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        g.bias.grad.data.numpy(), f.bias.grad.cached_data.numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        z.grad.data.numpy(), x.grad.cached_data.numpy(), rtol=1e-4, atol=1e-4
    )


op_conv_shapes = [
    ((3, 14, 14, 8), (3, 3, 8, 16), 1, 0),
    ((3, 14, 14, 8), (3, 3, 8, 16), 1, 1),
    ((3, 16, 16, 8), (3, 3, 8, 16), 1, 2),
    ((3, 16, 16, 8), (3, 3, 8, 14), 1, 0),
    ((3, 16, 16, 2), (3, 3, 2, 14), 1, 0),
    ((3, 14, 14, 8), (3, 3, 8, 16), 2, 0),
    ((3, 14, 14, 8), (3, 3, 8, 16), 2, 1),
    ((3, 16, 16, 8), (3, 3, 8, 16), 2, 2),
    ((3, 16, 16, 8), (3, 3, 8, 14), 2, 0),
    ((3, 16, 16, 2), (3, 3, 2, 14), 2, 0),
    ((3, 16, 16, 24), (3, 3, 24, 14), 1, 0),
    ((3, 14, 14, 8), (5, 5, 8, 16), 1, 0),
    ((3, 17, 17, 8), (5, 5, 8, 16), 1, 0),
    ((3, 17, 17, 1), (5, 5, 1, 16), 1, 0),
    ((3, 17, 17, 16), (5, 5, 16, 1), 1, 0),
    ((3, 17, 17, 16), (1, 1, 16, 1), 1, 0),
    ((1, 14, 14, 2), (3, 3, 2, 2), 1, 0),
]


@pytest.mark.parametrize(
    "Z_shape, W_shape, stride, padding",
    op_conv_shapes,
    ids=str,
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    _Z = rng.standard_normal(Z_shape, dtype=np.float32)
    _W = rng.standard_normal(W_shape, dtype=np.float32)

    Z = ndl.Tensor(_Z, device=device)
    W = ndl.Tensor(_W, device=device)

    Z_torch = torch.tensor(_Z, dtype=torch.float32, requires_grad=True)
    W_torch = torch.tensor(_W, dtype=torch.float32, requires_grad=True)

    y = ndl.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()

    out = torch.nn.functional.conv2d(
        Z_torch.permute(0, 3, 1, 2),
        W_torch.permute(3, 2, 0, 1),
        padding=padding,
        stride=stride,
    )
    out2 = out.sum()

    np.testing.assert_allclose(
        y.numpy(),
        out.permute(0, 2, 3, 1).contiguous().detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(y2.numpy(), out2.detach().numpy(), rtol=1e-4, atol=1e-4)

    if backward:
        out2.backward()
        y2.backward()
        np.testing.assert_allclose(
            Z.grad.numpy(),
            Z_torch.grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            W.grad.numpy(),
            W_torch.grad.numpy(),
            rtol=1e-2,
            atol=1e-2,
        )


# TODO: as_strided does not match Numpy's
def test_as_strided():
    # create an array, make it different using np.stride tricks and do the same
    # to the needle array
    # then compare the two arrays
    pass
