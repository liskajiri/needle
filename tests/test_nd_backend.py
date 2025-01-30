import needle as ndl
import numpy as np
import pytest
import torch
from needle import backend_ndarray as nd

from tests.gradient_check import backward_check

_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(),
        marks=pytest.mark.skipif(
            not ndl.cuda().enabled(),
            reason="No GPU",
        ),
    ),
]


EWISE_OPS = {"divide": lambda a, b: a / b, "subtract": lambda a, b: a - b}
EWISE_OP_FNS = [EWISE_OPS[k] for k in EWISE_OPS]
EWISE_OP_NAMES = list(EWISE_OPS)
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]


@pytest.mark.parametrize("fn", EWISE_OP_FNS, ids=EWISE_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)


SCALAR_OPS = {"divide": lambda a, b: a / b, "subtract": lambda a, b: a - b}
SCALAR_OP_FNS = [SCALAR_OPS[k] for k in SCALAR_OPS]
SCALAR_OP_NAMES = list(SCALAR_OPS)


@pytest.mark.parametrize("fn", SCALAR_OP_FNS, ids=SCALAR_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(1).astype(np.float32).item()
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, _B).numpy(), atol=1e-5, rtol=1e-5)


MATMUL_DIMS = [
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 32),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]


@pytest.mark.parametrize(("m", "n", "p"), MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_matmul(m, n, p, device, atol=1e-5, rtol=1e-5):
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    # for large matrices, relax the tolerance
    if m * n * p >= 128 * 128 * 128:
        atol = 1e-4
        rtol = 1e-4
    np.testing.assert_allclose(
        _A @ _B,
        (A @ B).numpy(),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_power(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randint(1)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(_A**_B, (A**_B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_log(shape, device):
    _A = np.random.randn(*shape).astype(np.float32) + 5.0
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.log(_A), ndl.log(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_exp(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.exp(_A), ndl.exp(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_relu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.maximum(_A, 0), ndl.relu(A).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_tanh(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.tanh(_A), ndl.tanh(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sqrt(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.sqrt(_A), ndl.sqrt(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_tanh_backward(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.tanh, A)


STACK_PARAMETERS = [
    ((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1, 5, 7), 2, 5),
    ((1, 3, 3), 0, 3),
    ((2, 2, 2, 2), 0, 3),
]


@pytest.mark.parametrize(("shape", "axis", "i"), STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack(shape, axis, i, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(i)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(i)]
    A_t = [torch.Tensor(_A[i]) for i in range(i)]
    out = ndl.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)
    np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("shape", "axis", "i"), STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.skipif(
    ndl.cpu().name == "numpy", reason="Numpy has different stack/split semantics"
)
def test_stack_backward(shape, axis, i, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(i)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(i)]
    A_t = [torch.Tensor(_A[i]) for i in range(i)]
    for idx in range(i):
        A_t[idx].requires_grad = True
    ndl.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()
    for idx in range(i):
        np.testing.assert_allclose(
            A_t[idx].grad.numpy(), A[idx].grad.numpy(), atol=1e-5, rtol=1e-5
        )


SUMMATION_PARAMETERS = [((1, 1, 1), None), ((5, 3), 0), ((8, 3, 2), 1), ((8, 3, 2), 2)]


@pytest.mark.parametrize(("shape", "axes"), SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.sum(_A, axes), ndl.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(("shape", "axes"), SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation_backward(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.summation, A, axes=axes)


BROADCAST_SHAPES = [((1, 1, 1), (3, 3, 3)), ((4, 1, 6), (4, 3, 6))]


@pytest.mark.parametrize(("shape", "shape_to"), BROADCAST_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.broadcast_to(_A, shape_to),
        ndl.broadcast_to(A, shape_to).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


RESHAPE_SHAPES = [((1, 1, 1), (1,)), ((4, 1, 6), (6, 4, 1))]


@pytest.mark.parametrize(("shape", "shape_to"), RESHAPE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.reshape(_A, shape_to), ndl.reshape(A, shape_to).numpy(), atol=1e-5, rtol=1e-5
    )


TRANSPOSE_SHAPES = [(1, 1, 1), (4, 5, 6)]
TRANSPOSE_AXES = [(0, 1), (0, 2), None]


@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("axes", TRANSPOSE_AXES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_transpose(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np_axes = (_A.ndim - 2, _A.ndim - 1) if axes is None else axes
    np.testing.assert_allclose(
        np.swapaxes(_A, np_axes[0], np_axes[1]),
        ndl.transpose(A, axes=axes).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(("shape", "axes"), SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsumexp(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    A_t = torch.Tensor(_A)
    t_axes = tuple(range(len(shape))) if axes is None else axes
    np.testing.assert_allclose(
        torch.logsumexp(A_t, dim=t_axes).numpy(),
        ndl.logsumexp(A, axes=axes).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


LOGSOFTMAX_SHAPES = (
    # shape, scale factor for values
    ((2, 3), 1),  # basic case
    ((5, 4), 1),  # different shape
    ((2, 3), 1000),  # large values
    ((10, 10), 0.001),  # small values
)


@pytest.mark.parametrize(
    ("shape", "scale"),
    LOGSOFTMAX_SHAPES,
    ids=[f"[{shape}-{scale}]" for shape, scale in LOGSOFTMAX_SHAPES],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsoftmax_forward(shape, scale, device):
    x = np.random.randn(*shape).astype(np.float32) * scale
    needle_x = ndl.Tensor(nd.array(x), device=device)
    torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    needle_out = ndl.ops.logsoftmax(needle_x)
    torch_out = torch.nn.functional.log_softmax(torch_x, dim=1)

    np.testing.assert_allclose(
        needle_out.numpy(), torch_out.detach().numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    ("shape", "scale"),
    LOGSOFTMAX_SHAPES,
    ids=[f"[{shape}-{scale}]" for shape, scale in LOGSOFTMAX_SHAPES],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsoftmax_backward(shape, scale, device):
    x = np.random.randn(*shape).astype(np.float32) * scale
    needle_x = ndl.Tensor(nd.array(x), device=device)
    torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    needle_out = ndl.ops.logsoftmax(needle_x)
    torch_out = torch.nn.functional.log_softmax(torch_x, dim=1)

    np.testing.assert_allclose(
        needle_out.numpy(), torch_out.detach().numpy(), rtol=1e-5, atol=1e-5
    )

    needle_out.sum().backward()
    torch_out.sum().backward()

    np.testing.assert_allclose(
        needle_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5
    )


def test_logsoftmax_invalid():
    # Test 1D input
    with pytest.raises(AssertionError):
        ndl.ops.logsoftmax(ndl.Tensor(np.array([1.0, 2.0])))

    # Test 3D input
    with pytest.raises(AssertionError):
        ndl.ops.logsoftmax(ndl.Tensor(np.random.randn(2, 3, 4)))
