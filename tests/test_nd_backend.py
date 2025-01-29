import needle as ndl
import numpy as np
import pytest

# TODO: skip torch tests if not able to import
import torch
from needle import backend_ndarray as nd

np.random.seed(1)


def to_torch_tensor(needle_tensor):
    """Convert needle tensor to torch tensor safely"""
    if hasattr(needle_tensor, "numpy"):
        data = needle_tensor.numpy()
    elif hasattr(needle_tensor, "realize_cached_data"):
        data = needle_tensor.realize_cached_data()
    else:
        data = needle_tensor
    return torch.tensor(data, requires_grad=True)


def handle_shape_op(op_name, torch_args, args, kwargs):
    """Handle shape manipulation operations"""
    if op_name == "reshape":
        shape = kwargs.get("shape") or args[1]
        return torch_args[0].reshape(shape)
    elif op_name == "transpose":
        axes = kwargs.get("axes") or args[1]
        if isinstance(axes, list | tuple):
            # Match dimensions count
            n_dims = torch_args[0].dim()
            if len(axes) != n_dims:
                axes = list(range(n_dims))
            return torch_args[0].permute(*axes)
        return torch_args[0].transpose(axes)
    elif op_name == "broadcast_to":
        shape = kwargs.get("shape") or args[1]
        return torch_args[0].expand(shape)
    return None


# TODO: is this really the same as checking numerically?
# TODO: move to a separate file - utils.py?
def backward_check(f, *args, tol: float = 1e-5, backward: bool = False, **kwargs):
    """Compare numerical and analytical gradients, with optional PyTorch comparison"""
    eps = 1e-4

    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    # 1. Compute numerical gradients
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flatten()[j] += eps
            f1 = float((f(*args, **kwargs).numpy() * c).sum())
            args[i].realize_cached_data().flatten()[j] -= 2 * eps
            f2 = float((f(*args, **kwargs).numpy() * c).sum())
            args[i].realize_cached_data().flatten()[j] += eps
            numerical_grads[i].flatten()[j] = (f1 - f2) / (2 * eps)

    # 2. Compute analytical gradients
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.numpy()
            for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]

    # 3. Try PyTorch comparison if possible
    try:
        torch_args = [to_torch_tensor(a) for a in args]
        op_name = f.__name__ if hasattr(f, "__name__") else f.__class__.__name__

        # Try shape operations first
        torch_out = handle_shape_op(op_name, torch_args, args, kwargs)

        if torch_out is None:
            # Standard operations
            op_map = {
                "matmul": torch.matmul,
                "multiply": torch.multiply,
                "divide": torch.divide,
                "divide_scalar": lambda x, scalar=None: x
                / (scalar or kwargs.get("scalar")),
                "add": torch.add,
                "sum": torch.sum,
                "summation": torch.sum,
                "negate": lambda x: -x,
                "exp": torch.exp,
                "log": torch.log,
                "softmax_loss": lambda x, y: torch.nn.functional.cross_entropy(x, y),
                "relu": torch.nn.functional.relu,
                "tanh": torch.nn.functional.tanh,
            }
            torch_f = op_map.get(op_name)
            if torch_f:
                # Handle divide_scalar specially
                if op_name == "divide_scalar":
                    scalar = kwargs.get("scalar")
                    torch_out = torch_f(torch_args[0], scalar=scalar)
                else:
                    torch_out = torch_f(*torch_args)
            else:
                print(f"Skipping PyTorch comparison for {op_name}")
                torch_out = None
                assert False

        # Compute gradients if we have an output
        if torch_out is not None:
            torch_out = torch_out.sum()
            torch_out.backward()
            torch_grads = [t.grad.numpy() for t in torch_args]

            # Compare gradients
            for i in range(len(args)):
                np.testing.assert_allclose(
                    computed_grads[i], torch_grads[i], rtol=tol, atol=tol
                )

    except Exception as e:
        raise e

    # numerical_tol = 1e-1
    # max_error = sum(
    #     np.linalg.norm(computed_grads[i] - numerical_grads[i])
    # for i in range(len(args))
    # )
    # assert max_error < numerical_tol, (
    #     f"Gradient check failed. Max error: {max_error:.4e}"
    # )

    return computed_grads


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
EWISE_OP_NAMES = [k for k in EWISE_OPS]
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
SCALAR_OP_NAMES = [k for k in SCALAR_OPS]


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


@pytest.mark.parametrize("m,n,p", MATMUL_DIMS)
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


@pytest.mark.parametrize("shape, axis, i", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack(shape, axis, i, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(i)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(i)]
    A_t = [torch.Tensor(_A[i]) for i in range(i)]
    out = ndl.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)
    np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape, axis, i", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.skipif(
    ndl.cpu().name == "numpy", reason="Numpy has different stack/split semantics"
)
def test_stack_backward(shape, axis, i, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(i)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(i)]
    A_t = [torch.Tensor(_A[i]) for i in range(i)]
    for i in range(i):
        A_t[i].requires_grad = True
    ndl.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()
    for i in range(i):
        np.testing.assert_allclose(
            A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5
        )


SUMMATION_PARAMETERS = [((1, 1, 1), None), ((5, 3), 0), ((8, 3, 2), 1), ((8, 3, 2), 2)]


@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(
        np.sum(_A, axes), ndl.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation_backward(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.summation, A, axes=axes)


BROADCAST_SHAPES = [((1, 1, 1), (3, 3, 3)), ((4, 1, 6), (4, 3, 6))]


@pytest.mark.parametrize("shape,shape_to", BROADCAST_SHAPES)
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


@pytest.mark.parametrize("shape,shape_to", RESHAPE_SHAPES)
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


@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsumexp(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    A_t = torch.Tensor(_A)
    t_axes = tuple(list(range(len(shape)))) if axes is None else axes
    np.testing.assert_allclose(
        torch.logsumexp(A_t, dim=t_axes).numpy(),
        ndl.logsumexp(A, axes=axes).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "test_case",
    [
        # shape, scale factor for values
        ((2, 3), 1),  # basic case
        ((5, 4), 1),  # different shape
        ((2, 3), 1000),  # large values
        ((10, 10), 0.001),  # small values
    ],
)
def test_logsoftmax(test_case):
    shape, scale = test_case

    # Setup
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate data
    x = np.random.randn(*shape) * scale
    needle_x = ndl.Tensor(x, dtype="float32")
    torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    # Forward pass
    needle_out = ndl.ops.logsoftmax(needle_x)
    torch_out = torch.nn.functional.log_softmax(torch_x, dim=1)

    # Test forward
    np.testing.assert_allclose(
        needle_out.numpy(), torch_out.detach().numpy(), rtol=1e-5, atol=1e-5
    )

    # TODO: backward check
    # # Test backward
    # needle_out.sum().backward()
    # torch_out.sum().backward()

    # np.testing.assert_allclose(
    #     needle_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5
    # )


def test_logsoftmax_invalid():
    # Test 1D input
    with pytest.raises(AssertionError):
        ndl.ops.logsoftmax(ndl.Tensor(np.array([1.0, 2.0])))

    # Test 3D input
    with pytest.raises(AssertionError):
        ndl.ops.logsoftmax(ndl.Tensor(np.random.randn(2, 3, 4)))
