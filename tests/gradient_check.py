import logging
from collections.abc import Callable

import needle as ndl
import numpy as np
import torch
from needle.tensor import Tensor

logger = logging.getLogger(__name__)


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
    if op_name == "transpose":
        axes = kwargs.get("axes") or args[1]
        if isinstance(axes, list | tuple):
            # Match dimensions count
            n_dims = torch_args[0].dim()
            if len(axes) != n_dims:
                axes = list(range(n_dims))
            return torch_args[0].permute(*axes)
        return torch_args[0].transpose(axes)
    if op_name == "broadcast_to":
        shape = kwargs.get("shape") or args[1]
        return torch_args[0].expand(shape)
    return None


def numerical_gradient(f, computed_grads, *args, tol: float = 1e-5, **kwargs):
    eps = 1e-4
    rng = np.random.default_rng()

    out = f(*args, **kwargs)
    c = rng.standard_normal(out.shape)
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flatten()[j] += eps
            f1 = float((f(*args, **kwargs).numpy() * c).sum())
            args[i].realize_cached_data().flatten()[j] -= 2 * eps
            f2 = float((f(*args, **kwargs).numpy() * c).sum())
            args[i].realize_cached_data().flatten()[j] += eps
            numerical_grads[i].flatten()[j] = (f1 - f2) / (2 * eps)

    numerical_tol = 1e-4
    max_error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert max_error < numerical_tol, (
        f"Gradient check failed. Max error: {max_error:.4e}"
    )


def handle_torch_op(
    f: Callable, args: tuple[Tensor], kwargs: dict, tol: float
) -> list[np.ndarray] | None:
    """Handle PyTorch operation mapping comparison."""
    torch_args = [to_torch_tensor(a) for a in args]
    op_name = f.__name__ if hasattr(f, "__name__") else f.__class__.__name__

    TORCH_OP_MAP = {
        "matmul": torch.matmul,
        "multiply": torch.multiply,
        "divide": torch.divide,
        "divide_scalar": lambda x, scalar=None: x / (scalar or kwargs.get("scalar")),
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

    # Try shape operations first
    torch_out = handle_shape_op(op_name, torch_args, args, kwargs)
    if torch_out is None:
        torch_f = TORCH_OP_MAP.get(op_name)
        if not torch_f:
            logger.error("Skipping PyTorch comparison for %s", op_name)
            return None

        # Handle special cases
        if op_name == "divide_scalar":
            torch_out = torch_f(torch_args[0], scalar=kwargs.get("scalar"))
        else:
            torch_out = torch_f(*torch_args)

    torch_out.sum().backward()
    return [t.grad.numpy() for t in torch_args]


def compute_ndl_gradients(
    f: Callable, args: tuple[Tensor, ...], backward: bool, **kwargs
) -> list[np.ndarray]:
    """Compute gradients using NDL."""
    out = f(*args, **kwargs)
    if not backward:
        return [
            x.numpy()
            for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)
        ]
    out.sum().backward()
    return [a.grad.numpy() for a in args]


# TODO: Improve this check
def backward_check(
    f, *args, tol: float = 1e-5, backward: bool = False, **kwargs
) -> list[np.ndarray]:
    """Compare numerical and analytical gradients, with optional PyTorch comparison"""
    torch_fn = kwargs.pop("torch_fn", None)
    torch_args = kwargs.pop("torch_args", None)

    computed_grads = compute_ndl_gradients(f, args, backward, **kwargs)
    if torch_fn:
        torch_args = [torch.tensor(arg.numpy(), requires_grad=True) for arg in args]

        # Run PyTorch function
        torch_out = torch_fn(*torch_args)
        torch_out.sum().backward()
        torch_grads = [t.grad.numpy() for t in torch_args]
    else:
        torch_grads = handle_torch_op(f, args, kwargs, tol)

    if torch_grads:
        # Compare gradients
        for i in range(len(args)):
            np.testing.assert_allclose(
                computed_grads[i], torch_grads[i], rtol=tol, atol=tol
            )

    # numerical_gradient(f, computed_grads, *args, tol=tol, **kwargs)

    return computed_grads
