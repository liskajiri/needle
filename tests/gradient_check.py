import logging
from collections.abc import Callable
from typing import ClassVar

import needle as ndl
import numpy as np
import torch
from needle.tensor import Tensor

logger = logging.getLogger(__name__)

rng = np.random.default_rng()


def numerical_gradient(f, out, computed_grads, *args, tol: float = 1e-4, **kwargs):
    eps = 1e-4
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

    max_error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert max_error < tol, f"Gradient check failed. Max error: {max_error:.4e}"


class TorchOps:
    OP_MAP: ClassVar = {
        "matmul": torch.matmul,
        "multiply": torch.multiply,
        "divide": torch.divide,
        "add": torch.add,
        "sum": torch.sum,
        "summation": torch.sum,
        "negate": lambda x: -x,
        "exp": torch.exp,
        "log": torch.log,
        "softmax_loss": lambda x, y: torch.nn.functional.cross_entropy(x, y),
        "relu": torch.nn.functional.relu,
        "tanh": torch.nn.functional.tanh,
        "divide_scalar": lambda x, scalar: x / scalar,
    }
    SHAPE_OPS: ClassVar = {
        # Shape manipulation
        "reshape": lambda t, shape: t.reshape(shape),
        "transpose": lambda t, axes: t.transpose(*axes),
        "broadcast_to": lambda t, shape: t.expand(shape),
        "flip": lambda t, dims: torch.flip(t, dims),
        "dilate": lambda t, axes: torch.nn.functional.pad(
            t,
            tuple(
                1 if i in axes else 0  # Add padding of dilation zeros between elements
                for dim in range(t.dim())
                for i in (dim, dim)
            ),
            mode="constant",
            value=0,
        ),
        "stack": lambda t, _: torch.stack(t),
    }

    @classmethod
    def get_op(cls, op_name: str) -> Callable:
        if op_name in cls.OP_MAP:
            return cls.OP_MAP[op_name]
        if op_name in cls.SHAPE_OPS:
            return cls.SHAPE_OPS[op_name]
        raise ValueError(f"PyTorch function not found for: {op_name}")


def handle_torch_op(
    f: Callable, torch_args: list[Tensor], kwargs: dict
) -> list[np.ndarray]:
    """Handle PyTorch operation mapping comparison."""
    op_name = f.__name__ if hasattr(f, "__name__") else f.__class__.__name__

    op = TorchOps.get_op(op_name)
    if op_name in TorchOps.SHAPE_OPS:
        # Handle shape manipulation operations
        shape_or_axes = (
            torch_args[1]
            if len(torch_args) > 1
            else kwargs.get("shape") or kwargs.get("axes")
        )
        if op_name == "stack":
            torch_out = op(torch_args, shape_or_axes)
        else:
            torch_out = op(torch_args[0], shape_or_axes)
    else:
        torch_out = op(*torch_args)

    torch_out.sum().backward()
    return [t.grad.numpy() for t in torch_args]  # type: ignore


def compute_ndl_gradients(out, args: tuple[Tensor], backward: bool) -> list[np.ndarray]:
    """Compute gradients using needle."""
    if not backward:
        backward_grad = out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)
        if isinstance(backward_grad[0], ndl.TensorTuple):  # TODO keep this?
            backward_grad = backward_grad[0].tuple()
        return [g.numpy() for g in backward_grad]
    out.sum().backward()
    return [a.grad.numpy() for a in args]


def _compute_torch_gradients(
    f: Callable, tensors: tuple[Tensor], torch_fn, kwargs: dict
) -> list[np.ndarray]:
    torch_args = []
    if isinstance(tensors[0], list):
        tensors = tensors[0]
    for t in tensors:
        if isinstance(t, float):
            torch_args.append(torch.tensor(t, requires_grad=True))
        else:
            torch_args.append(torch.tensor(t.numpy(), requires_grad=True))

    if torch_fn:
        # Run PyTorch function
        torch_out = torch_fn(*torch_args)
        torch_out.sum().backward()
        torch_grads = [t.grad.numpy() for t in torch_args]  # type: ignore
    else:
        torch_grads = handle_torch_op(f, torch_args, kwargs)

    return torch_grads


# TODO: Improve this check
# TODO: https://stackoverflow.com/questions/57627406/how-to-use-autograd-gradcheck-in-pytorch
def backward_check(
    f,
    *tensors,
    tol: float = 1e-4,
    backward: bool = False,
    **kwargs,
) -> list[np.ndarray]:
    """Compare numerical and analytical gradients, with optional PyTorch comparison"""
    torch_fn = kwargs.pop("torch_fn", None)

    out = f(*tensors, **kwargs)
    computed_grads = compute_ndl_gradients(out, tensors, backward)
    torch_grads = _compute_torch_gradients(f, tensors, torch_fn, kwargs)

    if torch_grads:
        # Compare gradients
        error = sum(
            np.linalg.norm(computed - torch_grad)
            for computed, torch_grad in zip(computed_grads, torch_grads, strict=False)
        )
        assert error <= tol, f"Gradient check failed. Error: {error:.4e}"
        logger.info(f"Gradient check passed. Error: {error:.4e}")

        for computed, torch_grad in zip(computed_grads, torch_grads, strict=False):
            np.testing.assert_allclose(computed, torch_grad, rtol=tol, atol=tol)

    # numerical_gradient(f, out, computed_grads, *args, tol=tol, **kwargs)

    return computed_grads
