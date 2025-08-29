import logging
import os
import random
from collections.abc import Callable

import needle as ndl
import numpy as np
import pytest
import torch
from needle import Tensor
from needle.typing.device import AbstractBackend

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = ["check_same_memory", "compare_strides", "set_random_seeds"]


RTOL = 1e-5
ATOL = 1e-5


DTYPE_FLOAT = np.float32
DTYPE_INT = np.int32


def set_random_seeds(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    ndl.cpu().set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Helper functions
def compare_strides(a_np: np.ndarray, a_nd: ndl.NDArray) -> None:
    """Check if strides match between numpy and ndarray arrays."""
    size = a_np.itemsize
    np_strides = tuple(x // size for x in a_np.strides)
    assert np_strides == a_nd.strides, (
        f"Strides {np_strides=} do not match {a_nd.strides=} "
    )


def check_same_memory(original: ndl.NDArray, view: ndl.NDArray) -> None:
    """Check if two arrays share the same memory."""
    assert original._handle.ptr() == view._handle.ptr()  # noqa: SLF001


def backward_forward():
    return pytest.mark.parametrize(
        "backward", [True, False], ids=["backward", "forward"]
    )


def generic_op_test(
    ndl_op: Callable,
    torch_op: Callable,
    inputs: list[np.ndarray],
    backward: bool,
    device: AbstractBackend,
    sum: bool = False,
) -> None:
    # Create Needle tensors
    ndl_inputs = [Tensor(arr, requires_grad=backward, device=device) for arr in inputs]
    ndl_out = ndl_op(*ndl_inputs)

    # Create Torch tensors
    torch_inputs = [
        torch.tensor(arr, dtype=torch.float32, requires_grad=backward) for arr in inputs
    ]
    torch_out = torch_op(*torch_inputs)

    # Forward check
    if not isinstance(ndl_out, Tensor):
        ndl_out = ndl_out[0]
        torch_out = torch_out[0]

    np.testing.assert_allclose(
        ndl_out.numpy(), torch_out.detach().numpy(), rtol=RTOL, atol=ATOL
    )

    assert ndl_out.device == device

    if backward:
        if not sum:
            ndl_out.backward()
        else:
            ndl_out.sum().backward()

        # check that the gradients are on the same device as the inputs
        for ndl_input in ndl_inputs:
            if not hasattr(ndl_input, "grad"):
                # This is because LogSoftmax does not produce gradients
                logging.error(f"Input tensor {ndl_input} has no gradient.")
                return
            else:
                assert ndl_input.grad.device == device

        # Backward Torch
        grad_torch = torch.autograd.grad(outputs=torch_out.sum(), inputs=torch_inputs)
        grad_torch = [g.detach().numpy() for g in grad_torch]

        ndl_grads = [t.grad.numpy() for t in ndl_inputs]

        # Gradient checks
        for g_ndl, g_torch in zip(ndl_grads, grad_torch):
            np.testing.assert_allclose(g_ndl, g_torch, rtol=RTOL, atol=ATOL)
