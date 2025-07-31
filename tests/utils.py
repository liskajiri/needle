import os
import random

import needle as ndl
import numpy as np
import pytest

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = ["check_same_memory", "compare_strides", "set_random_seeds"]


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
