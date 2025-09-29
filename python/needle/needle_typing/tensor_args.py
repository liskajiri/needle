from typing import TypedDict

from needle.needle_typing.device import AbstractBackend
from needle.needle_typing.types import DType


class TensorKwargs(TypedDict, total=False):
    """Type for Tensor keyword arguments."""

    device: AbstractBackend
    dtype: DType
    requires_grad: bool
