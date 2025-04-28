from typing import TypedDict

from needle.typing.device import AbstractBackend
from needle.typing.types import DType


class TensorKwargs(TypedDict, total=False):
    """Type for Tensor keyword arguments."""

    device: AbstractBackend
    dtype: DType
    requires_grad: bool
