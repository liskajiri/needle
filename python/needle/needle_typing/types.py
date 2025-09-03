from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from array import array as std_array

    # from needle.backend_selection import NDArray
    from needle.backend_ndarray.ndarray import NDArray
    from needle.tensor import Tensor

    # try:
    #     # from numpy.typing import NDArray as NP_NDArray
    #     # type np_ndarray = NP_NDArray

    #     from numpy.typing import NDArray as _NDArray  # type: ignore

    #     np_ndarray = _NDArray
    # except ImportError:
    #     # fallback if numpy is not installed
    #     pass

type np_ndarray = object

type DType = str
type Scalar = float | int
type Shape = tuple[int, ...]
type Axis = int | tuple[int, ...]
type Strides = Shape
type NDArrayLike = NDArray | Sequence[Scalar] | tuple[Scalar, ...] | Scalar | std_array

# TODO: proper type, this clashes with certain things, get something like nd.float32
float32: DType = "float32"

type BatchType = tuple[Tensor, ...]

type SingleIndex = int | slice
type ListTupleIndex = tuple[int, ...] | tuple[slice, ...] | list[int]
type ArrayIndex = ListTupleIndex | NDArray
type IndexType = SingleIndex | ArrayIndex
