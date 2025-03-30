from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

    from needle.tensor import Tensor
    from python.needle.backend_selection import NDArray


type DType = str
type Scalar = float | int
type Shape = tuple[int, ...]
type Strides = Shape

float32: DType = "float32"

type BatchType = tuple[Tensor, ...]
# TODO: better definition of index type
type IndexType = int | slice | tuple[int | slice, ...] | list[int] | NDArray | ndarray
# TODO: Type for axes

type np_ndarray = np.ndarray
