from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from numpy import ndarray

    from needle.backend_selection import NDArray
    from needle.tensor import Tensor


type DType = str
type Scalar = float | int
type Shape = tuple[int, ...]
type Strides = Shape

# TODO: proper type, this clashes with certain things, get something like nd.float32
float32: DType = "float32"

type BatchType = tuple[Tensor, ...]
# TODO: better definition of index type
type IndexType = int | slice | tuple[int | slice, ...] | list[int] | NDArray | ndarray
# TODO: Type for axes

type np_ndarray = ArrayLike
