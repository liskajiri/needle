from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from array import array as std_array

    # from needle.backend_selection import NDArray
    from needle.backend_ndarray.ndarray import NDArray
    from needle.tensor import Tensor

from numpy.typing import NDArray as NP_NDArray

type np_ndarray = NP_NDArray


type DType = str
type Scalar = float | int
type Shape = tuple[int, ...]
# TODO: default type values
type Axis = int | tuple[int, ...]
# TODO: Axes vs axis
type Strides = Shape
type NDArrayLike = (
    NDArray | np_ndarray | list[Scalar] | tuple[Scalar, ...] | Scalar | std_array
)

# TODO: proper type, this clashes with certain things, get something like nd.float32
float32: DType = "float32"

type BatchType = tuple[Tensor, ...]

type SingleIndex = int | slice  # TODO: add Ellipsis
type ListTupleIndex = list[int] | tuple[int, ...]
type ArrayIndex = Iterable[int | bool] | NDArray | np_ndarray
type IndexType = SingleIndex | ArrayIndex
