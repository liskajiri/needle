from typing import TYPE_CHECKING, TypedDict

from numpy.typing import ArrayLike

from needle.typing.device import AbstractBackend

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import Tensor

type np_ndarray = ArrayLike


type DType = str
type Scalar = float | int
type Shape = tuple[int, ...]
# TODO: default type values
type Axis = int | tuple[int, ...]
# TODO: Axes vs axis
type Strides = Shape
type NDArrayLike = NDArray | np_ndarray | list[Scalar] | tuple[Scalar, ...] | Scalar

# TODO: proper type, this clashes with certain things, get something like nd.float32
float32: DType = "float32"

type BatchType = tuple[Tensor, ...]
# TODO: better definition of index type
type IndexType = (
    int | slice | tuple[int | slice, ...] | list[int] | NDArray | np_ndarray
)
# TODO: Type for axes


class TensorKwargs(TypedDict, total=False):
    """Type for Tensor keyword arguments."""

    device: AbstractBackend
    dtype: DType
    requires_grad: bool
