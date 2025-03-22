from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle.tensor import Tensor


type DType = str
type Scalar = float | int
type Shape = tuple[int, ...]
type Strides = Shape

float32: DType = "float32"

type BatchType = tuple[Tensor, ...]
