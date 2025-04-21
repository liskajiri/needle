from __future__ import annotations

from typing import TYPE_CHECKING

from needle import init
from needle.ops.op import TensorOp, TensorTupleOp
from needle.tensor import Tensor, TensorTuple

if TYPE_CHECKING:
    from needle.backend_selection import NDArray


class MakeTensorTuple(TensorTupleOp):
    """
    Creates a TensorTuple from a sequence of tensors.
    This is the fundamental operation for creating tensor tuples in the autograd system.
    """

    def compute(self, *args: NDArray) -> tuple[NDArray, ...]:
        return tuple(args)

    def gradient(self, out_grad: TensorTuple, _node: Tensor) -> tuple[Tensor, ...]:
        assert isinstance(out_grad, TensorTuple)
        return tuple(out_grad[i] for i in range(len(out_grad)))


def make_tuple(*args: Tensor) -> TensorTuple:
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    """Extracts a single tensor from a TensorTuple at the specified index."""

    def __init__(self, index: int) -> None:
        self.index = index

    def __call__(self, a: TensorTuple, fold_const: bool = True) -> Tensor:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            # Get the input tensor at the specified index
            tensor = a.inputs[self.index]
            assert isinstance(tensor, Tensor)
            return tensor
        # Create a new op node with TensorTuple input
        return super().__call__(a)

    def compute(self, arr: list[NDArray] | tuple[NDArray, ...]) -> NDArray:
        return arr[self.index]

    def gradient(self, out_grad: Tensor, node: Tensor) -> TensorTuple:
        index = self.index
        in_grad = []
        input_tuple = node.inputs[0].realize_cached_data()
        for i in range(len(input_tuple)):
            if i != index:
                in_grad.append(init.zeros_like(input_tuple[i]))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value: TensorTuple, index: int) -> Tensor:
    """Extracts a tensor from a TensorTuple at the specified index."""
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    """Fuses two scalar addition operations into a single op that returns a TensorTuple.

    This optimization reduces memory allocations by computing both additions at once."""

    def __init__(self, c0: float, c1: float) -> None:
        self.c0 = c0
        self.c1 = c1

    def compute(self, a: NDArray) -> tuple[NDArray, NDArray]:
        """Returns (a + c0, a + c1) as a tuple."""
        return a + self.c0, a + self.c1

    def gradient(self, out_grad: TensorTuple, node: Tensor) -> Tensor:
        """Combines gradients from both outputs."""
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x: NDArray, c0: float, c1: float) -> TensorTuple:
    return FusedAddScalars(c0, c1)(x)
