from __future__ import annotations

from typing import TYPE_CHECKING

from needle import init
from needle.ops.op import TensorOp, TensorTupleOp
from needle.tensor import Tensor, TensorTuple

if TYPE_CHECKING:
    from needle.backend_selection import NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad: TensorTuple, _node: Tensor) -> tuple:
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args) -> TensorTuple:
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index: int) -> None:
        self.index = index

    def __call__(self, a: TensorTuple, fold_const: bool = True) -> Tensor:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, arr: list[NDArray] | tuple[NDArray, ...]) -> NDArray:
        return arr[self.index]

    def gradient(self, out_grad, node) -> TensorTuple:
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index: int) -> Tensor:
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float) -> None:
        self.c0 = c0
        self.c1 = c1

    def compute(self, a: NDArray) -> tuple[NDArray, NDArray]:
        return a + self.c0, a + self.c1

    def gradient(self, out_grad: Tensor, node) -> Tensor:
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x: NDArray, c0: float, c1: float) -> TensorTuple:
    return FusedAddScalars(c0, c1)(x)
