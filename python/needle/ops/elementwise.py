"""Element-wise operator implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from needle.ops.op import TensorOp

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.tensor import Tensor


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        return out_grad, out_grad


def add(a: Tensor, b: Tensor) -> Tensor:
    return EWiseAdd()(a, b)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * a.log()
        return grad_a, grad_b


def power(a: Tensor, b: Tensor) -> Tensor:
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, Tensor]:
        lhs, rhs = node.inputs
        return (
            divide(out_grad, rhs),
            divide(-out_grad * lhs, rhs**2),
        )


def divide(a: Tensor, b: Tensor) -> Tensor:
    return EWiseDiv()(a, b)
