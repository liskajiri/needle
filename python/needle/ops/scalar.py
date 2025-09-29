"""Scalar operator implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from needle.ops.op import TensorOp

if TYPE_CHECKING:
    from needle.backend_selection import NDArray
    from needle.needle_typing import Scalar
    from needle.tensor import Tensor


class AddScalar(TensorOp):
    def __init__(self, scalar: Scalar) -> None:
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad


def add_scalar(a: Tensor, scalar: Scalar) -> Tensor:
    return AddScalar(scalar)(a)


class MulScalar(TensorOp):
    def __init__(self, scalar: Scalar) -> None:
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * self.scalar


def mul_scalar(a: Tensor, scalar: Scalar) -> Tensor:
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: Scalar) -> None:
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        child = node.inputs[0]
        return self.scalar * out_grad * child ** (self.scalar - 1)


def power_scalar(a: Tensor, scalar: Scalar) -> Tensor:
    return PowerScalar(scalar)(a)


class DivScalar(TensorOp):
    """
    Divide a tensor by a scalar.
    """

    def __init__(self, scalar: float) -> None:
        if scalar == 0:
            raise ValueError("Cannot divide by 0")
        if isinstance(scalar, int):
            scalar = float(scalar)

        self.scalar = float(scalar)

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad / self.scalar


def divide_scalar(a: Tensor, scalar: float) -> Tensor:
    return DivScalar(scalar)(a)


class NegScalar(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return -out_grad


def neg_scalar(a: Tensor) -> Tensor:
    return NegScalar()(a)
