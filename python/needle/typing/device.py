from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from needle.typing.types import DType, IndexType, Scalar, Shape, Strides, np_ndarray


class AbstractBackend[T](ABC):
    """
    A backend device, wraps the implementation module.
    """

    def __init__(self, name: str, module: ModuleProtocol[T] | None = None) -> None:
        self.name = name
        # A module that implements the backend.
        self.module = module
        self.itemsize = 1
        self.__tile_size__ = 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBackend):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        return self.name + "()"

    def __getattr__(self, name: str) -> Callable:
        """Delegate all unknown operations to the backend module."""
        if not self.enabled():
            raise NotImplementedError(
                f"Backend module not available, cannot call {name}"
            )
        return getattr(self.module, name)

    def enabled(self) -> bool:
        """Check if backend module is available."""
        return self.module is not None

    @abstractmethod
    def rand(self, shape: Shape, dtype: DType = "float32") -> T:
        raise NotImplementedError

    @abstractmethod
    def randn(self, shape: Shape, dtype: DType) -> T:
        raise NotImplementedError

    @abstractmethod
    def one_hot(self, n: int, i: IndexType, dtype: DType) -> T:
        raise NotImplementedError

    @abstractmethod
    def zeros(self, shape: Shape, dtype: DType) -> T:
        raise NotImplementedError

    @abstractmethod
    def ones(self, shape: Shape, dtype: DType) -> T:
        raise NotImplementedError

    @abstractmethod
    def empty(self, shape: Shape, dtype: DType) -> T:
        raise NotImplementedError

    @abstractmethod
    def full(self, shape: Shape, fill_value: Scalar, dtype: DType) -> T:
        raise NotImplementedError


@runtime_checkable
class ModuleProtocol[T](Protocol):
    """Protocol defining the interface required from backend modules."""

    __tile_size__: int
    itemsize: int

    def Array(self, size: int) -> T: ...
    def from_numpy(self, numpy_array: np_ndarray, handle: T) -> None: ...
    def to_numpy(
        self, handle: T, shape: Shape, strides: Strides, offset: int
    ) -> np_ndarray: ...
    def fill(self, handle: T, value: Scalar) -> None: ...
    def compact(
        self, a: T, out: T, shape: Shape, strides: Strides, offset: int
    ) -> None: ...

    # Element-wise operations
    def ewise_setitem(
        self, b: T, out: T, shape: Shape, strides: Strides, offset: int
    ) -> None: ...
    def ewise_add(self, a: T, b: T, out: T) -> None: ...
    def ewise_mul(self, a: T, b: T, out: T) -> None: ...
    def ewise_div(self, a: T, b: T, out: T) -> None: ...
    def ewise_pow(self, a: T, b: T, out: T) -> None: ...
    def ewise_maximum(self, a: T, b: T, out: T) -> None: ...
    def ewise_eq(self, a: T, b: T, out: T) -> None: ...
    def ewise_ge(self, a: T, b: T, out: T) -> None: ...

    # Element-wise math functions
    def ewise_log(self, a: T, out: T) -> None: ...
    def ewise_exp(self, a: T, out: T) -> None: ...
    def ewise_tanh(self, a: T, out: T) -> None: ...

    # Scalar operations
    def scalar_setitem(
        self,
        size: int,
        val: Scalar,
        out: T,
        shape: Shape,
        strides: Strides,
        offset: int,
    ) -> None: ...
    def scalar_add(self, a: T, val: Scalar, out: T) -> None: ...
    def scalar_mul(self, a: T, val: Scalar, out: T) -> None: ...
    def scalar_div(self, a: T, val: Scalar, out: T) -> None: ...
    def scalar_power(self, a: T, val: Scalar, out: T) -> None: ...
    def scalar_maximum(self, a: T, val: Scalar, out: T) -> None: ...
    def scalar_eq(self, a: T, val: Scalar, out: T) -> None: ...
    def scalar_ge(self, a: T, val: Scalar, out: T) -> None: ...

    # Matrix operations
    def matmul(self, a: T, b: T, out: T, m: int, n: int, p: int) -> None: ...

    # Reduction operations
    def reduce_sum(self, a: T, out: T, size: int) -> None: ...
    def reduce_max(self, a: T, out: T, size: int) -> None: ...
