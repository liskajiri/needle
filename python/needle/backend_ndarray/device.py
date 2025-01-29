from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from typing import Any

    from needle.backend_selection import NDArray
    from needle.typing.utils import DType, Scalar, Shape


class AbstractBackend(ABC):
    """A backend device, wraps the implementation module."""

    def __init__(self, name: str, module: object = None) -> None:
        self.name = name
        # A module that implements the backend.
        self.module = module

    def __eq__(self, other: Self) -> bool:
        return self.name == other.name

    def __repr__(self) -> str:
        return self.name + "()"

    def __getattr__(self, name: str) -> "Any":
        return getattr(self.module, name)

    def enabled(self) -> bool:
        return self.module is not None

    @abstractmethod
    def rand(self, shape: "Shape", dtype: "DType" = "float32") -> "NDArray":
        raise NotImplementedError

    @abstractmethod
    def randn(self, shape: "Shape", dtype: "DType") -> "NDArray":
        raise NotImplementedError

    @abstractmethod
    def one_hot(self, n: int, i: int, dtype: "DType") -> "NDArray":
        """Create a one-hot vector.

        Args:
            n (int): Length of the vector.
            i (int): Index of the one-hot element.
            dtype (_type_, optional):

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            "NDArray": A one-hot vector.
        """
        raise NotImplementedError

    @abstractmethod
    def zeros(self, shape: "Shape", dtype: "DType") -> "NDArray":
        raise NotImplementedError

    @abstractmethod
    def ones(self, shape: "Shape", dtype: "DType") -> "NDArray":
        raise NotImplementedError

    @abstractmethod
    def empty(self, shape: "Shape", dtype: "DType") -> "NDArray":
        raise NotImplementedError

    @abstractmethod
    def full(self, shape: "Shape", fill_value: "Scalar", dtype: "DType") -> "NDArray":
        raise NotImplementedError
