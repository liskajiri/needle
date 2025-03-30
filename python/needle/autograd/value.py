from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from needle.backend_selection import NDArray
    from needle.ops.op import Op


LAZY_MODE = False


class Value(ABC):
    """
    A value in the computational graph.
    """

    # trace of computational graph
    op: Op | None
    inputs: list[Self]
    # The following fields are cached fields for dynamic computation
    cached_data: NDArray | None
    requires_grad: bool
    _counter: int = 0

    def _init(
        self,
        op: Op | None = None,
        inputs: Iterable[Self] = [],
        cached_data: NDArray | None = None,
        num_outputs: int = 1,
        requires_grad: bool | None = None,
    ) -> None:
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = list(inputs)
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad
        Value._counter += 1

    def realize_cached_data(self) -> NDArray:
        """Run compute to realize the cached data."""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        assert self.op is not None, "Cannot call realize_cached_data on a leaf node"
        for x in self.inputs:
            assert isinstance(x, Value), "Inputs must be of type Value"

        self.cached_data = self.op.compute(
            *tuple(x.realize_cached_data() for x in self.inputs)  # type: ignore
        )
        return self.cached_data

    @property
    def is_leaf(self) -> bool:
        return self.op is None

    def __del__(self) -> None:
        Value._counter -= 1

    @classmethod
    def make_const(cls, data: NDArray, requires_grad: bool = False):
        value = cls.__new__(cls)
        value._init(cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: Iterable[Self]) -> Self:
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value

    @abstractmethod
    def detach(self) -> Self:
        """Return new Value with same data but no gradient computation."""
        raise NotImplementedError
