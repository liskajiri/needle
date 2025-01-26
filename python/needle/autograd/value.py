from typing import TYPE_CHECKING, Self

from needle.backend_selection import NDArray

if TYPE_CHECKING:
    from needle.ops.op import Op


# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: "Op | None"
    inputs: list[Self]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    # TODO: Use more functools - cached_property, lru_cache...
    # @cached_property
    # def realize_cached_data(self) -> NDArray:
    #     """
    #     Computes and caches tensor data lazily.
    #     Returns cached result on subsequent calls.
    #     """
    #     # if self._cached_data is not None:
    #     #     return self._cached_data

    #     def input_data() -> Generator[NDArray, None, None]:
    #         for x in self.inputs:
    #             yield x.realize_cached_data

    #     self.cached_data = self.op.compute(*input_data())
    #     return self.cached_data

    def realize_cached_data(self) -> NDArray:
        """Run compute to realize the cached data."""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self) -> bool:
        return self.op is None

    def __del__(self) -> None:
        # TODO: Python weakref counter
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    # TODO: __init__, add call to super in Tensor
    def _init(
        self,
        op: "Op | None",
        inputs: list[Self],
        # TODO
        *,
        num_outputs: int = 1,
        cached_data: NDArray | None = None,
        requires_grad: bool | None = None,
    ) -> None:
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad: bool = False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: "Op", inputs: list[Self]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value
