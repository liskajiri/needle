from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from needle.tensor import Tensor, TensorTuple

if TYPE_CHECKING:
    from typing import Any

    from needle.autograd.value import Value
    from needle.backend_selection import NDArray


class Op(ABC):
    """Operator definition."""

    @abstractmethod
    def __call__(self, *args) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute(self, *arr: tuple[NDArray, ...]) -> NDArray:
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: NDArray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, out_grad: Value, _node: Value) -> Value | tuple[Value]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError

    def gradient_as_tuple(self, out_grad: Value, node: Value) -> tuple[Value]:
        """Convenience method to always return a tuple from gradient call."""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        if isinstance(output, list):
            return tuple(output)
        return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors."""

    def __call__(self, *args, **kwargs) -> Tensor:
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple."""

    # TODO: override OP compute parameters

    def __call__(self, *args, **kwargs) -> TensorTuple:
        return TensorTuple.make_from_op(self, args)
