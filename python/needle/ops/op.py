from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from needle.tensor import Tensor, TensorTuple

if TYPE_CHECKING:
    from needle.backend_selection import NDArray


class Op(ABC):
    """Operator definition."""

    # TODO: proper Typing of args

    @abstractmethod
    def __call__(self, *args: NDArray) -> object:
        raise NotImplementedError

    @abstractmethod
    def compute(self, *arr: NDArray) -> NDArray:
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
    def gradient(self, out_grad: Tensor, _node: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Tensor
            The adjoint wrt to the output value.

        _node: Tensor
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Tensor or Tuple[Tensor]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError

    def gradient_as_tuple(self, out_grad: Tensor, node: Tensor) -> tuple[Tensor, ...]:
        """Convenience method to always return a tuple from gradient call."""
        output = self.gradient(out_grad, node)
        if isinstance(output, (tuple | list)):
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
