"""Core data structures."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import needle as ndl
from needle.backend_selection import Device, NDArray, array_api, cpu, default_device

if TYPE_CHECKING:
    from typing import Any, Self

    from needle.autograd import Tensor, TensorTuple
    from needle.backend_selection import DType


# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

type Scalar = int | float


class Op:
    """Operator definition."""

    def __call__(self, *args) -> Any:
        raise NotImplementedError()

    # TODO: Always one arg, second optional
    def compute(self, *args: list[NDArray]) -> NDArray:
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: Value, node: Value) -> tuple[Value]:
        """Convenience method to always return a tuple from gradient call."""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors."""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple."""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Op | None
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
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Op | None,
        inputs: list[Tensor],
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
    def make_from_op(cls, op: Op, inputs: list[Self]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self) -> int:
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return ndl.ops.tuple_get_item(self, index)

    def tuple(self) -> tuple[Self]:
        return tuple([x for x in self])
        # return tuple(self)

    def __repr__(self) -> str:
        return "ndl.TensorTuple" + str(self.tuple())

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: TensorTuple) -> TensorTuple:
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return ndl.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self) -> TensorTuple:
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: Self

    def __init__(
        self,
        array: NDArray | Tensor,
        *,
        device: type[Device] | None = None,
        dtype: DType | None = None,
        requires_grad: bool = True,
        **kwargs,
    ) -> None:
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array: np.ndarray, device: Device, dtype: DType):
        if array_api is np:
            return np.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @classmethod
    def make_const(cls, data, *, requires_grad: bool = False) -> Self:
        if isinstance(data, Tensor):
            data = data.realize_cached_data()
        return super().make_const(data, requires_grad=requires_grad)

    @property
    def data(self) -> Tensor:
        return self.detach()

    @data.setter
    def data(self, value) -> None:
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, f"{value.dtype} {self.dtype}"
        self.cached_data = value.realize_cached_data()

    def detach(self) -> Tensor:
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is np:
            return cpu()
        return data.device

    def backward(self, out_grad: Tensor | None = None) -> None:
        out_grad = (
            out_grad
            if out_grad
            else ndl.init.init_basic.ones(
                *self.shape, dtype=self.dtype, device=self.device
            )
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self) -> str:
        return "ndl.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self) -> str:
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        # TODO: no need for this? numpy_api is patched
        if array_api is np:
            return data
        return data.numpy()

    def __add__(self, other: Tensor | Scalar) -> Tensor:
        if isinstance(other, Tensor):
            return ndl.ops.EWiseAdd()(self, other)
        return ndl.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return ndl.ops.EWiseMul()(self, other)
        return ndl.ops.MulScalar(other)(self)

    def __pow__(self, other):
        return ndl.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return ndl.ops.EWiseAdd()(self, ndl.ops.Negate()(other))
        return ndl.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return ndl.ops.EWiseDiv()(self, other)
        return ndl.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return ndl.ops.MatMul()(self, other)

    def matmul(self, other):
        return ndl.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return ndl.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return ndl.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return ndl.ops.Reshape(shape)(self)

    def __neg__(self):
        return ndl.ops.Negate()(self)

    def transpose(self, axes=None):
        return ndl.ops.Transpose(axes)(self)

    @property
    def T(self):
        return self.transpose()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    __radd__ = __add__
    __rmul__ = __mul__


def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor) -> None:
    """
    Take gradient of output node with respect to each node in node_list.
    Store the computed result in the grad field of each Variable.
    """

    def _sum_node_list(node_list: list[Tensor]) -> Tensor:
        """Avoid creating redundant nodes in Python sum implementation."""
        from functools import reduce
        from operator import add

        return reduce(add, node_list)

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads: defaultdict[Tensor, list[Tensor]] = defaultdict(list)
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are
    # taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        node.grad = _sum_node_list(node_to_output_grads[node])

        if node.op:
            # partial adjoints
            grads = node.op.gradient_as_tuple(node.grad, node)
            for input_node, grad in zip(node.inputs, grads):
                node_to_output_grads[input_node].append(grad)

            # Gradients do not need to be kept further in the AD graph
            node.grad = node.grad.detach()


def find_topo_sort(node_list: list[Tensor]) -> list[Tensor]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    topo_order = []
    visited = set()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node: Tensor, visited: set, topo_order: list[Tensor]) -> None:
    """Post-order DFS."""
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)
