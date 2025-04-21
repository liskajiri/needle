from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle.tensor import Tensor


def _sum_node_list(node_list: list[Tensor]) -> Tensor:
    """Avoid creating redundant nodes in Python sum implementation."""
    from functools import reduce
    from operator import add

    if len(node_list) == 0:
        raise ValueError("Cannot sum an empty list of nodes.")

    return reduce(add, node_list)


def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor) -> None:
    """
    Take gradient of output node with respect to each node in node_list.
    Store the computed result in the grad field of each Variable.
    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads: defaultdict[Tensor, list[Tensor]] = defaultdict(list)
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are
    # taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for indent, node in enumerate(reverse_topo_order):
        node.grad = _sum_node_list(node_to_output_grads[node])
        logging.debug(f"{indent * '='} Node: {node.__class__.__name__}")
        logging.debug(f"{indent * '='} Grad: {node.grad.__class__.__name__}")

        if node.op:
            logging.debug(f"{indent * '='} OP: {node.op.__class__.__name__}")
            # partial adjoints
            grads = node.op.gradient_as_tuple(node.grad, node)
            for input_node, grad in zip(node.inputs, grads, strict=False):
                node_to_output_grads[input_node].append(grad)  # type: ignore

            # Gradients do not need to be kept further in the AD graph
            node.grad = node.grad.detach()


def find_topo_sort(node_list: list[Tensor]) -> list[Tensor]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    topo_order: list[Tensor] = []
    visited: set[Tensor] = set()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node: Tensor, visited: set[Tensor], topo_order: list[Tensor]) -> None:
    """
    Post-order DFS.
    """
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)
