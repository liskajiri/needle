from collections import defaultdict

from needle.tensor import Tensor


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
