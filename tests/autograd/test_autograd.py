import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import (
    arrays,
)

from tests.gradient_check import backward_check

rng = np.random.default_rng(0)

DEFAULT_SHAPE = (5, 4)
TRANSPOSE_SHAPES = (3, 5, 4)
# matrix shapes
m, n, k = 5, 4, 3


# ##############################################################################
# # TESTS for find_topo_sort


@pytest.mark.parametrize(
    "input_tensors,expr_fn,expected_topo_order",
    [
        pytest.param(
            [
                np.asarray([[0.88282157]]),
                np.asarray([[0.90170084]]),
            ],
            lambda a, b: 3 * a * a + 4 * b * a - a,
            [
                np.array([[0.88282157]]),
                np.array([[2.64846471]]),
                np.array([[2.33812177]]),
                np.array([[0.90170084]]),
                np.array([[3.60680336]]),
                np.array([[3.1841638]]),
                np.array([[5.52228558]]),
                np.array([[-0.88282157]]),
                np.array([[4.63946401]]),
            ],
            id="scalar_operations",
        ),
        pytest.param(
            [
                np.asarray([[0.20914675], [0.65264178]]),
                np.asarray([[0.65394286, 0.08218317]]),
            ],
            lambda a, b: 3 * ((b @ a) + (2.3412 * b) @ a) + 1.5,
            [
                np.array([[0.65394286, 0.08218317]]),
                np.array([[0.20914675], [0.65264178]]),
                np.array([[0.19040619]]),
                np.array([[1.53101102, 0.19240724]]),
                np.array([[0.44577898]]),
                np.array([[0.63618518]]),
                np.array([[1.90855553]]),
                np.array([[3.40855553]]),
            ],
            id="matrix_multiplication_with_scalar",
        ),
        pytest.param(
            [
                np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
                np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
            ],
            lambda a, b: (a @ b + b - a) @ a,
            [
                np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
                np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
                np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
                np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
                np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
                np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
                np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]]),
            ],
            id="matrix_ops_with_subtraction",
        ),
    ],
)
def test_topo_sort(input_tensors, expr_fn, expected_topo_order) -> None:
    """Test topological sort functionality with different computational graphs."""
    tensors = [ndl.Tensor(arr) for arr in input_tensors]

    # Create the computational graph
    result = expr_fn(*tensors)

    topo_order = [x.numpy() for x in ndl.autograd.find_topo_sort([result])]

    assert len(expected_topo_order) == len(topo_order)

    for actual, expected in zip(topo_order, expected_topo_order, strict=False):
        np.testing.assert_allclose(actual, expected, rtol=1e-06, atol=1e-06)


# ##############################################################################
# # TESTS for compute_gradient_of_variables


@given(
    arrays(dtype=np.float32, shape=(m, n), elements=st.floats(0, 1)),
    arrays(dtype=np.float32, shape=(n, k), elements=st.floats(0, 1)),
    arrays(dtype=np.float32, shape=(m, k), elements=st.floats(0, 1)),
)
def test_compute_gradient_sum_matmul(a, b, c) -> None:
    atol = 1e-3
    # Needle implementation
    A_ndl = ndl.Tensor(a)
    B_ndl = ndl.Tensor(b)
    C_ndl = ndl.Tensor(c)

    out_ndl = ndl.ops.mathematic.summation(
        (A_ndl @ B_ndl + C_ndl) * (A_ndl @ B_ndl), axes=None
    )
    out_ndl.backward()

    # PyTorch implementation
    A_torch = torch.tensor(a, requires_grad=True)
    B_torch = torch.tensor(b, requires_grad=True)
    C_torch = torch.tensor(c, requires_grad=True)

    out_torch = ((A_torch @ B_torch + C_torch) * (A_torch @ B_torch)).sum()
    out_torch.backward()

    # Compare results
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), atol=atol)
    np.testing.assert_allclose(A_ndl.grad.numpy(), A_torch.grad.numpy(), atol=atol)  # type: ignore
    np.testing.assert_allclose(B_ndl.grad.numpy(), B_torch.grad.numpy(), atol=atol)  # type: ignore
    np.testing.assert_allclose(C_ndl.grad.numpy(), C_torch.grad.numpy(), atol=atol)  # type: ignore


@pytest.mark.parametrize(
    "ndl_fn,shapes,torch_fn",
    [
        pytest.param(
            lambda A, B, C: ndl.ops.mathematic.summation(
                (A @ B + C) * (A @ B), axes=None
            ),
            [(10, 9), (9, 8), (10, 8)],
            lambda A, B, C: ((A @ B + C) * (A @ B)).sum(),
            id="matmul_add_multiply",
        ),
        pytest.param(
            lambda A, B: ndl.ops.mathematic.summation(
                ndl.broadcast_to(A, shape=(10, 9)) * B, axes=None
            ),
            [(10, 1), (10, 9)],
            lambda A, B: (A.expand(10, 9) * B).sum(),
            id="broadcast_multiply",
        ),
        pytest.param(
            lambda A, B, C: ndl.ops.mathematic.summation(
                ndl.reshape(A, shape=(10, 10)) @ B / 5 + C, axes=None
            ),
            [(100,), (10, 5), (10, 5)],
            lambda A, B, C: (A.view(10, 10) @ B / 5 + C).sum(),
            id="reshape_matmul_divide",
        ),
    ],
)
def test_compute_gradients(ndl_fn, shapes, torch_fn) -> None:
    ndl_inputs = [ndl.Tensor(rng.standard_normal(shape)) for shape in shapes]

    backward_check(ndl_fn, *ndl_inputs, torch_fn=torch_fn, backward=True)


def test_compute_gradient_of_gradient() -> None:
    # check gradient of gradient
    x2 = ndl.Tensor([6])  # type: ignore[no-untyped-call]
    x3 = ndl.Tensor([0])  # type: ignore[no-untyped-call]

    y = x2 * x2 + x2 * x3
    y.backward()

    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()

    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad

    x2_val = x2.numpy()
    x3_val = x3.numpy()

    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1
