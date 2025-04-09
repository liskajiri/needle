import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    broadcastable_shapes,
)

from tests.gradient_check import backward_check

rng = np.random.default_rng(0)


DEFAULT_SHAPE = (5, 4)
TRANSPOSE_SHAPES = (3, 5, 4)
# matrix shapes
m, n, k = 5, 4, 3


@given(
    arrays(dtype=np.float32, shape=DEFAULT_SHAPE, elements=st.floats(-1e3, 1e3)),
    arrays(dtype=np.float32, shape=DEFAULT_SHAPE, elements=st.floats(1, 1e3)),
)
def test_divide_backward(a, b) -> None:
    backward_check(ndl.ops.divide, ndl.Tensor(a), ndl.Tensor(b))


@given(
    arrays(dtype=np.float32, shape=DEFAULT_SHAPE, elements=st.floats(-1e3, 1e3)),
    st.floats(1, 1e3),
)
def test_divide_scalar_backward(a, b) -> None:
    backward_check(ndl.ops.divide_scalar, ndl.Tensor(a), b)


@given(
    arrays(dtype=np.float32, shape=DEFAULT_SHAPE, elements=st.floats(-1e2, 1e2)),
    arrays(dtype=np.float32, shape=DEFAULT_SHAPE, elements=st.floats(-1e2, 1e2)),
)
def test_matmul_simple_backward(a, b) -> None:
    backward_check(
        ndl.matmul,
        ndl.Tensor(a),
        ndl.Tensor(b.T),
    )


@pytest.mark.parametrize(
    "batch_shapes",
    [
        ((6, 6, 5, 4), (6, 6, 4, 3)),  # batched @ batched (same batch dims)
        ((1, 2, 1, 2), (2, 1)),  # batched @ non-batched (smaller dims)
        ((6, 6, 5, 4), (4, 3)),  # batched @ non-batched
        ((5, 4), (6, 6, 4, 3)),  # non-batched @ batched
    ],
    ids=[
        "batched @ batched",
        "(batched @ non-batched)_small",
        "batched @ non-batched",
        "non-batched @ batched",
    ],
)
def test_matmul_batched_backward(batch_shapes) -> None:
    A_shape, B_shape = batch_shapes
    A = ndl.Tensor(rng.standard_normal(A_shape))
    B = ndl.Tensor(rng.standard_normal(B_shape))

    # Run backward check
    backward_check(ndl.matmul, A, B)


@given(
    arrays(dtype=np.float32, shape=DEFAULT_SHAPE),
)
def test_reshape_backward(a) -> None:
    backward_check(ndl.reshape, ndl.Tensor(a), shape=DEFAULT_SHAPE)


@given(arrays(dtype=np.float32, shape=DEFAULT_SHAPE))
def test_negate_backward(a) -> None:
    backward_check(ndl.negate, ndl.Tensor(a))


@given(
    arrays(dtype=np.float32, shape=DEFAULT_SHAPE, elements=st.floats(0.1, 10)),
)
def test_log_backward(a) -> None:
    backward_check(ndl.log, ndl.Tensor(a))


@given(arrays(dtype=np.float32, shape=DEFAULT_SHAPE, elements=st.floats(-10, 10)))
def test_relu_backward(a) -> None:
    backward_check(ndl.relu, ndl.Tensor(a))


@pytest.mark.parametrize(
    "axes",
    [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ],
    ids=lambda x: f"{x}",
)
@given(
    arrays(dtype=np.float32, shape=TRANSPOSE_SHAPES),
)
def test_transpose_backward(axes, a) -> None:
    backward_check(ndl.transpose, ndl.Tensor(a), axes=axes)


@given(
    data=st.data(),
)
def test_broadcast_to_backward_hypothesis(data) -> None:
    """
    Only tests sizes from 2, because there are some issues with broadcasting to (, 1)
    """
    base_shape = data.draw(array_shapes(min_dims=1, max_dims=5))

    array = data.draw(arrays(dtype=np.float32, shape=base_shape))

    target_shape = data.draw(
        broadcastable_shapes(shape=base_shape, min_dims=1, min_side=2).filter(
            # for some reason can generate smaller shapes
            lambda x: np.prod(x) > np.prod(base_shape) and len(x) >= len(base_shape)
        )
    )

    tensor = ndl.Tensor(array)
    backward_check(ndl.broadcast_to, tensor, shape=target_shape)


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        pytest.param((3, 1), (3, 3), id="broadcast_columns"),
        pytest.param((1, 3), (3, 3), id="broadcast_rows"),
        pytest.param((1,), (3, 3, 3), id="vector_to_3d"),
        pytest.param((), (3, 3, 3), id="scalar_to_3d"),
        pytest.param((5, 4, 1), (5, 4, 3), id="expand_last_dim"),
        pytest.param((5,), (1, 5), id="add_dim_preserve"),
        pytest.param((2, 3, 1, 5), (2, 3, 4, 5), id="expand_3d_to_4d"),
    ],
)
def test_broadcast_to_backward(input_shape, output_shape) -> None:
    input_data = rng.standard_normal(input_shape)
    backward_check(ndl.broadcast_to, ndl.Tensor(input_data), shape=output_shape)


@pytest.mark.parametrize(
    "shape,axes",
    [
        pytest.param((5, 4), (1,), id="sum_along_columns"),
        pytest.param((5, 4), (0,), id="sum_along_rows"),
        pytest.param((5, 4), (0, 1), id="sum_all_dimensions"),
        pytest.param((5, 4, 1), (0, 1), id="sum_first_two_dimensions"),
    ],
)
def test_summation_backward(shape, axes) -> None:
    tensor = ndl.Tensor(np.random.randn(*shape))
    backward_check(
        ndl.ops.mathematic.summation,
        tensor,
        axes=axes,
    )


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
