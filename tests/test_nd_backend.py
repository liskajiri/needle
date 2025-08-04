import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from needle import backend_ndarray as nd

from tests.devices import all_devices
from tests.gradient_check import backward_check

rng = np.random.default_rng(0)

# TODO: divide to _forward and _backward tests

# TODO: test that the results are on the same device as they started

OPS = {
    "divide": lambda a, b: a / b,
    "subtract": lambda a, b: a - b,
    "add": lambda a, b: a + b,
    "multiply": lambda a, b: a * b,
}
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]

SHAPE = (3, 3)


# Strategy to generate two different arrays with the same shape
@st.composite
def two_arrays_same_shape(draw, shape=SHAPE):
    non_zero_floats = st.floats(-10, 10).filter(lambda x: abs(x) >= 1e-10)
    array_strategy = arrays(dtype=np.float32, shape=shape, elements=non_zero_floats)
    a = draw(array_strategy)
    b = draw(array_strategy.filter(lambda x: not np.any(x == 0)))
    return a, b


@pytest.mark.parametrize("op_name", OPS.keys())
@given(arrays=two_arrays_same_shape())  # type: ignore
@all_devices()
def test_ewise_fn(device, op_name, arrays) -> None:
    arr1, arr2 = arrays

    op_fn = OPS[op_name]
    A = ndl.Tensor(nd.array(arr1), device=device)
    B = ndl.Tensor(nd.array(arr2), device=device)

    expected = op_fn(arr1, arr2)
    actual = op_fn(A, B).numpy()

    np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("op_name", OPS.keys())
@given(
    array=arrays(dtype=np.float32, shape=SHAPE, elements=st.floats(-10, 10)),
    scalar=st.floats(-10, 10).filter(lambda x: abs(x) >= 1e-10),
)
@all_devices()
def test_scalar_fn(device, op_name, array, scalar) -> None:
    op_fn = OPS[op_name]
    A = ndl.Tensor(nd.array(array), device=device)

    expected = op_fn(array, scalar)
    actual = op_fn(A, scalar).numpy()

    np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)


@st.composite
def compatible_matrices(draw):
    m = draw(st.integers(min_value=1, max_value=10))  # rows of A
    n = draw(st.integers(min_value=1, max_value=10))  # cols of A / rows of B
    p = draw(st.integers(min_value=1, max_value=10))  # cols of B

    A = draw(arrays(dtype=np.float32, shape=(m, n), elements=st.floats(-10, 10)))
    B = draw(arrays(dtype=np.float32, shape=(n, p), elements=st.floats(-10, 10)))
    return A, B


@given(compatible_matrices())  # type: ignore
@all_devices()
def test_matmul(device, matrices) -> None:
    atol = 1e-5
    rtol = 1e-5
    a, b = matrices
    A = ndl.Tensor(nd.array(a), device=device)
    B = ndl.Tensor(nd.array(b), device=device)
    np.testing.assert_allclose(a @ b, (A @ B).numpy(), atol=atol, rtol=rtol)


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
    power=st.integers(min_value=0, max_value=5),
)
@all_devices()
def test_power(device, array, power) -> None:
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(array**power, (A**power).numpy(), atol=1e-5, rtol=1e-5)


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(0.1, 100.0, allow_infinity=False, allow_nan=False),
    )
)
@all_devices()
def test_log(device, array) -> None:
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(np.log(array), ndl.log(A).numpy(), atol=1e-5, rtol=1e-5)


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    )
)
@all_devices()
def test_exp(device, array) -> None:
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(np.exp(array), ndl.exp(A).numpy(), atol=1e-5, rtol=1e-5)


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    )
)
@all_devices()
def test_relu(device, array):
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(
        np.maximum(array, 0), ndl.relu(A).numpy(), atol=1e-5, rtol=1e-5
    )


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    )
)
@all_devices()
def test_tanh(device, array):
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(
        np.tanh(array), ndl.tanh(A).numpy(), atol=1e-5, rtol=1e-5
    )


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(0.0, 10.0, allow_infinity=False, allow_nan=False),
    )
)
@all_devices()
def test_sqrt(device, array):
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(
        np.sqrt(array), ndl.sqrt(A).numpy(), atol=1e-5, rtol=1e-5
    )


@all_devices()
def test_sqrt_negative(device):
    array = np.array([-1.0], dtype=np.float32)
    A = ndl.Tensor(nd.array(array), device=device)
    with pytest.raises(ValueError):
        ndl.sqrt(A)


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    )
)
@all_devices()
def test_tanh_backward(device, array):
    A = ndl.Tensor(nd.array(array), device=device)
    backward_check(ndl.tanh, A)


@given(
    arrays=st.lists(
        arrays(
            dtype=np.float32,
            shape=SHAPE,
            elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
        ),
        min_size=1,
        max_size=5,
    ),
    axis=st.integers(min_value=0, max_value=1),
)
@all_devices()
def test_stack(device, arrays, axis):
    A = [ndl.Tensor(nd.array(arr), device=device) for arr in arrays]
    A_t = [torch.tensor(arr) for arr in arrays]

    out = ndl.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)

    np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    ndl.cpu().name == "numpy", reason="Numpy has different stack/split semantics"
)
@given(
    arrays=st.lists(
        arrays(
            dtype=np.float32,
            shape=SHAPE,
            elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
        ),
        min_size=1,
        max_size=5,
    ),
    axis=st.integers(min_value=0, max_value=1),
)
@all_devices()
def test_stack_backward(device, arrays, axis):
    A = [ndl.Tensor(nd.array(arr), device=device, requires_grad=True) for arr in arrays]
    A_t = [torch.tensor(arr, requires_grad=True) for arr in arrays]

    ndl.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()

    for a, a_t in zip(A, A_t):
        np.testing.assert_allclose(
            a.grad.numpy(), a_t.grad.numpy(), atol=1e-5, rtol=1e-5
        )


SUMMATION_PARAMETERS = [
    ((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2),
]


@pytest.mark.parametrize(
    ("shape", "axes"),
    SUMMATION_PARAMETERS,
    ids=[
        "single_elem_all_axes",
        "matrix_first_axis",
        "tensor_middle_axis",
        "tensor_last_axis",
    ],
)
@all_devices()
def test_summation_fixed(shape, axes, device):
    if axes is not None:
        axes = (axes,)
    a = rng.standard_normal(shape, dtype=np.float32)
    A = ndl.Tensor(nd.array(a), device=device)
    np.testing.assert_allclose(
        np.sum(a, axes), ndl.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    ("shape", "axes"),
    SUMMATION_PARAMETERS,
    ids=[
        "single_elem_all_axes",
        "matrix_first_axis",
        "tensor_middle_axis",
        "tensor_last_axis",
    ],
)
@all_devices()
def test_summation_backward_fixed(shape, axes, device):
    if axes is not None:
        axes = (axes,)
    a = rng.standard_normal(shape, dtype=np.float32)
    A = ndl.Tensor(nd.array(a), device=device)
    backward_check(ndl.summation, A, axes=axes)


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
    axes=st.one_of(st.none(), st.integers(0, 1)),
)
@all_devices()
def test_summation_hypothesis(device, array, axes):
    if axes is not None:
        axes = (axes,)
    print(axes)
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(
        np.sum(array, axes), ndl.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5
    )


@given(
    array=arrays(
        dtype=np.float32,
        shape=SHAPE,
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
    axes=st.one_of(st.none(), st.integers(0, 1)),
)
@all_devices()
def test_summation_backward_hypothesis(device, array, axes):
    if axes is not None:
        axes = (axes,)
    A = ndl.Tensor(nd.array(array), device=device)
    backward_check(ndl.summation, A, axes=axes)


BROADCAST_SHAPES = [
    ((1, 1, 1), (3, 3, 3), "expand_all"),
    ((4, 1, 6), (4, 3, 6), "expand_middle"),
]


@pytest.mark.parametrize(
    ("shape", "shape_to", "test_id"),
    BROADCAST_SHAPES,
    ids=lambda p: p[2] if isinstance(p, tuple) and len(p) > 2 else None,
)
@all_devices()
def test_broadcast_fixed(shape, shape_to, test_id, device):
    a = rng.standard_normal(shape, dtype=np.float32)
    A = ndl.Tensor(nd.array(a), device=device)
    np.testing.assert_allclose(
        np.broadcast_to(a, shape_to),
        ndl.broadcast_to(A, shape_to).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


# TODO: this could use some advanced constructs form hypothesis
@given(
    array=arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(1, 3), st.just(1), st.integers(1, 3)),
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
    expand_middle=st.integers(min_value=1, max_value=3),
)
@all_devices()
def test_broadcast_hypothesis(device, array, expand_middle):
    shape_to = (array.shape[0], expand_middle, array.shape[2])
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(
        np.broadcast_to(array, shape_to),
        ndl.broadcast_to(A, shape_to).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


RESHAPE_SHAPES = [
    {"shape": (1, 1, 1), "shape_to": (1,), "id": "flatten_to_scalar"},
    {"shape": (4, 1, 6), "shape_to": (6, 4, 1), "id": "reorder_3d"},
]


@pytest.mark.parametrize("params", RESHAPE_SHAPES, ids=lambda p: p["id"])
@all_devices()
def test_reshape_fixed(device, params):
    a = rng.standard_normal(params["shape"], dtype=np.float32)
    A = ndl.Tensor(nd.array(a), device=device)
    np.testing.assert_allclose(
        np.reshape(a, params["shape_to"]),
        ndl.reshape(A, params["shape_to"]).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


# TODO: this could use some advanced constructs form hypothesis
@given(
    array=arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(1, 4), st.integers(1, 4)),
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    )
)
@all_devices()
def test_reshape_hypothesis(device, array):
    # Flatten and then reshape back to 2D
    size = array.size
    shape_to = (size // 2, 2) if size % 2 == 0 else (size, 1)

    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(
        np.reshape(array, shape_to),
        ndl.reshape(A, shape_to).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


TRANSPOSE_SHAPES = [(1, 1, 1), (4, 5, 6)]
TRANSPOSE_AXES = [(0, 1), (0, 2), None]


@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES, ids=["ones", "rect"])
@pytest.mark.parametrize("axes", TRANSPOSE_AXES, ids=["1", "2", "default"])
@all_devices()
def test_transpose_fixed(shape, axes, device):
    a = rng.standard_normal(shape, dtype=np.float32)
    A = ndl.Tensor(nd.array(a), device=device)
    np_axes = (a.ndim - 2, a.ndim - 1) if axes is None else axes
    np.testing.assert_allclose(
        np.swapaxes(a, np_axes[0], np_axes[1]),
        ndl.transpose(A, axes=axes).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@given(
    array=arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(1, 4), st.integers(1, 4), st.integers(1, 4)),
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
    axis1=st.integers(0, 2),
    axis2=st.integers(0, 2),
)
@all_devices()
def test_transpose_hypothesis(device, array, axis1, axis2):
    # assume(axis1 != axis2)
    A = ndl.Tensor(nd.array(array), device=device)
    np.testing.assert_allclose(
        np.swapaxes(array, axis1, axis2),
        ndl.transpose(A, axes=(axis1, axis2)).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@given(
    array=arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(1, 4), st.integers(1, 4), st.integers(1, 4)),
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
    axis1=st.integers(0, 2),
    axis2=st.integers(0, 2),
)
@all_devices()
def test_transpose_backward_hypothesis(device, array, axis1, axis2):
    A = ndl.Tensor(nd.array(array), device=device)
    axes = (axis1, axis2)
    backward_check(ndl.transpose, A, axes=axes)


@pytest.mark.parametrize(("shape", "axes"), SUMMATION_PARAMETERS)
@all_devices()
def test_logsumexp(shape, axes, device):
    a = rng.standard_normal(shape, dtype=np.float32)
    A = ndl.Tensor(nd.array(a), device=device)
    A_t = torch.tensor(a)
    t_axes = tuple(range(len(shape))) if axes is None else axes
    np.testing.assert_allclose(
        torch.logsumexp(A_t, dim=t_axes).numpy(),
        ndl.logsumexp(A, axes=axes).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


LOGSOFTMAX_SHAPES = (
    # shape, scale factor for values
    ((2, 3), 1),  # basic case
    ((5, 4), 1),  # different shape
    ((2, 3), 1000),  # large values
    ((10, 10), 0.001),  # small values
)


@pytest.mark.parametrize(
    ("shape", "scale"),
    LOGSOFTMAX_SHAPES,
    ids=[f"[{shape}-{scale}]" for shape, scale in LOGSOFTMAX_SHAPES],
)
@all_devices()
def test_logsoftmax_forward(shape, scale, device):
    x = rng.standard_normal(shape, dtype=np.float32) * scale
    needle_x = ndl.Tensor(nd.array(x), device=device)
    torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    needle_out = ndl.ops.logsoftmax(needle_x)
    torch_out = torch.nn.functional.log_softmax(torch_x, dim=1)

    np.testing.assert_allclose(
        needle_out.numpy(), torch_out.detach().numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    ("shape", "scale"),
    LOGSOFTMAX_SHAPES,
    ids=[f"[{shape}-{scale}]" for shape, scale in LOGSOFTMAX_SHAPES],
)
@all_devices()
def test_logsoftmax_backward(shape, scale, device):
    x = rng.standard_normal(shape, dtype=np.float32) * scale
    needle_x = ndl.Tensor(nd.array(x), device=device)
    torch_x = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    needle_out = ndl.ops.logsoftmax(needle_x)
    torch_out = torch.nn.functional.log_softmax(torch_x, dim=1)

    np.testing.assert_allclose(
        needle_out.numpy(), torch_out.detach().numpy(), rtol=1e-5, atol=1e-5
    )

    needle_out.sum().backward()
    torch_out.sum().backward()

    np.testing.assert_allclose(
        needle_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5
    )


def test_logsoftmax_invalid():
    # Test 1D input
    with pytest.raises(AssertionError):
        ndl.ops.logsoftmax(ndl.Tensor(np.array([1.0, 2.0])))

    # Test 3D input
    with pytest.raises(AssertionError):
        ndl.ops.logsoftmax(ndl.Tensor(rng.standard_normal((2, 3, 4))))


@given(
    array=arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(1, 8), st.integers(1, 8)),
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
)
@all_devices()
def test_logsoftmax_forward_hypothesis(device, array):
    needle_x = ndl.Tensor(nd.array(array), device=device)
    torch_x = torch.tensor(array, dtype=torch.float32, requires_grad=True)

    needle_out = ndl.ops.logsoftmax(needle_x)
    torch_out = torch.nn.functional.log_softmax(torch_x, dim=1)

    np.testing.assert_allclose(
        needle_out.numpy(), torch_out.detach().numpy(), rtol=1e-5, atol=1e-5
    )


@given(
    array=arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(1, 8), st.integers(1, 8)),
        elements=st.floats(-10, 10, allow_infinity=False, allow_nan=False),
    ),
)
@all_devices()
def test_logsoftmax_backward_hypothesis(device, array):
    needle_x = ndl.Tensor(nd.array(array), device=device)
    torch_x = torch.tensor(array, dtype=torch.float32, requires_grad=True)

    needle_out = ndl.ops.logsoftmax(needle_x)
    torch_out = torch.nn.functional.log_softmax(torch_x, dim=1)

    needle_out.sum().backward()
    torch_out.sum().backward()

    np.testing.assert_allclose(
        needle_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5
    )
