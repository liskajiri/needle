import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, basic_indices
from needle import backend_ndarray as ndl
from needle.errors import BroadcastError

from tests.devices import all_devices
from tests.hypothesis_strategies import (
    array_shapes_strategy,
    array_strategy,
    broadcastable_arrays,
    reshape_examples,
    reshape_shapes,
    safe_float_strategy,
    scalar_slice_strategy,
    setitem_idx_strategy,
)
from tests.utils import DTYPE_FLOAT, check_same_memory, compare_strides

rng = np.random.default_rng()

# =================== Basic operations ===================


@given(data=array_strategy)
@all_devices()
def test_array_creation(data: np.ndarray, device: Any) -> None:
    """Test basic array creation and properties."""
    arr_nd = ndl.array(data, device=device)

    # Test basic properties
    assert arr_nd.shape == data.shape
    assert arr_nd.size == data.size
    assert arr_nd.ndim == data.ndim

    # Test array content
    np.testing.assert_allclose(arr_nd.numpy(), data, atol=1e-5, rtol=1e-5)
    compare_strides(data, arr_nd)


@given(data=array_strategy, scalar=safe_float_strategy)
@all_devices()
def test_fill(data: np.ndarray, scalar: float, device: Any) -> None:
    """Test array fill operation."""
    arr_nd = ndl.array(data, device=device)
    arr_nd.fill(scalar)

    expected = np.full_like(data, scalar)
    np.testing.assert_allclose(arr_nd.numpy(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "params",
    [
        {
            "shape": (4, 4),
            "np_fn": lambda X: X.transpose(),
            "nd_fn": lambda X: X.permute((1, 0)),
        },
        {
            "shape": (4, 1, 4),
            "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5, 4)),
            "nd_fn": lambda X: X.broadcast_to((4, 5, 4)),
        },
        {
            "shape": (4, 3),
            "np_fn": lambda X: X.reshape(2, 2, 3),
            "nd_fn": lambda X: X.reshape((2, 2, 3)),
        },
        {
            "shape": (16, 16),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(2, 4, 2, 2, 2, 2, 2),
            "nd_fn": lambda X: X.reshape((2, 4, 2, 2, 2, 2, 2)),
        },
        {
            "shape": (
                2,
                4,
                2,
                2,
                2,
                2,
                2,
            ),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(16, 16),
            "nd_fn": lambda X: X.reshape((16, 16)),
        },
        {"shape": (8, 8), "np_fn": lambda X: X[4:, 4:], "nd_fn": lambda X: X[4:, 4:]},
        {
            "shape": (8, 8, 2, 2, 2, 2),
            "np_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
            "nd_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
        },
        {
            "shape": (7, 8),
            "np_fn": lambda X: X.transpose()[3:7, 2:5],
            "nd_fn": lambda X: X.permute((1, 0))[3:7, 2:5],
        },
    ],
    ids=[
        "transpose",
        "broadcast_to",
        "reshape1",
        "reshape2",
        "reshape3",
        "getitem1",
        "getitem2",
        "transpose_getitem",
    ],
)
@all_devices()
def test_compact_after_op(params, device):
    shape, np_fn, nd_fn = params["shape"], params["np_fn"], params["nd_fn"]
    np_arr = np.random.randint(low=0, high=10, size=shape)
    a = ndl.array(np_arr, device=device)

    lhs = nd_fn(a).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(np_arr)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)


@given(data=array_strategy)
@all_devices()
def test_compact(data: np.ndarray, device: Any) -> None:
    """Test array compaction and is_compact check."""
    arr_nd = ndl.array(data, device=device)

    # Initial array should be compact
    assert arr_nd.is_compact()

    # Create a view by transposing
    transposed = arr_nd.permute(tuple(range(arr_nd.ndim))[::-1])

    compacted = transposed.compact()
    assert compacted.is_compact(), "Compacted array should be compact"
    np.testing.assert_allclose(
        compacted.numpy(), transposed.numpy(), atol=1e-5, rtol=1e-5
    )


# =================== Reshapes ===================


# Permute, broadcast_to, reshape, getitem, and combinations thereof.
@pytest.mark.parametrize(
    "params",
    [
        # Permute tests
        {
            "shape": (4, 4),
            "np_fn": lambda X: X.transpose(),
            "nd_fn": lambda X: X.permute((1, 0)),
        },
        {
            "shape": (2, 3, 4),
            "np_fn": lambda X: np.transpose(X, axes=(2, 1, 0)),
            "nd_fn": lambda X: X.permute((2, 1, 0)),
        },
        # Broadcast_to tests
        {
            "shape": (4, 1),
            "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5)),
            "nd_fn": lambda X: X.broadcast_to((4, 5)),
        },
        {
            "shape": (1, 3, 1),
            "np_fn": lambda X: np.broadcast_to(X, shape=(2, 3, 4)),
            "nd_fn": lambda X: X.broadcast_to((2, 3, 4)),
        },
        # Reshape tests
        {
            "shape": (6,),
            "np_fn": lambda X: X.reshape(2, 3),
            "nd_fn": lambda X: X.reshape((2, 3)),
        },
        {
            "shape": (2, 3, 4),
            "np_fn": lambda X: X.reshape(4, 6),
            "nd_fn": lambda X: X.reshape((4, 6)),
        },
        # Getitem tests
        {
            "shape": (8, 8),
            "np_fn": lambda X: X[2:6, 3:7],
            "nd_fn": lambda X: X[2:6, 3:7],
        },
        {
            "shape": (4, 5, 6),
            "np_fn": lambda X: X[1:3, :, 2:5],
            "nd_fn": lambda X: X[1:3, :, 2:5],
        },
        # Combination tests
        {
            "shape": (7, 8),
            "np_fn": lambda X: X.transpose()[1:5, 2:6],
            "nd_fn": lambda X: X.permute((1, 0))[1:5, 2:6],
        },
    ],
    ids=[
        "permute_2D",
        "permute_3D",
        "broadcast_to_2D",
        "broadcast_to_3D",
        "reshape_flat_to_2D",
        "reshape_3D_to_2D",
        "getitem_2D",
        "getitem_3D",
        "permute_getitem",
    ],
)
@all_devices()
def test_operations(params, device):
    shape, np_fn, nd_fn = params["shape"], params["np_fn"], params["nd_fn"]
    np_arr = np.random.randint(low=0, high=10, size=shape)
    a = ndl.array(np_arr, device=device)

    # Apply the function using the custom library and NumPy
    lhs = nd_fn(a).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(np_arr)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)


@given(reshape_data=reshape_examples())
@all_devices()
def test_reshape(device, reshape_data):
    array, new_shape = reshape_data

    a = ndl.array(array, device=device)

    lhs = array.reshape(new_shape)
    rhs = a.reshape(new_shape)

    np.testing.assert_allclose(rhs.numpy(), lhs, atol=1e-5, rtol=1e-5)
    # TODO: correctly fails, fix the issue with strides
    # compare_strides(lhs, rhs)
    check_same_memory(a, rhs)


@given(shapes=reshape_shapes())
@all_devices()
def test_reshape_and_inference(shapes: tuple, device: Any) -> None:
    """Test reshape operation with dimension inference."""
    orig_shape, new_shape = shapes
    data = np.arange(np.prod(orig_shape)).reshape(orig_shape)
    arr_nd = ndl.array(data, device=device)

    reshaped_np = data.reshape(new_shape)
    reshaped_nd = arr_nd.reshape(new_shape)

    np.testing.assert_allclose(reshaped_nd.numpy(), reshaped_np, atol=1e-5, rtol=1e-5)
    compare_strides(reshaped_np, reshaped_nd)
    check_same_memory(arr_nd, reshaped_nd)


@pytest.mark.parametrize(
    "params",
    [
        # Reshape with -1 inference
        {"shape": (2, 3, 4), "new_shape": (6, -1), "expected_shape": (6, 4)},
        {"shape": (24,), "new_shape": (-1, 6), "expected_shape": (4, 6)},
        {"shape": (8, 3), "new_shape": (2, -1, 3), "expected_shape": (2, 4, 3)},
        {"shape": (2, 3, 4), "new_shape": (-1,), "expected_shape": (24,)},  # Flatten
        {"shape": (4, 1, 3), "new_shape": (-1, 3), "expected_shape": (4, 3)},
        # Edge cases
        {"shape": (5, 1, 3), "new_shape": (5, -1), "expected_shape": (5, 3)},
        {"shape": (1, 1, 1), "new_shape": (-1, 1), "expected_shape": (1, 1)},
    ],
    ids=[
        "reshape_2D",
        "reshape_1D",
        "reshape_3D",
        "flatten",
        "reshape_2D_flatten",
        "reshape_edge_case_1",
        "reshape_edge_case_2",
    ],
)
@all_devices()
def test_reshape_with_inference(device, params):
    shape, new_shape = params["shape"], params["new_shape"]
    expected_shape = params["expected_shape"]

    np_a = np.arange(math.prod(shape) or 0).reshape(shape)
    a = ndl.array(np_a, device=device)

    lhs = np_a.reshape(expected_shape)
    rhs = a.reshape(new_shape)

    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)

    compare_strides(lhs, rhs)
    check_same_memory(a, rhs)


@pytest.mark.parametrize(
    "params",
    [
        {"shape": (4, 6), "new_shape": (5, 5)},  # Total size mismatch
        {"shape": (2, 3, 4), "new_shape": (2, -1, -1)},  # Multiple -1s
        {"shape": (24,), "new_shape": (-1, 7)},  # Size not divisible evenly
    ],
    ids=[
        "total_size_mismatch",
        "multiple_negatives",
        "size_not_divisible",
    ],
)
@all_devices()
def test_reshape_errors(device, params):
    shape, new_shape = params["shape"], params["new_shape"]
    np_a = np.random.randn(*shape)
    a = ndl.array(np_a, device=device)

    with pytest.raises(ValueError):
        a.reshape(new_shape)


@st.composite
def negative_stride_arrays(draw):
    """Generate arrays with negative strides via slicing."""
    shape = draw(array_shapes_strategy)
    arr = draw(arrays(dtype=DTYPE_FLOAT, shape=shape, elements=safe_float_strategy))

    # Generate negative step for a random dimension
    dim = draw(st.integers(min_value=0, max_value=len(shape) - 1))
    step = draw(st.integers(min_value=-3, max_value=-1))

    slices = [slice(None)] * len(shape)
    slices[dim] = slice(None, None, step)

    return arr, tuple(slices)


@given(data=negative_stride_arrays())
@all_devices()
def test_negative_strides(data: tuple[np.ndarray, tuple], device: Any) -> None:
    """Test array operations with negative strides."""
    arr, slices = data
    a_np = arr[slices]
    a_nd = ndl.array(arr, device=device)[slices]

    np.testing.assert_allclose(a_nd.numpy(), a_np, atol=1e-5, rtol=1e-5)
    compare_strides(a_np, a_nd)


@given(shape=array_shapes_strategy)
@all_devices()
def test_permute(device, shape):
    axes = tuple(rng.permutation(len(shape)).tolist())
    np_a = rng.normal(size=shape)

    a = ndl.array(np_a, device=device)

    lhs = np.transpose(np_a, axes=axes)
    rhs = a.permute(axes)

    np.testing.assert_allclose(lhs, rhs.numpy())
    compare_strides(lhs, rhs)
    check_same_memory(a, rhs)


# ==================== Setitem ====================


@given(case=setitem_idx_strategy())
@all_devices()
def test_setitem_ewise(case, device):
    shape = case["shape"]
    lhs_idx = case["lhs_idx"]
    rhs_idx = case["rhs_idx"]

    np_a = rng.normal(size=shape)
    np_b = rng.normal(size=shape)

    a = ndl.array(np_a, device=device)
    b = ndl.array(np_b, device=device)

    a[lhs_idx] = b[rhs_idx]
    np_a[lhs_idx] = np_b[rhs_idx]

    np.testing.assert_allclose(a.numpy(), np_a, atol=1e-5, rtol=1e-5)
    compare_strides(np_a, a)


@given(case=scalar_slice_strategy())
@all_devices()
@pytest.mark.skip(reason="This test should pass, but fails due to a bug.")
def test_setitem_scalar(case, device):
    shape, slices = case

    np_a = rng.normal(size=shape)
    a = ndl.array(np_a, device=device)

    np_a[slices] = 4.0
    a[slices] = 4.0

    np.testing.assert_allclose(a.numpy(), np_a, atol=1e-5, rtol=1e-5)
    compare_strides(np_a, a)


# ==================== Getitem ====================


@given(data=st.data())
def test_getitem(data):
    arr = data.draw(array_strategy)
    nd_array = ndl.array(arr)

    idx = data.draw(basic_indices(arr.shape))
    nd_indexed = nd_array[idx]
    np_indexes = arr[idx]

    np.testing.assert_array_equal(nd_indexed, np_indexes)


@given(data=st.data())
def test_getitem_multi_index(data):
    arr = data.draw(array_strategy)
    nd_array = ndl.array(arr)

    n_axes = data.draw(st.integers(min_value=1, max_value=arr.ndim))

    # 3. For each axis i in [0, n_axes), draw a valid index for that axis
    indices = tuple(
        data.draw(st.integers(min_value=0, max_value=arr.shape[axis] - 1))
        for axis in range(n_axes)
    )

    # Compare numpy and ndarray indexing
    np_result = arr[indices]
    nd_result = nd_array[indices]
    np.testing.assert_array_equal(nd_result, np_result)


@given(data=st.data())
def test_getitem_raises_out_of_range_index(data):
    arr = data.draw(array_strategy)
    nd_array = ndl.array(arr)

    bad_idx_strategy = st.one_of(
        st.integers(min_value=-arr.size - 100, max_value=-arr.size - 1),
        st.integers(min_value=arr.size, max_value=arr.size + 100),
    )
    bad_idx = data.draw(bad_idx_strategy)

    with pytest.raises(IndexError):
        _ = nd_array[bad_idx]
    with pytest.raises(IndexError):
        _ = arr[bad_idx]


@given(data=st.data())
def test_getitem_raises_too_many_indices(data):
    # Generate an array with 1-3 dimensions
    arr = data.draw(array_strategy)
    nd_array = ndl.array(arr)

    # Generate number of indices (must be more than ndim)
    n_indices = arr.ndim + data.draw(st.integers(min_value=1, max_value=3))

    # Generate list of integers as indices
    indices = tuple(
        data.draw(st.integers(min_value=0, max_value=max(arr.shape) - 1))
        for _ in range(n_indices)
    )

    with pytest.raises(IndexError):
        _ = nd_array[indices]
    with pytest.raises(IndexError):
        _ = arr[indices]


# ==================== Broadcasting ===================


@given(data=broadcastable_arrays(num_shapes=2))
@all_devices()
def test_broadcast_ops(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test broadcasting operations with arrays of different ranks."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)

    # Test various operations with broadcasting
    operations = [
        (lambda x, y: x + y, lambda x, y: x + y),
        (lambda x, y: x * y, lambda x, y: x * y),
        (lambda x, y: x.maximum(y), lambda x, y: np.maximum(x, y)),
    ]

    for nd_op, np_op in operations:
        np.testing.assert_allclose(
            nd_op(a, b).numpy(),
            np_op(arr1, arr2),
            atol=1e-5,
            rtol=1e-5,
        )


@all_devices()
@given(data=broadcastable_arrays(num_shapes=5))
def test_broadcast_shapes(device, data):
    ndl_arrays = [ndl.array(arr, device=device) for arr in data]

    result = ndl.broadcast_shapes(*[arr.shape for arr in ndl_arrays])
    np.testing.assert_array_equal(
        np.array(result), np.broadcast_shapes(*[arr.shape for arr in ndl_arrays])
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # Incompatible dimensions
        [(2, 3), (2, 4)],
        [(3,), (4,)],
        [(2, 1), (8, 4, 3)],
        [(5, 6), (5, 7)],
        # Complex incompatible cases
        [(2, 3, 4), (2, 2, 4)],
        [(8, 1, 3, 1), (7, 2, 3)],
    ],
    ids=[
        "incompatible_dimensions_1",
        "incompatible_dimensions_2",
        "incompatible_dimensions_3",
        "incompatible_dimensions_4",
        "complex_incompatible_case_1",
        "complex_incompatible_case_2",
    ],
)
def test_broadcast_shapes_error(shapes):
    """Test that broadcast_shapes raises ValueError for incompatible shapes."""
    with pytest.raises(BroadcastError, match="Incompatible shapes for broadcasting"):
        ndl.broadcast_shapes(*shapes)


@pytest.mark.parametrize(
    "params",
    [
        # Incompatible shapes
        {"shape": (3,), "broadcast_shape": (2, 2)},
        {"shape": (2, 3), "broadcast_shape": (2, 2)},
        {"shape": (3, 2), "broadcast_shape": (2, 2)},
        # Shrinking non-1 dimensions
        {"shape": (2, 3), "broadcast_shape": (1, 3)},
        # Wrong number of dimensions
        {"shape": (2, 2, 2), "broadcast_shape": (2, 2)},
    ],
    ids=[
        "incompatible_shapes",
        "incompatible_shapes_2",
        "incompatible_shapes_3",
        "shrinking_non_1_dimensions",
        "wrong_number_of_dimensions",
    ],
)
@all_devices()
def test_broadcast_to_errors(device, params):
    shape = params["shape"]
    broadcast_shape = params["broadcast_shape"]

    np_a = np.random.randn(*shape)
    a = ndl.array(np_a, device=device)

    with pytest.raises(BroadcastError):
        a.broadcast_to(broadcast_shape)
