from typing import Any

import numpy as np
from hypothesis import given
from needle import backend_ndarray as ndl

from tests.devices import all_devices
from tests.hypothesis_strategies import (
    array_and_axis,
    array_strategy,
    division_arrays,
    matmul_arrays,
    nonzero_float_strategy,
    positive_array_strategy,
    power_float_strategy,
    safe_float_strategy,
    same_shape_arrays,
    small_array_strategy,
)

# ============ Elementwise operations ============


@all_devices()
@given(data=same_shape_arrays())
def test_add(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test array addition."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)
    np.testing.assert_allclose((a + b).numpy(), arr1 + arr2, atol=1e-5, rtol=1e-5)


@all_devices()
@given(data=same_shape_arrays())
def test_multiply(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test array multiplication."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)
    np.testing.assert_allclose((a * b).numpy(), arr1 * arr2, atol=1e-5, rtol=1e-5)


@all_devices()
@given(data=division_arrays())
def test_divide(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test array division."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)
    np.testing.assert_allclose((a / b).numpy(), arr1 / arr2, atol=1e-5, rtol=1e-5)


@given(data=same_shape_arrays())
@all_devices()
def test_subtract(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test array subtraction."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)
    np.testing.assert_allclose((a - b).numpy(), arr1 - arr2, atol=1e-5, rtol=1e-5)


@given(data=same_shape_arrays())
@all_devices()
def test_equal(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test array equality."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)
    np.testing.assert_allclose(a == b, arr1 == arr2)


@given(data=same_shape_arrays())
@all_devices()
def test_greater_equal(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test array equality."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)
    np.testing.assert_allclose(a >= b, arr1 >= arr2)


@all_devices()
@given(data=same_shape_arrays())
def test_elementwise_maximum(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test elementwise maximum operation."""
    arr1, arr2 = data
    a = ndl.array(arr1, device=device)
    b = ndl.array(arr2, device=device)
    np.testing.assert_allclose(a.maximum(b).numpy(), np.maximum(arr1, arr2))


@all_devices()
@given(data=positive_array_strategy)
def test_log(data: np.ndarray, device: Any) -> None:
    """Test log operation."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose(arr_nd.log().numpy(), np.log(data), atol=1e-5, rtol=1e-5)


@all_devices()
@given(data=small_array_strategy)
def test_exp(data: np.ndarray, device: Any) -> None:
    """Test exp operation."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose(arr_nd.exp().numpy(), np.exp(data), atol=1e-5, rtol=1e-5)


@all_devices()
@given(data=array_strategy)
def test_tanh(data: np.ndarray, device: Any) -> None:
    """Test tanh operation."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose(
        arr_nd.tanh().numpy(), np.tanh(data), atol=1e-5, rtol=1e-5
    )


# ============ Scalar operations ============


@all_devices()
@given(data=array_strategy, scalar=safe_float_strategy)
def test_scalar_mul(data: np.ndarray, scalar: float, device: Any) -> None:
    """Test scalar multiplication."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose(
        (arr_nd * scalar).numpy(), data * scalar, atol=1e-5, rtol=1e-5
    )


@all_devices()
@given(data=array_strategy, scalar=nonzero_float_strategy)
def test_scalar_div(data: np.ndarray, scalar: float, device: Any) -> None:
    """Test scalar division."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose(
        (arr_nd / scalar).numpy(), data / scalar, atol=1e-5, rtol=1e-5
    )


@all_devices()
@given(data=positive_array_strategy, scalar=power_float_strategy)
def test_scalar_power(data: np.ndarray, scalar: float, device: Any) -> None:
    """Test scalar power operation."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose(
        (arr_nd**scalar).numpy(),
        data**scalar,
        atol=1e-5,
        rtol=1e-5,
    )


@all_devices()
@given(data=array_strategy, scalar=nonzero_float_strategy)
def test_scalar_maximum(data: np.ndarray, scalar: float, device: Any) -> None:
    """Test scalar maximum operation."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose(
        (arr_nd.maximum(scalar)).numpy(), np.maximum(data, scalar)
    )


@all_devices()
@given(data=array_strategy, scalar=nonzero_float_strategy)
def test_scalar_equal(data: np.ndarray, scalar: float, device: Any) -> None:
    """Test scalar equality operation."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose((arr_nd == scalar).numpy(), (data == scalar))


@all_devices()
@given(data=array_strategy, scalar=nonzero_float_strategy)
def test_scalar_greater_equal(data: np.ndarray, scalar: float, device: Any) -> None:
    """Test scalar greater than or equal operation."""
    arr_nd = ndl.array(data, device=device)
    np.testing.assert_allclose((arr_nd >= scalar).numpy(), (data >= scalar))


# ============ General operations ============


@given(data=matmul_arrays())
@all_devices()
def test_matmul(data: tuple[np.ndarray, np.ndarray], device: Any) -> None:
    """Test matrix multiplication."""
    a_np, b_np = data
    a_nd = ndl.array(a_np, device=device)
    b_nd = ndl.array(b_np, device=device)

    np.testing.assert_allclose((a_nd @ b_nd).numpy(), a_np @ b_np, atol=1e-5, rtol=1e-5)


# ============ Reductions ============


@given(a_np_and_axis=array_and_axis())
@all_devices()
def test_reduce_sum(a_np_and_axis, device):
    a_np, axis = a_np_and_axis

    # wrap into your framework
    a = ndl.array(a_np, device=device)

    # compare sum(keepdims=True)
    expected = a_np.sum(axis=axis, keepdims=True)
    actual = a.sum(axis=axis, keepdims=True).numpy()

    np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)


@given(a_np_and_axis=array_and_axis())
@all_devices()
def test_reduce_max(a_np_and_axis, device):
    a_np, axis = a_np_and_axis

    # wrap into your framework
    a = ndl.array(a_np, device=device)

    # compare max(keepdims=True)
    expected = a_np.max(axis=axis, keepdims=True)
    actual = a.max(axis=axis, keepdims=True).numpy()

    np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)
