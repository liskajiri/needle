import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from needle import NDArray, array_api

from tests.devices import all_devices
from tests.hypothesis_strategies import (
    DTYPE_FLOAT,
    array_strategy,
    flip_params,
    stack_params,
)

rng = np.random.default_rng()


@given(params=flip_params())
@all_devices()
def test_flip(params, device) -> None:
    """Test flip operation forward pass"""
    shape, axes = params["shape"], params["axes"]
    input_array = rng.standard_normal(shape)
    ndl_array = NDArray(input_array, device=device)

    target = np.flip(input_array, axes)
    result = array_api.flip(ndl_array, axis=axes)

    np.testing.assert_allclose(result.numpy(), target, rtol=1e-6)


PAD_PARAMS = [
    pytest.param(
        {"shape": (10, 32, 32, 8), "padding": ((0, 0), (2, 2), (2, 2), (0, 0))},
        id="symmetric_padding",
    ),
    pytest.param(
        {"shape": (10, 32, 32, 8), "padding": ((0, 0), (0, 0), (0, 0), (0, 0))},
        id="no_padding",
    ),
    pytest.param(
        {"shape": (10, 32, 32, 8), "padding": ((0, 1), (2, 0), (2, 1), (0, 0))},
        id="asymmetric_padding",
    ),
]


@pytest.mark.parametrize(
    "params",
    PAD_PARAMS,
)
@all_devices()
def test_pad(params, device) -> None:
    """Test padding operation forward pass"""
    shape, padding = params["shape"], params["padding"]
    input_array = rng.standard_normal(shape)
    ndl_array = NDArray(input_array, device=device)

    target = np.pad(input_array, padding)
    result = array_api.pad(ndl_array, padding)

    np.testing.assert_allclose(result.numpy(), target, rtol=1e-6)


@given(params=stack_params())
@all_devices()
def test_stack(params, device) -> None:
    """Test stack forward pass"""
    shape, n, axis = params["shape"], params["n"], params["axis"]
    to_stack_ndl = []
    to_stack_npy = []
    for i in range(n):
        a = rng.standard_normal(shape)
        to_stack_ndl += [NDArray(a, device=device)]
        to_stack_npy += [a]

    target = np.stack(to_stack_npy, axis=axis)
    result = array_api.stack(to_stack_ndl, axis=axis)

    np.testing.assert_allclose(result.numpy(), target, rtol=1e-6)


@given(data=st.data())
@all_devices()
@pytest.mark.skip(reason="This test should pass, but fails due to a bug.")
def test_array_split(data, device):
    arr = data.draw(array_strategy)
    nd_array = NDArray(arr, device=device)

    # Generate random indices for splitting
    indices = data.draw(st.integers(min_value=1, max_value=max(1, arr.shape[0] - 1)))

    # Split the array using numpy and needle
    np_split = np.array_split(arr, indices)
    nd_split = array_api.split(nd_array, indices)

    # Compare the results
    for i in range(len(np_split)):
        np.testing.assert_array_equal(nd_split[i].numpy(), np_split[i])


@given(arr=array_strategy)
@all_devices()
def test_transpose(arr, device) -> None:
    nd_arr = NDArray(arr, device=device)
    permutations = np.random.permutation(arr.ndim).tolist()

    np_trans = arr.transpose(permutations)
    nd_trans = array_api.transpose(nd_arr, permutations)
    np.testing.assert_allclose(nd_trans.numpy(), np_trans, rtol=1e-5, atol=1e-8)


@given(data=st.data())
@all_devices()
def test_concatenate(data, device) -> None:
    arrs = [data.draw(arrays(dtype=DTYPE_FLOAT, shape=(3, 3))) for i in range(5)]
    arr_dim = arrs[0].ndim
    axis = data.draw(st.integers(min_value=-arr_dim, max_value=arr_dim - 1))

    arrs_ndl = [NDArray(arr, device=device) for arr in arrs]

    np_result = np.concatenate(arrs, axis=axis)
    ndl_result = array_api.concatenate(tuple(arrs_ndl), axis=axis)

    np.testing.assert_allclose(ndl_result.numpy(), np_result, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "array_data, shape, strides",
    [
        (np.arange(1, 10, dtype=np.int32), (7, 3), (4, 4)),
        (np.arange(1, 10, dtype=np.int32), (5,), (2 * 4,)),
        (np.arange(1, 10, dtype=np.int32).reshape(3, 3), (2, 2), (4, 8)),
        (np.arange(1, 10, dtype=np.int32).reshape(1, 3, 3), (2, 2), (4, 8)),
    ],
    ids=[
        "1D_array",
        "1D_array_2D_shape",
        "2D_array_with_strides",
        "3D_array_with_strides",
    ],
)
def test_as_strided(array_data, shape, strides) -> None:
    """
    Test _as_strided against NumPy's as_strided.
    """
    ndl_array = array_api.array(array_data)
    result = array_api._as_strided(ndl_array, shape=shape, strides=strides)

    expected = np.lib.stride_tricks.as_strided(array_data, shape=shape, strides=strides)

    np.testing.assert_array_equal(result.numpy(), expected)
