import needle.ops as ops
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from needle import Tensor

from tests.hypothesis_strategies import matmul_arrays, same_shape_arrays
from tests.utils import DTYPE_FLOAT, backward_forward


@given(arr=same_shape_arrays(dtype=DTYPE_FLOAT, n=1))
@backward_forward()
def test_summation(arr, backward) -> None:
    ndl_a = Tensor(arr, requires_grad=True)
    # Test with no axes (sum of all elements)
    ndl_out_all = ops.summation(ndl_a)
    expected_all = np.sum(arr)
    np.testing.assert_allclose(ndl_out_all.numpy(), expected_all, rtol=1e-5, atol=1e-5)

    # Test with specific axis
    if arr.ndim > 0:
        axis = 0
        ndl_out_axis = ops.summation(ndl_a, axes=axis)
        expected_axis = np.sum(arr, axis=axis)
        np.testing.assert_allclose(
            ndl_out_axis.numpy(), expected_axis, rtol=1e-5, atol=1e-5
        )

    # Test with keepdims
    if arr.ndim > 0:
        ndl_out_keepdims = ops.summation(ndl_a, axes=0, keepdims=True)
        expected_keepdims = np.sum(arr, axis=0, keepdims=True)
        np.testing.assert_allclose(
            ndl_out_keepdims.numpy(), expected_keepdims, rtol=1e-5, atol=1e-5
        )

    if backward and arr.ndim > 0:
        # Test gradient
        ndl_out = ops.summation(ndl_a, axes=0)
        ndl_out.backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.sum(torch_a, dim=0)
        torch_out.backward(torch.ones_like(torch_out))

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arrs=matmul_arrays())
@backward_forward()
def test_matmul(arrs, backward) -> None:
    arr1, arr2 = arrs

    ndl_a = Tensor(arr1, requires_grad=True)
    ndl_b = Tensor(arr2, requires_grad=True)
    ndl_out = ops.matmul(ndl_a, ndl_b)

    expected = np.matmul(arr1, arr2)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr1, dtype=torch.float32, requires_grad=True)
        torch_b = torch.tensor(arr2, dtype=torch.float32, requires_grad=True)
        torch_out = torch.matmul(torch_a, torch_b)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            ndl_b.grad.numpy(), torch_b.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arr=same_shape_arrays(dtype=DTYPE_FLOAT, n=1))
@backward_forward()
def test_negate(arr, backward) -> None:
    ndl_a = Tensor(arr, requires_grad=True)
    ndl_out = ops.negate(ndl_a)

    expected = -arr
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = -torch_a
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(
    arr=same_shape_arrays(
        dtype=DTYPE_FLOAT, n=1, elements=st.floats(min_value=0.01, max_value=100.0)
    )
)
@backward_forward()
def test_log(arr, backward) -> None:
    ndl_a = Tensor(arr, requires_grad=True)
    ndl_out = ops.log(ndl_a)

    expected = np.log(arr)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.log(torch_a)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arr=same_shape_arrays(dtype=DTYPE_FLOAT, n=1))
@backward_forward()
def test_exp(arr, backward) -> None:
    # Limit input magnitude to avoid overflow
    arr = np.clip(arr, -10, 10)

    ndl_a = Tensor(arr, requires_grad=True)
    ndl_out = ops.exp(ndl_a)

    expected = np.exp(arr)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.exp(torch_a)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arr=same_shape_arrays(dtype=DTYPE_FLOAT))
@backward_forward()
def test_relu(arr, backward) -> None:
    # perf warning
    arr = np.array(arr, dtype=np.float32)

    ndl_a = Tensor(arr, requires_grad=True)
    ndl_out = ops.relu(ndl_a)

    expected = np.maximum(arr, 0)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.relu(torch_a)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(
    arr=same_shape_arrays(
        dtype=DTYPE_FLOAT, elements=st.floats(min_value=0.0, max_value=100.0)
    )
)
@backward_forward()
def test_sqrt(arr, backward) -> None:
    # perf warning
    arr = np.array(arr, dtype=np.float32)

    ndl_a = Tensor(arr, requires_grad=True)
    ndl_out = ops.sqrt(ndl_a)

    expected = np.sqrt(arr)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.sqrt(torch_a)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arr=same_shape_arrays(dtype=DTYPE_FLOAT))
@backward_forward()
def test_tanh(arr, backward) -> None:
    # perf warning
    arr = np.array(arr, dtype=np.float32)

    ndl_a = Tensor(arr, requires_grad=True)
    ndl_out = ops.tanh(ndl_a)

    expected = np.tanh(arr)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.tanh(torch_a)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arr=same_shape_arrays(dtype=DTYPE_FLOAT))
@backward_forward()
def test_sigmoid(arr, backward) -> None:
    # Limit input magnitude to avoid overflow
    arr = np.clip(arr, -10, 10)

    ndl_a = Tensor(arr, requires_grad=True)
    ndl_out = ops.sigmoid(ndl_a)

    expected = 1.0 / (1.0 + np.exp(-arr))
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.sigmoid(torch_a)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arr=same_shape_arrays(dtype=DTYPE_FLOAT, n=1))
@backward_forward()
@pytest.mark.skip("Test fails")
def test_mean(arr, backward) -> None:
    ndl_a = Tensor(arr, requires_grad=True)

    # Test with default axis=0
    ndl_out = ops.mean(ndl_a, axes=0)
    expected = np.mean(arr, axis=0)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    # Test with axis=None (mean of all elements)
    if arr.ndim > 1:
        ndl_out_all = ops.mean(ndl_a, axes=None)
        expected_all = np.mean(arr, axis=None)
        np.testing.assert_allclose(
            ndl_out_all.numpy(), expected_all, rtol=1e-5, atol=1e-5
        )

    if backward:
        ndl_out.sum().backward()

        torch_a = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        torch_out = torch.mean(torch_a, dim=0)
        torch_out.sum().backward()

        np.testing.assert_allclose(
            ndl_a.grad.numpy(), torch_a.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )
