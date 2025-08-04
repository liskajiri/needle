import needle.ops as ops
import numpy as np
import torch
from hypothesis import given
from needle import Tensor

from tests.hypothesis_strategies import same_shape_arrays
from tests.utils import backward_forward


@given(arrs=same_shape_arrays())
@backward_forward()
def test_add(arrs, backward) -> None:
    arr1, arr2 = arrs
    ndl_a = Tensor(arr1, requires_grad=True)
    ndl_b = Tensor(arr2, requires_grad=True)
    ndl_out = ops.add(ndl_a, ndl_b)

    expected = np.add(arr1, arr2)
    np.testing.assert_allclose(ndl_out.numpy(), expected, rtol=1e-5, atol=1e-5)

    if backward:
        ndl_out.sum().backward()
        grad_a = ndl_a.grad.numpy()
        grad_b = ndl_b.grad.numpy()

        torch_a = torch.tensor(arr1, dtype=torch.float32, requires_grad=True)
        torch_b = torch.tensor(arr2, dtype=torch.float32, requires_grad=True)
        torch_out = torch_a + torch_b

        # grab grads directly via autograd.grad
        grad_t_a, grad_t_b = torch.autograd.grad(
            outputs=torch_out.sum(), inputs=(torch_a, torch_b)
        )

        # Compare gradients
        np.testing.assert_allclose(
            grad_a, grad_t_a.detach().numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            grad_b, grad_t_b.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(arrs=same_shape_arrays())
def test_multiply(arrs) -> None:
    arr1, arr2 = arrs
    ndl_result = ops.multiply(Tensor(arr1), Tensor(arr2))
    expected = np.multiply(arr1, arr2)
    np.testing.assert_allclose(ndl_result.numpy(), expected, rtol=1e-5, atol=1e-5)


# @given(arrs=same_shape_arrays(dtype=DTYPE_INT, elements=st.integers(1, 10)))
# def test_power(arrs) -> None:
#     arr1, arr2 = arrs
#     ndl_result = ops.power(Tensor(arr1), Tensor(arr2))
#     expected = np.power(arr1, arr2)
#     np.testing.assert_allclose(ndl_result.numpy(), expected, rtol=1e-5, atol=1e-5)


@given(arrs=same_shape_arrays())
def test_divide(arrs) -> None:
    arr1, arr2 = arrs
    # Avoid division by zero
    arr2 = arr2 + 1e-1
    ndl_result = ops.divide(Tensor(arr1), Tensor(arr2))
    expected = np.divide(arr1, arr2)
    np.testing.assert_allclose(ndl_result.numpy(), expected, rtol=1e-5, atol=1e-5)
