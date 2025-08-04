import needle as ndl
import numpy as np
import torch
from hypothesis import given

from tests.devices import all_devices
from tests.gradient_check import backward_check
from tests.hypothesis_strategies import (
    flip_params,
    stack_params,
)

rng = np.random.default_rng()


@given(params=stack_params())
@all_devices()
def test_stack_backward(params, device) -> None:
    """Test stack backward pass"""
    shape, n, axis = params["shape"], params["n"], params["axis"]
    tensors = [ndl.Tensor(rng.standard_normal(shape), device=device) for _ in range(n)]
    backward_check(ndl.stack, tensors, axis=axis)


@all_devices()
def test_stack_vs_pytorch(device) -> None:
    """Test stack operation against PyTorch implementation with gradients"""
    A = rng.standard_normal((5, 5))
    B = rng.standard_normal((5, 5))
    C = rng.standard_normal((5, 5))
    D = rng.standard_normal((15, 5))

    A_ndl = ndl.Tensor(A, requires_grad=True, device=device)
    B_ndl = ndl.Tensor(B, requires_grad=True, device=device)
    C_ndl = ndl.Tensor(C, requires_grad=True, device=device)
    D_ndl = ndl.Tensor(D, requires_grad=True, device=device)

    A_torch = torch.tensor(A, requires_grad=True)
    B_torch = torch.tensor(B, requires_grad=True)
    C_torch = torch.tensor(C, requires_grad=True)
    D_torch = torch.tensor(D, requires_grad=True)

    X_ndl = ndl.stack([A_ndl, C_ndl @ B_ndl, C_ndl], axis=1)
    X_torch = torch.stack([A_torch, C_torch @ B_torch, C_torch], dim=1)

    assert X_ndl.shape == X_torch.shape
    np.testing.assert_allclose(
        X_ndl.numpy(), X_torch.detach().numpy(), rtol=1e-4, atol=1e-4
    )

    Y_ndl = (D_ndl @ X_ndl.reshape((5, 15)) @ D_ndl).sum()
    Y_torch = (D_torch @ X_torch.reshape(5, 15) @ D_torch).sum()

    np.testing.assert_allclose(
        Y_ndl.numpy(), Y_torch.detach().numpy(), rtol=1e-4, atol=1e-4
    )

    Y_ndl.backward()
    Y_torch.backward()

    # Check gradients match PyTorch
    for ndl_tensor, torch_tensor, name in [
        (A_ndl, A_torch, "A"),
        (B_ndl, B_torch, "B"),
        (C_ndl, C_torch, "C"),
        (D_ndl, D_torch, "D"),
    ]:
        np.testing.assert_allclose(
            ndl_tensor.grad.numpy(),
            torch_tensor.grad.detach().numpy(),  # type: ignore
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Gradient mismatch for tensor {name}",
        )


@given(params=flip_params())
@all_devices()
def test_flip_backward(params, device) -> None:
    """Test flip operation backward pass"""
    shape, axes = params["shape"], params["axes"]
    backward_check(
        ndl.flip, ndl.Tensor(rng.standard_normal(shape), device=device), axes=axes
    )
