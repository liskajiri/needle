import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.gradient_check import backward_check

rng = np.random.default_rng()


@st.composite
def stack_params(draw):
    """Generate valid parameters for stack tests"""
    # Generate shape with 2-4 dimensions, each dimension 1-10
    shape = tuple(draw(st.integers(1, 10)) for _ in range(draw(st.integers(2, 4))))
    # Number of tensors to stack: 1-5
    n = draw(st.integers(1, 5))
    # Axis can be 0 to len(shape)
    axis = draw(st.integers(0, len(shape)))
    return {"shape": shape, "n": n, "axis": axis}


@given(params=stack_params())
@all_devices()
def test_stack_backward(params, device):
    """Test stack backward pass"""
    shape, n, axis = params["shape"], params["n"], params["axis"]
    tensors = [ndl.Tensor(rng.standard_normal(shape), device=device) for _ in range(n)]
    backward_check(ndl.stack, tensors, axis=axis)


@given(params=stack_params())
@all_devices()
def test_stack_forward(params, device):
    """Test stack forward pass"""
    shape, n, axis = params["shape"], params["n"], params["axis"]
    to_stack_ndl = []
    to_stack_npy = []
    for i in range(n):
        a = rng.standard_normal(shape)
        to_stack_ndl += [ndl.Tensor(a, device=device)]
        to_stack_npy += [a]

    target = np.stack(to_stack_npy, axis=axis)
    result = ndl.stack(to_stack_ndl, axis=axis)

    np.testing.assert_allclose(result.numpy(), target, rtol=1e-6)


@all_devices()
def test_stack_vs_pytorch(device):
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
            torch_tensor.grad.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Gradient mismatch for tensor {name}",
        )


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
def test_pad_forward(params, device):
    """Test padding operation forward pass"""
    shape, padding = params["shape"], params["padding"]
    input_array = rng.standard_normal(shape)
    ndl_array = ndl.NDArray(input_array, device=device)

    target = np.pad(input_array, padding)
    result = ndl.array_api.pad(ndl_array, padding)

    np.testing.assert_allclose(result.numpy(), target, rtol=1e-6)


@st.composite
def flip_params(draw):
    """Generate valid parameters for flip tests.

    Generates:
    - Shape with 2-4 dimensions
    - Valid axes for flipping
    """
    # Generate shape with 2-4 dimensions
    n_dims = draw(st.integers(min_value=2, max_value=4))
    shape = tuple(draw(st.integers(min_value=2, max_value=32)) for _ in range(n_dims))

    # Generate number of axes to flip (1 to n_dims)
    n_axes = draw(st.integers(min_value=1, max_value=n_dims))

    # Generate unique sorted axes
    axes = tuple(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=n_dims - 1),
                min_size=n_axes,
                max_size=n_axes,
                unique=True,
            )
        )
    )

    return {"shape": shape, "axes": axes}


@given(params=flip_params())
@all_devices()
def test_flip_forward(params, device):
    """Test flip operation forward pass"""
    shape, axes = params["shape"], params["axes"]
    input_array = rng.standard_normal(shape)
    ndl_array = ndl.Tensor(input_array, device=device)

    target = np.flip(input_array, axes)
    result = ndl.flip(ndl_array, axes=axes)

    np.testing.assert_allclose(result.numpy(), target, rtol=1e-6)


@given(params=flip_params())
@all_devices()
def test_flip_backward(params, device):
    """Test flip operation backward pass"""
    shape, axes = params["shape"], params["axes"]
    backward_check(
        ndl.flip, ndl.Tensor(rng.standard_normal(shape), device=device), axes=axes
    )


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
def test_as_strided(array_data, shape, strides):
    """
    Test _as_strided against NumPy's as_strided.
    """
    ndl_array = ndl.array_api.array(array_data)
    result = ndl.array_api._as_strided(ndl_array, shape=shape, strides=strides)

    expected = np.lib.stride_tricks.as_strided(array_data, shape=shape, strides=strides)

    np.testing.assert_array_equal(result.numpy(), expected)
