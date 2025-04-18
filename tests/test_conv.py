import needle as ndl
import numpy as np
import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.gradient_check import backward_check

rng = np.random.default_rng()


def backward_forward():
    return pytest.mark.parametrize(
        "backward", [True, False], ids=["backward", "forward"]
    )


# ====================== DILATE ======================


@pytest.mark.parametrize(
    "input,dilation,axes,expected",
    [
        pytest.param(
            np.array([[6.0, 1.0, 4.0, 4.0, 8.0], [4.0, 6.0, 3.0, 5.0, 8.0]]),
            0,
            (0,),
            np.array([[6.0, 1.0, 4.0, 4.0, 8.0], [4.0, 6.0, 3.0, 5.0, 8.0]]),
            id="2d_no_dilation_axis0",
        ),
        pytest.param(
            np.array([[7.0, 9.0, 9.0, 2.0, 7.0], [8.0, 8.0, 9.0, 2.0, 6.0]]),
            1,
            (0,),
            np.array([
                [7.0, 9.0, 9.0, 2.0, 7.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 8.0, 9.0, 2.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            id="2d_dilation1_axis0",
        ),
        pytest.param(
            np.array([[9.0, 5.0, 4.0, 1.0, 4.0], [6.0, 1.0, 3.0, 4.0, 9.0]]),
            1,
            (1,),
            np.array([
                [9.0, 0.0, 5.0, 0.0, 4.0, 0.0, 1.0, 0.0, 4.0, 0.0],
                [6.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 9.0, 0.0],
            ]),
            id="2d_dilation1_axis1",
        ),
        pytest.param(
            np.array([[2.0, 4.0, 4.0, 4.0, 8.0], [1.0, 2.0, 1.0, 5.0, 8.0]]),
            1,
            (0, 1),
            np.array([
                [2.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 5.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            id="2d_dilation1_axis01",
        ),
        pytest.param(
            np.array([[4.0, 3.0], [8.0, 3.0]]),
            2,
            (0, 1),
            np.array([
                [4.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            id="2d_dilation2_axis01",
        ),
        pytest.param(
            np.array([
                [[[1.0, 1.0], [5.0, 6.0]], [[6.0, 7.0], [9.0, 5.0]]],
                [[[2.0, 5.0], [9.0, 2.0]], [[2.0, 8.0], [4.0, 7.0]]],
            ]),
            1,
            (1, 2),
            np.array([
                [
                    [[1.0, 1.0], [0.0, 0.0], [5.0, 6.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[6.0, 7.0], [0.0, 0.0], [9.0, 5.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[2.0, 5.0], [0.0, 0.0], [9.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[2.0, 8.0], [0.0, 0.0], [4.0, 7.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
            ]),
            id="4d_dilation1_axis12",
        ),
    ],
)
@all_devices()
def test_dilate_forward(input, dilation, axes, expected, device):
    A = ndl.Tensor(input, device=device)
    result = ndl.dilate(A, dilation=dilation, axes=axes).numpy()

    # Values are not changed, so tolerance=0
    np.testing.assert_array_equal(result, expected)


@st.composite
def dilate_params(draw):
    """Generate valid parameters for dilate tests.

    Generates:
    - Shape with 2-4 dimensions
    - Dilation value (0-3)
    - Valid unique axes for dilation
    """
    # Generate shape with 2-4 dimensions
    n_dims = draw(st.integers(min_value=2, max_value=4))
    shape = tuple(draw(st.integers(min_value=2, max_value=8)) for _ in range(n_dims))

    # Generate dilation value (0-3)
    dilation = draw(st.integers(min_value=0, max_value=3))

    # Generate number of axes to dilate (1 to n_dims)
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

    return {"shape": shape, "d": dilation, "axes": axes}


@given(params=dilate_params())
@all_devices()
def test_dilate_backward(params, device):
    """Test dilate operation backward pass"""
    shape, d, axes = params["shape"], params["d"], params["axes"]
    backward_check(
        ndl.dilate,
        ndl.Tensor(rng.standard_normal(shape), device=device),
        dilation=d,
        axes=axes,
        tol=0,
    )


# ====================== NN_CONV ======================


@st.composite
def conv_params(draw):
    """
    Generate valid parameters for Conv layer testing.
    """
    batch_size = draw(st.integers(min_value=1, max_value=2))
    in_channels = draw(st.integers(min_value=1, max_value=8))
    height = draw(st.integers(min_value=1, max_value=32))
    width = draw(st.integers(min_value=1, max_value=32))

    out_channels = draw(st.integers(min_value=1, max_value=8))
    kernel = draw(st.integers(min_value=1, max_value=3))
    stride = draw(st.integers(min_value=1, max_value=2))

    return {
        "batch_size": batch_size,
        "in_channels": in_channels,
        "height": height,
        "width": width,
        "out_channels": out_channels,
        "kernel": kernel,
        "stride": stride,
    }


@given(params=conv_params())
@all_devices()
@backward_forward()
def test_nn_conv(params, device, backward):
    """
    Test convolution forward and backward passes against PyTorch
    """

    height = params["height"]
    width = params["width"]
    in_channels = params["in_channels"]
    out_channels = params["out_channels"]
    kernel = params["kernel"]
    stride = params["stride"]
    batch_size = params["batch_size"]

    # Initialize Needle convolution
    needle_conv = ndl.nn.Conv(
        in_channels, out_channels, kernel, stride=stride, device=device
    )
    input_tensor = ndl.init.rand(
        (batch_size, in_channels, height, width),
        device=device,
        requires_grad=backward,  # Only need gradients for backward pass
    )

    # Initialize equivalent PyTorch convolution
    torch_conv = torch.nn.Conv2d(
        in_channels, out_channels, kernel, stride=stride, padding=kernel // 2
    )
    torch_conv.weight.data = torch.tensor(
        needle_conv.weight.realize_cached_data().numpy().transpose(3, 2, 0, 1)
    )
    torch_conv.bias.data = torch.tensor(needle_conv.bias.realize_cached_data().numpy())
    torch_input = torch.tensor(input_tensor.cached_data.numpy(), requires_grad=backward)

    # Forward pass
    needle_output = needle_conv(input_tensor)
    torch_output = torch_conv(torch_input)

    # Check forward pass
    np.testing.assert_allclose(
        needle_output.realize_cached_data().numpy(),
        torch_output.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )

    if backward:
        # Backward pass
        needle_sum = needle_output.sum()
        torch_sum = torch_output.sum()

        torch_sum.backward()
        needle_sum.backward()

        # Check gradients
        np.testing.assert_allclose(
            needle_conv.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1),
            torch_conv.weight.grad.data.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            needle_conv.bias.grad.cached_data.numpy(),
            torch_conv.bias.grad.data.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            input_tensor.grad.cached_data.numpy(),
            torch_input.grad.data.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )


# ====================== OP_CONV ======================

OP_CONV_PARAMS = [
    ((3, 14, 14, 8), (3, 3, 8, 16), 1, 0),  # basic case
    ((3, 14, 14, 8), (3, 3, 8, 16), 1, 1),  # with padding
    ((3, 16, 16, 8), (3, 3, 8, 16), 1, 2),  # more padding
    ((3, 16, 16, 8), (3, 3, 8, 14), 1, 0),  # different output channels
    ((3, 16, 16, 2), (3, 3, 2, 14), 1, 0),  # fewer input channels
    ((3, 14, 14, 8), (3, 3, 8, 16), 2, 0),  # stride 2
    ((3, 14, 14, 8), (3, 3, 8, 16), 2, 1),  # stride 2 with padding
    ((3, 16, 16, 8), (3, 3, 8, 16), 2, 2),  # stride 2 more padding
    ((3, 16, 16, 8), (3, 3, 8, 14), 2, 0),  # stride 2 different outputs
    ((3, 16, 16, 2), (3, 3, 2, 14), 2, 0),  # stride 2 fewer channels
    ((3, 16, 16, 24), (3, 3, 24, 14), 1, 0),  # more channels
    ((3, 14, 14, 8), (5, 5, 8, 16), 1, 0),  # larger kernel
    ((3, 17, 17, 8), (5, 5, 8, 16), 1, 0),  # odd size input
    ((3, 17, 17, 1), (5, 5, 1, 16), 1, 0),  # single input channel
    ((3, 17, 17, 16), (5, 5, 16, 1), 1, 0),  # single output channel
    ((3, 17, 17, 16), (1, 1, 16, 1), 1, 0),  # 1x1 conv
    ((1, 14, 14, 2), (3, 3, 2, 2), 1, 0),  # batch size 1
]


@pytest.mark.parametrize(
    "input_shape,kernel_shape,stride,padding",
    OP_CONV_PARAMS,
    ids=[
        f"{s}-kernel={k}-stride={stride}-padding={p}"
        for s, k, stride, p in OP_CONV_PARAMS
    ],
)
@all_devices()
@backward_forward()
def test_op_conv(input_shape, kernel_shape, stride, padding, backward, device):
    """Test convolution operation against PyTorch implementation"""
    # Initialize input and kernel tensors
    input_array = rng.standard_normal(input_shape, dtype=np.float32)
    kernel_array = rng.standard_normal(kernel_shape, dtype=np.float32)

    # Create Needle tensors
    input_ndl = ndl.Tensor(input_array, device=device)
    kernel_ndl = ndl.Tensor(kernel_array, device=device)

    # Create PyTorch tensors
    input_torch = torch.tensor(input_array, dtype=torch.float32, requires_grad=True)
    kernel_torch = torch.tensor(kernel_array, dtype=torch.float32, requires_grad=True)

    # Forward pass
    result_ndl = ndl.conv(input_ndl, kernel_ndl, padding=padding, stride=stride)
    result_sum_ndl = result_ndl.sum()

    # Equivalent PyTorch operation with channel order adjustment
    result_torch = torch.nn.functional.conv2d(
        input_torch.permute(0, 3, 1, 2),
        kernel_torch.permute(3, 2, 0, 1),
        padding=padding,
        stride=stride,
    )
    result_sum_torch = result_torch.sum()

    # Check forward pass results
    np.testing.assert_allclose(
        result_ndl.numpy(),
        result_torch.permute(0, 2, 3, 1).contiguous().detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result_sum_ndl.numpy().item(),
        result_sum_torch.detach().numpy().item(),
        rtol=1e-4,
        atol=1e-4,
    )

    if backward:
        # Backward pass
        result_sum_torch.backward()
        result_sum_ndl.backward()

        # Check input gradients
        np.testing.assert_allclose(
            input_ndl.grad.numpy(),
            input_torch.grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        # Check kernel gradients
        np.testing.assert_allclose(
            kernel_ndl.grad.numpy(),
            kernel_torch.grad.numpy(),
            rtol=1e-2,
            atol=1e-2,
        )


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    height=st.integers(min_value=8, max_value=32),
    width=st.integers(min_value=8, max_value=32),
    in_channels=st.integers(min_value=1, max_value=16),
    out_channels=st.integers(min_value=1, max_value=16),
    kernel_size=st.integers(min_value=1, max_value=5),
    stride=st.integers(min_value=1, max_value=2),
    padding=st.integers(min_value=0, max_value=2),
)
@all_devices()
@backward_forward()
def test_op_conv_proptest(
    batch_size,
    height,
    width,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    device,
    backward,
):
    """Property-based test for convolution operation against PyTorch implementation.

    This test generates random but valid combinations of:
    - Input tensor shape (batch, height, width, channels)
    - Kernel shape (size, size, in_channels, out_channels)
    - Stride and padding values

    And verifies that needle's convolution matches PyTorch's implementation
    for both forward and backward passes.
    """
    # Create shapes from generated dimensions
    input_shape = (batch_size, height, width, in_channels)
    kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)

    # Compute potential output shape to validate it
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    assume(out_height > 0)
    assume(out_width > 0)

    # Initialize input and kernel tensors
    input_array = rng.standard_normal(input_shape, dtype=np.float32)
    kernel_array = rng.standard_normal(kernel_shape, dtype=np.float32)

    # Create Needle tensors
    input_ndl = ndl.Tensor(input_array, device=device)
    kernel_ndl = ndl.Tensor(kernel_array, device=device)

    # Create PyTorch tensors with gradients enabled
    input_torch = torch.tensor(input_array, dtype=torch.float32, requires_grad=True)
    kernel_torch = torch.tensor(kernel_array, dtype=torch.float32, requires_grad=True)

    # Forward pass - wrapped in try since not all combinations might be valid
    result_ndl = ndl.conv(input_ndl, kernel_ndl, padding=padding, stride=stride)
    result_sum_ndl = result_ndl.sum()

    # Equivalent PyTorch operation with channel order adjustment
    result_torch = torch.nn.functional.conv2d(
        input_torch.permute(0, 3, 1, 2),
        kernel_torch.permute(3, 2, 0, 1),
        padding=padding,
        stride=stride,
    )
    result_sum_torch = result_torch.sum()

    # Check forward pass results match
    np.testing.assert_allclose(
        result_ndl.numpy(),
        result_torch.permute(0, 2, 3, 1).contiguous().detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )

    if backward:
        # Backward pass
        result_sum_torch.backward()
        result_sum_ndl.backward()

        # Check gradients match
        np.testing.assert_allclose(
            input_ndl.grad.numpy(),
            input_torch.grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            kernel_ndl.grad.numpy(),
            kernel_torch.grad.numpy(),
            rtol=1e-2,
            atol=1e-2,
        )
