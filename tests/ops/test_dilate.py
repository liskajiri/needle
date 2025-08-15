import needle as ndl
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.gradient_check import backward_check

rng = np.random.default_rng()

# ====================== DILATE ======================
# TODO


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
            np.array(
                [
                    [7.0, 9.0, 9.0, 2.0, 7.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [8.0, 8.0, 9.0, 2.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            id="2d_dilation1_axis0",
        ),
        pytest.param(
            np.array([[9.0, 5.0, 4.0, 1.0, 4.0], [6.0, 1.0, 3.0, 4.0, 9.0]]),
            1,
            (1,),
            np.array(
                [
                    [9.0, 0.0, 5.0, 0.0, 4.0, 0.0, 1.0, 0.0, 4.0, 0.0],
                    [6.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 9.0, 0.0],
                ]
            ),
            id="2d_dilation1_axis1",
        ),
        pytest.param(
            np.array([[2.0, 4.0, 4.0, 4.0, 8.0], [1.0, 2.0, 1.0, 5.0, 8.0]]),
            1,
            (0, 1),
            np.array(
                [
                    [2.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 8.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 5.0, 0.0, 8.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            id="2d_dilation1_axis01",
        ),
        pytest.param(
            np.array([[4.0, 3.0], [8.0, 3.0]]),
            2,
            (0, 1),
            np.array(
                [
                    [4.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            id="2d_dilation2_axis01",
        ),
        pytest.param(
            np.array(
                [
                    [[[1.0, 1.0], [5.0, 6.0]], [[6.0, 7.0], [9.0, 5.0]]],
                    [[[2.0, 5.0], [9.0, 2.0]], [[2.0, 8.0], [4.0, 7.0]]],
                ]
            ),
            1,
            (1, 2),
            np.array(
                [
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
                ]
            ),
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


# TODO: remove backward check after refactor


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
