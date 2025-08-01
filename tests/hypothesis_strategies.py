import numpy as np
import torch
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    basic_indices,
    mutually_broadcastable_shapes,
)

from tests.utils import DTYPE_FLOAT

rng = np.random.default_rng()

# Define common strategies
array_shapes_strategy = array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=5)

float_strategy = st.floats(
    min_value=-10.0, max_value=10.0, allow_infinity=False, allow_nan=False
)

safe_float_strategy = float_strategy.filter(lambda x: abs(x) >= 1e-4)

int_strategy = st.integers(min_value=-10, max_value=10)

array_strategy = arrays(
    dtype=DTYPE_FLOAT,
    shape=array_shapes_strategy,
    elements=float_strategy,
)

# Strategy for non-zero floats (small range to avoid overflow)
nonzero_float_strategy = st.one_of(
    st.floats(min_value=0.1, max_value=2.0),
    st.floats(min_value=-2.0, max_value=-0.1),
)

# Strategy for small power values
power_float_strategy = st.floats(
    min_value=-2.0, max_value=2.0, allow_infinity=False, allow_nan=False
)

# Strategy for positive arrays (for log)
positive_array_strategy = arrays(
    dtype=DTYPE_FLOAT,
    shape=array_shapes_strategy,
    elements=st.floats(
        min_value=0.1, max_value=2.0, allow_infinity=False, allow_nan=False
    ),
)

# Strategy for small arrays (for exp)
small_array_strategy = arrays(
    dtype=DTYPE_FLOAT,
    shape=array_shapes_strategy,
    elements=st.floats(
        min_value=-2.0, max_value=2.0, allow_infinity=False, allow_nan=False
    ),
)


# Strategy for arrays of the same shape
@st.composite
def same_shape_arrays(
    draw,
    n=2,
    dtype=DTYPE_FLOAT,
    shape=None,
    elements=safe_float_strategy,
) -> list[np.ndarray]:
    """Generate two arrays of the same shape."""
    # this ensures that shapes of all arrays are the same
    if shape is None:
        shape = draw(array_shapes_strategy)
    arrs = [draw(arrays(dtype=dtype, shape=shape, elements=elements)) for _ in range(n)]
    if n == 1:
        return arrs[0]
    return arrs


@st.composite
def torch_tensors(
    draw,
    n=2,
    dtype=DTYPE_FLOAT,
    shape=array_shapes_strategy,
    elements=safe_float_strategy,
):
    """Generate a list of tensors with the same shape."""
    arrs = draw(
        same_shape_arrays(
            n=n,
            dtype=dtype,
            shape=shape,
            elements=elements,
        )
    )
    return [torch.tensor(arr, requires_grad=True, dtype=torch.double) for arr in arrs]


# Strategy for division arrays (guaranteed same shape, non-zero denominator)
@st.composite
def division_arrays(draw):
    """Generate two arrays suitable for division."""
    return draw(
        same_shape_arrays(n=2, dtype=DTYPE_FLOAT, elements=nonzero_float_strategy)
    )


# Strategy for matrix multiplication shapes
@st.composite
def matmul_arrays(draw, min_dims=2, max_dims=6):
    """Generate arrays suitable for matmul with optional batch dimensions."""
    # Keep matrix dimensions small to avoid overflow
    m = draw(st.integers(min_value=1, max_value=3))
    n = draw(st.integers(min_value=1, max_value=3))
    p = draw(st.integers(min_value=1, max_value=3))

    # Decide if we want both arrays batched or one batched and one not
    both_batched = draw(st.booleans())

    if both_batched:
        # both arrays have the same batch dimensions
        n_batch_dims = draw(st.integers(min_value=min_dims - 2, max_value=max_dims - 2))
        batch_dims = tuple(
            draw(st.integers(min_value=1, max_value=3)) for _ in range(n_batch_dims)
        )

        shape1 = (*batch_dims, m, n)
        shape2 = (*batch_dims, n, p)
    else:
        # New case: one array is batched, the other is not
        # Decide which array gets the batch dimensions
        first_is_batched = draw(st.booleans())

        # Generate batch dimensions for the batched array
        n_batch_dims = draw(st.integers(min_value=1, max_value=max_dims - 2))
        batch_dims = tuple(
            draw(st.integers(min_value=1, max_value=3)) for _ in range(n_batch_dims)
        )

        if first_is_batched:
            # First array is batched, second is 2D
            shape1 = (*batch_dims, m, n)
            shape2 = (n, p)
        else:
            # First array is 2D, second is batched
            shape1 = (m, n)
            shape2 = (*batch_dims, n, p)

    arr1 = draw(arrays(dtype=DTYPE_FLOAT, shape=shape1, elements=safe_float_strategy))
    arr2 = draw(arrays(dtype=DTYPE_FLOAT, shape=shape2, elements=safe_float_strategy))
    return arr1, arr2


# Strategy for broadcasting with different ranks
@st.composite
def broadcastable_arrays(draw, num_shapes=2):
    """Generate arrays with different ranks that can be broadcast together."""
    # Get shapes that can be broadcast
    shapes = draw(
        mutually_broadcastable_shapes(num_shapes=num_shapes, min_dims=1, max_dims=4)
    )
    arrs = [
        draw(arrays(dtype=DTYPE_FLOAT, shape=shape, elements=safe_float_strategy))
        for shape in shapes.input_shapes
    ]

    return arrs


# Strategy for shapes with dimension inference (-1)
@st.composite
def reshape_shapes(draw):
    """Generate shape pairs for testing reshape with -1."""
    # Original size between 1 and 12
    size = draw(st.integers(min_value=1, max_value=12))

    # Generate a valid divisor for the size
    factors = [i for i in range(1, size + 1) if size % i == 0]

    # Choose how to split the size
    if len(factors) > 1:
        factor = draw(
            st.sampled_from(factors[1:])
        )  # Skip 1 to ensure multiple dimensions
        # Original shape can be either (size,) or (factor, size//factor)
        orig_shape = draw(st.sampled_from([(size,), (factor, size // factor)]))
        # New shape uses -1 to infer one dimension
        dim = draw(st.sampled_from(factors))
        new_shape = (dim, -1) if size // dim > 0 else (-1, dim)
    else:
        # If size is prime, use simple shapes
        orig_shape = (size,)
        new_shape = (-1,)

    return orig_shape, new_shape


@st.composite
def array_and_axis(draw):
    # draw an array from your pre-existing generator
    a_np = draw(array_strategy)

    # now pick a valid axis: 0 <= axis < ndim
    axis = draw(st.integers(min_value=0, max_value=a_np.ndim - 1))

    return a_np, axis


@st.composite
def reshape_examples(draw):
    """Strategy to generate arrays and valid new shapes for reshape testing."""
    # Generate original array shape and data
    orig_shape = draw(array_shapes_strategy)
    array = draw(arrays(dtype=np.float32, shape=orig_shape))

    total_elements = array.size

    # Choose number of dimensions for new shape
    new_ndim = draw(st.integers(min_value=1, max_value=4))

    # Generate a valid new shape
    # For simplicity, we'll use a helper function to find factors
    def get_valid_shape(size, ndim):
        # Base case
        if ndim == 1:
            return (size,)

        # Choose a factor for the first dimension
        valid_factors = [i for i in range(1, min(8, size + 1)) if size % i == 0]
        if not valid_factors:
            return (1,) * (ndim - 1) + (size,)

        factor = draw(st.sampled_from(valid_factors))
        # Recursively fill the rest
        return (factor, *get_valid_shape(size // factor, ndim - 1))

    new_shape = get_valid_shape(total_elements, new_ndim)

    # Ensure we're not testing the same shape (uninteresting)
    assume(new_shape != orig_shape)

    return array, new_shape


@st.composite
def setitem_idx_strategy(draw):
    shape = draw(array_shapes_strategy)
    lhs_idx = draw(basic_indices(shape))
    rhs_idx = draw(basic_indices(shape))

    np_arr = np.zeros(shape, dtype=DTYPE_FLOAT)

    try:
        lhs_view = np_arr[lhs_idx]
        rhs_view = np_arr[rhs_idx]
        assume(lhs_view.shape == rhs_view.shape and lhs_view.size > 0)
    except IndexError:
        assume(False)

    return {
        "shape": shape,
        "lhs_idx": lhs_idx,
        "rhs_idx": rhs_idx,
    }


@st.composite
def scalar_slice_strategy(draw):
    shape = draw(array_shapes_strategy)
    idx = draw(basic_indices(shape))

    np_a = rng.normal(size=shape)
    try:
        view = np_a[idx]
        assume(view.size > 0)
    except IndexError:
        assume(False)

    if not isinstance(idx, int | slice):
        idx = tuple(idx)
    return shape, idx


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
