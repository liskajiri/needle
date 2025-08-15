import random

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from needle.backend_ndarray import NDArray, array_api


@given(
    st.lists(
        st.floats(min_value=0.1, max_value=10),
        min_size=2,
        max_size=100,
    )
)
def test_zero_copy_array_interface(lst_array):
    arr = NDArray(lst_array)
    # Get NumPy view without copy
    np_arr = np.array(arr, copy=False)
    # Modify the NumPy array

    for i in range(5):
        random_index = random.randint(0, len(lst_array) - 1)
        random_value = random.random()

        np_arr[random_index] = random_value

        # Check if the NDArray was updated
        assert arr.numpy()[random_index] == random_value, (
            "Memory is not shared between NumPy and NDArray"
        )

    for i in range(5):
        random_index = random.randint(0, len(lst_array) - 1)
        random_value = random.random()

        # Modify the NDArray
        arr[random_index] = random_value
        # Check if NumPy array reflects the change
        assert np_arr[random_index] == random_value, (
            "Memory is not shared between NDArray and NumPy"
        )

    nd_ptr = arr._handle.ptr()
    np_ptr = np_arr.__array_interface__["data"][0]
    assert nd_ptr == np_ptr, "Arrays don't share memory (different addresses)"


@given(
    st.lists(
        st.floats(min_value=-10, max_value=10),
        min_size=2,
        max_size=100,
    )
)
def test_dlpack_numpy_interop(lst_array):
    arr = NDArray(lst_array)
    # Create a NumPy array from DLPack
    np_arr = np.from_dlpack(arr)
    # Create another NDArray from the DLPack capsule
    c = array_api.from_dlpack(arr)

    for i in range(5):
        random_index = random.randint(0, len(lst_array) - 1)
        random_value = random.random()

        # Modify the NDArray
        arr[random_index] = random_value
        # Check if NumPy array reflects the change
        assert np_arr[random_index] == random_value == c[random_index], f"""
            Memory is not shared between NDArray and NumPy
            {arr=}
            {np_arr=}
            {c=}
            """


@given(
    arrays(
        dtype=np.float32,
        shape=(3, 3),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    ),
)
def test_ndarray_to_numpy_conversion(arr):
    # Convert to NDArray
    nd_arr = NDArray(arr)
    np_arr_back = nd_arr.numpy()

    np.testing.assert_array_equal(
        arr, np_arr_back, "Conversion to NumPy failed", strict=True
    )
