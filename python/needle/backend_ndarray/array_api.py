from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

# from needle.backend_ndarray.device import default_device
from needle.backend_ndarray.ndarray import NDArray, default_device, make

if TYPE_CHECKING:
    from needle.typing import AbstractBackend, DType, Shape, Strides, np_ndarray


def from_numpy(a: np_ndarray) -> NDArray:
    return NDArray(a)


def array(
    a: np_ndarray | NDArray | list,
    dtype="float32",
    device: AbstractBackend = default_device,
) -> NDArray:
    """Convenience methods to match numpy a bit more closely."""
    if dtype != "float32":
        logging.warning(f"Only support float32 for now, got {dtype}")
        logging.warning(f"Converting to numpy array with dtype {dtype}")
        dtype = "float32"
    assert dtype == "float32"
    return NDArray(a, dtype=dtype, device=device)


def empty(
    shape: Shape, dtype: DType = "float32", device: AbstractBackend = default_device
) -> NDArray:
    return device.empty(shape, dtype)


def zeros(
    shape: Shape, dtype: DType = "float32", device: AbstractBackend = default_device
) -> NDArray:
    return device.zeros(shape, dtype)


def ones(
    shape: Shape, dtype: DType = "float32", device: AbstractBackend = default_device
) -> NDArray:
    return device.ones(shape, dtype)


def full(
    shape: Shape,
    fill_value: float,
    dtype: DType = "float32",
    device: AbstractBackend = default_device,
) -> NDArray:
    return device.full(shape, fill_value, dtype)


def broadcast_to(array: NDArray, new_shape: Shape) -> NDArray:
    return array.broadcast_to(new_shape)


def max(array: NDArray, axis=None, keepdims: bool = False) -> NDArray:
    return array.max(axis=axis, keepdims=keepdims)


def reshape(array: NDArray, new_shape: Shape) -> NDArray:
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray) -> NDArray:
    return a.maximum(b)


def log(a: NDArray) -> NDArray:
    return a.log()


def exp(a: NDArray) -> NDArray:
    return a.exp()


def tanh(a: NDArray) -> NDArray:
    return a.tanh()


def sum(a: NDArray, axis: tuple = (), keepdims: bool = False) -> NDArray:
    return a.sum(axis=axis, keepdims=keepdims)


def flip(a: NDArray, axis: tuple[int, ...] | int) -> NDArray:
    """
    Flip this ndarray along the specified axes.
    Note: compacts the array before returning.

    Args:
        axes: Tuple or int specifying the axes to flip

    Returns:
        NDArray: New array with flipped axes
    """
    # Handle single axis case
    if isinstance(axis, int):
        axis = (axis,)

    # Validate axes
    for ax in axis:
        if ax < -a.ndim or ax >= a.ndim:
            raise ValueError(
                f"Axis {ax} is out of bounds for array of dimension {a.ndim}"
            )

    # Normalize negative axes
    # TODO: this will be common in array ops, make it a function
    # (convert neg to pos  idx)
    axis = tuple(ax if ax >= 0 else a.ndim + ax for ax in axis)

    # Create new view with modified strides and offset
    new_strides = list(a._strides)
    offset = a._offset

    # For each axis to flip:
    # 1. Make stride negative to traverse in reverse order
    # 2. Adjust offset to start from end of axis
    for ax in axis:
        new_strides[ax] = -a._strides[ax]
        offset += a._strides[ax] * (a._shape[ax] - 1)

    out = make(
        a._shape,
        strides=tuple(new_strides),
        device=a.device,
        handle=a._handle,
        offset=offset,
    )
    # Return compacted array to ensure standard memory layout
    # TODO: Copies memory, if negative strides are supported, this can be avoided
    return out.compact()


def pad(a, axes: tuple[tuple[int, int], ...]) -> NDArray:
    """
    Pad this ndarray by zeros by the specified amount in `axes`,
    which lists for _all_ axes the left and right padding amount, e.g.,

    axes = ( (0, 0), (1, 1), (0, 0) )
    pads the middle axis with a 0 on the left and right side.

    Note: This has to create a new array and copy the data over.

    Args:
        axes: Tuple of tuples specifying padding amount for each axis

    Returns:
        NDArray: New array with padding added

    Raises:
        ValueError: If padding axes do not match array dimensions

    >>> import needle as ndl
    >>> import numpy as np
    >>> a = NDArray(np.array([[1, 2], [3, 4]]))
    >>> print(ndl.array_api.pad(a, ((1, 1), (1, 1))))
    [[0. 0. 0. 0.]
     [0. 1. 2. 0.]
     [0. 3. 4. 0.]
     [0. 0. 0. 0.]]
    """
    if len(axes) != a.ndim:
        raise ValueError(f"Padding axes {axes} must match array dimensions {a.ndim}")

    # Calculate new shape after padding
    new_shape = tuple(
        dim + left + right for dim, (left, right) in zip(a.shape, axes, strict=False)
    )

    # Create output array filled with zeros
    out = a.device.zeros(new_shape, dtype=a.dtype)

    # Create slices to insert original data
    slices = tuple(
        slice(left, left + dim) for dim, (left, _) in zip(a.shape, axes, strict=False)
    )
    # Copy data into padded array
    out[slices] = a
    return out


def stack(arrays: tuple[NDArray] | list[NDArray], axis: int = 0) -> NDArray:
    """Stack a list of arrays along specified axis.

    Args:
        arrays: List of NDArrays with identical shapes
        axis: Integer axis along which to stack (default=0)

    Returns:
        NDArray: Stacked array with shape expanded on specified axis

    Raises:
        AssertionError: If arrays list is empty or shapes don't match
        ValueError: If axis is out of bounds
    """
    if not arrays:
        raise AssertionError("Cannot stack empty list of arrays")

    base_array = arrays[0]
    if not all(array.shape == base_array.shape for array in arrays):
        raise AssertionError("All arrays must have identical shapes")

    if axis < -(base_array.ndim + 1) or axis > base_array.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for arrays of dimension {base_array.ndim}"
        )
    if axis < 0:
        axis += base_array.ndim + 1

    output_shape = list(base_array.shape)
    output_shape.insert(axis, len(arrays))
    output_shape = tuple(output_shape)

    out = empty(output_shape, device=base_array.device)

    slice_spec: list = [slice(None)] * out.ndim

    for idx, array in enumerate(arrays):
        slice_spec[axis] = idx
        out[tuple(slice_spec)] = array

    return out


def split(
    arr: NDArray, sections: int | list[int] | None = None, axis: int = 0
) -> list[NDArray]:
    """Split array into multiple sub-arrays along specified axis.

    Args:
        arr: NDArray to split
        indices_or_sections: If int, number of equal sections to split into.
                            If list/tuple, the indices indicating split points.
        axis: Integer axis along which to split (default=0)

    Returns:
        List of NDArrays: Split arrays along specified axis

    Raises:
        ValueError: If axis is out of bounds
    """
    # Handle negative axis indexing
    if axis < 0:
        axis += arr.ndim
    if not 0 <= axis < arr.ndim:
        raise ValueError(f"Axis {axis} out of bounds")
    if sections is None:
        sections = arr.shape[axis]

    if isinstance(sections, int):
        # Handle N sections case
        section_size = arr.shape[axis] // sections
        remainder = arr.shape[axis] % sections
        indices = []
        acc = 0
        for i in range(sections - 1):
            acc += section_size + (1 if i < remainder else 0)
            indices.append(acc)
    else:
        indices = sections

    # Create split points
    split_points = [0, *list(indices), arr.shape[axis]]
    out = []
    slice_spec = [slice(None)] * arr.ndim

    for start, end in itertools.pairwise(split_points):
        slice_spec[axis] = slice(start, end)
        # Get the sub-array
        sub_array = arr[tuple(slice_spec)]
        # For single-element slices, reshape to remove the dimension
        if end - start == 1:
            out_shape = list(arr.shape)
            out_shape.pop(axis)
            sub_array = sub_array.compact().reshape(tuple(out_shape))
        out.append(sub_array)

    return out


def transpose(a: NDArray, axes: tuple = ()) -> NDArray:
    if not axes:
        axes = tuple(range(a.ndim))[::-1]
    return a.permute(axes)


def concatenate(arrays: tuple[NDArray], axis: int = 0) -> NDArray:
    """Concatenate arrays along an existing axis.

    Args:
        arrays: List of NDArrays with identical shapes except in the concat dimension
        axis: Integer axis along which to concatenate (default=0)

    Returns:
        NDArray: Concatenated array
    """
    from builtins import sum as py_sum

    base_array = arrays[0]

    # Handle negative axis
    if axis < 0:
        axis += base_array.ndim

    if not 0 <= axis < base_array.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for array of dimension {base_array.ndim}"
        )

    # Validate shapes - all dimensions except concat axis must match
    for arr in arrays[1:]:
        for i in range(base_array.ndim):
            if i != axis and arr.shape[i] != base_array.shape[i]:
                raise ValueError(
                    f"All arrays must have same shape except in the concatenation axis."
                    f"Got {base_array.shape} and {arr.shape}"
                )

    # Calculate new shape
    new_shape = list(base_array.shape)
    new_shape[axis] = py_sum(arr.shape[axis] for arr in arrays)

    # Create output array
    out = empty(tuple(new_shape), device=base_array.device)

    # Copy data into position
    offset = 0
    slice_spec = [slice(None)] * base_array.ndim
    for arr in arrays:
        size = arr.shape[axis]
        slice_spec[axis] = slice(offset, offset + size)
        out[tuple(slice_spec)] = arr
        offset += size

    return out


def _as_strided(array: NDArray, shape: Shape, strides: Strides) -> NDArray:
    """
    Create a view into the array with the given shape and strides.

    Args:
        array: Input array
        shape: The shape of the new array
        strides: The strides of the new array (in bytes)

    Returns:
        NDArray: A view into the array with the given shape and strides

    Example:
        >>> import needle as ndl
        >>> a = NDArray([[1, 2], [3, 4]])
        >>> b = _as_strided(a, (2, 2), (8, 4))
        >>> print(b)
        [[1. 2.]
         [3. 4.]]

        >>> b.strides
        (2, 1)
        >>> b.shape
        (2, 2)
    """
    elem_strides = tuple(s // array.device.itemsize for s in strides)
    return array.as_strided(shape, elem_strides)
