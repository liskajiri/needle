import itertools

import numpy as np

from numpy_backend.device import AbstractBackend, Axis, DType, Shape, Strides
from numpy_backend.numpy_backend import NDArray, default_device


def array(
    a: NDArray,
    dtype: DType = "float32",
    device: AbstractBackend = default_device,
) -> NDArray:
    return NDArray(a, dtype=dtype, device=device)


def zeros(
    shape: Shape, dtype: DType = "float32", device: AbstractBackend = default_device
) -> NDArray:
    return NDArray(np.zeros(shape, dtype=dtype), dtype=dtype, device=device)


def ones(
    shape: Shape, dtype: DType = "float32", device: AbstractBackend = default_device
) -> NDArray:
    return NDArray(np.ones(shape, dtype=dtype), dtype=dtype, device=device)


def max(array: NDArray, axis: Axis | None = None, keepdims: bool = False) -> NDArray:
    return array.max(axis=axis, keepdims=keepdims)


def min(array: NDArray, axis: Axis | None = None, keepdims: bool = False) -> NDArray:
    arr = -array
    return -arr.max(axis=axis, keepdims=keepdims)


def reshape(array: NDArray, new_shape: Shape) -> NDArray:
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray) -> NDArray:
    return a.maximum(b)


def log(a: NDArray) -> NDArray:
    return NDArray(np.log(a.array), device=a.device)


def exp(a: NDArray) -> NDArray:
    return NDArray(np.exp(a.array), device=a.device)


def tanh(a: NDArray) -> NDArray:
    return NDArray(np.tanh(a.array), device=a.device)


def sum(a: NDArray, axis: Axis | None = None, keepdims: bool = False) -> NDArray:
    return a.sum(axis=axis, keepdims=keepdims)


def flip(a: NDArray, axis: Axis) -> NDArray:
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
    # (convert neg to pos idx)
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

    out = np.empty(a._shape, dtype=a.dtype)
    return out


def pad(a: NDArray, axes: tuple[tuple[int, int], ...]) -> NDArray:
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
    if not arrays:
        raise AssertionError("Cannot stack empty list of arrays")

    base_arr = arrays[0]
    if not all(array.shape == base_arr.shape for array in arrays):
        raise AssertionError("All arrays must have identical shapes")

    if axis < -(base_arr.ndim + 1) or axis > base_arr.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for arrays of dimension {base_arr.ndim}"
        )
    if axis < 0:
        axis += base_arr.ndim + 1

    out_shape = (*base_arr.shape[:axis], len(arrays), *base_arr.shape[axis:])
    out = NDArray(np.empty(out_shape, dtype=base_arr.dtype), device=base_arr.device)

    slice_spec: list = [slice(None)] * out.ndim

    for idx, array in enumerate(arrays):
        slice_spec[axis] = idx
        out[tuple(slice_spec)] = array

    return out


def split(
    arr: NDArray, sections: int | list[int] | None = None, axis: int = 0
) -> list[NDArray]:
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


def transpose(a: NDArray, axes: tuple[int, ...] | None = None) -> NDArray:
    if axes is None:
        axes = tuple(range(a.ndim))[::-1]
    return a.permute(axes)


def concatenate(arrays: tuple[NDArray, ...], axis: int = 0) -> NDArray:
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
    out = NDArray(
        np.empty(tuple(new_shape), dtype=base_array.dtype), device=base_array.device
    )

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
    elem_strides = tuple(s // array.device.itemsize for s in strides)
    return array.as_strided(shape, elem_strides)


def broadcast_to(array: NDArray, new_shape: Shape) -> NDArray:
    return np.broadcast_to(array, new_shape)
