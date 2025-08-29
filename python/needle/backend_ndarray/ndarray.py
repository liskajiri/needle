from __future__ import annotations

import logging
import math
import random
from functools import cached_property
from typing import TYPE_CHECKING, override

import numpy as np

from needle.errors import BroadcastError
from needle.typing import AbstractBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from needle.typing import (
        Axis,
        DType,
        IndexType,
        NDArrayLike,
        Scalar,
        Shape,
        Strides,
        np_ndarray,
    )
    from needle.typing.device import ModuleProtocol, NDArrayBackendProtocol
    from needle.typing.dlpack import DLPackDeviceId, DLPackDeviceType

logger = logging.getLogger(__name__)

# TODO: reference hw3.ipynb for future optimizations
# TODO: investigate usage of __slots__, Python's array.array for NDArray class


class BackendDevice(AbstractBackend):
    # TODO: dtype?

    def __init__(
        self, name: str, module: ModuleProtocol[NDArray] | None = None
    ) -> None:
        if module is None:
            super().__init__(name, module, -1, -1)
        else:
            super().__init__(
                name,
                module,
                tile_size=module.__tile_size__,
                itemsize=module.itemsize,
            )

    def randn(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        """
        Generate random array from standard normal distribution
        Uses native backend implementation when available, falls back to Python RNG.
        """
        if isinstance(shape, int):
            shape = (shape,)

        size = (math.prod(shape),)
        arr = self.empty(size, dtype=dtype)

        # Use native backend RNG if available
        if self.enabled():
            # backend.randn expects an AlignedArray/handle and fills it in-place
            self.module.randn(arr._handle)
        else:
            # deterministic fallback for tests / platforms without native backend
            random.seed(0)
            for i in range(arr.size):
                arr[i] = random.gauss(0.0, 1.0)

        return arr.reshape(shape)

    def rand(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        """
        Generate random samples from uniform distribution [0,1).
        Uses native backend implementation when available, falls back to Python RNG.
        """
        if isinstance(shape, int):
            shape = (shape,)

        size = (math.prod(shape),)
        arr = self.empty(size, dtype=dtype)

        if self.enabled():
            self.module.rand(arr._handle)
        else:
            random.seed(0)
            for i in range(arr.size):
                arr[i] = random.uniform(0.0, 1.0)

        return arr.reshape(shape)

    @override
    def one_hot(self, n: int, i: IndexType, dtype: DType) -> NDArray:
        if self.enabled():
            # allocate output on device with shape (*idx.shape, n)
            i = NDArray(i, device=self)
            i_shape = (*i.shape, n)
            out = make(i_shape, device=self)

            assert self.module is not None
            self.module.one_hot(out._handle, i._handle, n)
            return out
        else:
            raise NotImplementedError()

        # # Fallback: pure-python / numpy implementation
        # idx = np.asarray(i)
        # out = np.zeros((*idx.shape, n), dtype=dtype)

        # # Fill the hot positions efficiently
        # # (idx.size, n)
        # flat = out.reshape(-1, n)
        # flat[np.arange(idx.size), idx.ravel()] = 1

        # return NDArray(out, device=self)

    # def one_hot(self, n: int, i: IndexType, dtype: DType) -> NDArray:
    #     """Create a one-hot vector.

    #     Args:
    #         n (int): Length of the vector.
    #         i (int): Index of the one-hot element.
    #         dtype (DType): Data type of the array.

    #     Returns:
    #         NDArray: A one-hot vector.
    #     """
    #     arr = self.zeros((n,), dtype)
    #     arr[i] = 1.0
    #     return arr

    @override
    def empty(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        return make(shape, device=self)

    @override
    def set_seed(self, seed: int | None = None) -> None:
        # Set Python RNGs
        random.seed(seed)
        if seed is not None:
            np.random.seed(seed)
        # Propagate seed to native backend if available
        if seed is not None and self.enabled():
            # backend.set_seed expects an unsigned int
            self.module.set_seed(int(seed))

    @staticmethod
    def _tiled_matmul(arr: NDArray, other: NDArray, m: int, n: int, p: int) -> NDArray:
        def _tile(a: NDArray, tile: int) -> NDArray:
            """
            Transforms a matrix [k, n] into a
            matrix [k // tile, n // tile, tile, tile].
            """
            return a.as_strided(
                (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                (a.shape[1] * tile, tile, a.shape[1], 1),
            ).compact()

        t = arr.device.__tile_size__
        a = _tile(arr.compact(), t)
        b = _tile(other.compact(), t)
        out = make((a.shape[0], b.shape[1], t, t), device=arr.device)
        arr.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

        return (
            out.permute((0, 2, 1, 3)).compact().reshape((arr.shape[0], other.shape[1]))
        )


def cuda() -> AbstractBackend:
    """Return cuda device."""
    try:
        from needle.backend_ndarray import ndarray_backend_cuda  # type: ignore

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy() -> AbstractBackend:
    """Return numpy device."""
    try:
        from needle import backend_numpy
        from needle.backend_numpy import NumpyBackend

        return NumpyBackend("cpu_numpy", backend_numpy)  # type: ignore
    except ImportError:
        raise ImportError("Numpy backend not available")


def cpu() -> AbstractBackend:
    """Return cpu device."""
    try:
        from backends.cpu import ndarray_backend_cpu  # type: ignore
        # import ndarray_backend_cpu

        return BackendDevice("cpu", ndarray_backend_cpu)  # type: ignore
    except ImportError:
        raise ImportError("CPU backend not available")


def all_devices() -> list[AbstractBackend]:
    """Return a list of all available devices."""
    return [cpu(), cuda(), cpu_numpy()]


default_device = cpu()


def make(
    shape: Shape,
    strides: Strides | None = None,
    device: AbstractBackend = default_device,
    handle: NDArrayBackendProtocol | None = None,
    offset: int = 0,
) -> NDArray:
    """
    Create a new NDArray with the given properties.
    Allocates a new array if handle is not provided.

    Args:
        shape: Tuple specifying dimensions of the array
        strides: Optional tuple specifying stride for each dimension
        device: Device backend for the array (defaults to CPU)
        handle: Existing handle to use for memory (allocates new if None)
        offset: Memory offset for the array (default: 0)

    Returns:
        NDArray: New array with requested properties

    Raises:
        ValueError: If shape contains invalid dimensions

    Examples:
        >>> make((2, 3)).shape
        (2, 3)
        >>> make((2, 3), strides=(1, 2)).strides
        (1, 2)
        >>> make((2, 3), strides=(1, 2))._offset
        0
        >>> make((2, 3), strides=(1, 2), offset=5)._offset
        5
        >>> make((2, 3), strides=(1, 2), device=cpu()).device
        cpu()
    """

    def prod(shape: Shape) -> int:
        """Calculate product of shape tuple, handling nested tuples."""
        result = 1
        for dim in shape:
            if isinstance(dim, tuple):
                result *= prod(dim)
            else:
                result *= dim
        return result

    array = NDArray.__new__(NDArray)
    array._shape = shape
    array._strides = NDArray.compact_strides(shape) if strides is None else strides
    array._offset = offset
    array._device = device

    array_size = prod(shape)
    if handle is None:
        if array_size < 0:
            raise ValueError(f"Array size cannot be negative, Invalid shape: {shape}")
        array._handle = array.device.Array(array_size)
    else:
        array._handle = handle
    return array


def broadcast_shapes(*shapes: Shape) -> Shape:
    """
    Return broadcasted shape for multiple input shapes.

    Broadcasting rules (numpy-style):
        1. Start with the trailing (rightmost) dimensions and continue left.
        2. Two dimensions are compatible when:
           - They are equal
           - One of them is 1

    Args:
        *shapes: one or more shapes as tuples

    Returns:
        tuple: broadcast-compatible shape

    Raises:
        BroadcastError: If shapes cannot be broadcast together


    Examples:
        >>> broadcast_shapes((2, 3), (1, 3))
        (2, 3)
        >>> broadcast_shapes((2, 3), (3,))
        (2, 3)
        >>> broadcast_shapes((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5))
        (8, 7, 6, 5)
        >>> broadcast_shapes((2, 3), (2, 4))
        Traceback (most recent call last):
        ...
        BroadcastError: Incompatible shapes for broadcasting: ((2, 3), (2, 4))
    """
    # If only one shape provided, return it
    if len(shapes) == 1:
        return shapes[0]

    max_dims = max(len(shape) for shape in shapes)
    # Left-pad shorter shapes with 1s to align dimensions
    aligned_shapes = [(1,) * (max_dims - len(s)) + s for s in shapes]

    # Determine output dimension for each position
    result = []
    for dims in zip(*aligned_shapes, strict=False):
        max_dim = max(dims)
        for d in dims:
            if d != 1 and d != max_dim:
                raise BroadcastError(shapes)
        result.append(max_dim)

    return tuple(result)


class NDArray:  # noqa: PLR0904 = too many public methods
    """
    A generic ND array class that may contain multiple different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    For now, for simplicity the class only supports float32 types.

    # TODO: Create array from list, tuple, or scalar
    Examples:
        >>> # Create from Python list
        >>> arr = NDArray([[1, 2], [3, 4]])
        >>> arr.shape
        (2, 2)
        >>> arr.ndim
        2
        >>> arr.size
        4

        >>> # Create from numpy array
        >>> import numpy as np
        >>> arr = NDArray(np.zeros((3, 2)))
        >>> arr.shape
        (3, 2)

        >>> # Create copy on specific device
        >>> cpu_arr = NDArray([1, 2, 3], device=cpu())
        >>> cpu_arr.device
        cpu()
    """

    def __init__(
        self,
        other: NDArrayLike | None = None,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        """Initialize an NDArray by copying from another array-like object.

        Args:
            other: Source array-like object to copy from. Can be:
                - Another NDArray (creates a copy)
                - A numpy ndarray
                - A Python list/tuple that can be converted to ndarray
            device: The backend device to create the array on. Defaults to CPU.
            dtype: Data type for the array. Currently only float32 is supported.

        >>> # Create from Python list
        >>> arr = NDArray([1, 2])
        >>> arr.shape
        (2,)
        >>> arr.ndim
        1
        >>> arr.size
        2

        >>> # Create from Python list
        >>> arr = NDArray([[1, 2], [3, 4]])
        >>> arr.shape
        (2, 2)
        >>> arr.ndim
        2
        >>> arr.size
        4
        """
        if isinstance(other, NDArray):
            # Create a copy of existing NDArray
            if device != other.device:
                logger.error(
                    f"Creating NDArray with different device {device} != {other.device}"
                )
            # create a copy
            array = other.to(device) + 0.0
        elif isinstance(other, np.ndarray):
            array = make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
        # TODO: from_tuple
        elif isinstance(other, list):
            other, shape = NDArray._flatten_iterable(other)

            array = make(shape=shape, device=device)
            # empty tuple
            array.device.from_list(other, array._handle)
        else:
            # see if we can create a numpy array from input
            np_arr = np.array(other)
            array = NDArray(np_arr, device=device)

        self._shape: Shape = array._shape
        self._strides: Strides = array._strides
        # TODO: clarify if this is items or bytes
        self._offset: int = array._offset
        self._device: AbstractBackend = array._device
        self._handle: NDArrayBackendProtocol = array._handle

    @staticmethod
    def _flatten_iterable(lst: list | tuple) -> tuple[list, Shape]:
        """
        Recursively flattens a nested iterable and returns its flattened version
        along with the original shape (dimensions).

        Args:
            lst (list | tuple): A nested iterable of arbitrary depth,
            e.g., `[[1, 2], [3, 4]]`

        Returns:
            tuple: `(flat_list, shape)`, where
                - flat_list is a `1D` list of all scalar elements
                - shape is a list of integers representing the dimensions

        Raises:
            ValueError: If the input has inconsistent dimensions

        Examples:
            >>> NDArray._flatten_iterable([[1, 2], [3, 4]])
            ([1, 2, 3, 4], (2, 2))

            >>> NDArray._flatten_iterable([[[1], [2]], [[3], [4]]])
            ([1, 2, 3, 4], (2, 2, 1))

            >>> NDArray._flatten_iterable([1, 2, 3])
            ([1, 2, 3], (3,))

            >>> NDArray._flatten_iterable([])
            ([], (0,))

            >>> NDArray._flatten_iterable(
            ...     [
            ...         [1, 2],
            ...         [3],
            ...     ]
            ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Inconsistent dimensions
        """
        flat = []
        shape = []

        def _flatten(sublist: list | tuple) -> list:
            if isinstance(sublist, list):
                if not sublist:
                    return [0]
                dims = [_flatten(item) for item in sublist]

                # Ensure all sub-dimensions match
                if any(d != dims[0] for d in dims):
                    raise ValueError("Inconsistent dimensions")
                flat_shape = [len(sublist)] + dims[0]
                return flat_shape
            else:
                flat.append(sublist)
                return []

        shape = _flatten(lst)
        return flat, tuple(shape)

    # ==================== Properties and string representations

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def strides(self) -> Strides:
        return self._strides

    @property
    def device(self) -> AbstractBackend:
        return self._device

    @property
    def dtype(self) -> DType:
        # only support float32 for now
        return "float32"

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @cached_property
    def size(self) -> int:
        return int(math.prod(self._shape))

    def __repr__(self) -> str:
        return self.__str__()

    # TODO: implement __str__ to print the data in the array
    def __str__(self) -> str:
        """String representation of the NDArray. (Inspired by numpy's __str__ method)

        >>> arr = NDArray([[1, 2], [3, 4]])
        >>> print(arr)
        [[1. 2.]
         [3. 4.]]
        """
        # TODO: Check that this is zero-copy
        data = self.numpy()
        return data.__str__()

    def __len__(self) -> int:
        """Returns the size of the first dimension.

        Returns:
            int: The size of the first dimension.
        """
        if len(self.shape) == 0:
            return 1
        return self.shape[0]

    # ==================== Basic array manipulation

    def fill(self, value: Scalar) -> None:
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device: AbstractBackend) -> NDArray:
        """
        Convert between devices, using to/from numpy calls as the unifying bridge.

        Args:
            device (AbstractBackend): The target backend device to convert to.

        Returns:
            NDArray: A new NDArray instance on the target device.
        """
        if device == self.device:
            return self
        return NDArray(self.numpy(), device=device)

    # ==================== Numpy/Array/DLPack Interop
    def numpy(self) -> np_ndarray:
        """Convert to a numpy array.

        Returns:
            np.ndarray: A numpy array with the same shape and data as the NDArray.
        """
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    @staticmethod
    def from_numpy(
        a: np_ndarray,
    ) -> NDArray:
        """
        Copy from a numpy array.

        Args:
            a (np.ndarray): The numpy array to copy from.

        Returns:
            NDArray: A NDArray with the same shape and data as the numpy array.
        """
        array = make(a.shape)
        array.device.from_numpy(np.ascontiguousarray(a), array._handle)
        return array

    def __array__(self, copy: bool = False) -> np_ndarray:
        """Convert to a numpy array.

        Returns:
            np.ndarray: A numpy array with the same shape and data as the NDArray.
        """
        return self.numpy()

    def __dlpack_device__(self) -> tuple[DLPackDeviceType, DLPackDeviceId]:
        """
        Returns a tuple of (device_type, device_id) representing the DLPack device.

        Device types follow DLPack:

        Returns:
            tuple: (device_type, device_id)
        """
        return self._handle.__dlpack_device__()

    def __dlpack__(
        self,
        *,
        max_version: tuple[int, int] = (2024, 12),
        stream: int | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool = False,
    ) -> object:
        """Export array as a DLPack capsule.

        Args:
            max_version: Maximum version of DLPack to use.
            stream: Optional CUDA stream (unused for CPU arrays)
            dl_device: Optional device ID for DLPack (unused for CPU arrays)
            copy: If True, a copy of the array is made. Defaults to False.

        Returns:
            A DLPack capsule that can be consumed by other frameworks.
            The capsule owns a copy of the array data to ensure safety.
        """
        return self._handle.__dlpack__(self._shape, self._strides, self._offset)

    # ==================== Shapes and strides

    @staticmethod
    def compact_strides(shape: Shape) -> Strides:
        """
        For a contiguous array, this calculates how many elements to skip to move
        one step along each dimension.
        Computation starts with the (rightmost) dimension and works outward.

        Args:
            shape: The shape of the array as a tuple of dimensions

        Returns:
            A tuple of strides, one for each dimension in the shape

        Examples:
            >>> NDArray.compact_strides((2, 3, 4))
            (12, 4, 1)
            >>> NDArray.compact_strides((5,))
            (1,)
            >>> NDArray.compact_strides(())
            ()
        """
        stride = 1
        strides = []
        for dim in reversed(shape):
            strides.append(stride)
            stride *= dim

        return tuple(reversed(strides))

    def is_compact(self) -> bool:
        """
        Check if the array is compact in memory.

        A compact array has contiguous memory layout and the internal size matches
        the product of its shape dimensions.

        Returns:
            bool: True if the array is compact, False otherwise.
        """
        return (
            self._strides == self.compact_strides(self._shape)
            and self.size == self._handle.size
        )

    def compact(self) -> NDArray:
        """
        Convert a matrix to be compact.

        Returns:
            NDArray: A new NDArray that is compact in memory.
        """
        if self.is_compact():
            return self
        out = make(self.shape, device=self.device)
        self.device.compact(
            self._handle,
            out._handle,
            self.shape,
            self.strides,
            self._offset,
        )
        return out

    def as_strided(self, shape: Shape, strides: Strides) -> NDArray:
        """
        Re-stride the matrix without copying memory.

        Args:
            shape (Shape): Target shape for the array
            strides (Strides): Strides for each dimension

        Returns:
            NDArray: NDArray that is a view of the original with new shape and strides.

        Raises:
            DimensionError: If the length of shape and strides do not match.
        """
        if len(shape) != len(strides):
            raise ValueError(
                "Shape and strides must have same number of dimensions, "
                f"got {len(shape)} vs {len(strides)}"
            )
        return make(
            shape,
            strides=strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset,
        )

    def astype(self, dtype: DType = "float32") -> NDArray:
        """
        Return a copy of the array with a different type.

        Args:
            dtype (DType): The desired data type for the array. Defaults to float32.

        Returns:
            NDArray: A new NDArray with the specified data type.
        """
        if dtype != "float32":
            logger.warning("Only support float32 for now", extra={"dtype": dtype})
            a = self.numpy().astype("float32")
            dtype = f"{a.dtype}"
            logger.warning(
                "Converting to numpy array with dtype",
                extra={"dtype": dtype},
            )
        assert dtype == "float32", f"Only support float32 for now, got {dtype}"
        return self + 0.0

    def flatten(self) -> NDArray:
        """
        Return a 1D view of the array.

        Returns:
            NDArray: A new NDArray that is a flattened view of the original.

        Example:
            >>> x = NDArray([[1, 2], [3, 4]])
            >>> x.flatten().shape
            (4,)
        """
        return self.reshape((self.size,))

    def reshape(self, new_shape: Shape) -> NDArray:
        """
        Reshape the matrix without copying memory.

        Returns a new array that points to the same memory but with a different shape.
        The total number of elements must remain the same and the array must be compact.

        Args:
            new_shape (Shape): Target shape for the array

        Returns:
            NDArray: Reshaped view of the array, sharing the same memory

        Raises:
            ValueError: If:
                - Total size changes between shapes
                - Array is not compact
                - Multiple -1 dimensions specified
                - Size is not divisible when using -1 dimension

        Examples:
            >>> x = NDArray(np.array([1, 2, 3, 4, 5, 6]))
            >>> x.reshape((2, 3)).shape
            (2, 3)
            >>> x.reshape((3, 2)).shape
            (3, 2)
            >>> x.reshape((2, -1)).shape  # Automatic dimension calculation
            (2, 3)
        """
        if self.shape == new_shape:
            return self

        # Special case: Convert 1D array to column vector
        if self._is_1d_to_column_vector(new_shape):
            return make(
                new_shape,
                device=self.device,
                handle=self._handle,
                strides=(self.strides[0], 0),
            )
        if not self.is_compact():
            raise ValueError("Cannot reshape non-compact array. Call compact() first.")

        # Handle automatic dimension calculation with -1
        new_shape = self._resolve_negative_dimensions(new_shape, self.size)

        # Verify total size remains constant
        if self.size != math.prod(new_shape):
            raise ValueError(
                f"Cannot reshape array of size {self.size} into shape {new_shape}"
            )

        # Create reshaped view with new strides
        return self.as_strided(new_shape, self.compact_strides(new_shape))

    def _is_1d_to_column_vector(self, new_shape: Shape) -> bool:
        """Check if reshaping from 1D array to column vector.

        Args:
            new_shape (Shape): New shape for the NDArray

        Returns:
        """
        return (
            len(self.shape) == 1
            and len(new_shape) == 2
            and self.shape[0] == new_shape[0]
            and new_shape[1] == 1
        )

    @staticmethod
    def _resolve_negative_dimensions(new_shape: Shape, size: int) -> Shape:
        """
        Calculate actual shape when -1 dimension is used.

        Args:
            new_shape: Proposed shape with possible -1 dimension

        Returns:
            Shape: Resolved shape with -1 replaced by calculated dimension

        Raises:
            ValueError: If multiple -1 dimensions or size mismatch
        """
        new_dims_count = new_shape.count(-1)
        if new_dims_count == 0:
            return new_shape

        # Verify only one -1 dimension
        if new_dims_count > 1:
            raise ValueError(
                f"Cannot deduce shape with multiple -1 dimensions in {new_shape}"
            )

        # Calculate missing dimension
        neg_idx = new_shape.index(-1)
        other_dims_product = math.prod(new_shape[:neg_idx] + new_shape[neg_idx + 1 :])

        if size % other_dims_product != 0:
            raise ValueError(
                f"Cannot reshape array of size {size} into shape {new_shape}. "
                f"Size must be divisible by {other_dims_product}."
            )

        missing_dim = size // other_dims_product
        return (*new_shape[:neg_idx], missing_dim, *new_shape[neg_idx + 1 :])

    def permute(self, new_axes: Shape) -> NDArray:
        """
        Permute the order of array dimensions without copying memory.

        Reorders dimensions by adjusting the shape and strides
        of the array, returning a view that points to the same memory.

        Args:
            new_axes (Shape): permutation order of the dimensions

        Returns:
            NDArray: A view of the array with permuted dimensions.

        Raises:
            ValueError: If the length of new_axes doesn't match ndim or
                    if new_axes isn't a valid permutation.

        Examples:
            >>> x = NDArray(np.arange(24).reshape(2, 3, 4))
            >>> x.shape
            (2, 3, 4)
            >>> # Transpose last two dimensions
            >>> y = x.permute((0, 2, 1))
            >>> y.shape
            (2, 4, 3)

            >>> # Simple matrix transpose
            >>> z = NDArray([[1, 2, 3], [4, 5, 6]])
            >>> z.permute((1, 0)).shape
            (3, 2)
            >>> # Permute 4D tensor from BHWC to BCHW format
            >>> t = NDArray(np.ones((8, 16, 16, 3)))  # BHWC format
            >>> t.permute((0, 3, 1, 2)).shape  # BCHW format
            (8, 3, 16, 16)
        """
        if len(new_axes) != self.ndim:
            raise ValueError(
                f"New axes {new_axes} has different number of axes than {self.ndim}"
            )

        # gets new shape
        new_shape = tuple(self._shape[d] for d in new_axes)
        new_strides = tuple(self._strides[d] for d in new_axes)

        return self.as_strided(new_shape, new_strides)

    def broadcast_to(self, new_shape: Shape) -> NDArray:
        """
        Broadcast an array to a new shape without copying memory.

        Broadcasting follows NumPy's rules:
        1. Dimensions are aligned from right to left
        2. Size-1 dimensions can be stretched to any size
        3. Missing dimensions are treated as size 1

        This operation changes the view (strides) of the array without copying data.

        Args:
            new_shape (Shape): Target shape to broadcast to

        Returns:
            NDArray: A view with the new broadcast shape pointing to the same memory

        Raises:
            BroadcastError: If shapes are incompatible for broadcasting

        Examples:
            >>> x = NDArray(np.array([[1], [2]]))  # Shape (2, 1)
            >>> x.broadcast_to((2, 3)).shape
            (2, 3)

            >>> # Add leading dimensions
            >>> y = NDArray(np.array([5]))  # Shape (1,)
            >>> y.broadcast_to((3, 2, 1)).shape
            (3, 2, 1)

            >>> # Error: can't broadcast dimension with size != 1
            >>> z = NDArray(np.array([[1, 2], [3, 4]]))  # Shape (2, 2)
            >>> z.broadcast_to((2, 3))
            Traceback (most recent call last):
            ...
            BroadcastError: Cannot broadcast shape (2, 2) to (2, 3)
        """
        if self.shape == new_shape:
            return self

        # Cannot broadcast to a smaller shape
        num_different_dims = len(new_shape) - len(self._shape)
        if num_different_dims < 0:
            raise BroadcastError(
                f"Cannot broadcast shape {self._shape} to {new_shape}: "
                f"target has fewer dimensions."
            )

        # Pad original shape/strides with size-1 dims at the front
        padded_shape = (1,) * num_different_dims + self._shape
        padded_strides = (0,) * num_different_dims + self._strides

        # Validate broadcasting and adjust strides
        new_strides = list(padded_strides)
        for i, (orig_dim, target_dim) in enumerate(zip(padded_shape, new_shape)):
            if orig_dim == target_dim:
                continue
            elif orig_dim == 1:
                # Broadcasted dimension → stride becomes 0
                new_strides[i] = 0
            else:
                raise BroadcastError(
                    f"Cannot broadcast shape {self._shape} to {new_shape}: "
                    f"size mismatch at dimension {i} (original: {orig_dim}, "
                    f"target: {target_dim})"
                )

        return self.as_strided(new_shape, tuple(new_strides))

    # ====================  Get and set elements

    def _process_slice(self, sl: slice, dim: int) -> slice:
        """Convert a slice to an explicit start/stop/step

        Returns:
            slice: A slice object with explicit start, stop, and step values.
                Also handles negative indices and adjusts start/stop accordingly.
                Ranges of values are:
                - start: [0, size)
                - stop: [-1 - size, size)]
        """
        size = self.shape[dim]
        step = sl.step if sl.step is not None else 1

        # Handle negative indices
        if sl.start is not None:
            start = sl.start if sl.start >= 0 else size + sl.start
        else:
            start = size - 1 if step < 0 else 0
        # start: [0, size)

        if sl.stop is not None:
            stop = sl.stop if sl.stop >= 0 else size + sl.stop
        else:
            stop = size if step > 0 else -1
        # stop: [-1 - size, size)]

        return slice(start, stop, step)

    def _prepare_indices(
        self,
        idxs: int | tuple[int | slice, ...] | slice,
    ) -> tuple[tuple[slice, ...], set[int]]:
        """
        Convert input indices to tuple of slices and track squeeze dimensions.
        """
        if idxs is Ellipsis:
            return tuple(slice(0, d, 1) for d in self.shape), set()

        # Convert single integer index to tuple
        if isinstance(idxs, int | slice):
            orig_idxs = (idxs,)
        elif isinstance(idxs, tuple):
            if Ellipsis in idxs:
                ellipsis_idx = idxs.index(Ellipsis)
                extra_dims = self.ndim - (len(idxs) - 1)
                expanded_slices = (slice(None),) * extra_dims
                orig_idxs = (
                    idxs[:ellipsis_idx] + expanded_slices + idxs[ellipsis_idx + 1 :]
                )
            else:
                orig_idxs = idxs
        else:
            raise TypeError(f"Invalid index type: {type(idxs)}")

        # Track dimensions to squeeze (from integer indexing)
        squeeze_dims = set()
        processed_idxs = []

        # Process each index
        for i, idx in enumerate(orig_idxs):
            if isinstance(idx, int):
                # Handle negative indices
                if idx < 0:
                    idx += self.shape[i]
                if not 0 <= idx < self.shape[i]:
                    raise IndexError(f"Index {idx} is out of bounds for axis {i}")
                processed_idxs.append(slice(idx, idx + 1, 1))
                squeeze_dims.add(i)
            else:
                processed_idxs.append(self._process_slice(idx, i))

        # Fill in remaining dimensions
        processed_idxs.extend(
            slice(0, self.shape[i], 1) for i in range(len(processed_idxs), self.ndim)
        )

        return tuple(processed_idxs), squeeze_dims

    def _compute_view_shape(
        self, slices: tuple[slice[int, int, int], ...], squeeze_dims: set[int]
    ) -> tuple[Shape, Strides, int]:
        """Compute shape, strides and offset for the sliced view."""
        new_shape = []
        new_strides = []
        new_offset = self._offset

        for i, slc in enumerate(slices):
            new_offset += self._strides[i] * slc.start

            if i in squeeze_dims:
                continue

            start, stop, step = slc.start, slc.stop, slc.step
            if (step > 0 and start >= stop) or (step < 0 and start <= stop):
                n_elements = 0
                new_strides.append(self._strides[i])
            else:
                n_elements = stop - start + step
                if step > 0:
                    n_elements -= 1
                else:
                    n_elements += 1
                n_elements //= step
                new_strides.append(self._strides[i] * step)

            new_shape.append(n_elements)

        return tuple(new_shape), tuple(new_strides), new_offset

    def _handle_array_indexing(self, idxs: NDArray) -> NDArray:
        """
        Handle advanced indexing with a NDArray of arbitrary shape.
        Gathers elements at the specified indices along the first axis.

        Examples
        --------
        >>> import needle as ndl
        >>> x = ndl.backend_ndarray.NDArray(
        ...     [
        ...         [10, 11],
        ...         [20, 21],
        ...         [30, 31],
        ...         [40, 41],
        ...     ]
        ... )
        >>> idx = ndl.backend_ndarray.NDArray([[2, 0], [1, 3]])
        >>> y = x._handle_array_indexing(idx)
        >>> y.numpy()
        array([[[30., 31.],
                [10., 11.]],
        <BLANKLINE>
               [[20., 21.],
                [40., 41.]]], dtype=float32)
        """
        # idxs: shape (...), values are indices into axis 0
        out_shape = idxs.shape + self.shape[1:]
        out = make(out_shape, device=self.device)

        # Recursively iterate over all indices in idxs
        def recursive_assign(idxs_array, prefix):
            if len(idxs_array.shape) == 0:
                idx = (
                    int(idxs_array.item())
                    if hasattr(idxs_array, "item")
                    else int(idxs_array)
                )
                src_idx = (idx,) + (slice(None),) * (self.ndim - 1)
                dst_idx = tuple(prefix) + (slice(None),) * (self.ndim - 1)
                out[dst_idx] = self[src_idx]  # type: ignore
            else:
                for i in range(idxs_array.shape[0]):
                    recursive_assign(idxs_array[i], [*prefix, i])

        recursive_assign(idxs, [])
        return out.compact()

    def __getitem__(self, idxs: IndexType) -> NDArray:
        """
        Get a view into the array based on the given indices.

        Implements Python's indexing protocol for array access. The method supports:
        - Integer indexing (single value)
        - Slice indexing (ranges of values)
        - Multiple dimension indexing (tuples of indices/slices)
        - Integer array indexing

        The resulting array shares memory with the original when possible.

        Args:
            idxs: Index specification, which can be:
                - An integer (single element)
                - A slice (range of elements)
                - A tuple of integers/slices (multi-dimensional indexing)
                - An array of integers (advanced indexing)
                - List of integers (advanced indexing)

        Returns:
            NDArray: A view of the selected elements

        Note:
            - Negative indices are supported and wrap around
            - Step values in slices must be positive
            - The number of indices must match the number of dimensions

        Raises:
            AssertionError: If the number of idxs does not match number of dimensions.

        Examples:
            >>> x = NDArray([[1, 2, 3], [4, 5, 6]])
            >>> x[0, 1]  # Integer indexing
            2.0
            >>> x[0:2, 1:]  # Slice indexing
            [[2. 3.]
             [5. 6.]]

            # TODO: Advanced indexing with a NDArray
            # >>> x[[0, 1], [1, 2]]  # Advanced indexing
            # [2. 6.]
        """

        # Indexing with an Array
        if isinstance(idxs, list | NDArray | np.ndarray):
            # Convert to NDArray if needed
            if not isinstance(idxs, NDArray):
                idxs = NDArray(idxs, device=self.device)  # type: ignore
            return self._handle_array_indexing(idxs)

        # Process indices and track which dimensions need squeezing
        slices, squeeze_dims = self._prepare_indices(idxs)
        if len(slices) != self.ndim:
            raise IndexError(
                f"""Need indexes equal to number of dimensions,
                trying to select {idxs} from {self.ndim} dimensions"""
            )

        new_shape, new_strides, new_offset = self._compute_view_shape(
            slices, squeeze_dims
        )

        return make(
            new_shape,
            strides=new_strides,
            device=self._device,
            handle=self._handle,
            offset=new_offset,
        )

    def __setitem__(
        self,
        idxs: IndexType,
        other: NDArrayLike,
    ) -> None:
        """
        Set the value of the array at the specified indices.

        For supported index types, see __getitem__.

        Args:
            idxs (IndexType): Indices to set the value at.:
            other (NDArrayLike): Value to set at the specified indices.

        Raises:
            ValueError: If the size of the view and other do not match.

        Examples:
            >>> x = NDArray([[1, 2], [3, 4]])
            >>> x[0, 1] = 5
            >>> x
            [[1. 5.]
             [3. 4.]]

            >>> x[0:2, 1:] = NDArray([[6], [7]])
            >>> x
            [[1. 6.]
             [3. 7.]]
        """
        view = self.__getitem__(idxs)
        if isinstance(other, np.ndarray):
            other = NDArray(other, device=self.device)

        if isinstance(other, NDArray):
            if view.size != other.size:
                raise ValueError(f"Size mismatch: {view.size} != {other.size}")
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        elif isinstance(other, float | int):
            self.device.scalar_setitem(
                view.size, other, view._handle, view.shape, view.strides, view._offset
            )

    def item(self, index: int | tuple[int, ...] | None = None) -> Scalar:
        """Copy an element of an array to a standard Python scalar and return it.

        Parameters:
            index: The index of the element to copy. Can be:
                None: Only works for arrays with one element (a.size == 1),
                int | tuple[int, ...]: Indices of the element to return.

        Returns:
            Scalar: A copy of the specified element of the array as a Python scalar

        Examples:
            >>> x = NDArray([1])
            >>> x.item()
            1.0

            >>> x = NDArray([[1, 2], [3, 4]])
            >>> x.item(0)
            1.0
            >>> x.item((1, 0))
            3.0
            >>> x.item((0, 1))
            2.0
            >>> x.item()  # Only works for arrays with one element
            Traceback (most recent call last):
            ...
            ValueError: Can only convert an array of size 1 to a Python scalar
        """
        if index is None:
            if self.size != 1:
                raise ValueError(
                    "Can only convert an array of size 1 to a Python scalar"
                )
            return self.device.scalar_item(self._handle, self._offset)

        if isinstance(index, int):
            if not 0 <= index < self.size:
                raise IndexError("Index out of bounds")
            compact = self.compact()
            return self.device.scalar_item(compact._handle, index)

        if isinstance(index, tuple):
            if len(index) != len(self.shape):
                raise IndexError(
                    f"Tuple index must have exactly {len(self.shape)} elements"
                )
            offset = sum(i * s for i, s in zip(index, self.strides))
            if not 0 <= offset < self.size:
                raise IndexError("Index out of bounds")
            compact = self.compact()
            return self.device.scalar_item(compact._handle, compact._offset + offset)

    # ====================  Element-wise and scalar functions

    def ewise_or_scalar(
        self,
        other: NDArrayLike,
        ewise_func: Callable[
            [NDArrayBackendProtocol, NDArrayBackendProtocol, NDArrayBackendProtocol],
            None,
        ],
        scalar_func: Callable[
            [NDArrayBackendProtocol, Scalar, NDArrayBackendProtocol], None
        ],
    ) -> NDArray:
        """
        Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar.

        Args:
            other (NDArrayLike): The other operand for the operation.
            ewise_func (Callable[[NDArray, NDArray, NDArray], None]):
                Element-wise function to apply.
            scalar_func (Callable[[NDArray, Scalar, NDArray], None]):
                Scalar function to apply.

        Returns:
            NDArray: The result of the operation.

        Raises:
            TypeError: If `other` is not an NDArrayLike object.
        """
        if isinstance(other, NDArray):
            if other.shape != self.shape:
                # Broadcast to the larger shape
                larger_shape = broadcast_shapes(self.shape, other.shape)
                other = other.broadcast_to(larger_shape)
                self = self.broadcast_to(larger_shape)

            out = make(self.shape, device=self.device)
            other = other.broadcast_to(self.shape)
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        elif isinstance(other, float | int):
            out = make(self.shape, device=self.device)
            scalar_func(self.compact()._handle, other, out._handle)
        elif isinstance(other, np.ndarray):
            return self.ewise_or_scalar(
                NDArray(other, device=self.device), ewise_func, scalar_func
            )
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return out

    def __add__(self, other: NDArray | Scalar) -> NDArray:
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other: NDArray | Scalar) -> NDArray:
        return self + (-other)

    def __rsub__(self, other: NDArray | Scalar) -> NDArray:
        return other + (-self)

    def __mul__(self, other: NDArray | Scalar) -> NDArray:
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other: NDArray | Scalar) -> NDArray:
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __rtruediv__(self, other: NDArray | Scalar) -> NDArray:
        if isinstance(other, int | float):
            out = make(self.shape, device=self.device)
            out.fill(other)
            return out / self
        return NDArray(other, device=self.device) / self

    def __rfloordiv__(self, other: NDArray | Scalar) -> NDArray:
        if isinstance(other, int | float):
            out = make(self.shape, device=self.device)
            out.fill(other)
            return out // self
        return NDArray(other, device=self.device) // self

    def __neg__(self) -> NDArray:
        return self * (-1)

    def __pow__(self, other: NDArray | Scalar) -> NDArray:
        out = make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            self.device.ewise_pow(
                self.compact()._handle, other.compact()._handle, out._handle
            )
        elif isinstance(other, int | float):
            self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other: NDArrayLike) -> NDArray:
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    # Element-wise functions

    def log(self) -> NDArray:
        out = make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self) -> NDArray:
        out = make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self) -> NDArray:
        out = make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    # ====================  Comparison operators

    # def __hash__(self) -> int:
    #     return hash(self._handle)

    def __eq__(self, other: NDArrayLike) -> NDArray:
        if isinstance(other, np.ndarray):
            return NDArray(np.equal(self.numpy(), other).astype("float32"))
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: NDArrayLike) -> NDArray:
        if isinstance(other, np.ndarray):
            return NDArray(np.greater_equal(self.numpy(), other).astype("float32"))
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: NDArrayLike) -> NDArray:
        return 1.0 - (self == other)

    def __gt__(self, other: NDArrayLike) -> NDArray:
        return (self >= other) * (1.0 - (self == other))

    def __lt__(self, other: NDArrayLike) -> NDArray:
        return 1.0 - (self >= other)

    def __le__(self, other: NDArrayLike) -> NDArray:
        return 1.0 - (self > other)

    # TODO: breaks array interop
    # def __bool__(self) -> bool:
    #     """Convert array to boolean value (True or False)."""
    #     if self.size == 1:
    #         return bool(self.item())
    #     raise ValueError(
    #         "Truth value of NDArray with more than one element is ambiguous"
    #     )

    # ====================  Matrix multiplication
    def _check_matrix_shapes(self, other: NDArray) -> None:
        if self.ndim < 2:
            raise ValueError(
                f"Matrix multiplication needs at least 2D arrays, got {self.shape}"
            )
        if other.ndim < 2:
            raise ValueError(
                f"Matrix multiplication needs at least 2D arrays, got {other.shape}"
            )
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(
                "Matrix multiplication needs compatible shapes, "
                f"got {self.shape} and {other.shape}"
            )

    def _batched_matmul(self, other: NDArray) -> NDArray:
        m, k1 = self.shape[-2:]
        k2, n = other.shape[-2:]

        a_batch_shape = self.shape[:-2] if self.ndim > 2 else (1,)
        b_batch_shape = other.shape[:-2] if other.ndim > 2 else (1,)

        batch_shape = broadcast_shapes(a_batch_shape, b_batch_shape)
        batch_size = math.prod(batch_shape)

        # broadcast shapes of axis
        a = self.broadcast_to((*batch_shape, m, k1)).compact()
        b = other.broadcast_to((*batch_shape, k2, n)).compact()

        a = a.reshape((-1, m, k1))
        b = b.reshape((-1, k2, n))
        if a.shape[0] != b.shape[0]:
            raise AssertionError(f"Batched matmul: {a.shape[0]} != {b.shape[0]}")

        # Create output
        out = make((batch_size, m, n), device=self.device)
        for i in range(batch_size):
            out[i] = a[i] @ b[i]

        # Restore batch dimensions
        return out.reshape((*batch_shape, m, n))

    def __matmul__(self, other: NDArray) -> NDArray:
        """
        Matrix multiplication of two arrays.

        Args:
            other: The second array to multiply with. Must be 2D or 3D.

        Returns:
            NDArray: The result of the matrix multiplication.

        Raises:
            ValueError: If the shapes are not compatible for matrix multiplication.
            AssertionError: If the batch sizes of the arrays do not match.

        Note:
            - For N-D arrays, the last two dimensions are treated as matrices.
            - The batch size is determined by the leading dimensions of the arrays.

        Raises:

        Examples:
            >>> a = NDArray([[1, 2], [3, 4]])
            >>> b = NDArray([[5, 6], [7, 8]])
            >>> c = a @ b
            >>> c.shape
            (2, 2)
            >>> c
            [[19. 22.]
             [43. 50.]]

            >>> a = NDArray(np.random.rand(2, 2, 3, 4))
            >>> b = NDArray(np.random.rand(2, 2, 4, 5))
            >>> c = a @ b
            >>> c.shape
            (2, 2, 3, 5)
        """
        self._check_matrix_shapes(other)

        if self.ndim > 2 or other.ndim > 2:
            return self._batched_matmul(other)

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # For smaller matrices, the overhead of tiling and reshaping is too large
        matrix_is_large = m * n * p > 64**3

        if (
            matrix_is_large
            and hasattr(self.device, "matmul_tiled")
            and all(d % self.device.__tile_size__ == 0 for d in (m, n, p))
        ):
            return self.device._tiled_matmul(self, other, m, n, p)

        out = make((m, p), device=self.device)
        self.device.matmul(
            self.compact()._handle, other.compact()._handle, out._handle, m, n, p
        )
        return out

    # ====================  Reductions over all element or over given axis

    def reduce_view_out(
        self, axis: Axis | None, *, keepdims: bool = False
    ) -> tuple[NDArray, NDArray]:
        """
        Return a view to the array set up for reduction functions and output array.

        Args:
            axis: Axes to reduce over. None to reduce all axes, or int/tuple of axes.
            keepdims: If true, reduced axes are kept with size 1

        Returns:
            tuple(NDArray, NDArray): A tuple of two NDArray objects:
                - The first is a view of the array set up for reduction.
                - The second is an output array for the result of the reduction.

        Raises:
            ValueError: If axis is empty or invalid.

        Examples:
            >>> x = NDArray([[1, 2], [3, 4]])
            >>> view, out = x.reduce_view_out(axis=0)
            >>> view.shape
            (2, 2)
            >>> out.shape
            (2,)
        """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (self.size,))
            out_shape = (1,) * self.ndim if keepdims else (1,)
            out = make(out_shape, device=self.device)
            return view, out

        if isinstance(axis, int):
            axis = (axis,)

        # handle negative axes - probably not needed
        axis = tuple(sorted([ax if ax >= 0 else self.ndim + ax for ax in axis]))

        # move reduction axes to the end
        other_axes = tuple(i for i in range(self.ndim) if i not in axis)
        view = self.permute(other_axes + axis)

        if keepdims:
            new_shape = tuple(1 if i in axis else s for i, s in enumerate(self.shape))
        else:
            new_shape = tuple(s for i, s in enumerate(self.shape) if i not in axis)

        out = make(new_shape, device=self.device)

        # reshape reduction axes to a single axis
        reduce_size = math.prod(self.shape[i] for i in axis)
        view_shape = (*view.shape[: -len(axis)], reduce_size)
        view = view.compact().reshape(view_shape)

        return view, out

    def sum(self, axis: Axis | None = None, *, keepdims: bool = False) -> NDArray:
        """
        Sum the elements of the array along the specified axis.

        Args:
            axis (Axis or None): Axis or axes to sum over. If None, sum all elements.
            keepdims (bool): If true, keep reduced dimensions with size 1.

        Returns:
            NDArray: The sum of the elements along the specified axis.

        Raises:
            ValueError: If axis is empty or invalid.

        Examples:
            >>> x = NDArray([[1, 2], [3, 4]])
            >>> x.sum(axis=0)
            [4. 6.]
            >>> x.sum(axis=1)
            [3. 7.]
            >>> x.sum()
            [10.]
            >>> x.sum(axis=(0, 1))
            10.0
            >>> x.sum(keepdims=True)
            [[10.]]
            >>> x.sum(axis=0, keepdims=True)
            [[4. 6.]]
        """  # noqa: DOC502
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis: Axis | None = None, *, keepdims: bool = False) -> NDArray:
        """
        Find the maximum value in the array along the specified axis.

        Args:
            axis (Axis or None):
                Axes to find the maximum over.
                If None, find the maximum of all elements.
            keepdims (bool): If true, keep reduced dimensions with size 1.

        Returns:
            NDArray: The maximum value along the specified axis.

        Raises:
            ValueError: If axis is empty or invalid.

        Examples:
            >>> x = NDArray([[1, 2], [3, 4]])
            >>> x.max(axis=0)
            [3. 4.]
            >>> x.max(axis=1)
            [2. 4.]
            >>> x.max()
            [4.]
            >>> x.max(axis=(0, 1))
            4.0
            >>> x.max(keepdims=True)
            [[4.]]
            >>> x.max(axis=0, keepdims=True)
            [[3. 4.]]
        """  # noqa: DOC502
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out
