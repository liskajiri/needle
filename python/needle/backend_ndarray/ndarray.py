from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from needle.typing import AbstractBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from needle.typing import DType, IndexType, Scalar, Shape, Strides, np_ndarray

logger = logging.getLogger(__name__)

# TODO: reference hw3.ipynb for future optimizations
# TODO: investigate usage of __slots__, Python's array.array for NDArray class

if True:

    class BackendDevice(AbstractBackend):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton

        # TODO: move to c++ backend
        def randn(self, shape: Shape, dtype: DType = "float32") -> NDArray:
            # import random

            # random_values = [random.random() for _ in range(math.prod(shape))]

            # arr = make(shape, device=self)
            # for i, value in enumerate(random_values):
            #     arr[i] = value

            # return arr
            if isinstance(shape, int):
                shape = (shape,)
            return NDArray(np.random.randn(*shape).astype(dtype), device=self)

        def rand(self, shape: Shape, dtype: DType = "float32") -> NDArray:
            # random_values = [random.uniform(0, 1) for _ in range(math.prod(shape))]

            # arr = make(shape, device=self)
            # for i, value in enumerate(random_values):
            #     arr._handle[i] = value

            # return arr
            if isinstance(shape, int):
                shape = (shape,)
            return NDArray(np.random.rand(*shape).astype(dtype), device=self)

        def one_hot(self, n: int, i: IndexType, dtype: DType) -> NDArray:
            """Create a one-hot vector.

            Args:
                n (int): Length of the vector.
                i (int): Index of the one-hot element.
                dtype (_type_, optional):

            Raises:
                NotImplementedError: If the method is not implemented.

            Returns:
                NDArray: A one-hot vector.
            """
            return NDArray(np.eye(n, dtype=dtype)[i], device=self)

        def zeros(self, shape: Shape, dtype: DType) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(0.0)
            return arr

        def ones(self, shape: Shape, dtype: DType) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(1.0)
            return arr

        def constant(self, shape: Shape, value: Scalar, dtype: DType) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(value)
            return arr

        def empty(self, shape: Shape, dtype: DType = "float32") -> NDArray:
            return make(shape, device=self)

        def full(
            self, shape: Shape, fill_value: Scalar, dtype: DType = "float32"
        ) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(fill_value)
            return arr

else:
    import random

    class BackendDevice(AbstractBackend):
        # TODO: dtype?
        def randn(self, shape: Shape, dtype: DType = "float32") -> NDArray:
            """
            Generate random array from standard normal distribution
            """
            random.seed(0)

            size = (math.prod(shape),)
            arr = self.empty(size, dtype=dtype)
            for i in range(arr.size):
                arr[i] = random.gauss(0.0, 1.0)
            return arr.reshape(shape)

        def rand(self, shape: Shape, dtype: DType = "float32") -> NDArray:
            """
            Generate random samples from uniform distribution [0,1).
            """
            random.seed(0)

            size = (math.prod(shape),)
            arr = self.empty(size, dtype=dtype)
            for i in range(arr.size):
                arr[i] = random.uniform(0.0, 1.0)
            return arr.reshape(shape)

        def one_hot(self, n: int, i: IndexType, dtype: DType) -> NDArray:
            """Create a one-hot vector.

            Args:
                n (int): Length of the vector.
                i (int): Index of the one-hot element.
                dtype (DType): Data type of the array.

            Returns:
                NDArray: A one-hot vector.
            """
            arr = self.zeros((n,), dtype)
            arr[i] = 1.0
            return arr

        def zeros(self, shape: Shape, dtype: DType) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(0.0)
            return arr

        def ones(self, shape: Shape, dtype: DType) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(1.0)
            return arr

        def constant(self, shape: Shape, value: Scalar, dtype: DType) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(value)
            return arr

        def empty(self, shape: Shape, dtype: DType = "float32") -> NDArray:
            return make(shape, device=self)

        def full(
            self, shape: Shape, fill_value: Scalar, dtype: DType = "float32"
        ) -> NDArray:
            arr = self.empty(shape, dtype=dtype)
            arr.fill(fill_value)
            return arr


def cuda() -> AbstractBackend:
    """Return cuda device."""
    try:
        from needle.backend_ndarray import ndarray_backend_cuda  # type: ignore

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda")


def cpu_numpy() -> AbstractBackend:
    """Return numpy device."""
    try:
        from needle import backend_numpy
        from needle.backend_numpy import NumpyBackend

        return NumpyBackend("cpu_numpy", backend_numpy)
    except ImportError:
        raise ImportError("Numpy backend not available")


def cpu() -> AbstractBackend:
    """Return cpu device."""
    try:
        from needle.backend_ndarray import ndarray_backend_cpu  # type: ignore

        return BackendDevice("cpu", ndarray_backend_cpu)
    except ImportError:
        raise ImportError("CPU backend not available")


def all_devices() -> list[AbstractBackend]:
    """Return a list of all available devices."""
    return [cpu(), cuda(), cpu_numpy()]


default_device = cpu()


def make(
    shape: tuple,
    strides: tuple | None = None,
    device: AbstractBackend = default_device,
    handle: NDArray | None = None,
    offset: int = 0,
) -> NDArray:
    """
    Create a new NDArray with the given properties
    This will allocate the memory if handle=None,
    otherwise it will use the handle of an existing array.
    """

    def prod(shape: tuple) -> int:
        # TODO: flatten?
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
    if handle is None:
        array_size = prod(shape)
        if array_size <= 0:
            raise ValueError(f"Array size cannot be negative, Invalid shape: {shape}")
        array._handle = array.device.Array(array_size)
    else:
        array._handle = handle
    return array


def broadcast_shapes(*shapes: tuple[Shape, ...]) -> tuple:
    """Return broadcasted shape for multiple input shapes.

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
        ValueError: If shapes cannot be broadcast together


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
        ValueError: Incompatible shapes for broadcasting: ((2, 3), (2, 4))
    """
    # If only one shape provided, return it
    if len(shapes) == 1:
        return shapes[0]

    from builtins import max as pymax

    max_dims = pymax([len(shape) for shape in shapes])
    # Left-pad shorter shapes with 1s to align dimensions
    aligned_shapes = [(1,) * (max_dims - len(s)) + s for s in shapes]

    # Determine output dimension for each position
    result = []
    for dims in zip(*aligned_shapes, strict=False):
        max_dim = pymax(dims)
        for d in dims:
            if d != 1 and d != max_dim:
                raise ValueError(f"Incompatible shapes for broadcasting: {shapes}")
        result.append(max_dim)

    return tuple(result)


class NDArray:
    """A generic ND array class that may contain multiple different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.
    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(
        self,
        other: NDArray | np_ndarray | list,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        # TODO: allow creating by itself and from list[]
        """Create by copying another NDArray, or from numpy."""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device != other.device:
                logger.error(
                    f"Creating NDArray with different device {device} != {other.device}"
                )
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            array = make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: NDArray) -> None:
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape: Shape) -> Strides:
        """Utility function to compute compact strides."""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

        # Properties and string representations

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
        """Return number of dimensions."""
        return len(self._shape)

    @cached_property
    def size(self) -> int:
        return int(math.prod(self._shape))

    def __repr__(self) -> str:
        return f"NDArray( shape: {self.shape} | device={self.device}"

    def __str__(self) -> str:
        # return f"NDArray( shape: {self.shape} | device={self.device}"
        return self.numpy().__str__()

    def __len__(self) -> int:
        """Returns the size of the first dimension.

        Returns:
            int: The size of the first dimension.
        """
        if len(self.shape) == 0:
            return 1
        return self.shape[0]

    # Basic array manipulation
    def fill(self, value: float) -> None:
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device: AbstractBackend) -> NDArray:
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        return NDArray(self.numpy(), device=device)

    def numpy(self) -> np_ndarray:
        """Convert to a numpy array."""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    @staticmethod
    def from_numpy(
        a: np_ndarray,
    ) -> NDArray:
        """Copy from a numpy array."""
        array = make(a.shape)
        array.device.from_numpy(np.ascontiguousarray(a), array._handle)
        return array

    def __array__(
        self,
        dtype: DType | None = None,
        copy: bool = False,
    ):
        """Allow implicit conversion to numpy array"""
        return self.numpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy ufuncs by converting to numpy first"""
        arrays = []
        for input in inputs:
            if isinstance(input, NDArray):
                arrays.append(input.numpy())
            else:
                arrays.append(input)
        return getattr(ufunc, method)(*arrays, **kwargs)

    # Shapes and strides

    def is_compact(self) -> bool:
        """
        Return true if array is compact in memory and
        internal size equals math.product of the shape dimensions.
        """
        return (
            self._strides == self.compact_strides(self._shape)
            and self.size == self._handle.size
        )

    def compact(self) -> NDArray:
        """Convert a matrix to be compact."""
        if self.is_compact():
            return self
        out = make(self.shape, device=self.device)
        self.device.compact(
            self._handle, out._handle, self.shape, self.strides, self._offset
        )
        return out

    def as_strided(self, shape: Shape, strides: Strides) -> NDArray:
        """Re-stride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return make(
            shape,
            strides=strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset,
        )

    def astype(self, dtype: DType = "float32") -> NDArray:
        """Return a copy of the array with a different type."""
        if dtype != "float32":
            logger.warning(f"Only support float32 for now, got {dtype}")
            a = np.array(self, dtype="float32")
            dtype = a.dtype
            logger.warning(f"Converting to numpy array with dtype {dtype}")
        assert dtype == "float32", f"Only support float32 for now, got {dtype}"
        return self + 0.0

    def flatten(self) -> NDArray:
        """Return a 1D view of the array."""
        return self.reshape((self.size,))

    def reshape(self, new_shape: Shape) -> NDArray:
        """Reshape the matrix without copying memory.

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
        """
        if self.shape == new_shape:
            return self

        # Special case: Convert 1D array to column vector
        if self._is_1d_to_column_vector(new_shape):
            return self._make_column_vector(new_shape)

        if not self.is_compact():
            raise ValueError(
                f"""Cannot reshape non-compact array of shape {self.shape}.
                Call compact() first."""
            )

        # Handle automatic dimension calculation with -1
        new_shape = self._resolve_negative_dimensions(new_shape)

        # Verify total size remains constant
        if self.size != math.prod(new_shape):
            raise ValueError(
                f"Cannot reshape array of size {self.size} into shape {new_shape}."
                f"Total elements must remain constant."
            )

        # Create reshaped view with new strides
        return self.as_strided(new_shape, self.compact_strides(new_shape))

    def _is_1d_to_column_vector(self, new_shape: Shape) -> bool:
        """Check if reshaping from 1D array to column vector."""
        return (
            len(self.shape) == 1
            and len(new_shape) == 2
            and self.shape[0] == new_shape[0]
            and new_shape[1] == 1
        )

    def _make_column_vector(self, new_shape: Shape) -> NDArray:
        """Convert 1D array to column vector with optimized strides."""
        return make(
            new_shape,
            device=self.device,
            handle=self._handle,
            strides=(self.strides[0], 0),
        )

    def _resolve_negative_dimensions(self, new_shape: Shape) -> Shape:
        """Calculate actual shape when -1 dimension is used.

        Args:
            new_shape: Proposed shape with possible -1 dimension

        Returns:
            Shape: Resolved shape with -1 replaced by calculated dimension

        Raises:
            ValueError: If multiple -1 dimensions or size mismatch
        """
        if -1 not in new_shape:
            return new_shape

        # Verify only one -1 dimension
        if new_shape.count(-1) > 1:
            raise ValueError(
                f"Cannot deduce shape with multiple -1 dimensions in {new_shape}"
            )

        # Calculate missing dimension
        neg_idx = new_shape.index(-1)
        other_dims_product = math.prod(new_shape[:neg_idx] + new_shape[neg_idx + 1 :])

        if self.size % other_dims_product != 0:
            raise ValueError(
                f"Cannot reshape array of size {self.size} into shape {new_shape}. "
                f"Size must be divisible by {other_dims_product}."
            )

        missing_dim = self.size // other_dims_product
        return new_shape[:neg_idx] + (missing_dim,) + new_shape[neg_idx + 1 :]

    def permute(self, new_axes: tuple) -> NDArray:
        """Permute order of the dimensions.  new_axes describes a permutation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.

        Args:
            new_axes (tuple): permutation order of the dimensions
        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).

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
        """Broadcast an array to a new shape.
        New_shape's elements must be the same as the original shape,
        except for dimensions in the self where the size = 1
        (which can then be broadcast to any size). As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Args:
            new_shape (tuple): shape to broadcast to
        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        Raises:
            ValueError: If shapes are incompatible for broadcasting
            AssertionError if new_shape[i] != shape[i] for all i where
            shape[i] != 1
        """
        if self.shape == new_shape:
            return self

        new_size = math.prod(new_shape)
        if len(new_shape) < len(self._shape) or new_size % self.size != 0:
            raise ValueError(f"Cannot broadcast shape {self._shape} to {new_shape}")

        leading_dims = len(new_shape) - len(self._shape)
        broadcast_strides = (0,) * leading_dims + self._strides

        # Zero out strides for dimensions with size 1
        new_strides = list(broadcast_strides)
        for i, dim_size in enumerate(reversed(self._shape)):
            if dim_size == 1:
                new_strides[-(i + 1)] = 0

        return self.as_strided(new_shape, tuple(new_strides))

    # === Get and set elements

    def process_slice(self, sl: slice, dim: int) -> slice:
        """Convert a slice to an explicit start/stop/step."""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim]

        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop

        if step is None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    # TODO: standardize idxs type: int | iterable[int] | slice
    # TODO: getitem is not really working like numpy's
    def __getitem__(self, idxs: IndexType) -> NDArray:  # noqa: C901
        """
        The __getitem__ operator in Python allows access to elements of the
        array.
        When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).
        Slices have three elements: (.start .stop .step), which can be None
        or have negative entries.
        For this tuple of slices, return an array that subsets the desired
        elements.
        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple[slice], a tuple of slice elements
            corresponding to the subset of the matrix to get
        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.
        """

        def _prepare_indices(
            idxs: int | tuple[int | slice, ...],
        ) -> tuple[tuple[slice, ...], set[int]]:
            """
            Convert input indices to tuple of slices and track squeeze dimensions.
            """
            # Convert single integer index to tuple
            orig_idxs = (idxs,) if not isinstance(idxs, tuple) else idxs
            if not isinstance(orig_idxs, tuple):
                raise TypeError(f"Invalid index type: {type(orig_idxs)}")

            # Track dimensions to squeeze (from integer indexing)
            squeeze_dims = {
                i for i, idx in enumerate(orig_idxs) if isinstance(idx, int)
            }

            # Convert each index to slice
            processed_idxs = []
            for i, idx in enumerate(orig_idxs):
                if isinstance(idx, slice):
                    processed_idxs.append(self.process_slice(idx, i))
                elif isinstance(idx, int):
                    processed_idxs.append(slice(idx, idx + 1, 1))
                else:
                    processed_idxs.append(idx)

            # Pad with full slices
            final_idxs = tuple(
                processed_idxs + [slice(None)] * (self.ndim - len(processed_idxs))
            )

            return final_idxs, squeeze_dims

        def _handle_array_indexing(idxs: NDArray) -> NDArray:
            """
            Handle indexing with a list or NDArray
            """
            out_shape = idxs.shape + self.shape[1:]
            out = make(out_shape, device=self.device)

            # Copy selected elements
            for i, idx in enumerate(idxs.numpy().flatten()):
                src_idx = (int(idx),) + (slice(None),) * (self.ndim - 1)
                dst_idx = (i,) + (slice(None),) * (self.ndim - 1)
                out[dst_idx] = self[src_idx]
            return out.compact()

        def _compute_view_shape(
            slices: tuple[slice, ...], squeeze_dims: set[int]
        ) -> tuple[Shape, Strides, int]:
            """
            Compute the shape and strides of the view resulting from slicing.
            Removes dimensions that should be squeezed due to integer indexing.
            """
            new_shape = []
            new_strides = list(self._strides)
            new_offset = self._offset

            for i, ax in enumerate(slices):
                start = ax.start if ax.start is not None else 0
                stop = ax.stop if ax.stop is not None else self.shape[i]
                step = ax.step if ax.step is not None else 1
                if i not in squeeze_dims:
                    n_shape = (stop - start - 1) // step + 1
                    new_shape.append(n_shape)
                new_strides[i] *= step
                new_offset += self._strides[i] * start

            # Remove strides for squeezed dimensions
            final_strides = tuple(
                stride for i, stride in enumerate(new_strides) if i not in squeeze_dims
            )

            return tuple(new_shape), final_strides, new_offset

        # Indexing with a list or NDArray
        if isinstance(idxs, list | np.ndarray | NDArray):
            # Convert to NDArray if needed
            if not isinstance(idxs, NDArray):
                idxs = NDArray(idxs, device=self.device)
            return _handle_array_indexing(idxs)

        # Process indices and track which dimensions need squeezing
        slices, squeeze_dims = _prepare_indices(idxs)
        if len(slices) != self.ndim:
            raise AssertionError(
                f"""Need indexes equal to number of dimensions,
                trying to select {idxs} from {self.ndim} dimensions"""
            )

        new_shape, new_strides, new_offset = _compute_view_shape(slices, squeeze_dims)

        # TODO: This array is smaller and so its size is smaller
        # That is not however changed, so the check is_compact fails, because
        # the sub array still thinks it is the larger array
        # This leads to needed unnecessary calls to compact
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
        other: NDArray | np.ndarray | Scalar,
    ) -> None:
        """
        Set the values of a view into an array,
        using the same semantics as __getitem__().
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
        else:
            self.device.scalar_setitem(
                view.size, other, view._handle, view.shape, view.strides, view._offset
            )

    # Collection of element-wise and scalar function: add, multiply, boolean, etc
    # TODO: probably must implement in the backend
    def item(self) -> Scalar:
        raise NotImplementedError("item() not implemented")

    def ewise_or_scalar(
        self, other: NDArray | Scalar, ewise_func: Callable, scalar_func: Callable
    ) -> NDArray:
        """
        Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar.
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
            raise ValueError(f"Unsupported type {type(other)}")
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __rtruediv__(self, other) -> NDArray:
        if isinstance(other, int | float):
            out = make(self.shape, device=self.device)
            out.fill(other)
            return out / self
        return NDArray(other, device=self.device) / self

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other: Scalar) -> NDArray:
        out = make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    # Binary operators all return (0.0, 1.0) floating point values
    # TODO: could of course be optimized

    def __eq__(self, other: object) -> bool:
        """Support == comparison with numpy arrays and scalars"""
        if isinstance(other, np.ndarray):
            return np.array_equal(self.numpy(), other)
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: NDArray | Scalar) -> bool:
        if isinstance(other, np.ndarray):
            np.greater_equal(self.numpy(), other)
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: NDArray) -> bool:
        return 1 - (self == other)

    def __gt__(self, other: NDArray) -> bool:
        return (self >= other) * (self != other)

    def __lt__(self, other: NDArray) -> bool:
        return 1 - (self >= other)

    def __le__(self, other: NDArray | Scalar) -> NDArray:
        return 1 - (self > other)

    # Element-wise functions

    # TODO: auto derive inplace functions
    def log(self) -> NDArray:
        out = make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    # Matrix multiplication
    def __matmul__(self, other: NDArray) -> NDArray:  # noqa: C901
        """
        Matrix multiplication of two arrays.
        This requires that both arrays be 2D
        (i.e., we don't handle batch matrix multiplication),
        and that the sizes match up properly for matrix multiplication.
        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.
        In this case, the code below will re-stride and compact the matrix
        into tiled form, and then pass to the relevant CPU backend.
        For the CPU version we will just fall back to the naive CPU implementation
        if the array shape is not a multiple of the tile size.
        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        def _check_matrix_shapes() -> None:
            if self.ndim < 2:
                raise ValueError(
                    f"Matrix multiplication needs at least 2D arrays, got {self.shape}"
                )
            if other.ndim < 2:
                raise ValueError(
                    f"Matrix multiplication needs at least 2D arrays, got {other.shape}"
                )
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(f"""
                    Matrix multiplication requires inner dimensions to match,
                    but A.cols != B.rows: {self.shape[-1]} != {other.shape[-2]}
                    for shapes {self.shape} and {other.shape}
                    """)

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

        def _tiled_matmul(self, other: NDArray) -> NDArray:
            def _tile(a: NDArray, tile: int) -> NDArray:
                """
                Transforms a matrix [k, n] into a
                matrix [k // tile, n // tile, tile, tile].
                """
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                ).compact()

            t = self.device.__tile_size__
            a = _tile(self.compact(), t)
            b = _tile(other.compact(), t)
            out = make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        # Main matmul function

        _check_matrix_shapes()

        if self.ndim > 2 or other.ndim > 2:
            return _batched_matmul(self, other)

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # For smaller matrices, the overhead of tiling and reshaping is too large
        # TODO: More scientific study of this
        matrix_is_large = m * n * p > 64**3

        if (
            matrix_is_large
            and hasattr(self.device, "matmul_tiled")
            and all(d % self.device.__tile_size__ == 0 for d in (m, n, p))
        ):
            return _tiled_matmul(self, other)

        out = make((m, p), device=self.device)
        self.device.matmul(
            self.compact()._handle, other.compact()._handle, out._handle, m, n, p
        )
        return out

    # Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(
        self, axis: tuple | None, keepdims: bool = False
    ) -> tuple[NDArray, NDArray]:
        """
        Return a view to the array set up for reduction functions and output array.

        Args:
            axis: Axes to reduce over. Either None to reduce all axes, or tuple of axes.
            keepdims: If true, reduced axes are kept with size 1

        Returns:
            tuple of (view, out) where:
            - view is arranged for reduction
            - out is the output array
        """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (self.size,))
            out = make((1,), device=self.device)
            return view, out

        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, list):
            axis = tuple(axis)

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
        view_shape = view.shape[: -len(axis)] + (reduce_size,)
        view = view.compact().reshape(view_shape)

        return view, out

    def sum(self, axis: tuple | None = None, keepdims: bool = False) -> NDArray:
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis: tuple | None = None, keepdims: bool = False) -> NDArray:
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out
