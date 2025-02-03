from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from needle.backend_ndarray.device import AbstractBackend

if TYPE_CHECKING:
    from needle.typing.types import DType, Scalar, Shape, Strides

logger = logging.getLogger(__name__)

# TODO: reference hw3.ipynb for future optimizations
# TODO: investigate usage of __slots__, Python's array.array for NDArray class


class BackendDevice(AbstractBackend):
    # note: numpy doesn't support types within standard random routines, and
    # .astype("float32") does work if we're generating a singleton

    # TODO: move to c++ backend
    def randn(self, *shape: Shape, dtype: DType = "float32") -> NDArray:
        # random_values = [random.gauss(0, 1) for _ in range(math.prod(shape))]

        # arr = NDArray.make(shape, device=self)
        # for i, value in enumerate(random_values):
        #     arr[i] = value

        # return arr

        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape: Shape, dtype: DType = "float32") -> NDArray:
        # random_values = [random.uniform(0, 1) for _ in range(math.prod(shape))]

        # arr = NDArray.make(shape, device=self)
        # for i, value in enumerate(random_values):
        #     arr._handle[i] = value

        # return arr
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n: int, i: int, dtype: DType) -> NDArray:
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
        # TODO: why are we constructing a NxN matrix to select a vector?
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def zeros(self, shape: Shape, dtype: DType) -> NDArray:
        arr = self.empty(shape, dtype=dtype)
        arr.fill(0.0)
        return arr

    def ones(self, shape: Shape, dtype: DType) -> NDArray:
        arr = self.empty(shape, dtype=dtype)
        arr.fill(1.0)
        return arr

    def empty(self, shape: Shape, dtype: DType = "float32") -> NDArray:
        return NDArray.make(shape, device=self)

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
        from needle.backend_ndarray import BackendDevice as NumpyBackend

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


default_device = cpu()


def all_devices() -> list[AbstractBackend]:
    """Return a list of all available devices."""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multiple different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.
    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(
        self,
        other: NDArray | np.ndarray | list,
        device: AbstractBackend = default_device,
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
            # device = device if device is not None else default_device
            # device = default_device
            array = self.make(other.shape, device=device)
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

    @staticmethod
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
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properties and string representations
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
        return self.shape[0]

    ### Basic array manipulation
    def fill(self, value: float) -> None:
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device: AbstractBackend) -> NDArray:
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        return NDArray(self.numpy(), device=device)

    def numpy(self) -> np.ndarray:
        """Convert to a numpy array."""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    @staticmethod
    def from_numpy(
        a: np.ndarray,
    ) -> NDArray:
        """Copy from a numpy array."""
        array = NDArray.make(a.shape)
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

    ### Shapes and strides

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
        out = NDArray.make(self.shape, device=self.device)
        self.device.compact(
            self._handle, out._handle, self.shape, self.strides, self._offset
        )
        return out

    def as_strided(self, shape: Shape, strides: Strides) -> NDArray:
        """Re-stride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
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
        new_strides = self.compact_strides(new_shape)
        return self.as_strided(new_shape, new_strides)

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
        return NDArray.make(
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
        new_shape = new_shape[:neg_idx] + (missing_dim,) + new_shape[neg_idx + 1 :]
        return new_shape

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

    ### Get and set elements

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
    def __getitem__(self, idxs: int | tuple[int] | tuple[slice] | NDArray) -> NDArray:
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

        def _convert_tuple_to_slice(idxs: int | tuple[int]) -> tuple[slice, ...]:
            """
            Convert input idxs from tuple to tuple of slices in each dimension
            """
            if isinstance(idxs, int):
                idxs = (idxs,)
            new_idxs = idxs + (slice(None),) * (self.ndim - len(idxs))

            idx = tuple(
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(new_idxs)
            )
            return idx

        def _handle_array_indexing(idxs: NDArray) -> NDArray:
            """
            Handle indexing with a list or NDArray
            """
            out_shape = idxs.shape + self.shape[1:]
            out = NDArray.make(out_shape, device=self.device)

            # Copy selected elements
            for i, idx in enumerate(idxs.numpy().flatten()):
                src_idx = (int(idx),) + (slice(None),) * (self.ndim - 1)
                dst_idx = (i,) + (slice(None),) * (self.ndim - 1)
                out[dst_idx] = self[src_idx]
            return out.compact()

        def _compute_view_shape() -> tuple[Shape, Strides, int]:
            """
            Compute the shape and strides of the view resulting from slicing.
            """
            new_shape = []
            new_strides = list(self._strides)
            new_offset = self._offset
            for i, ax in enumerate(slices):
                # stop - start guaranteed to be positive
                # slice(0, 3, 2) = [0, 2], but when using (3 - 0) // 2 = 1
                new_shape.append((ax.stop - ax.start - 1) // ax.step + 1)
                new_strides[i] *= ax.step
                # distance to the first element
                new_offset += self._strides[i] * ax.start
            return tuple(new_shape), tuple(new_strides), new_offset

        # Indexing with a list or NDArray
        if isinstance(idxs, list | np.ndarray | NDArray):
            # Convert to NDArray if needed
            if not isinstance(idxs, NDArray):
                idxs = NDArray(idxs, device=self.device)
            return _handle_array_indexing(idxs)

        slices = _convert_tuple_to_slice(idxs)
        if len(slices) != self.ndim:
            raise AssertionError(
                f"""Need indexes equal to number of dimensions,
                trying to select {idxs} from {self.ndim} dimensions"""
            )

        new_shape, new_strides, new_offset = _compute_view_shape()

        # TODO: This array is smaller and so its size is smaller
        # That is not however changed, so the check is_compact fails, because
        # the sub array still thinks it is the larger array
        # This leads to needed unnecessary calls to compact
        return NDArray.make(
            new_shape,
            strides=new_strides,
            device=self._device,
            handle=self._handle,
            offset=new_offset,
        )

    def __setitem__(
        self,
        idxs: int | tuple[int] | tuple[slice] | NDArray,
        other: NDArray | Scalar,
    ) -> None:
        """
        Set the values of a view into an array,
        using the same semantics as __getitem__().
        """
        view = self.__getitem__(idxs)
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

    ### Collection of element-wise and scalar function: add, multiply, boolean, etc
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

            out = NDArray.make(self.shape, device=self.device)
            other = other.broadcast_to(self.shape)
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        elif isinstance(other, float | int):
            out = NDArray.make(self.shape, device=self.device)
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

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other: Scalar) -> NDArray:
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    # Binary operators all return (0.0, 1.0) floating point values
    # TODO: could of course be optimized

    def __eq__(self, other) -> bool:
        """Support == comparison with numpy arrays and scalars"""
        if isinstance(other, np.ndarray):
            return np.array_equal(self.numpy(), other)
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: NDArray) -> bool:
        if isinstance(other, np.ndarray):
            np.greater_equal(self.numpy(), other)
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: NDArray) -> bool:
        return 1 - (self == other)

    def __gt__(self, other: NDArray) -> bool:
        return (self >= other) * (self != other)

    def __lt__(self, other: NDArray) -> bool:
        return 1 - (self >= other)

    def __le__(self, other: NDArray) -> bool:
        return 1 - (self > other)

    ### Element-wise functions

    # TODO: auto derive inplace functions
    def log(self) -> NDArray:
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other: NDArray) -> NDArray:
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
        # TODO: tests for batched matmul
        # TODO: efficiency : probably crazy inefficient
        if self.ndim > 2 or other.ndim > 2:
            # Broadcast batch dimensions
            batch_shape = tuple(broadcast_shapes(self.shape[:-2], other.shape[:-2]))
            m, n = self.shape[-2], other.shape[-1]
            assert self.shape[-1] == other.shape[-2], f"""
            Inner batch-matmul dimensions must match,
            got {self.shape[-1]} != {other.shape[-2]}
            """
            k = self.shape[-1]

            # Reshape to 3D
            a_new_shape = (*batch_shape, m, k)
            b_new_shape = (*batch_shape, k, n)
            a = self.broadcast_to(a_new_shape).compact()
            b = other.broadcast_to(b_new_shape).compact()

            # Flatten batch dims
            batch_size = math.prod(batch_shape)
            a = a.reshape((batch_size, m, k))
            b = b.reshape((batch_size, k, n))

            # Create output
            out = NDArray.make((batch_size, m, n), device=self.device)
            for i in range(batch_size):
                a_i = a[i].compact().reshape((m, k))
                b_i = b[i].compact().reshape((k, n))
                out[i] = a_i @ b_i

            # Restore batch dimensions
            return out.reshape((*batch_shape, m, n))

        assert self.ndim == 2, (
            f"Matrix multiplication requires 2D arrays, got {self.ndim} in {self.shape}"
        )
        assert other.ndim == 2, f"""Matrix multiplication requires 2D arrays,
            got {other.ndim} in {other.shape}"""
        assert (
            self.shape[1] == other.shape[0]
        ), f"""Matrix multiplication requires inner dimensions to match,
        but A.cols != B.rows: {self.shape[1]} != {other.shape[0]}"""

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # For smaller matrices, the overhead of tiling and reshaping is too large
        matrix_is_large = m * n * p > 64**3

        if (
            matrix_is_large
            and hasattr(self.device, "matmul_tiled")
            and all(d % self.device.__tile_size__ == 0 for d in (m, n, p))
        ):

            def tile(a: NDArray, tile: int) -> NDArray:
                """
                Transforms a matrix [k, n] into a
                matrix [k // tile, n // tile, tile, tile].
                """
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            out = (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )
        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
        return out

    ### Reductions, i.e., sum/max over all element or over given axis
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
            out = NDArray.make((1,), device=self.device)
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

        out = NDArray.make(new_shape, device=self.device)

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

    ### Other functions

    def flip(self, axes: tuple[int] | int) -> NDArray:
        """
        Flip this ndarray along the specified axes.
        Note: compacts the array before returning.

        Args:
            axes: Tuple or int specifying the axes to flip

        Returns:
            NDArray: New array with flipped axes
        """
        # Handle single axis case
        if isinstance(axes, int):
            axes = (axes,)

        # Validate axes
        for ax in axes:
            if ax < -self.ndim or ax >= self.ndim:
                raise ValueError(
                    f"Axis {ax} is out of bounds for array of dimension {self.ndim}"
                )

        # Normalize negative axes
        # TODO: this will be common in array ops, make it a function
        axes = tuple(ax if ax >= 0 else self.ndim + ax for ax in axes)

        # Create new view with modified strides and offset
        new_strides = list(self._strides)
        offset = self._offset

        # For each axis to flip:
        # 1. Make stride negative to traverse in reverse order
        # 2. Adjust offset to start from end of axis
        for ax in axes:
            new_strides[ax] = -self._strides[ax]
            offset += self._strides[ax] * (self._shape[ax] - 1)

        out = NDArray.make(
            self._shape,
            strides=tuple(new_strides),
            device=self.device,
            handle=self._handle,
            offset=offset,
        )
        # Return compacted array to ensure standard memory layout
        # TODO: Copies memory, if negative strides are supported, this can be avoided
        out = out.compact()

        return out

    def pad(self, axes: tuple[tuple[int, int]]) -> NDArray:
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

        >>> a = NDArray(np.array([[1, 2], [3, 4]]))
        >>> a.pad(((1, 1), (1, 1)))
        array([[0., 0., 0., 0.],
               [0., 1., 2., 0.],
               [0., 3., 4., 0.],
               [0., 0., 0., 0.]], dtype=float32)
        """
        if len(axes) != self.ndim:
            raise ValueError(
                f"Padding axes {axes} must match array dimensions {self.ndim}"
            )

        # Calculate new shape after padding
        new_shape = tuple(
            dim + left + right for dim, (left, right) in zip(self.shape, axes)
        )

        # Create output array filled with zeros
        out = self.device.zeros(new_shape, dtype=self.dtype)

        # Create slices to insert original data
        slices = tuple(
            slice(left, left + dim) for dim, (left, _) in zip(self.shape, axes)
        )
        # Copy data into padded array
        out[slices] = self
        return out


# TODO: really needed?
def broadcast_shapes(*shapes: tuple) -> tuple:
    """
    Return broadcasted shape for multiple input shapes.

    Args:
        *shapes: one or more shapes as tuples
    Returns:
        tuple: broadcast compatible shape
    Raises:
        ValueError: If shapes cannot be broadcast together
    """
    # If only one shape provided, return it
    if len(shapes) == 1:
        return shapes[0]

    from builtins import max as pymax

    # Convert all shapes to lists and left-pad shorter ones with 1s
    max_dims = pymax([len(shape) for shape in shapes])
    padded = [[1] * (max_dims - len(shape)) + list(shape) for shape in shapes]

    # Compute output shape according to broadcasting rules
    result = []
    for dims in zip(*padded):
        non_ones = set(d for d in dims if d != 1)
        if len(non_ones) > 1:
            if len(set(non_ones)) > 1:
                raise ValueError(f"Incompatible shapes for broadcasting: {shapes}")
            result.append(non_ones.pop())
        else:
            result.append(pymax(dims))

    return tuple(result)


def from_numpy(a):
    return NDArray(a)


def array(a, dtype="float32", device: AbstractBackend = default_device) -> NDArray:
    """Convenience methods to match numpy a bit more closely."""
    if dtype != "float32":
        # logger.warning(f"Only support float32 for now, got {dtype}")
        a = np.array(a, dtype="float32")
        dtype = a.dtype
        # logger.warning(f"Converting to numpy array with dtype {dtype}")
    assert dtype == "float32"
    return NDArray(a, device=device)


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


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def max(array, axis=None, keepdims=False):
    return array.max(axis=axis, keepdims=keepdims)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a: NDArray) -> NDArray:
    return a.tanh()


def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def flip(a: NDArray, axes: tuple[int] | int) -> NDArray:
    return a.flip(axes)


def stack(arrays: list[NDArray], axis: int = 0) -> NDArray:
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

    if axis < -base_array.ndim - 1 or axis > base_array.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for arrays of dimension {base_array.ndim}"
        )

    output_shape = list(base_array.shape)
    output_shape.insert(axis, len(arrays))
    output_shape = tuple(output_shape)

    out = empty(output_shape, device=base_array.device)

    slice_spec: list = [slice(None)] * out.ndim

    for idx, array in enumerate(arrays):
        slice_spec[axis] = idx
        out[tuple(slice_spec)] = array

    return out


def split(arr: NDArray, axis: int = 0) -> list[NDArray]:
    """Split an array into multiple sub-arrays along the specified axis.

    Args:
        array: NDArray to split
        axis: Integer axis along which to split (default=0)

    Returns:
        List of NDArrays: Split arrays along specified axis

    Raises:
        ValueError: If axis is out of bounds
    """
    out_shape = list(arr.shape)
    out_shape.pop(axis)
    out_shape = tuple(out_shape)
    out = []

    slice_spec: list = [slice(None)] * arr.ndim
    for i in range(arr.shape[axis]):
        slice_spec[axis] = i
        sub_array = arr[tuple(slice_spec)].compact().reshape(out_shape)
        out.append(sub_array)

    return out


def array_split(
    arr: NDArray, indices_or_sections: int | list[int], axis: int = 0
) -> list[NDArray]:
    """Split array into multiple sub-arrays, allowing uneven divisions."""
    if axis < 0:
        axis += arr.ndim
    if not 0 <= axis < arr.ndim:
        raise ValueError(f"Axis {axis} out of bounds")

    if isinstance(indices_or_sections, int):
        # Handle N sections case
        section_size = arr.shape[axis] // indices_or_sections
        remainder = arr.shape[axis] % indices_or_sections
        indices = []
        acc = 0
        for i in range(indices_or_sections - 1):
            acc += section_size + (1 if i < remainder else 0)
            indices.append(acc)
    else:
        indices = indices_or_sections

    # Create split points
    split_points = [0, *list(indices), arr.shape[axis]]
    out = []
    slice_spec = [slice(None)] * arr.ndim

    for start, end in itertools.pairwise(split_points):
        slice_spec[axis] = slice(start, end)
        out.append(arr[tuple(slice_spec)])

    return out


def transpose(a: NDArray, axes: tuple | None = None) -> NDArray:
    if axes is None:
        axes = tuple(range(a.ndim))[::-1]
    return a.permute(axes)
