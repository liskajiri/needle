from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from needle.backend_ndarray.device import AbstractBackend

if TYPE_CHECKING:
    from needle.backend_ndarray.utils import DType, Scalar, Shape

logger = logging.getLogger(__name__)

# TODO: get rid of .compact calls - copies memory
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
        from . import ndarray_backend_cuda  # type: ignore

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda")


def cpu_numpy() -> AbstractBackend:
    """Return numpy device."""
    try:
        from . import backend_numpy

        return BackendDevice("cpu_numpy", backend_numpy)
    except ImportError:
        raise ImportError("Numpy backend not available")


def cpu() -> AbstractBackend:
    """Return cpu device."""
    try:
        from . import ndarray_backend_cpu  # type: ignore

        return BackendDevice("cpu", ndarray_backend_cpu)
    except ImportError:
        raise ImportError("CPU backend not available")


def default_device() -> AbstractBackend:
    return cpu_numpy()


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
        device: AbstractBackend = default_device(),
    ) -> None:
        # TODO: allow creating by itself and from list[]
        """Create by copying another NDArray, or from numpy."""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            # device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: NDArray):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape: tuple):
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
        device: AbstractBackend = default_device(),
        handle: NDArray | None = None,
        offset: int = 0,
    ):
        """
        Create a new NDArray with the given properties
        This will allocate the memory if handle=None,
        otherwise it will use the handle of an existing array.
        """
        array = NDArray.__new__(NDArray)
        array._shape = shape
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device
        if handle is None:
            array._handle = array.device.Array(math.prod(shape))
        else:
            array._handle = handle
        return array

    ### Properties and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return math.prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__()  # + f", device={self.device})"
        # return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value: float):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device: BackendDevice):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        return NDArray(self.numpy(), device=device)

    def numpy(self):
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

    # TODO:
    def __array__(
        self,
        dtype: str | None = None,
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

    def is_compact(self):
        """
        Return true if array is compact in memory and
        internal size equals math.product of the shape dimensions.
        """
        return (
            self._strides == self.compact_strides(self._shape)
            and math.prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact."""
        if self.is_compact():
            return self
        out = NDArray.make(self.shape, device=self.device)
        self.device.compact(
            self._handle, out._handle, self.shape, self.strides, self._offset
        )
        return out

    def as_strided(self, shape, strides):
        """Re-stride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    def astype(self, dtype):
        # TODO: support other types
        """Return a copy of the array with a different type."""
        if dtype != "float32":
            logger.warning(f"Only support float32 for now, got {dtype}")
            a = np.array(self, dtype="float32")
            dtype = a.dtype
            logger.warning(f"Converting to numpy array with dtype {dtype}")
        assert dtype == "float32", f"Only support float32 for now, got {dtype}"
        return self + 0.0

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape: tuple) -> NDArray:
        """Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises
        ------
            ValueError if math.product of current shape is not equal to the math.product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray : reshaped array; this will point to the same memory
            as the original NDArray.

        """
        # TODO: simplify this
        if (
            len(self.shape) == 1
            and len(new_shape) == 2
            and self.shape[0] == new_shape[0]
            and new_shape[1] == 1
        ):
            return NDArray.make(
                new_shape,
                device=self.device,
                handle=self._handle,
                strides=(self.strides[0], 0),
            )

        if self.shape == new_shape:
            return self
        if not self.is_compact():
            raise ValueError("Cannot reshape array that is not compact")

        # flatten the array
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError(
                    f"""Reshape can not deduce shape with multiple -1,
                    but called with {new_shape}"""
                )

            neg_idx = new_shape.index(-1)
            other_dims_product = math.prod(
                new_shape[:neg_idx] + new_shape[neg_idx + 1 :]
            )

            # remainder has to be divisible
            if self.size % other_dims_product != 0:
                raise ValueError(
                    f"Cannot reshape array of size {self.size} into shape {new_shape}"
                )
            missing_dim = self.size // other_dims_product
            temp_shape = new_shape[:neg_idx] + (missing_dim,) + new_shape[neg_idx + 1 :]
            logger.info(
                f"""Reshaping with -1:
                from {self.shape} to {new_shape} | got {temp_shape}"""
            )
            new_shape = temp_shape

        if self.size != math.prod(new_shape):
            raise ValueError(f"Cannot reshape from {self.shape} to {new_shape}")

        new_strides = self.compact_strides(new_shape)
        return self.as_strided(new_shape, new_strides)

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

    def broadcast_to(self, new_shape: tuple[int]) -> NDArray:
        """Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
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
        curr_size = math.prod(self._shape)
        new_size = math.prod(new_shape)
        if len(new_shape) < len(self._shape) or new_size % curr_size != 0:
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

    def process_slice(self, sl, dim):
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

    # TODO: errors out
    def __getitem__(self, idxs: int | tuple[int] | NDArray) -> NDArray:
        """The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.
        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory
        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            corresponding to the subset of the matrix to get
        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.

        """
        # handle singleton as tuple, everything as slices
        # TODO: allow idxs: NDArray
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )  # type: ignore
        assert (
            len(idxs) == self.ndim
        ), f"""Need indexes equal to number of dimensions, trying to
            select {idxs} from {self.ndim} dimensions"""

        # idxs correspond to slices in each dimension
        idxs: tuple[slice]

        new_shape = []
        new_strides = list(self._strides)
        new_offset = self._offset
        for i, ax in enumerate(idxs):
            # stop - start guaranteed to be positive
            # slice(0, 3, 2) = [0, 2], but when using (3 - 0) // 2 = 1
            new_shape.append((ax.stop - ax.start - 1) // ax.step + 1)
            new_strides[i] *= ax.step
            # distance to the first element
            new_offset += self._strides[i] * ax.start

        return NDArray.make(
            tuple(new_shape),
            strides=tuple(new_strides),
            device=self._device,
            handle=self._handle,
            offset=new_offset,
        )

    def __setitem__(self, idxs: tuple, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__().
        """
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert math.prod(view.shape) == math.prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                math.prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of element-wise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(
        self, other: NDArray | Scalar, ewise_func: Callable, scalar_func: Callable
    ) -> NDArray:
        """
        Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar.
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, f"""
                operation needs two equal-sized arrays,
                got {self.shape} and {other.shape}
                """
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        elif isinstance(other, float | int):
            scalar_func(self.compact()._handle, other, out._handle)
        else:
            logger.error(f"Unsupported type {type(other)} for ewise_or_scalar")
            scalar_func(self.compact()._handle, other, out._handle)
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

    def __eq__(self, other):
        """Support == comparison with numpy arrays and scalars"""
        if isinstance(other, np.ndarray):
            return np.array_equal(self.numpy(), other)
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        if isinstance(other, np.ndarray):
            np.greater_equal(self.numpy(), other)
            # return self.numpy() >= (
            #     other.numpy() if isinstance(other, NDArray) else other
            # )
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    # TODO: auto-derive comparisons - total_ordering
    # def __lt__(self, other):
    # if isinstance(other, np.ndarray):
    #     return self.numpy() < (
    #         other.numpy() if isinstance(other, NDArray) else other
    #     )
    # else:
    # return 1 - (self >= other)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Element-wise functions

    # TODO: auto derive inplace functions
    def log(self):
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
        """Matrix multiplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.
        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will re-stride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size
        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """
        assert (
            self.ndim == 2
        ), f"Matrix multiplication requires 2D arrays, got {self.ndim} in {self.shape}"
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
            # if the matrix is aligned, use tiled matrix multiplication
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

    # TODO: Reductions over multiple axes
    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(
        self, axis: tuple | None, keepdims: bool = False
    ) -> tuple[NDArray, NDArray]:
        """
        Return a view to the array set up for reduction functions and output array.
        TODO: Doc, doctest
        """
        if isinstance(axis, tuple) and not axis:
            msg = "Empty axis in reduce"
            raise ValueError(msg)

        if axis is None:
            # view: (1, 1, ..., N)
            # out: (1, )
            view = self.compact().reshape(
                (1,) * (self.ndim - 1) + (math.prod(self.shape),)
            )
            out = NDArray.make((1,), device=self.device)

        else:
            if isinstance(axis, tuple | list):
                assert (
                    len(axis) == 1
                ), f"Only support reduction over a single axis, got {axis}"
                # axis = axis[0]

            # view: (x, y, z, ...) without axis
            # out: same, or 1 instead of axis if keepdims
            print(f"Reducing {self.shape} with axis {axis}")

            view = self.permute(
                (*tuple([a for a in range(self.ndim) if a != axis]), axis)
            )

            if keepdims:
                new_shape = tuple(
                    [1 if i == axis else s for i, s in enumerate(self.shape)]
                )
            else:
                new_shape = tuple([s for i, s in enumerate(self.shape) if i != axis])

            print(f"Keep dims; {keepdims} | new_shape: {new_shape}")
            out = NDArray.make(new_shape, device=self.device)

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

    def flip(self, axes):
        """Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        # TODO:
        raise NotImplementedError

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0))
        pads the middle axis with a 0 on the left and right side.
        """
        # TODO:
        raise NotImplementedError


def from_numpy(a):
    return NDArray(a)


def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    if dtype != "float32":
        logger.warning(f"Only support float32 for now, got {dtype}")
        a = np.array(a, dtype="float32")
        dtype = a.dtype
        logger.warning(f"Converting to numpy array with dtype {dtype}")
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    # TODO: get rid of this by default params
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
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


# TODO: sum over multiple axis
def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def flip(a, axes):
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


def split(array: NDArray, axis: int = 0) -> list[NDArray]:
    """Split an array into multiple sub-arrays along the specified axis.

    Args:
        array: NDArray to split
        axis: Integer axis along which to split (default=0)

    Returns:
        List of NDArrays: Split arrays along specified axis

    Raises:
        ValueError: If axis is out of bounds
    """
    out_shape = list(array.shape)
    out_shape.pop(axis)
    out_shape = tuple(out_shape)
    out = []

    slice_spec: list = [slice(None)] * array.ndim
    for i in range(array.shape[axis]):
        slice_spec[axis] = i
        sub_array = array[tuple(slice_spec)].compact().reshape(out_shape)
        out.append(sub_array)

    return out


def transpose(a, axes: tuple | None = None):
    if axes is None:
        axes = tuple(range(a.ndim))[::-1]
    return a.permute(axes)
