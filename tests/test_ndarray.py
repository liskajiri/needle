import math

import numpy as np
import pytest
from needle import backend_ndarray as ndl

_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]


def compare_strides(a_np, a_nd) -> None:
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides


def check_same_memory(original, view) -> None:
    assert original._handle.ptr() == view._handle.ptr()  # noqa: SLF001


@pytest.mark.parametrize(
    "params",
    [
        {
            "shape": (4, 4),
            "np_fn": lambda X: X.transpose(),
            "nd_fn": lambda X: X.permute((1, 0)),
        },
        {
            "shape": (4, 1, 4),
            "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5, 4)),
            "nd_fn": lambda X: X.broadcast_to((4, 5, 4)),
        },
        {
            "shape": (4, 3),
            "np_fn": lambda X: X.reshape(2, 2, 3),
            "nd_fn": lambda X: X.reshape((2, 2, 3)),
        },
        {
            "shape": (16, 16),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(2, 4, 2, 2, 2, 2, 2),
            "nd_fn": lambda X: X.reshape((2, 4, 2, 2, 2, 2, 2)),
        },
        {
            "shape": (
                2,
                4,
                2,
                2,
                2,
                2,
                2,
            ),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(16, 16),
            "nd_fn": lambda X: X.reshape((16, 16)),
        },
        {"shape": (8, 8), "np_fn": lambda X: X[4:, 4:], "nd_fn": lambda X: X[4:, 4:]},
        {
            "shape": (8, 8, 2, 2, 2, 2),
            "np_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
            "nd_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
        },
        {
            "shape": (7, 8),
            "np_fn": lambda X: X.transpose()[3:7, 2:5],
            "nd_fn": lambda X: X.permute((1, 0))[3:7, 2:5],
        },
    ],
    ids=[
        "transpose",
        "broadcast_to",
        "reshape1",
        "reshape2",
        "reshape3",
        "getitem1",
        "getitem2",
        "transposegetitem",
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_compact(params, device):
    shape, np_fn, nd_fn = params["shape"], params["np_fn"], params["nd_fn"]
    _A = np.random.randint(low=0, high=10, size=shape)
    A = ndl.array(_A, device=device)

    lhs = nd_fn(A).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(_A)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)


### ====


# Permute, broadcast_to, reshape, getitem, and combinations thereof.
@pytest.mark.parametrize(
    "params",
    [
        # Permute tests
        {
            "shape": (4, 4),
            "np_fn": lambda X: X.transpose(),
            "nd_fn": lambda X: X.permute((1, 0)),
        },
        {
            "shape": (2, 3, 4),
            "np_fn": lambda X: np.transpose(X, axes=(2, 1, 0)),
            "nd_fn": lambda X: X.permute((2, 1, 0)),
        },
        # Broadcast_to tests
        {
            "shape": (4, 1),
            "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5)),
            "nd_fn": lambda X: X.broadcast_to((4, 5)),
        },
        {
            "shape": (1, 3, 1),
            "np_fn": lambda X: np.broadcast_to(X, shape=(2, 3, 4)),
            "nd_fn": lambda X: X.broadcast_to((2, 3, 4)),
        },
        # Reshape tests
        {
            "shape": (6,),
            "np_fn": lambda X: X.reshape(2, 3),
            "nd_fn": lambda X: X.reshape((2, 3)),
        },
        {
            "shape": (2, 3, 4),
            "np_fn": lambda X: X.reshape(4, 6),
            "nd_fn": lambda X: X.reshape((4, 6)),
        },
        # Getitem tests
        {
            "shape": (8, 8),
            "np_fn": lambda X: X[2:6, 3:7],
            "nd_fn": lambda X: X[2:6, 3:7],
        },
        {
            "shape": (4, 5, 6),
            "np_fn": lambda X: X[1:3, :, 2:5],
            "nd_fn": lambda X: X[1:3, :, 2:5],
        },
        # Combination tests
        {
            "shape": (7, 8),
            "np_fn": lambda X: X.transpose()[1:5, 2:6],
            "nd_fn": lambda X: X.permute((1, 0))[1:5, 2:6],
        },
    ],
    ids=[
        "permute_2D",
        "permute_3D",
        "broadcast_to_2D",
        "broadcast_to_3D",
        "reshape_flat_to_2D",
        "reshape_3D_to_2D",
        "getitem_2D",
        "getitem_3D",
        "permute_getitem",
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_operations(params, device):
    shape, np_fn, nd_fn = params["shape"], params["np_fn"], params["nd_fn"]
    _A = np.random.randint(low=0, high=10, size=shape)
    A = ndl.array(_A, device=device)

    # Apply the function using the custom library and NumPy
    lhs = nd_fn(A).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(_A)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)


### ====


reduce_params = [
    {"dims": (10,), "axis": 0},
    {"dims": (4, 5, 6), "axis": 0},
    {"dims": (4, 5, 6), "axis": 1},
    {"dims": (4, 5, 6), "axis": 2},
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("params", reduce_params)
def test_reduce_sum(params, device):
    dims, axis = params["dims"], params["axis"]
    _A = np.random.randn(*dims)
    A = ndl.array(_A, device=device)
    np.testing.assert_allclose(
        _A.sum(axis=axis, keepdims=True),
        A.sum(axis=axis, keepdims=True).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("params", reduce_params)
def test_reduce_max(params, device):
    dims, axis = params["dims"], params["axis"]
    _A = np.random.randn(*dims)
    A = ndl.array(_A, device=device)
    np.testing.assert_allclose(
        _A.max(axis=axis, keepdims=True),
        A.max(axis=axis, keepdims=True).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


"""
For converting slice notation to slice objects
to make some proceeding tests easier to read
"""


class _ShapeAndSlices(ndl.NDArray):
    def __getitem__(self, idxs):
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        return self.shape, idxs


def ShapeAndSlices(*shape):
    return _ShapeAndSlices(np.ones(shape))


@pytest.mark.parametrize(
    "params",
    [
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:2, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:3, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6],
            "rhs": ShapeAndSlices(7, 7, 7)[:2, :3, :4],
        },
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_ewise(params, device):
    lhs_shape, lhs_slices = params["lhs"]
    rhs_shape, rhs_slices = params["rhs"]
    _A = np.random.randn(*lhs_shape)
    _B = np.random.randn(*rhs_shape)
    A = ndl.array(_A, device=device)
    B = ndl.array(_B, device=device)
    start_ptr = A._handle.ptr()  # noqa: SLF001
    A[lhs_slices] = B[rhs_slices]
    _A[lhs_slices] = _B[rhs_slices]
    end_ptr = A._handle.ptr()  # noqa: SLF001
    assert start_ptr == end_ptr, "you should modify in-place"
    compare_strides(_A, A)
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)


# Ex: We want arrays of size (4, 5, 6) setting element(s) [1:4, 2, 3] to a scalar
@pytest.mark.parametrize(
    "params",
    [
        ShapeAndSlices(4, 5, 6)[1, 2, 3],
        ShapeAndSlices(4, 5, 6)[1:4, 2, 3],
        ShapeAndSlices(4, 5, 6)[:4, 2:5, 3],
        ShapeAndSlices(4, 5, 6)[1::2, 2:5, ::2],
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_scalar(params, device):
    shape, slices = params
    _A = np.random.randn(*shape)
    A = ndl.array(_A, device=device)
    # probably tear these out using lambdas
    start_ptr = A._handle.ptr()  # noqa: SLF001
    _A[slices] = 4.0
    A[slices] = 4.0
    end_ptr = A._handle.ptr()  # noqa: SLF001
    assert start_ptr == end_ptr, "you should modify in-place"
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
    compare_strides(_A, A)


matmul_tiled_shapes = [(1, 1, 1), (2, 2, 3), (1, 2, 1), (3, 3, 3)]


@pytest.mark.parametrize(("m", "n", "p"), matmul_tiled_shapes)
def test_matmul_tiled(m, n, p):
    device = ndl.cpu()
    assert hasattr(device, "matmul_tiled")
    t = device.__tile_size__
    A = ndl.array(np.random.randn(m, n, t, t), device=ndl.cpu())
    B = ndl.array(np.random.randn(n, p, t, t), device=ndl.cpu())
    C = ndl.NDArray.make((m, p, t, t), device=ndl.cpu())
    device.matmul_tiled(A._handle, B._handle, C._handle, m * t, n * t, p * t)  # noqa: SLF001

    lhs = A.numpy().transpose(0, 2, 1, 3).flatten().reshape(
        m * t, n * t
    ) @ B.numpy().transpose(0, 2, 1, 3).flatten().reshape(n * t, p * t)
    rhs = C.numpy().transpose(0, 2, 1, 3).flatten().reshape(m * t, p * t)

    np.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


OPS = {
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "equal": lambda a, b: a == b,
    "greater_than": lambda a, b: a >= b,
}
OP_FNS = [OPS[k] for k in OPS]
OP_NAMES = list(OPS)

ewise_shapes = [(1, 1, 1), (4, 5, 6)]


@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = ndl.array(_A, device=device)
    B = ndl.array(_B, device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_max(shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = ndl.array(_A, device=device)
    B = ndl.array(_B, device=device)
    np.testing.assert_allclose(
        np.maximum(_A, _B), A.maximum(B).numpy(), atol=1e-5, rtol=1e-5
    )


permute_params = [
    {"dims": (4, 5, 6), "axes": (0, 1, 2)},
    {"dims": (4, 5, 6), "axes": (1, 0, 2)},
    {"dims": (4, 5, 6), "axes": (2, 1, 0)},
]


@pytest.mark.parametrize("params", permute_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_permute(device, params):
    dims = params["dims"]
    axes = params["axes"]
    _A = np.random.randn(*dims)
    A = ndl.array(_A, device=device)
    lhs = np.transpose(_A, axes=axes)
    rhs = A.permute(axes)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


reshape_params = [
    {"shape": (8, 16), "new_shape": (2, 4, 16)},
    {"shape": (8, 16), "new_shape": (8, 4, 2, 2)},
]


@pytest.mark.parametrize("params", reshape_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape(device, params):
    shape = params["shape"]
    new_shape = params["new_shape"]
    _A = np.random.randn(*shape)
    A = ndl.array(_A, device=device)
    lhs = _A.reshape(*new_shape)
    rhs = A.reshape(new_shape)
    np.testing.assert_allclose(rhs.numpy(), lhs, atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


@pytest.mark.parametrize(
    "params",
    [
        # Reshape with -1 inference
        {"shape": (2, 3, 4), "new_shape": (6, -1), "expected_shape": (6, 4)},
        {"shape": (24,), "new_shape": (-1, 6), "expected_shape": (4, 6)},
        {"shape": (8, 3), "new_shape": (2, -1, 3), "expected_shape": (2, 4, 3)},
        {"shape": (2, 3, 4), "new_shape": (-1,), "expected_shape": (24,)},  # Flatten
        {"shape": (4, 1, 3), "new_shape": (-1, 3), "expected_shape": (4, 3)},
        # Edge cases
        {"shape": (5, 1, 3), "new_shape": (5, -1), "expected_shape": (5, 3)},
        {"shape": (1, 1, 1), "new_shape": (-1, 1), "expected_shape": (1, 1)},
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape_with_inference(device, params):
    shape, new_shape = params["shape"], params["new_shape"]
    expected_shape = params["expected_shape"]

    _A = np.arange(math.prod(shape) or 0).reshape(shape)
    A = ndl.array(_A, device=device)

    lhs = _A.reshape(expected_shape)
    rhs = A.reshape(new_shape)

    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)

    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


@pytest.mark.parametrize(
    "params",
    [
        {"shape": (4, 6), "new_shape": (5, 5)},  # Total size mismatch
        {"shape": (2, 3, 4), "new_shape": (2, -1, -1)},  # Multiple -1s
        {"shape": (24,), "new_shape": (-1, 7)},  # Size not divisible evenly
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape_errors(device, params):
    shape, new_shape = params["shape"], params["new_shape"]
    _A = np.random.randn(*shape)
    A = ndl.array(_A, device=device)

    with pytest.raises((ValueError, AssertionError)):
        A.reshape(new_shape)


@pytest.mark.parametrize(
    "params",
    [
        # Add new dimensions
        {"shape": (3,), "broadcast_shape": (2, 3)},
        {"shape": (5,), "broadcast_shape": (4, 5)},
        # Expand singleton dimensions
        {"shape": (1, 3), "broadcast_shape": (2, 3)},
        {"shape": (3, 1), "broadcast_shape": (3, 4)},
        {"shape": (1, 1, 3), "broadcast_shape": (2, 3, 3)},
        # Identity cases
        {"shape": (2, 3), "broadcast_shape": (2, 3)},
        # Edge cases
        {"shape": (1,), "broadcast_shape": (5,)},
        {"shape": (1, 1), "broadcast_shape": (4, 4)},
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to_2(device, params):
    shape = params["shape"]
    to_shape = params["broadcast_shape"]

    _A = np.random.randn(*shape)
    A = ndl.array(_A, device=device)

    lhs = np.broadcast_to(_A, shape=to_shape)
    rhs = A.broadcast_to(to_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


@pytest.mark.parametrize(
    "params",
    [
        # Incompatible shapes
        {"shape": (3,), "broadcast_shape": (2, 2)},
        {"shape": (2, 3), "broadcast_shape": (2, 2)},
        {"shape": (3, 2), "broadcast_shape": (2, 2)},
        # Shrinking non-1 dimensions
        {"shape": (2, 3), "broadcast_shape": (1, 3)},
        # Wrong number of dimensions
        {"shape": (2, 2, 2), "broadcast_shape": (2, 2)},
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to_errors(device, params):
    shape = params["shape"]
    broadcast_shape = params["broadcast_shape"]

    _A = np.random.randn(*shape)
    A = ndl.array(_A, device=device)

    with pytest.raises((ValueError, AssertionError)):
        A.broadcast_to(broadcast_shape)


getitem_params = [
    {"shape": (8, 16), "fn": lambda X: X[3:4, 3:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:2, 1:3]},
    {"shape": (8, 16), "fn": lambda X: X[3:4, 1:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:4, 3:4]},
]


@pytest.mark.parametrize("params", getitem_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem(device, params):
    fn = params["fn"]
    _A = np.random.randn(5, 5)
    A = ndl.array(_A, device=device)
    lhs = fn(_A)
    rhs = fn(A)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


# TODO: Investigate using 2d indices
multi_index_getitem_params = [
    {"fn": lambda X: X[[1, 2]]},
    {"fn": lambda X: X[[1, 2, 3]]},
    {"fn": lambda X: X[[0, 3, 4, 1]]},
]


@pytest.mark.parametrize("params", multi_index_getitem_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem_multi_index(device, params):
    fn = params["fn"]
    _A = np.random.randn(5, 5)
    A = ndl.array(_A, device=device)
    lhs = fn(_A)
    rhs = fn(A)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    # Cannot have the same memory
    # check_same_memory(A, rhs)


broadcast_params = [
    {"from_shape": (1, 3, 4), "to_shape": (6, 3, 4)},
]


@pytest.mark.parametrize("params", broadcast_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to(device, params):
    from_shape, to_shape = params["from_shape"], params["to_shape"]
    _A = np.random.randn(*from_shape)
    A = ndl.array(_A, device=device)

    lhs = np.broadcast_to(_A, shape=to_shape)
    rhs = A.broadcast_to(to_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


matmul_dims = [
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (8, 8, 8),
    (16, 16, 16),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(("m", "n", "p"), matmul_dims)
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n)
    _B = np.random.randn(n, p)
    A = ndl.array(_A, device=device)
    B = ndl.array(_B, device=device)
    np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_mul(device):
    A = np.random.randn(5, 5)
    B = ndl.array(A, device=device)
    np.testing.assert_allclose(A * 5.0, (B * 5.0).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_div(device):
    A = np.random.randn(5, 5)
    B = ndl.array(A, device=device)
    np.testing.assert_allclose(A / 5.0, (B / 5.0).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_power(device):
    A = np.random.randn(5, 5)
    A = np.abs(A)
    B = ndl.array(A, device=device)
    np.testing.assert_allclose(np.power(A, 5.0), (B**5.0).numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.power(A, 0.5), (B**0.5).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_maximum(device):
    A = np.random.randn(5, 5)
    B = ndl.array(A, device=device)
    C = (np.max(A) + 1.0).item()
    np.testing.assert_allclose(
        np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5
    )
    C = (np.max(A) - 1.0).item()
    np.testing.assert_allclose(
        np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_eq(device):
    A = np.random.randn(5, 5)
    B = ndl.array(A, device=device)
    C = A[0, 1].item()
    np.testing.assert_allclose(A == C, (B == C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_ge(device):
    A = np.random.randn(5, 5)
    B = ndl.array(A, device=device)
    C = A[0, 1].item()
    np.testing.assert_allclose(A >= C, (B >= C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_log(device):
    A = np.abs(np.random.randn(5, 5))
    B = ndl.array(A, device=device)
    np.testing.assert_allclose(np.log(A), (B.log()).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_exp(device):
    A = np.random.randn(5, 5)
    B = ndl.array(A, device=device)
    np.testing.assert_allclose(np.exp(A), (B.exp()).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_tanh(device):
    A = np.random.randn(5, 5)
    B = ndl.array(A, device=device)
    np.testing.assert_allclose(np.tanh(A), (B.tanh()).numpy(), atol=1e-5, rtol=1e-5)


# examples from numpy documentation for np.array_split
SPLITTED_SHAPES = [(np.arange(8.0), 3), (np.arange(9), 4)]


@pytest.mark.parametrize(("x", "indices"), SPLITTED_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_array_split(x, indices, device):
    ndl_x = ndl.array(x, device=device)
    splitted_numpy = np.array_split(x, indices)
    splitted_ndl = ndl.array_split(ndl_x, indices)
    for i in range(3):
        np.testing.assert_allclose(
            splitted_numpy[i], splitted_ndl[i].numpy(), atol=1e-5, rtol=1e-5
        )


def Prepare(A):
    return (A.numpy().flatten()[:128], A.strides, A.shape)


def Rand(*shape, device=ndl.default_device, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    _A = np.random.randint(low=1, high=100, size=shape)
    return ndl.array(_A, device=device)


def RandC(*shape, entropy=1):
    if ndl.cuda().enabled():
        return Rand(*shape, device=ndl.cuda(), entropy=2)
    msg = "You need a GPU to run these tests."
    raise NotImplementedError(msg)
