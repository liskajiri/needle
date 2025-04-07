import needle as ndl
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

rng = np.random.default_rng(0)

CONCATENATE_SHAPE = (2, 2)

# TODO: Copy from test_autograd backwards tests


@given(
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    )
)
def test_divide(data: np.ndarray) -> None:
    # Avoid division by zero by using data + 1 as denominator
    result = ndl.ops.divide(ndl.Tensor(data), ndl.Tensor(data + 1)).numpy()
    expected = np.divide(data, data + 1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "scalar", [2.0, 10.0, 0.5], ids=["divide_by_2", "divide_by_10", "divide_by_half"]
)
@given(
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    )
)
def test_divide_scalar(scalar: float, data: np.ndarray) -> None:
    result = ndl.ops.divide_scalar(ndl.Tensor(data), scalar=scalar).numpy()
    expected = data / scalar
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    ),
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    ),
)
def test_matmul(a: np.ndarray, b: np.ndarray) -> None:
    result = ndl.matmul(ndl.Tensor(a), ndl.Tensor(b))
    expected = np.matmul(a, b)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "axes", [None, 0, 1], ids=["full_reduction", "reduce_rows", "reduce_cols"]
)
@given(
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    )
)
def test_summation(axes: int | None, data: np.ndarray) -> None:
    result = ndl.ops.mathematic.summation(ndl.Tensor(data), axes=axes).numpy()
    expected = np.sum(data, axis=axes)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_broadcast_to_forward():
    np.testing.assert_allclose(
        ndl.broadcast_to(ndl.Tensor([[1.85, 0.85, 0.6]]), shape=(3, 3, 3)).numpy(),
        np.array(
            [
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
            ]
        ),
    )


def test_reshape_forward():
    np.testing.assert_allclose(
        ndl.reshape(
            ndl.Tensor(
                [
                    [2.9, 2.0, 2.4],
                    [3.95, 3.95, 4.65],
                    [2.1, 2.5, 2.7],
                    [1.9, 4.85, 3.25],
                    [3.35, 3.45, 3.45],
                ]
            ),
            shape=(15,),
        ).numpy(),
        np.array(
            [
                2.9,
                2.0,
                2.4,
                3.95,
                3.95,
                4.65,
                2.1,
                2.5,
                2.7,
                1.9,
                4.85,
                3.25,
                3.35,
                3.45,
                3.45,
            ]
        ),
    )
    np.testing.assert_allclose(
        ndl.reshape(
            ndl.Tensor(
                [
                    [[4.1, 4.05, 1.35, 1.65], [3.65, 0.9, 0.65, 4.15]],
                    [[4.7, 1.4, 2.55, 4.8], [2.8, 1.75, 2.8, 0.6]],
                    [[3.75, 0.6, 0.0, 3.5], [0.15, 1.9, 4.75, 2.8]],
                ]
            ),
            shape=(2, 3, 4),
        ).numpy(),
        np.array(
            [
                [
                    [4.1, 4.05, 1.35, 1.65],
                    [3.65, 0.9, 0.65, 4.15],
                    [4.7, 1.4, 2.55, 4.8],
                ],
                [[2.8, 1.75, 2.8, 0.6], [3.75, 0.6, 0.0, 3.5], [0.15, 1.9, 4.75, 2.8]],
            ]
        ),
    )


@given(
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    )
)
def test_negate(data: np.ndarray) -> None:
    result = ndl.negate(ndl.Tensor(data)).numpy()
    expected = -data
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "axes",
    [(1, 0), (0, 1), None],
    ids=["swap_axes", "transpose_dims", "default_transpose"],
)
@given(
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    )
)
def test_transpose(axes: tuple | None, data: np.ndarray) -> None:
    result = ndl.transpose(ndl.Tensor(data), axes=axes).numpy()
    expected = np.transpose(data, axes=axes)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(
    arrays(
        dtype=np.float32, shape=(3, 3), elements=st.floats(min_value=-10, max_value=10)
    )
)
def test_relu(data: np.ndarray) -> None:
    result = ndl.relu(ndl.Tensor(data)).numpy()
    expected = np.maximum(data, 0)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "axis",
    [0, 1, -1],
    ids=["axis_0", "axis_1", "axis_-1"],
)
@given(data=st.data())
def test_concatenate(axis, data) -> None:
    arrs = [
        data.draw(arrays(dtype=np.float32, shape=CONCATENATE_SHAPE)) for _ in range(3)
    ]
    arrs_ndl = [ndl.NDArray(arr) for arr in arrs]

    np.testing.assert_allclose(
        ndl.array_api.concatenate(
            arrs_ndl,
            axis=axis,
        ).numpy(),
        np.concatenate(arrs, axis=axis),
    )
