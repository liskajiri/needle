import numpy as np
import pytest
from needle import backend_ndarray as nd

import backends.numpy as backend_numpy

# Constants for testing
DEFAULT_SHAPE = (4, 5)
STATISTICAL_SAMPLE_SIZE = 10000
TOLERANCE = 0.1
DTYPES = ["float32", "float64"]


def get_backends():
    """Get both CPU and NumPy backends for testing."""
    backends = [
        (nd.cpu(), "cpu"),
        (backend_numpy, "numpy"),
    ]
    if nd.cuda().enabled():
        backends.append((nd.cuda(), "cuda"))
    return backends


def convert_to_numpy(array):
    """Convert array to numpy array if needed."""
    return array if isinstance(array, np.ndarray) else array.numpy()


def assert_same_results(cpu_result, numpy_result):
    """Compare results from different backends using NumPy testing."""
    np.testing.assert_equal(
        convert_to_numpy(cpu_result), convert_to_numpy(numpy_result)
    )


def assert_statistical_properties(
    data, expected_mean, expected_std, tolerance=TOLERANCE
):
    """Check if statistical properties match expectations using NumPy testing."""
    np.testing.assert_allclose(
        data.mean(),
        expected_mean,
        atol=tolerance,
        err_msg=f"Expected mean close to {expected_mean}, got {data.mean()}",
    )
    np.testing.assert_allclose(
        data.std(),
        expected_std,
        atol=tolerance,
        err_msg=f"Expected std close to {expected_std}, got {data.std()}",
    )


def backend_id(backend_tuple):
    """Generate ID for backend parameter."""
    _, name = backend_tuple
    return f"{name}"


def shape_id(shape):
    """Generate ID for shape parameter."""
    return "".join(str(shape))


def idx_id(idx_tuple):
    """Generate ID for (n,idx) parameter."""
    n, idx = idx_tuple
    return f"n={n}_idx={idx}"


class TestRandom:
    shapes = (DEFAULT_SHAPE, (10,), (2, 3, 4))
    shape_ids = (shape_id(s) for s in shapes)

    @pytest.mark.parametrize(
        "backend,name", get_backends(), ids=[backend_id(b) for b in get_backends()]
    )
    @pytest.mark.parametrize("shape", shapes, ids=shape_ids)
    def test_randn_shape(self, backend, name, shape):
        """Test random normal distribution shape."""
        result = backend.randn(shape)
        np.testing.assert_equal(
            result.shape, shape, err_msg=f"Expected shape {shape}, got {result.shape}"
        )

    @pytest.mark.parametrize(
        "backend,name", get_backends(), ids=[backend_id(b) for b in get_backends()]
    )
    def test_randn_distribution(self, backend, name):
        """Test random normal distribution properties."""
        samples = backend.randn((STATISTICAL_SAMPLE_SIZE,))
        data = convert_to_numpy(samples)

        # Normal distribution should have mean ≈ 0 and std ≈ 1
        assert_statistical_properties(data, expected_mean=0, expected_std=1)

    @pytest.mark.parametrize(
        "backend,name", get_backends(), ids=[backend_id(b) for b in get_backends()]
    )
    @pytest.mark.parametrize("shape", shapes, ids=shape_ids)
    def test_rand_shape(self, backend, name, shape):
        """Test uniform random distribution shape."""
        result = backend.rand(shape)
        np.testing.assert_equal(
            result.shape, shape, err_msg=f"Expected shape {shape}, got {result.shape}"
        )

    @pytest.mark.parametrize(
        "backend,name", get_backends(), ids=[backend_id(b) for b in get_backends()]
    )
    def test_rand_distribution(self, backend, name):
        """Test uniform random distribution properties."""
        samples = backend.rand((STATISTICAL_SAMPLE_SIZE,))
        data = convert_to_numpy(samples)

        # Uniform [0,1] distribution properties
        np.testing.assert_array_less(
            -1e-6, data, err_msg="Values should be greater than or equal to 0"
        )
        np.testing.assert_array_less(
            data, 1 + 1e-6, err_msg="Values should be less than or equal to 1"
        )
        assert_statistical_properties(
            data, expected_mean=0.5, expected_std=1 / np.sqrt(12)
        )


class TestOneHot:
    one_hot_cases = ((5, 2), (10, 0), (3, 2))
    one_hot_ids = (f"n={n},idx={i}" for n, i in one_hot_cases)

    @pytest.mark.parametrize(
        "backend,name", get_backends(), ids=[backend_id(b) for b in get_backends()]
    )
    @pytest.mark.parametrize("n,idx", one_hot_cases, ids=one_hot_ids)
    def test_one_hot(self, backend, name, n, idx, dtype="float32"):
        """Test one-hot vector creation with different sizes and dtypes."""
        result = backend.one_hot(n, idx, dtype)

        # Test shape
        np.testing.assert_equal(
            result.shape, (n,), err_msg=f"Expected shape ({n},), got {result.shape}"
        )

        # Test values
        data = convert_to_numpy(result)
        np.testing.assert_equal(
            data.dtype,
            np.dtype(dtype),
            err_msg=f"Expected dtype {dtype}, got {data.dtype}",
        )

        # Create expected one-hot array and compare
        expected = np.zeros(n, dtype=dtype)
        expected[idx] = 1.0
        np.testing.assert_equal(
            data, expected, err_msg=f"One-hot encoding incorrect at index {idx}"
        )


class TestBackendConsistency:
    """Test consistency between CPU and NumPy backends."""

    one_hot_cases = ((5, 2), (10, 5))
    one_hot_ids = (f"n={n},idx={i}" for n, i in one_hot_cases)
    dtype_ids = (f"{dt}" for dt in DTYPES)
    SHAPES = (DEFAULT_SHAPE, (10,), (2, 3, 4))
    shape_ids = (shape_id(s) for s in SHAPES)

    def setup_method(self):
        self.cpu_backend = nd.cpu()
        self.numpy_backend = backend_numpy

    @pytest.mark.parametrize("n,idx", one_hot_cases, ids=one_hot_ids)
    @pytest.mark.parametrize("dtype", DTYPES, ids=dtype_ids)
    def test_one_hot_consistency(self, n, idx, dtype):
        """Test one_hot creates same results across backends."""
        cpu_result = self.cpu_backend.one_hot(n, idx, dtype)
        numpy_result = self.numpy_backend.one_hot(n, idx, dtype)
        assert_same_results(cpu_result, numpy_result)

    @pytest.mark.parametrize("shape", SHAPES, ids=shape_ids)
    def test_randn_consistency(self, shape):
        """Test randn distribution shape consistency."""
        # We can only check shapes as values will differ due to random generation
        cpu_randn = self.cpu_backend.randn(shape)
        numpy_randn = self.numpy_backend.randn(shape)
        assert cpu_randn.shape == numpy_randn.shape, (
            f"Expected shape {shape}, got {cpu_randn.shape} and {numpy_randn.shape}"
        )

    @pytest.mark.parametrize("shape", SHAPES, ids=shape_ids)
    def test_rand_consistency(self, shape):
        """Test rand distribution shape consistency."""
        # We can only check shapes as values will differ due to random generation
        cpu_rand = self.cpu_backend.rand(shape)
        numpy_rand = self.numpy_backend.rand(shape)
        assert cpu_rand.shape == numpy_rand.shape, (
            f"Expected shape {shape}, got {cpu_rand.shape} and {numpy_rand.shape}"
        )
