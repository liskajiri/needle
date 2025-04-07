import numpy as np
import pytest
from needle import backend_ndarray as ndl
from needle.backend_selection import NDArray

rng = np.random.default_rng(0)

_ALL_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

# CPU: vs Numpy, Torch, Torch.compile

matmul_dims = [
    (8, 8, 8),
    (64, 64, 64),
    (128, 128, 128),
]


def matmul(A: NDArray, B: NDArray) -> NDArray:
    return A @ B


@pytest.mark.parametrize("device", _ALL_DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(("m", "n", "p"), matmul_dims)
@pytest.mark.benchmark(
    max_time=1,
    min_rounds=1000,
    disable_gc=True,
    warmup=True,
    warmup_iterations=100,
)
def test_matmul(benchmark, m, n, p, device) -> None:
    a = rng.standard_normal((m, n))
    b = rng.standard_normal((n, p))

    A = ndl.array(a, device=device)
    B = ndl.array(b, device=device)

    out = benchmark(matmul, A, B)
    np.testing.assert_allclose(out.numpy(), a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("m", "n", "p"), [(128, 128, 128)])
@pytest.mark.benchmark(
    max_time=1,
    min_rounds=1000,
    disable_gc=True,
    warmup=True,
    warmup_iterations=100,
)
def test_matmul_numpy(benchmark, m, n, p) -> None:
    device = ndl.cpu_numpy()
    a = rng.standard_normal((m, n))
    b = rng.standard_normal((n, p))

    A = ndl.array(a, device=device)
    B = ndl.array(b, device=device)

    out = benchmark(matmul, A, B)
    np.testing.assert_allclose(out.numpy(), a @ b, rtol=1e-4, atol=1e-4)


large_matmul_dims = [
    (256, 256, 256),
    # (512, 512, 512),
    # (1024, 1024, 1024),
]


@pytest.mark.parametrize("device", _ALL_DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(("m", "n", "p"), large_matmul_dims)
@pytest.mark.benchmark(
    max_time=3,
    min_rounds=100,
    disable_gc=True,
    warmup=True,
    warmup_iterations=10,
)
def test_matmul_large(benchmark, m, n, p, device) -> None:
    a = rng.standard_normal((m, n))
    b = rng.standard_normal((n, p))
    A = ndl.array(a, device=device)
    B = ndl.array(b, device=device)

    out = benchmark(matmul, A, B)
    np.testing.assert_allclose(out.numpy(), a @ b, rtol=1e-4, atol=1e-4)
