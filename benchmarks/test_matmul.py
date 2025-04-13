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
    (256, 256, 256),
    (512, 512, 512),
]


@pytest.mark.parametrize("device", _ALL_DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(("m", "n", "p"), matmul_dims)
def test_matmul(benchmark, m, n, p, device) -> None:
    def matmul(A: NDArray, B: NDArray) -> NDArray:
        return A @ B

    a = rng.standard_normal((m, n))
    b = rng.standard_normal((n, p))

    A = ndl.array(a, device=device)
    B = ndl.array(b, device=device)

    _out = benchmark(matmul, A, B)
    # np.testing.assert_allclose(out.numpy(), a @ b, rtol=1e-4, atol=1e-4)
