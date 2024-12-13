import numpy as np
import pytest
from needle import backend_ndarray as nd

_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")
    ),
]

_ALL_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")
    ),
    nd.cpu_numpy(),
]

##### ===
##### === Matmul benchmark
##### === - Graph of results is in benchmark_results/matmul_graph
##### === - Full results are in .benchmarks
##### ===

matmul_dims = [
    (8, 8, 8),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    # TODO: Fails on precision
    # (256, 256, 256),
]


@pytest.mark.parametrize("device", _ALL_DEVICES, ids=["cpu", "cuda", "np"])
@pytest.mark.parametrize("m,n,p", matmul_dims)
@pytest.mark.benchmark(
    max_time=0.5,
    min_rounds=10,
    disable_gc=True,
    warmup=True,
    warmup_iterations=3,
)
def test_matmul(benchmark, m, n, p, device):
    def matmul(A, B):
        return A @ B

    _A = np.random.randn(m, n)
    _B = np.random.randn(n, p)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)

    out = benchmark(matmul, A, B)
    np.testing.assert_allclose(out.numpy(), _A @ _B, rtol=1e-5, atol=1e-5)


# TODO:
# CPU: vs Numpy, Torch, Torch.compile
# GPU: Torch, Torch.compile
# Maybe add Triton backend?
