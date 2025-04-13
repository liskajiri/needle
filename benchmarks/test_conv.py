import needle as ndl
import numpy as np
import pytest

rng = np.random.default_rng(0)

BATCH_SIZE = 32

_ALL_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

conv_configs = [
    # (spatial_size, in_channels, out_channels, kernel_size, stride)
    # (32, 16, 32, 3, 1),  # Larger channels
    (64, 3, 16, 3, 2),  # Image-like dimensions
]


@pytest.mark.parametrize("device", _ALL_DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize(
    "s,in_channels,out_channels,k,stride",
    conv_configs,
    ids=["image-like"],
)
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_conv(
    benchmark, s, in_channels, out_channels, k, stride, device, backward
) -> None:
    def run_forward_backward(loss):
        loss.backward()
        return loss

    conv = ndl.nn.Conv(in_channels, out_channels, k, stride=stride, device=device)
    x = ndl.init.rand(
        (BATCH_SIZE, in_channels, s, s), device=device, requires_grad=backward
    )

    if not backward:
        out = benchmark(conv, x)
    else:
        out = conv(x)
        loss = out.sum()
        _loss = benchmark(run_forward_backward, loss)

        assert x.grad.shape == x.shape
        assert conv.weight.grad.shape == conv.weight.shape
        assert conv.bias.grad.shape == conv.bias.shape
