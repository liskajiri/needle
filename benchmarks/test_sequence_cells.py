import needle as ndl
import numpy as np
import pytest

BATCH_SIZE = 16
INPUT_SIZE = 32
HIDDEN_SIZE = 32

rng = np.random.default_rng(3)

DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]


@pytest.mark.parametrize("device", DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@pytest.mark.benchmark(
    min_rounds=2,
    disable_gc=True,
    warmup=True,
    warmup_iterations=1,
)
def test_sequence_cell(benchmark, device, cell_type, backward) -> None:
    """Benchmark RNN/LSTM cell operations

    Args:
        cell_type: Type of cell to benchmark ('rnn' or 'lstm')
        backward: Whether to benchmark forward pass only
    """
    x = ndl.Tensor(
        rng.standard_normal((BATCH_SIZE, INPUT_SIZE), dtype=np.float32), device=device
    )
    h0 = ndl.Tensor(
        rng.standard_normal((BATCH_SIZE, HIDDEN_SIZE), dtype=np.float32), device=device
    )

    if cell_type == "rnn":
        model = ndl.nn.RNNCell(INPUT_SIZE, HIDDEN_SIZE, device=device)
        hidden = h0
    else:  # lstm
        model = ndl.nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE, device=device)
        c0 = ndl.Tensor(
            rng.standard_normal((BATCH_SIZE, HIDDEN_SIZE), dtype=np.float32),
            device=device,
        )
        hidden = (h0, c0)

    def step_backward():
        if isinstance(h, tuple):
            h[0].sum().backward()  # For LSTM, backprop through h
        else:
            h.sum().backward()  # For RNN
        return model.W_ih.grad.numpy()

    def forward(x, hidden):
        h = model(x, hidden)
        return h

    if not backward:
        benchmark(forward, x, hidden)
    else:
        h = forward(x, hidden)
        benchmark(step_backward)
