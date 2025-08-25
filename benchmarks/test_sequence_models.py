import needle as ndl
import numpy as np
import pytest
from needle.nn import LSTM, RNN

from tests.devices import all_devices

BATCH_SIZE = 8
INPUT_SIZE = 16
HIDDEN_SIZE = 16
SEQ_LEN = 32
NUM_LAYERS = 1

rng = np.random.default_rng(0)


@pytest.mark.parametrize("model_type", ["rnn", "lstm"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@all_devices()
def test_full_sequence_model(benchmark, device, model_type, backward) -> None:
    """Benchmark full RNN/LSTM sequence models on long input sequences."""
    x = ndl.Tensor(
        rng.standard_normal((SEQ_LEN, BATCH_SIZE, INPUT_SIZE), dtype=np.float32),
        device=device,
    )
    if model_type == "rnn":
        model = RNN(INPUT_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, device=device)
        h0 = ndl.Tensor(
            rng.standard_normal(
                (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE), dtype=np.float32
            ),
            device=device,
        )
        hidden = h0
    else:
        model = LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, device=device)
        h0 = ndl.Tensor(
            rng.standard_normal(
                (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE), dtype=np.float32
            ),
            device=device,
        )
        c0 = ndl.Tensor(
            rng.standard_normal(
                (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE), dtype=np.float32
            ),
            device=device,
        )
        hidden = (h0, c0)

    def step_backward():
        out = model(x, hidden)
        # For LSTM, out is (output, (h_n, c_n)), for RNN, (output, h_n)
        if isinstance(out, tuple):
            out[0].sum().backward()
        else:
            out.sum().backward()

    def forward():
        out = model(x, hidden)
        if isinstance(out, tuple):
            return out[0]
        return out

    if not backward:
        benchmark(forward)
    else:
        benchmark(step_backward)
