import needle as ndl
import needle.nn as nn
import numpy as np
import pytest
import torch

from tests.devices import all_devices
from tests.nn.sequence_test_config import (
    BATCH_SIZES,
    BIAS,
    HIDDEN_SIZES,
    INIT_HIDDEN,
    INPUT_SIZES,
    NONLINEARITIES,
    NUM_LAYERS,
    SEQ_LENGTHS,
)

rng = np.random.default_rng(3)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@all_devices()
@pytest.mark.slow
def test_rnn(
    seq_length,
    num_layers,
    batch_size,
    input_size,
    hidden_size,
    bias,
    init_hidden,
    nonlinearity,
    device,
) -> None:
    x = rng.standard_normal((seq_length, batch_size, input_size), dtype=np.float32)
    h0 = rng.standard_normal((num_layers, batch_size, hidden_size), dtype=np.float32)

    model_ = torch.nn.RNN(
        input_size,
        hidden_size,
        num_layers=num_layers,
        bias=bias,
        nonlinearity=nonlinearity,
    )
    if init_hidden:
        output_, h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        output_, h_ = model_(torch.tensor(x), None)

    model = nn.RNN(
        input_size,
        hidden_size,
        num_layers,
        bias,
        device=device,
        nonlinearity=nonlinearity,
    )
    for k in range(num_layers):
        model.rnn_cells[k].W_ih = ndl.Tensor(
            getattr(model_, f"weight_ih_l{k}").detach().numpy().transpose(),
            device=device,
        )
        model.rnn_cells[k].W_hh = ndl.Tensor(
            getattr(model_, f"weight_hh_l{k}").detach().numpy().transpose(),
            device=device,
        )
        if bias:
            model.rnn_cells[k].bias_ih = ndl.Tensor(
                getattr(model_, f"bias_ih_l{k}").detach().numpy(), device=device
            )
            model.rnn_cells[k].bias_hh = ndl.Tensor(
                getattr(model_, f"bias_hh_l{k}").detach().numpy(), device=device
            )
    if init_hidden:
        output, h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        output, h = model(ndl.Tensor(x, device=device), None)

    assert output.device == device
    assert h.device == device

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5
    )

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(
        model.rnn_cells[0].W_ih.grad.detach().numpy(),
        model_.weight_ih_l0.grad.numpy().transpose(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@all_devices()
@pytest.mark.slow
def test_lstm(
    seq_length,
    num_layers,
    batch_size,
    input_size,
    hidden_size,
    bias,
    init_hidden,
    device,
) -> None:
    x = rng.standard_normal((seq_length, batch_size, input_size), dtype=np.float32)
    h0 = rng.standard_normal((num_layers, batch_size, hidden_size), dtype=np.float32)
    c0 = rng.standard_normal((num_layers, batch_size, hidden_size), dtype=np.float32)

    model_ = torch.nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)
    if init_hidden:
        output_, (h_, c_) = model_(
            torch.tensor(x), (torch.tensor(h0), torch.tensor(c0))
        )
    else:
        output_, (h_, c_) = model_(torch.tensor(x), None)

    model = nn.LSTM(input_size, hidden_size, num_layers, bias, device=device)
    for k in range(num_layers):
        model.lstm_cells[k].W_ih = ndl.Tensor(
            getattr(model_, f"weight_ih_l{k}").detach().numpy().transpose(),
            device=device,
        )
        model.lstm_cells[k].W_hh = ndl.Tensor(
            getattr(model_, f"weight_hh_l{k}").detach().numpy().transpose(),
            device=device,
        )
        if bias:
            model.lstm_cells[k].bias_ih = ndl.Tensor(
                getattr(model_, f"bias_ih_l{k}").detach().numpy(), device=device
            )
            model.lstm_cells[k].bias_hh = ndl.Tensor(
                getattr(model_, f"bias_hh_l{k}").detach().numpy(), device=device
            )
    if init_hidden:
        output, (h, c) = model(
            ndl.Tensor(x, device=device),
            (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)),
        )
    else:
        output, (h, c) = model(ndl.Tensor(x, device=device), None)

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5
    )

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(
        model.lstm_cells[0].W_ih.grad.detach().numpy(),
        model_.weight_ih_l0.grad.numpy().transpose(),
        atol=1e-5,
        rtol=1e-5,
    )
