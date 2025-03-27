import needle as ndl
import needle.nn as nn
import numpy as np
import pytest
import torch

np.random.seed(3)


_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]


BATCH_SIZES = [1, 15]
INPUT_SIZES = [1, 11]
HIDDEN_SIZES = [1, 12]
BIAS = [True, False]
INIT_HIDDEN = [True, False]
NONLINEARITIES = ["tanh", "relu"]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rnn_cell(
    batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device
):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNNCell(
        input_size, hidden_size, nonlinearity=nonlinearity, bias=bias
    )
    if init_hidden:
        h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        h_ = model_(torch.tensor(x), None)

    model = nn.RNNCell(
        input_size, hidden_size, device=device, bias=bias, nonlinearity=nonlinearity
    )
    model.W_ih = ndl.Tensor(
        model_.weight_ih.detach().numpy().transpose(), device=device
    )
    model.W_hh = ndl.Tensor(
        model_.weight_hh.detach().numpy().transpose(), device=device
    )
    if bias:
        model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)
    if init_hidden:
        h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        h = model(ndl.Tensor(x, device=device), None)
    assert h.device == device
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(
        model_.weight_ih.grad.detach().numpy().transpose(),
        model.W_ih.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_lstm_cell(batch_size, input_size, hidden_size, bias, init_hidden, device):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
    if init_hidden:
        h_, c_ = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        h_, c_ = model_(torch.tensor(x), None)

    model = nn.LSTMCell(input_size, hidden_size, device=device, bias=bias)

    model.W_ih = ndl.Tensor(
        model_.weight_ih.detach().numpy().transpose(), device=device
    )
    model.W_hh = ndl.Tensor(
        model_.weight_hh.detach().numpy().transpose(), device=device
    )
    if bias:
        model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)

    if init_hidden:
        h, c = model(
            ndl.Tensor(x, device=device),
            (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)),
        )
    else:
        h, c = model(ndl.Tensor(x, device=device), None)
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)

    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(
        model_.weight_ih.grad.detach().numpy().transpose(),
        model.W_ih.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


SEQ_LENGTHS = [1, 13]
NUM_LAYERS = [1, 2]


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
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
):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

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
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_lstm(
    seq_length,
    num_layers,
    batch_size,
    input_size,
    hidden_size,
    bias,
    init_hidden,
    device,
):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

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
