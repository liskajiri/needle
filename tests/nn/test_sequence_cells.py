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
)

rng = np.random.default_rng(3)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@all_devices()
def test_rnn_cell(
    batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device
) -> None:
    x = rng.standard_normal((batch_size, input_size), dtype=np.float32)
    h0 = rng.standard_normal((batch_size, hidden_size), dtype=np.float32)

    model_ = torch.nn.RNNCell(
        input_size, hidden_size, nonlinearity=nonlinearity, bias=bias
    )
    torch_input = torch.tensor(x)
    torch_h0 = torch.tensor(h0) if init_hidden else None
    h_ = model_(torch_input, torch_h0)

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

    needle_input = ndl.Tensor(x, device=device)
    needle_h0 = ndl.Tensor(h0, device=device) if init_hidden else None
    h = model(needle_input, needle_h0)

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
@all_devices()
def test_lstm_cell(
    batch_size, input_size, hidden_size, bias, init_hidden, device
) -> None:
    x = rng.standard_normal((batch_size, input_size), dtype=np.float32)
    h0 = rng.standard_normal((batch_size, hidden_size), dtype=np.float32)
    c0 = rng.standard_normal((batch_size, hidden_size), dtype=np.float32)

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

    assert h.device == device
    assert c.device == device
    np.testing.assert_allclose(h.numpy(), h_.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c.numpy(), c_.detach().numpy(), atol=1e-5, rtol=1e-5)

    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(
        model.W_ih.grad.numpy(),
        model_.weight_ih.grad.detach().numpy().transpose(),
        atol=1e-5,
        rtol=1e-5,
    )
