from math import sqrt
from typing import TypedDict

import needle.init as init
import needle.nn as nn
import needle.ops as ops
from needle.backend_selection import default_device
from needle.nn import Module
from needle.tensor import Tensor
from needle.typing import AbstractBackend, DType


class RNNInitConfig(TypedDict):
    low: float
    high: float
    device: AbstractBackend
    dtype: DType


class RNNCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k))
        where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        match nonlinearity:
            case "tanh":
                self.nonlinearity = nn.Tanh()
            case "relu":
                self.nonlinearity = nn.ReLU()
            case _:
                raise ValueError(
                    f"Invalid nonlinearity '{nonlinearity}'. Must be 'tanh' or 'relu'."
                )

        bound = 1 / sqrt(hidden_size)
        config = RNNInitConfig(low=-bound, high=bound, device=device, dtype=dtype)

        self.W_ih = nn.Parameter(init.rand((input_size, hidden_size), **config))
        self.W_hh = nn.Parameter(init.rand((hidden_size, hidden_size), **config))
        if bias:
            self.bias_ih = nn.Parameter(init.rand((hidden_size,), **config))
            self.bias_hh = nn.Parameter(init.rand((hidden_size,), **config))
        else:
            self.bias_ih, self.bias_hh = 0, 0

    def forward(self, X: Tensor, h: Tensor | None = None) -> Tensor:
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        assert len(X.shape) == 2, "Input must be 2D"
        batch_size, input_size = X.shape

        if h is None:
            h = init.zeros(
                (batch_size, self.hidden_size), device=X.device, dtype=X.dtype
            )

        # h' = nonlinearity(X @ W_ih + bias_ih + h @ W_hh + bias_hh)
        hidden = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            hidden += self.bias_ih + self.bias_hh

        h = self.nonlinearity(hidden)
        return h


class RNN(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.rnn_cells = [
            # First layer takes the input_size as input
            RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)
        ]

        for _ in range(1, num_layers):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            )

    def forward(self, X: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Inputs:
        X of shape (seq_len, bs, input_size) with the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) with the initial
          hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs:
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) with the
        final hidden state for each element in the batch.
        """
        assert X.ndim == 3, "Input must be 3D"
        seq_len, batch_size, _hidden_size = X.shape

        if h0 is None:
            h0 = init.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=X.device,
                dtype=X.dtype,
            )
        hs = []
        outputs = []

        for t in range(seq_len):
            # x_t: (batch_size, input_size)
            x_t = X[t]

            for layer in range(self.num_layers):
                h_prev = h0[layer] if t == 0 else hs[layer]

                if layer == 0:
                    # First layer takes the input
                    h_next = self.rnn_cells[layer](x_t, h_prev)
                else:
                    h_next = self.rnn_cells[layer](h_next, h_prev)

                if t == 0:
                    hs.append(h_next)
                else:
                    hs[layer] = h_next

            outputs.append(h_next)

        # Output: (seq_len, batch_size, hidden_size)
        output = ops.stack(outputs, axis=0)

        # hs: (num_layers, batch_size, hidden_size)
        hs = ops.stack(hs, axis=0)

        return output, hs


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION
