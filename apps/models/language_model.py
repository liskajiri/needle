import needle as ndl
import needle.nn as nn
from needle.backend_selection import default_device
from needle.tensor import Tensor
from needle.typing import AbstractBackend, DType


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        output_size: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        seq_model: str = "rnn",
        seq_len: int = 40,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype,
        )

        model = nn.RNN if seq_model == "rnn" else nn.LSTM
        self.seq_model = model(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

        self.linear = nn.Linear(
            hidden_size,
            output_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        """
        Given sequence (and the previous hidden state if given),
        returns probabilities of next word
        (along with the last hidden state from the sequence model).

        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)

        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        x = self.embedding(x)
        x, h = self.seq_model(x, h)
        x = x.reshape((-1, self.hidden_size))
        x = self.linear(x)
        return x, h
