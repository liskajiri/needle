from __future__ import annotations

from typing import TYPE_CHECKING

import needle.init as init
from needle.backend_selection import default_device
from needle.nn.core import Module, Parameter

if TYPE_CHECKING:
    from needle.tensor import Tensor
    from needle.typing import AbstractBackend, DType


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: AbstractBackend = default_device,
        dtype: DType = "float32",
    ) -> None:
        """Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(
            init.randn((num_embeddings, embedding_dim)), device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """Maps word indices to one-hot vectors, and projects to embedding vectors

        Args:
            Tensor: x: Input tensor of shape (seq_len, bs) containing word indices

        Returns:
            Tensor: Tensor of shape (seq_len, bs, embedding_dim) containing embeddings
        """
        seq_len, batch_size = x.shape

        x_one_hot = init.one_hot(
            self.num_embeddings,
            x,
            device=self.device,
            dtype=self.dtype,
        )
        out = x_one_hot @ self.weight
        return out.reshape((seq_len, batch_size, self.embedding_dim))
