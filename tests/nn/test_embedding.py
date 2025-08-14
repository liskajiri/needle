import needle as ndl
import needle.nn as nn
import numpy as np
import pytest
import torch

from tests.devices import all_devices
from tests.nn.sequence_test_config import (
    BATCH_SIZES,
    EMBEDDING_DIMS,
    NUM_EMBEDDINGS,
    SEQ_LENGTHS,
)

rng = np.random.default_rng(3)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_embeddings", NUM_EMBEDDINGS)
@pytest.mark.parametrize("embedding_dim", EMBEDDING_DIMS)
@all_devices()
def test_embedding(
    seq_length, batch_size, num_embeddings, embedding_dim, device
) -> None:
    """Tests the Embedding module against PyTorch's implementation.

    Verifies both forward pass correctness and gradient computation.
    """
    # Generate random word indices with values in range [0, num_embeddings-1]
    x = rng.integers(0, num_embeddings, size=(seq_length, batch_size))

    model_ = torch.nn.Embedding(num_embeddings, embedding_dim)
    output_ = model_(torch.tensor(x))

    model = nn.Embedding(num_embeddings, embedding_dim, device=device)
    model.weight = ndl.Tensor(model_.weight.detach().numpy(), device=device)

    output = model(ndl.Tensor(x, device=device, dtype="int32"))

    assert output.device == device

    np.testing.assert_allclose(
        output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5
    )

    output.sum().backward()
    output_.sum().backward()

    np.testing.assert_allclose(
        model_.weight.grad.detach().numpy(),
        model.weight.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )

    # verify specific indices map to correct embeddings
    test_indices = rng.integers(0, num_embeddings, size=(5, 1), dtype=np.int32)

    needle_test = model(ndl.Tensor(test_indices, device=device, dtype="int32"))
    torch_test = model_(torch.tensor(test_indices))

    np.testing.assert_allclose(
        torch_test.detach().numpy(), needle_test.numpy(), atol=1e-5, rtol=1e-5
    )
