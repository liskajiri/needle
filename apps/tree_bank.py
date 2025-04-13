import logging

import needle as ndl
from models.language_model import LanguageModel
from needle import nn
from needle.backend_selection import default_device
from needle.data.nlp import Corpus
from needle.typing import AbstractBackend, DType


def epoch_general_ptb(
    data: ndl.Tensor,
    model: nn.Module,
    seq_len: int = 40,
    loss_fn: nn.Module = nn.SoftmaxLoss(),
    opt: ndl.optim.Optimizer | None = None,
    clip: float | None = None,
    device: AbstractBackend = default_device,
    dtype: DType = "float32",
) -> tuple[float, float]:
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: Data of shape (n_batch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: Max norm of gradients (optional)
        device: Device to run the computation on
        dtype: Data type for tensors

    Returns:
        tuple: (avg_acc, avg_loss) where:
            avg_acc: Average accuracy over dataset
            avg_loss: Average loss over dataset
    """
    if opt is not None:
        model.train()
    else:
        model.eval()

    _n_batch, batch_size = data.shape
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i in range(0, data.shape[0] - 1, seq_len):
        X, y = ndl.data.nlp.get_batch(data, i, seq_len, device=device, dtype=dtype)

        # Forward pass
        out, h = model(X)
        loss = loss_fn()(out, y)
        total_loss += loss.numpy() * seq_len * batch_size

        # Calculate accuracy
        predictions = out.numpy().argmax(axis=1)
        targets = y.numpy().reshape(-1)
        total_correct += (predictions == targets).sum()
        total_samples += targets.size

        if opt is not None:
            opt.reset_grad()
            loss.backward()

            # Gradient clipping if specified
            if clip is not None:
                model.clip_grad_norm(clip)

            opt.step()

        # detach hidden state to avoid back-propagating through entire history
        h = tuple([h_i.detach() for h_i in h]) if isinstance(h, tuple) else h.detach()
        logging.info(
            f"""Batch {i // seq_len}: Loss: {loss.numpy()},
              Accuracy: {total_correct / total_samples}"""
        )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_acc, avg_loss


def train_ptb(
    model: nn.Module,
    data: ndl.Tensor,
    seq_len: int = 40,
    n_epochs: int = 1,
    optimizer: type[ndl.optim.Optimizer] = ndl.optim.SGD,
    lr: float = 4.0,
    weight_decay: float = 0.0,
    loss_fn: type[nn.Module] = nn.SoftmaxLoss,
    clip: float | None = None,
    device: AbstractBackend = default_device,
    dtype: DType = "float32",
) -> tuple[float, float]:
    """
    Performs n_epochs epochs of training.

    Args:
        model: LanguageModel instance
        data: Data of shape (n_batch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: Number of epochs (int)
        optimizer: Optimizer class
        lr: Learning rate (float)
        weight_decay: Weight decay (float)
        loss_fn: nn.Module class
        clip: Max norm of gradients (optional)
        device: Device to run the computation on
        dtype: Data type for tensors

    Returns:
        tuple: (avg_acc, avg_loss) where:
            avg_acc: Average accuracy over dataset from last epoch of training
            avg_loss: Average loss over dataset from last epoch of training
    """
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc, train_loss = 0.0, 0.0

    for _epoch in range(n_epochs):
        train_acc, train_loss = epoch_general_ptb(
            data=data,
            model=model,
            seq_len=seq_len,
            loss_fn=loss_fn,
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype,
        )

    return train_acc, train_loss


def evaluate_ptb(
    model: nn.Module,
    data: ndl.Tensor,
    seq_len: int = 40,
    loss_fn: type[nn.Module] = nn.SoftmaxLoss,
    device: AbstractBackend = default_device,
    dtype: DType = "float32",
) -> tuple[float, float]:
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: Data of shape (n_batch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class
        device: Device to run the computation on
        dtype: Data type for tensors

    Returns:
        tuple: (avg_acc, avg_loss) where:
            avg_acc: Average accuracy over dataset
            avg_loss: Average loss over dataset
    """
    return epoch_general_ptb(
        data=data,
        model=model,
        seq_len=seq_len,
        loss_fn=loss_fn,
        opt=None,  # No optimizer for evaluation
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    corpus = Corpus()
    train_data = corpus.train
    test_data = corpus.test

    # Create batches
    batch_size = 100
    train_batches = ndl.data.nlp.batchify(train_data, batch_size)
    test_batches = ndl.data.nlp.batchify(test_data, batch_size)

    # Create model
    embedding_size = 32
    output_size = len(corpus.dictionary)
    hidden_size = 32
    num_layers = 2
    seq_len = 35
    model = LanguageModel(
        embedding_size=embedding_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_len=seq_len,
    )
    # Train model
    n_epochs = 1
    optimizer = ndl.optim.Adam
    loss_fn = nn.SoftmaxLoss
    logging.info("Starting training...")
    train_acc, train_loss = train_ptb(
        model=model,
        data=train_batches,
        seq_len=seq_len,
        n_epochs=n_epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    logging.info(f"Train Accuracy: {train_acc}, Train Loss: {train_loss}")
