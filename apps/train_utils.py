import logging
from typing import NewType

import needle as ndl
import numpy as np
from needle import nn

try:
    from tqdm import tqdm  # type: ignore

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

Accuracy = NewType("Accuracy", float)
Loss = NewType("Loss", float)


def epoch(
    dataloader: ndl.data.DataLoader,
    model: nn.Module,
    opt: ndl.optim.Optimizer | None = None,
    loss_fn: nn.Module = nn.SoftmaxLoss(),
    log_interval: int = 100,
    mode: str = "train",
) -> tuple[Accuracy, Loss]:
    """
    Iterates over the dataloader.
    If optimizer is not None, sets the model to train mode,
    and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    if mode == "train":
        model.train()
    else:
        model.eval()

    n_batches = len(dataloader)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = (
        tqdm(dataloader, desc=f"{mode} batch: ") if TQDM_AVAILABLE else dataloader  # type: ignore
    )
    for batch_idx, (x, y) in enumerate(progress_bar):
        logits = model(x)
        loss = loss_fn(logits, y)

        predictions = logits.numpy().argmax(axis=1)
        batch_correct = np.count_nonzero(predictions == y.numpy())
        batch_size = y.shape[0]
        batch_acc = batch_correct / batch_size

        total_correct += batch_correct
        total_samples += batch_size
        total_loss += loss.numpy() * batch_size

        # Update progress bar with current metrics
        avg_acc = total_correct / total_samples
        curr_loss = (total_loss / total_samples).item()

        if opt:
            opt.zero_grad()
            # loss is a tensor
            loss.backward()
            opt.step()

        if TQDM_AVAILABLE:
            progress_bar.set_postfix(
                {
                    "loss": f"{curr_loss:.3f}",
                    "batch_acc": f"{batch_acc:.3f}",
                    "avg_acc": f"{avg_acc:.3f}",
                }
            )
        else:
            if batch_idx % log_interval == 0:
                logging.info(
                    "Batch %d/%d | Loss: %.2f | Acc: %.2f",
                    batch_idx,
                    n_batches,
                    curr_loss,
                    avg_acc,
                )

    avg_acc = total_correct / total_samples
    avg_loss = total_loss / total_samples

    return Accuracy(avg_acc), Loss(avg_loss)
