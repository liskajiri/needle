import logging
from typing import NewType

import needle as ndl
from models.resnet9 import ResNet9
from needle import nn

try:
    from tqdm import tqdm  # type: ignore

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


Loss = NewType("Loss", float)
Accuracy = NewType("Accuracy", float)


# === CIFAR-10 training ===
def epoch_general_cifar10(
    dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None, log_interval: int = 1
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
    if opt is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Batch: ") if TQDM_AVAILABLE else dataloader  # type: ignore
    for batch_idx, batch in enumerate(progress_bar):
        x, y = batch[0], batch[1]
        logits = model(x)
        loss = loss_fn(logits, y)

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

        predictions = logits.numpy().argmax(axis=1)
        batch_correct = (predictions == y.numpy()).sum().item()
        batch_size = y.shape[0]
        batch_acc = batch_correct / batch_size

        total_correct += batch_correct
        total_samples += batch_size
        total_loss += loss.numpy() * batch_size

        # Update progress bar with current metrics
        avg_acc = total_correct / total_samples
        curr_loss = (total_loss / total_samples).item()

        if TQDM_AVAILABLE:
            progress_bar.set_postfix(
                {
                    "loss": f"{curr_loss:.4f}",
                    "batch_acc": f"{batch_acc:.4f}",
                    "avg_acc": f"{avg_acc:.4f}",
                }
            )
        else:
            if batch_idx % log_interval == 0:
                logging.info(
                    f"Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"Avg Acc: {avg_acc:.4f} | "
                    f"Avg loss: {curr_loss:.4f}"
                )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return Accuracy(avg_acc), Loss(avg_loss)


def train_cifar10(
    model,
    dataloader,
    n_epochs=1,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    loss_fn=nn.SoftmaxLoss,
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    loss_fn_instance = loss_fn()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    avg_acc = 0
    avg_loss = 0

    logging.info(f"Starting training for n_epochs={n_epochs}...")
    for _epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(
            dataloader=dataloader, model=model, loss_fn=loss_fn_instance, opt=opt
        )

    return avg_acc, avg_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    loss_fn_instance = loss_fn()

    avg_acc, avg_loss = epoch_general_cifar10(
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn_instance,
        opt=None,
    )

    return avg_acc, avg_loss


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    batch_size = 16

    train_dataset = ndl.data.datasets.CIFAR10Dataset(train=True)
    test_dataset = ndl.data.datasets.CIFAR10Dataset(train=False)

    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = ResNet9(in_features=3, out_features=10)

    logging.info("Training CIFAR-10 model...")
    train_cifar10(model=model, dataloader=train_dataloader)
