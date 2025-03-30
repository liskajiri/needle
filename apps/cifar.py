import logging

import needle as ndl
from models.resnet9 import ResNet9
from needle import nn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# === CIFAR-10 training ===
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
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

    for batch in dataloader:
        x, y = batch[0], batch[1]
        logits = model(x)
        loss = loss_fn(logits, y)

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

        predictions = logits.numpy().argmax(axis=1)
        total_correct += (predictions == y.numpy()).sum()
        total_samples += y.shape[0]
        total_loss += loss.numpy() * y.shape[0]
        logging.info(f"Total acc: {total_correct / total_samples}")
        logging.info(f"Avg loss: {total_loss / total_samples}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_acc, avg_loss


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
    batch_size = 32

    train_dataset = ndl.data.datasets.CIFAR10Dataset(train=True)
    test_dataset = ndl.data.datasets.CIFAR10Dataset(train=False)

    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = ResNet9(in_features=3, out_features=10)

    train_cifar10(model=model, dataloader=train_dataloader)
