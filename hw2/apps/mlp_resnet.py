import sys
from typing import NewType

sys.path.append("../python")

import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)


def ResidualBlock(
    dim: int, hidden_dim: int, norm: nn.Module = nn.BatchNorm1d, drop_prob: float = 0.1
) -> nn.Module:
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )


def MLPResNet(
    dim: int,
    hidden_dim: int = 100,
    num_blocks: int = 3,
    num_classes: int = 10,
    norm: nn.Module = nn.BatchNorm1d,
    drop_prob: float = 0.1,
) -> nn.Module:
    # important to use tuples and unpacking - won't work with lists
    residual_blocks = (
        ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
        for _ in range(num_blocks)
    )
    return nn.Sequential(
        *(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            *residual_blocks,
            nn.Linear(hidden_dim, num_classes),
        )
    )


ErrorRate = NewType("error_rate", float)
Loss = NewType("loss", float)
Accuracy = NewType("accuracy", float)


def epoch(
    dataloader: ndl.data.DataLoader, model: nn.Module, opt: ndl.optim.Optimizer = None
) -> tuple[ErrorRate, Loss]:
    np.random.seed(4)

    if opt:
        model.train()
    else:
        model.eval()

    correct_preds, avg_loss = 0.0, 0.0
    n_rows = dataloader.dataset.X.shape[0]
    softmax_loss = nn.SoftmaxLoss()

    n_batches = 0
    for x, y in dataloader:
        x = x.reshape((x.shape[0], -1))

        preds = model(x)
        loss = softmax_loss(preds, y)

        correct_preds += (preds.numpy().argmax(axis=1) == y.numpy()).sum()
        avg_loss += loss.numpy()

        if opt:
            opt.zero_grad()
            # loss is a tensor
            loss.backward()
            opt.step()
        n_batches += 1

    # error rate = 1 - accuracy
    avg_err_rate = 1 - (correct_preds / n_rows)
    avg_loss /= n_batches

    return ErrorRate(avg_err_rate), Loss(avg_loss)


def train_mnist(
    batch_size: int = 100,
    epochs: int = 10,
    optimizer: ndl.optim.Optimizer = ndl.optim.Adam,
    lr: float = 0.001,
    weight_decay: float = 0.001,
    hidden_dim: int = 100,
    data_dir: str = "data",
) -> tuple[Accuracy, Loss, Accuracy, Loss]:
    np.random.seed(4)

    train_filenames = [
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    ]
    test_filenames = [
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
    ]
    mnist_train = ndl.data.MNISTDataset(*train_filenames)
    mnist_test = ndl.data.MNISTDataset(*test_filenames)

    train_loader = ndl.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(mnist_test, batch_size=batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        test_acc, test_loss = epoch(test_loader, model)

    return (
        Accuracy(train_acc),
        Loss(train_loss),
        Accuracy(test_acc),
        Loss(test_loss),
    )


if __name__ == "__main__":
    train_mnist(data_dir="../data")
