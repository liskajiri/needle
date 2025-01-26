from typing import NewType

import needle as ndl
import numpy as np
from needle import nn
from needle.data.datasets.mnist import MNISTPaths

np.random.seed(0)


def ResidualBlock(
    dim: int,
    hidden_dim: int,
    norm: nn.Module = nn.norms.BatchNorm1d,
    drop_prob: float = 0.1,
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
    norm: nn.Module = nn.norms.BatchNorm1d,
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
    dataloader: ndl.data.DataLoader,
    model: nn.Module,
    opt: ndl.optim.Optimizer | None = None,
) -> tuple[ErrorRate, Loss]:  # type: ignore
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
) -> tuple[Accuracy, Loss, Accuracy, Loss]:  # type: ignore
    np.random.seed(4)

    INPUT_DIM = 784
    train_filenames = [
        MNISTPaths.TRAIN_IMAGES,
        MNISTPaths.TRAIN_LABELS,
    ]
    test_filenames = [
        MNISTPaths.TEST_IMAGES,
        MNISTPaths.TEST_LABELS,
    ]
    mnist_train = ndl.data.MNISTDataset(train_filenames[0], train_filenames[1])
    mnist_test = ndl.data.MNISTDataset(test_filenames[0], test_filenames[1])

    train_loader = ndl.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(mnist_test, batch_size=batch_size)

    model = MLPResNet(INPUT_DIM, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _epoch_idx in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        test_acc, test_loss = epoch(test_loader, model)

    return (
        Accuracy(train_acc),
        Loss(train_loss),
        Accuracy(test_acc),
        Loss(test_loss),
    )


if __name__ == "__main__":
    # TODO: Some memory leak in the code
    train_mnist()
