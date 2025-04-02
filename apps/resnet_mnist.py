import argparse
import logging
from typing import NewType

import needle as ndl
from models.resnet9 import MLPResNet
from needle import nn
from needle.data.datasets.mnist import MNISTPaths

Loss = NewType("Loss", float)
Accuracy = NewType("Accuracy", float)


def epoch(
    dataloader: ndl.data.DataLoader,
    model: nn.Module,
    opt: ndl.optim.Optimizer | None = None,
) -> tuple[Accuracy, Loss]:
    if opt:
        model.train()
    else:
        model.eval()

    correct_preds, avg_loss = 0.0, 0.0
    n_rows = len(dataloader.dataset)
    softmax_loss = nn.SoftmaxLoss()

    n_batches = 0
    for x, y in dataloader:
        x = x.reshape((x.shape[0], -1))

        preds = model(x)
        loss = softmax_loss(preds, y)

        correct_preds += (preds.numpy().argmax(axis=1) == y.numpy()).sum()
        avg_loss += loss.numpy()[0]  # 1d tensor

        if opt:
            opt.zero_grad()
            # loss is a tensor
            loss.backward()
            opt.step()
        n_batches += 1

        logging.info(
            "Batch %d/%d | Loss: %.2f | Acc: %.2f",
            n_batches,
            n_rows // dataloader.batch_size,
            loss.numpy()[0],
            correct_preds / ((n_batches) * dataloader.batch_size),
        )

    acc = correct_preds / n_rows
    avg_loss /= n_batches

    return Accuracy(acc), Loss(avg_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLPResNet on MNIST")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training (default: 10)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay (default: 0.001)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=10, help="Hidden dimension size (default: 10)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam"],
        help="Optimizer to use (default: adam)",
    )
    parser.add_argument(
        "--log", type=str, default="ERROR", help="Logging level (default: ERROR)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    INPUT_DIM = 784

    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    hidden_dim = args.hidden_dim
    optimizer = args.optimizer

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.ERROR),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    train_filenames = [MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS]
    test_filenames = [MNISTPaths.TEST_IMAGES, MNISTPaths.TEST_LABELS]
    mnist_train = ndl.data.MNISTDataset(train_filenames[0], train_filenames[1])
    mnist_test = ndl.data.MNISTDataset(test_filenames[0], test_filenames[1])

    train_loader = ndl.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(mnist_test, batch_size=batch_size)

    model = MLPResNet(INPUT_DIM, hidden_dim)
    if not optimizer:
        optimizer = ndl.optim.Adam
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc, train_loss, test_acc, test_loss = 0.0, 0.0, 0.0, 0.0
    logging.info("Starting training...")
    for _epoch_idx in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        test_acc, test_loss = epoch(test_loader, model)
        logging.info(
            "Epoch %d/%d\n\t"
            "Train Acc: %.2f\n\tTrain Loss: %.2f\n\t"
            "Test Acc: %.2f\n\tTest Loss: %.2f\n",
            _epoch_idx + 1,
            epochs,
            train_acc,
            train_loss,
            test_acc,
            test_loss,
        )

    logging.info("Training completed. Evaluating final model performance...")
    logging.info(
        "Final Train Acc: %.2f\nFinal Train Loss: %.2f\n"
        "Final Test Acc: %.2f\nFinal Test Loss: %.2f",
        train_acc,
        train_loss,
        test_acc,
        test_loss,
    )
