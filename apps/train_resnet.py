import argparse
import logging

import needle as ndl
from models.resnet9 import MLPResNet, ResNet9

from apps.train_utils import epoch

try:
    from tqdm import tqdm  # type: ignore

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLPResNet on MNIST")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10"],
        help="Dataset to use (default: cifar10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
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
        "--hidden-dim",
        type=int,
        default=100,
        help="Hidden dimension size (default: 10)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Optimizer to use (default: adam)",
    )
    parser.add_argument(
        "--log", type=str, default="ERROR", help="Logging level (default: ERROR)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    hidden_dim = args.hidden_dim
    optimizer = args.optimizer
    logging_level = getattr(logging, args.log.upper(), logging.ERROR)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    if args.dataset == "mnist":
        train_dataset = ndl.data.MNISTDataset(train=True)
        test_dataset = ndl.data.MNISTDataset(train=False)
        data_shape = train_dataset.x.shape[1]
        model = MLPResNet(input_dim=data_shape, hidden_dim=hidden_dim)
    elif args.dataset == "cifar10":
        train_dataset = ndl.data.datasets.CIFAR10Dataset(train=True)
        test_dataset = ndl.data.datasets.CIFAR10Dataset(train=False)
        model = ResNet9(in_features=3, out_features=10)

    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    if args.optimizer == "sgd":
        optimizer = ndl.optim.SGD
    elif args.optimizer == "adam":
        optimizer = ndl.optim.Adam

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc, train_loss, test_acc, test_loss = 0.0, 0.0, 0.0, 0.0
    logging.info(f"Training Resnet on {args.dataset} dataset...")
    logging.info(f"Started training for {epochs} epochs...")

    training_bar = (
        tqdm(range(epochs), desc="Epochs ") if TQDM_AVAILABLE else range(epochs)  # type: ignore
    )
    for _epoch_idx in training_bar:
        train_acc, train_loss = epoch(train_dataloader, model, opt, mode="train")
        test_acc, test_loss = epoch(test_dataloader, model, mode="test")

        if not TQDM_AVAILABLE:
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
        "Final Train Acc: %.2f | Final Train Loss: %.2f\n"
        "Final Test Acc: %.2f | Final Test Loss: %.2f",
        train_acc,
        train_loss,
        test_acc,
        test_loss,
    )
