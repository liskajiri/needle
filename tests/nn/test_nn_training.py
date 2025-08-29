import needle as ndl
import pytest
from needle import nn
from needle.data.datasets.synthetic_mnist import SyntheticMNIST
from needle.optim import SGD, Adam
from needle.optim.base import Optimizer

from apps.train_utils import epoch


@pytest.mark.parametrize("optimizer", [SGD, Adam])
def test_synthetic_mnist_training(
    optimizer: type[Optimizer],
    input_shape: tuple[int, int, int] = (1, 28, 28),
    num_classes: int = 10,
    num_samples: int = 100,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_epochs: int = 10,
) -> None:
    """
    Train a small network on SyntheticMNIST and ensure training reduces loss.
    """

    dataset = SyntheticMNIST(
        num_samples=num_samples,
        num_classes=num_classes,
        image_shape=input_shape,
        seed=42,
    )
    dataloader = ndl.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_shape[1] * input_shape[2], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    )

    # Evaluate initial loss (no optimizer => eval mode in epoch)
    model.eval()
    initial_acc, initial_loss = epoch(dataloader, model, opt=None)

    opt = optimizer(model.parameters(), lr=0.1)  # type: ignore
    model.train()

    curr_loss = -1
    curr_acc = 0

    for _epoch in range(num_epochs):
        curr_acc, curr_loss = epoch(dataloader, model, opt)

        print(f"Epoch {_epoch + 1}: Loss = {curr_loss:.4f}, Acc = {curr_acc:.4f}")

    assert float(curr_loss) <= float(initial_loss), (
        "Training did not reduce loss on synthetic dataset as expected"
    )
    assert float(curr_acc) > float(initial_acc), (
        "Training did not improve accuracy on synthetic dataset as expected"
    )
