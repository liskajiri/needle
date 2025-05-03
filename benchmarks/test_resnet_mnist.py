import os

import needle as ndl
import pytest
from models.resnet9 import MLPResNet

from apps.train_utils import epoch

INPUT_DIM = 784
HIDDEN_DIM = 10
BATCH_SIZE = 128
LR = 0.01
WEIGHT_DECAY = 0.1


try:
    mnist_train = ndl.data.MNISTDataset(train=True)
    mnist_test = ndl.data.MNISTDataset(train=False)
except FileNotFoundError:
    pytest.skip(
        "MNIST dataset not found. Please download the dataset.",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "dataloader, mode",
    [
        pytest.param(
            ndl.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True),
            "train",
        ),
        pytest.param(ndl.data.DataLoader(mnist_test, batch_size=BATCH_SIZE), "test"),
    ],
    ids=["train", "test"],
)
@pytest.mark.parametrize("optimizer", [ndl.optim.Adam, ndl.optim.SGD])
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Benchmark skipped in CI")
def test_mnist_epoch(benchmark, dataloader, mode, optimizer) -> None:
    model = MLPResNet(INPUT_DIM, HIDDEN_DIM)
    optimizer = optimizer(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if mode == "test":
        optimizer = None
    benchmark(epoch, dataloader, model, optimizer, mode=mode)
