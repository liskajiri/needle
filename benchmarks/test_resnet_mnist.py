import needle as ndl
import pytest
from models.resnet9 import MLPResNet
from needle.data.datasets.mnist import MNISTPaths

from apps.resnet_mnist import epoch

INPUT_DIM = 784
HIDDEN_DIM = 10
BATCH_SIZE = 128
LR = 0.01
WEIGHT_DECAY = 0.1


try:
    mnist_train = ndl.data.MNISTDataset(
        MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS
    )
    mnist_test = ndl.data.MNISTDataset(MNISTPaths.TEST_IMAGES, MNISTPaths.TEST_LABELS)
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
@pytest.mark.skip("MNIST benchmark test too slow for now.")
def test_mnist_epoch(benchmark, dataloader, mode, optimizer) -> None:
    model = MLPResNet(INPUT_DIM, HIDDEN_DIM)
    optimizer = optimizer(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    benchmark(epoch, dataloader, model, optimizer, mode)
