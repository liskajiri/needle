import needle as ndl
import pytest
from models.resnet9 import MLPResNet
from needle import nn
from needle.data.datasets.synthetic_mnist import SyntheticMNIST

from apps.train_utils import epoch

# Benchmark configuration
IMAGE_DIMENSION = 14
NUM_CLASSES = 10
BATCH_SIZE = 128
HIDDEN_SIZE = 64
LR = 0.01
TRAIN_SAMPLES = 200

simple_mlp = nn.Sequential(
    nn.Flatten(),  # (N, C*H*W)
    nn.Linear(IMAGE_DIMENSION * IMAGE_DIMENSION, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
)

resnet = nn.Sequential(
    nn.Flatten(),
    MLPResNet(
        IMAGE_DIMENSION * IMAGE_DIMENSION,
        hidden_dim=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
    ),
)


@pytest.mark.parametrize("optimizer", [ndl.optim.SGD, ndl.optim.Adam])
@pytest.mark.parametrize("model", [simple_mlp, resnet], ids=["mlp", "resnet"])
def test_training_epoch(benchmark, optimizer, model):
    """Benchmark a single epoch of training on the synthetic MNIST dataset.

    The benchmark measures the time to run one epoch (train mode) over a
    SyntheticMNIST dataset. It mirrors other repository benchmarks but uses the
    fresh synthetic dataset so it does not require external downloads.
    """
    ds = SyntheticMNIST(
        num_samples=TRAIN_SAMPLES,
        num_classes=NUM_CLASSES,
        image_shape=(1, IMAGE_DIMENSION, IMAGE_DIMENSION),
        seed=42,
    )
    dataloader = ndl.data.DataLoader(dataset=ds, batch_size=BATCH_SIZE)

    opt = optimizer(model.parameters(), lr=LR)

    # Run and measure one epoch
    benchmark(epoch, dataloader, model, opt, mode="train")
