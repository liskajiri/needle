import needle as ndl
import pytest
from models.resnet9 import MLPResNet
from needle.data.datasets.artificial_mnist import ArtificialMNIST

from apps.resnet_mnist import epoch

IMAGE_DIMENSION = 28
HIDDEN_DIM = 10
NUM_CLASSES = 10
BATCH_SIZE = 128
LR = 0.01
WEIGHT_DECAY = 0.1
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000


def create_datasets(image_dim: int):
    """Create train and test datasets with specified dimensions."""
    input_dim = image_dim * image_dim
    return (
        ArtificialMNIST(
            num_samples=TRAIN_SAMPLES, image_dim=image_dim, num_classes=NUM_CLASSES
        ),
        ArtificialMNIST(
            num_samples=TEST_SAMPLES, image_dim=image_dim, num_classes=NUM_CLASSES
        ),
        input_dim,
    )


@pytest.mark.parametrize(
    "dataset_type",
    ["train", "test"],
    ids=["train", "test"],
)
@pytest.mark.parametrize("optimizer", [ndl.optim.Adam, ndl.optim.SGD])
def test_artificial_mnist_epoch(
    benchmark, dataset_type, optimizer, image_dim=IMAGE_DIMENSION
) -> None:
    """Benchmark training loop on artificial MNIST-like dataset.

    This test measures performance using datasets of different dimensions, where
    each image's label corresponds to its pixel density (label 0 = all zeros,
    label 9 = all ones, labels 1-8 = proportional random pixels).

    Args:
        image_dim: Size of images (image_dim x image_dim)
        dataset_type: Whether to use training or test dataset
        optimizer: Optimizer class to use (Adam or SGD)
    """
    train_data, test_data, input_dim = create_datasets(image_dim)

    # Select dataset based on type
    dataset = train_data if dataset_type == "train" else test_data
    dataloader = ndl.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    # Create model and optimizer
    model = MLPResNet(input_dim, HIDDEN_DIM)
    optimizer = optimizer(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    benchmark(epoch, dataloader, model, optimizer, dataset_type)
