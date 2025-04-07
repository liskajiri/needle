import needle as ndl
import pytest
from models.resnet9 import MLPResNet
from needle.data.datasets.mnist import MNISTPaths

from apps.resnet_mnist import epoch

INPUT_DIM = 784
HIDDEN_DIM = 10


@pytest.fixture(scope="session")
def mnist_train(train_filenames=[MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS]):
    """Create MNIST training dataset"""
    return ndl.data.MNISTDataset(train_filenames[0], train_filenames[1])


@pytest.fixture(scope="session")
def mnist_test(test_filenames=[MNISTPaths.TEST_IMAGES, MNISTPaths.TEST_LABELS]):
    """Create MNIST test dataset"""
    return ndl.data.MNISTDataset(test_filenames[0], test_filenames[1])


@pytest.fixture
def batch_size():
    """Create batch size for testing"""
    return 128


@pytest.fixture
def train_loader(mnist_train, batch_size):
    """Create training data loader with different batch sizes"""
    return ndl.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)


@pytest.fixture
def test_loader(mnist_test, batch_size):
    """Create test data loader with different batch sizes"""
    return ndl.data.DataLoader(mnist_test, batch_size=batch_size)


@pytest.fixture(scope="session")
def model():
    """Create a model for testing"""
    return MLPResNet(INPUT_DIM, HIDDEN_DIM)


@pytest.fixture(params=["adam", "sgd"])
def optimizer(request, model):
    """Create an optimizer for testing"""
    lr = 0.01
    weight_decay = 0.001
    if request.param == "adam":
        return ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return ndl.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)


@pytest.mark.benchmark(
    min_rounds=2,
    disable_gc=True,
    warmup=True,
    warmup_iterations=1,
)
def test_train_epoch(benchmark, train_loader, model, optimizer):
    benchmark(epoch, train_loader, model, optimizer)


@pytest.mark.benchmark(
    min_rounds=3,
    disable_gc=True,
    warmup=True,
    warmup_iterations=1,
)
def test_test_epoch(benchmark, test_loader, model, optimizer):
    benchmark(epoch, test_loader, model, optimizer)
