import needle as ndl
import numpy as np
import pytest
from needle import backend_ndarray as nd
from needle.backend_selection import NDArray
from needle.data.datasets import CIFAR10Dataset, CIFARPath

np.random.seed(2)


_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

TRAIN = [True, False]


@pytest.mark.parametrize("train", TRAIN)
def test_cifar10_dataset(train):
    dataset = CIFAR10Dataset(CIFARPath, train=train)
    if train:
        assert len(dataset) == 50000
    else:
        assert len(dataset) == 10000
    example = dataset[np.random.randint(len(dataset))]
    assert isinstance(example, tuple)
    X, y = example
    assert isinstance(X, NDArray)
    assert X.shape == (3, 32, 32)


BATCH_SIZES = [1, 15]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_cifar10_loader(batch_size, train, device):
    cifar10_train_dataset = CIFAR10Dataset(CIFARPath, train=train)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, batch_size)
    for X, y in train_loader:
        break
    assert isinstance(X.cached_data, nd.NDArray)
    assert isinstance(X, ndl.Tensor)
    assert isinstance(y, ndl.Tensor)
    assert X.dtype == "float32"
