import needle as ndl
import numpy as np
import pytest
from needle import backend_ndarray as nd
from needle.backend_selection import NDArray
from needle.data.datasets import CIFAR10Dataset, CIFARPath

from tests.devices import all_devices

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
    X, _y = example
    assert isinstance(X, NDArray)
    assert X.shape == (3, 32, 32)


BATCH_SIZES = [1, 15]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("train", TRAIN)
@all_devices()
def test_cifar10_loader(batch_size, train, device):
    cifar10_train_dataset = CIFAR10Dataset(CIFARPath, train=train)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, batch_size, device=device)
    for _X, _y in train_loader:
        assert isinstance(_X.cached_data, nd.NDArray)
        assert isinstance(_X, ndl.Tensor)
        assert isinstance(_y, ndl.Tensor)
        assert _X.dtype == "float32"
        break
