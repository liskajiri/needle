import needle as ndl
import numpy as np
import pytest
from needle.backend_selection import NDArray
from needle.data.datasets.mnist import MNISTDataset, MNISTPaths

from tests.utils import set_random_seeds


@pytest.fixture(scope="module")
def mnist_train():
    return ndl.data.MNISTDataset(train=True)


@pytest.fixture(scope="module")
def mnist_test():
    return ndl.data.MNISTDataset(train=False)


@pytest.fixture(params=[1])
def train_dataloader(mnist_train, request) -> tuple[ndl.data.DataLoader, int]:
    batch_size = request.param
    return (
        ndl.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=False),
        batch_size,
    )


@pytest.fixture(params=[5])
def test_dataloader(mnist_test, request) -> tuple[ndl.data.DataLoader, int]:
    batch_size = request.param
    return (
        ndl.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False),
        batch_size,
    )


def test_mnist_dataset_stats(
    mnist_train: ndl.data.MNISTDataset, mnist_test: ndl.data.MNISTDataset
):
    train_X = mnist_train.x
    test_X = mnist_test.x

    assert train_X.dtype == "float32"
    assert mnist_train.y.dtype == np.uint8
    assert test_X.dtype == "float32"
    assert mnist_test.y.dtype == np.uint8

    # assert train_X.shape == (60000, 784)
    assert mnist_train.y.shape == (60000,)
    # assert test_X.shape == (10000, 784)
    assert mnist_test.y.shape == (10000,)

    np.testing.assert_allclose(np.linalg.norm(train_X[:10]), 27.892084)
    np.testing.assert_allclose(
        np.linalg.norm(train_X[:1000]),
        293.0717,
        err_msg="""If you failed this test but not the previous one,
        you are probably normalizing incorrectly. You should normalize
        w.r.t. the whole dataset, _not_ individual images.""",
        rtol=1e-6,
    )
    np.testing.assert_equal(mnist_train.y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])


def test_mnist_test_dataset_size(mnist_test):
    assert len(mnist_test) == 10000


def test_mnist_train_dataset_size(mnist_train):
    assert len(mnist_train) == 60000


@pytest.mark.parametrize(
    "test_id, indices, expected_norms, expected_labels",
    [
        (
            "standard_samples",
            [1, 42, 1000, 2000, 3000, 4000, 5000, 5005],
            [
                10.188792,
                6.261355,
                8.966858,
                9.4346485,
                9.086626,
                9.214664,
                10.208544,
                10.649756,
            ],
            [0, 7, 0, 5, 9, 7, 7, 8],
        )
    ],
    ids=["standard_samples"],
)
def test_train_dataset_samples(
    mnist_train, test_id, indices, expected_norms, expected_labels
):
    sample_norms = [np.linalg.norm(mnist_train[idx][0]) for idx in indices]
    sample_labels = [mnist_train[idx][1] for idx in indices]
    np.testing.assert_allclose(sample_norms, expected_norms)
    np.testing.assert_allclose(sample_labels, expected_labels)


def test_mnist_train_sample_norms_and_labels(mnist_train):
    sample_indices = [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
    sample_norms = np.array(
        [np.linalg.norm(mnist_train[idx][0]) for idx in sample_indices]
    )
    compare_against = np.array(
        [
            10.188792,
            6.261355,
            8.966858,
            9.4346485,
            9.086626,
            9.214664,
            10.208544,
            10.649756,
        ]
    )
    sample_labels = np.array([mnist_train[idx][1] for idx in sample_indices])
    compare_labels = np.array([0, 7, 0, 5, 9, 7, 7, 8])

    np.testing.assert_allclose(sample_norms, compare_against)
    np.testing.assert_allclose(sample_labels, compare_labels)


def test_mnist_test_sample_norms_and_labels(mnist_test):
    sample_indices = [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
    sample_norms = np.array(
        [np.linalg.norm(mnist_test[idx][0]) for idx in sample_indices]
    )
    compare_against = np.array(
        [
            9.857545,
            8.980832,
            8.57207,
            6.891522,
            8.192135,
            9.400087,
            8.645003,
            7.405202,
        ]
    )
    sample_labels = np.array([mnist_test[idx][1] for idx in sample_indices])
    compare_labels = np.array([2, 4, 9, 6, 6, 9, 3, 1])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)


@pytest.mark.slow
def test_mnist_normalization(
    mnist_train: ndl.data.MNISTDataset, mnist_test: ndl.data.MNISTDataset
):
    """Test that MNIST dataset values are properly normalized to [0.0, 1.0]"""

    # Check train data normalization
    train_min = mnist_train.x.numpy().min()
    train_max = mnist_train.x.numpy().max()

    # Check test data normalization
    test_min = mnist_test.x.numpy().min()
    test_max = mnist_test.x.numpy().max()

    # Assert values are properly normalized to [0.0, 1.0] range
    # MNIST has those values
    np.testing.assert_allclose(train_min, 0.0, atol=1e-6)
    np.testing.assert_allclose(train_max, 1.0, atol=1e-6)
    np.testing.assert_allclose(test_min, 0.0, atol=1e-6)
    np.testing.assert_allclose(test_max, 1.0, atol=1e-6)


SAMPLE_INDICES = [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
EXPECTED_LABELS = [0, 7, 0, 5, 9, 7, 7, 8]


# TODO: tests for only transforms with random data
def test_mnist_train_crop28_flip():
    """Test MNIST dataset with RandomCrop(28) + RandomFlip"""
    # Reset all random states
    set_random_seeds(0)

    transforms = [ndl.data.RandomCrop(28), ndl.data.RandomFlipHorizontal()]

    expected_norms = [
        0.0,
        0.0,
        8.966858,
        0.0,
        9.086626,
        6.415914,
        0.0,
        3.090034,
    ]

    dataset = ndl.data.MNISTDataset(train=True, transforms=transforms)
    dataset.x = dataset.x.reshape((-1, 28, 28, 1))
    sample_norms = [np.linalg.norm(dataset[idx][0]) for idx in SAMPLE_INDICES]
    sample_labels = [dataset[idx][1] for idx in SAMPLE_INDICES]

    np.testing.assert_allclose(sample_norms, expected_norms, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, EXPECTED_LABELS)


def test_mnist_train_crop12_flip():
    """Test MNIST dataset with RandomCrop(12) + RandomFlip(0.4)"""
    set_random_seeds(0)

    transforms = [ndl.data.RandomCrop(12), ndl.data.RandomFlipHorizontal(0.4)]

    expected_norms = [
        8.231772,
        5.044336,
        8.966858,
        9.434648,
        8.686448,
        7.313684,
        9.727224,
        9.565062,
    ]

    dataset = ndl.data.MNISTDataset(train=True, transforms=transforms)
    dataset.x = dataset.x.reshape((-1, 28, 28, 1))
    sample_norms = [np.linalg.norm(dataset[idx][0]) for idx in SAMPLE_INDICES]
    sample_labels = [dataset[idx][1] for idx in SAMPLE_INDICES]

    np.testing.assert_allclose(sample_norms, expected_norms, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, EXPECTED_LABELS)


@pytest.mark.slow
def test_train_dataloader(
    mnist_train: ndl.data.MNISTDataset, train_dataloader: ndl.data.DataLoader
):
    loader, batch_size = train_dataloader
    for i, batch in enumerate(loader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        truth = mnist_train[i * batch_size : (i + 1) * batch_size]
        if isinstance(truth[0], NDArray):
            truth_x = truth[0].numpy()
            truth_x = truth_x if truth_x.shape[0] > 1 else truth_x.reshape(-1)
        else:
            truth_x = truth[0] if truth[0].shape[0] > 1 else truth[0].reshape(-1)
        truth_y = truth[1] if truth[1].shape[0] > 1 else truth[1].reshape(-1)

        np.testing.assert_allclose(truth_x, batch_x.flatten())
        np.testing.assert_allclose(batch_y, truth_y)


@pytest.mark.slow
def test_test_dataloader(
    mnist_test: ndl.data.MNISTDataset, test_dataloader: ndl.data.DataLoader
):
    loader, batch_size = test_dataloader
    for i, batch in enumerate(loader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        truth = mnist_test[i * batch_size : (i + 1) * batch_size]
        truth_x = truth[0]
        truth_y = truth[1]

        np.testing.assert_allclose(truth_x, batch_x)
        np.testing.assert_allclose(batch_y, truth_y)


# TODO: investigate what tests should be marked slow (< 1s is ok?)


@pytest.mark.slow
def test_shuffle(mnist_test: ndl.data.MNISTDataset):
    not_shuffled = ndl.data.DataLoader(dataset=mnist_test, batch_size=10, shuffle=False)
    shuffled = ndl.data.DataLoader(dataset=mnist_test, batch_size=10, shuffle=True)
    for i, j in zip(shuffled, not_shuffled, strict=True):
        assert i != j, "Shuffling had no effect on the dataloader."


def test_softmax_loss_ndl():
    _X, y = MNISTDataset.parse_mnist(MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS)
    np.random.seed(0)
    Z = ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
    y = ndl.Tensor(y)
    softmax = ndl.nn.SoftmaxLoss()

    np.testing.assert_allclose(
        softmax(Z, y).numpy(),
        2.3025850,
        rtol=1e-6,
        atol=1e-6,
    )
    Z = ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
    np.testing.assert_allclose(softmax(Z, y).numpy(), 2.7291998, rtol=1e-6, atol=1e-6)
