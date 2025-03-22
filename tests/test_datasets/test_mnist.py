import needle as ndl
import numpy as np
import pytest
from needle.data.datasets.mnist import MNISTPaths


@pytest.fixture(scope="module")
def mnist_train():
    return ndl.data.MNISTDataset(MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS)


@pytest.fixture(scope="module")
def mnist_test():
    return ndl.data.MNISTDataset(MNISTPaths.TEST_IMAGES, MNISTPaths.TEST_LABELS)


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
    train_X = mnist_train.X
    test_X = mnist_test.X

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
    ("indices", "expected_norms", "expected_labels"),
    [
        (
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
)
def test_train_dataset_samples(mnist_train, indices, expected_norms, expected_labels):
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


SAMPLE_INDICES = [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
EXPECTED_LABELS = [0, 7, 0, 5, 9, 7, 7, 8]


def test_mnist_train_crop28_flip():
    """Test MNIST dataset with RandomCrop(28) + RandomFlip"""
    # Reset all random states
    np.random.seed(0)

    transforms = [ndl.data.RandomCrop(28), ndl.data.RandomFlipHorizontal()]

    expected_norms = [
        2.0228338,
        0.0,
        7.4892044,
        0.0,
        0.0,
        3.8012788,
        9.583429,
        4.2152724,
    ]

    dataset = ndl.data.MNISTDataset(
        MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS, transforms=transforms
    )
    sample_norms = [np.linalg.norm(dataset[idx][0]) for idx in SAMPLE_INDICES]
    sample_labels = [dataset[idx][1] for idx in SAMPLE_INDICES]

    np.testing.assert_allclose(sample_norms, expected_norms, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, EXPECTED_LABELS)


# TODO: BREAKS if randomness is set
# def test_mnist_train_crop12_flip():
#     """Test MNIST dataset with RandomCrop(12) + RandomFlip(0.4)"""

#     transforms = [ndl.data.RandomCrop(12), ndl.data.RandomFlipHorizontal(0.4)]

#     expected_norms = [
#         5.369537,
#         5.5454974,
#         8.966858,
#         7.547235,
#         8.785921,
#         7.848442,
#         7.1654058,
#         9.361828,
#     ]

#     dataset = ndl.data.MNISTDataset(
#         MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS, transforms=transforms
#     )
#     sample_norms = [np.linalg.norm(dataset[idx][0]) for idx in SAMPLE_INDICES]
#     sample_labels = [dataset[idx][1] for idx in SAMPLE_INDICES]

#     np.testing.assert_allclose(sample_norms, expected_norms, rtol=1e-5, atol=1e-5)
#     np.testing.assert_allclose(sample_labels, EXPECTED_LABELS)


@pytest.mark.slow
def test_train_dataloader(
    mnist_train: ndl.data.MNISTDataset, train_dataloader: ndl.data.DataLoader
):
    loader, batch_size = train_dataloader
    for i, batch in enumerate(loader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        truth = mnist_train[i * batch_size : (i + 1) * batch_size]
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


@pytest.mark.slow
def test_shuffle(mnist_test: ndl.data.MNISTDataset):
    not_shuffled = ndl.data.DataLoader(dataset=mnist_test, batch_size=10, shuffle=False)
    shuffled = ndl.data.DataLoader(dataset=mnist_test, batch_size=10, shuffle=True)
    for i, j in zip(shuffled, not_shuffled, strict=True):
        assert i != j, "Shuffling had no effect on the dataloader."


def test_dataloader_ndarray():
    for batch_size in [1, 10, 100]:
        np.random.seed(0)

        train_dataset = ndl.data.NDArrayDataset(np.random.rand(100, 10, 10))
        train_dataloader = ndl.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=False
        )

        for i, batch in enumerate(train_dataloader):
            batch_x = batch[0].numpy()
            truth_x = train_dataset[i * batch_size : (i + 1) * batch_size][0].reshape(
                (
                    batch_size,
                    10,
                    10,
                )
            )
            np.testing.assert_allclose(truth_x, batch_x)

    batch_size = 1
    np.random.seed(0)
    train_dataset = ndl.data.NDArrayDataset(
        np.arange(
            100,
        )
    )
    train_dataloader = iter(
        ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    )

    elements = np.array([next(train_dataloader)[0].numpy().item() for _ in range(10)])
    np.testing.assert_allclose(
        elements, np.array([26, 86, 2, 55, 75, 93, 16, 73, 54, 95])
    )

    batch_size = 10
    train_dataset = ndl.data.NDArrayDataset(
        np.arange(
            100,
        )
    )
    train_dataloader = iter(
        ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    )

    elements = np.array(
        [np.linalg.norm(next(train_dataloader)[0].numpy()) for _ in range(10)]
    )
    np.testing.assert_allclose(
        elements,
        np.array(
            [
                164.805946,
                173.43875,
                169.841102,
                189.050258,
                195.880065,
                206.387984,
                209.909504,
                185.776748,
                145.948621,
                160.252925,
            ]
        ),
    )
