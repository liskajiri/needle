import needle as ndl
import numpy as np
import pytest

from needle.backend_ndarray.ndarray import NDArray

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes


@pytest.mark.proptest
@given(arrays(dtype=np.float32, shape=(5, 5, 5), elements=st.floats(0, 1)))
def test_flip_horizontal_hypothesis(a: NDArray):
    transform = ndl.data.RandomFlipHorizontal(p=1)

    np.random.seed(0)
    b = np.flip(a, axis=1)

    result = transform(a)

    np.testing.assert_allclose(result, b)

    # The transform should be reversible by applying it twice
    double_transform = transform(result)
    np.testing.assert_allclose(double_transform, a)


# ==============================


def numpy_crop(img: NDArray, padding: int = 3) -> NDArray:
    """Zero pad and then randomly crop an image.
    Args:
            img: H x W x C NDArray of an image
    Return
        H x W x C NDArray of clipped image
    Note: generate the image shifted by shift_x, shift_y specified below
    """
    assert img.ndim == 3
    shift_x, shift_y = np.random.randint(low=-padding, high=padding + 1, size=2)
    H, W, C = img.shape

    H += 2 * padding
    W += 2 * padding

    img_padded = np.zeros((H, W, C))
    # copy img to the padded array
    img_padded[padding : H - padding, padding : W - padding, :] = img

    return img_padded[
        padding + shift_x : H - padding + shift_x,
        padding + shift_y : W - padding + shift_y,
        :,
    ]


@pytest.mark.proptest
@given(
    arrays(
        dtype=np.float32,
        shape=array_shapes(min_dims=3, max_dims=3),
        elements=st.floats(0, 1),
    ),
    st.integers(1, 8),
)
def test_random_crop_hypothesis(a: NDArray, padding: int):
    transform = ndl.data.RandomCrop(padding)

    # set seeds to ensure reproducibility
    np.random.seed(0)
    b = numpy_crop(a, padding)
    np.random.seed(0)

    np.testing.assert_allclose(transform(a), b)


def test_mnist_dataset():
    # Test dataset sizing
    mnist_train_dataset = ndl.data.MNISTDataset(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    assert len(mnist_train_dataset) == 60000

    sample_norms = np.array(
        [
            np.linalg.norm(mnist_train_dataset[idx][0])
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
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
    sample_labels = np.array(
        [
            mnist_train_dataset[idx][1]
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
    )
    compare_labels = np.array([0, 7, 0, 5, 9, 7, 7, 8])

    np.testing.assert_allclose(sample_norms, compare_against)
    np.testing.assert_allclose(sample_labels, compare_labels)

    mnist_train_dataset = ndl.data.MNISTDataset(
        "data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"
    )
    assert len(mnist_train_dataset) == 10000

    sample_norms = np.array(
        [
            np.linalg.norm(mnist_train_dataset[idx][0])
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
    )
    compare_against = np.array(
        [9.857545, 8.980832, 8.57207, 6.891522, 8.192135, 9.400087, 8.645003, 7.405202]
    )
    sample_labels = np.array(
        [
            mnist_train_dataset[idx][1]
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
    )
    compare_labels = np.array([2, 4, 9, 6, 6, 9, 3, 1])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)

    # test a transform
    np.random.seed(0)
    tforms = [ndl.data.RandomCrop(28), ndl.data.RandomFlipHorizontal()]
    mnist_train_dataset = ndl.data.MNISTDataset(
        "data/train-images-idx3-ubyte.gz",
        "data/train-labels-idx1-ubyte.gz",
        transforms=tforms,
    )

    sample_norms = np.array(
        [
            np.linalg.norm(mnist_train_dataset[idx][0])
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
    )
    compare_against = np.array(
        [2.0228338, 0.0, 7.4892044, 0.0, 0.0, 3.8012788, 9.583429, 4.2152724]
    )
    sample_labels = np.array(
        [
            mnist_train_dataset[idx][1]
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
    )
    compare_labels = np.array([0, 7, 0, 5, 9, 7, 7, 8])

    np.testing.assert_allclose(sample_norms, compare_against)
    np.testing.assert_allclose(sample_labels, compare_labels)

    # test a transform
    tforms = [ndl.data.RandomCrop(12), ndl.data.RandomFlipHorizontal(0.4)]
    mnist_train_dataset = ndl.data.MNISTDataset(
        "data/train-images-idx3-ubyte.gz",
        "data/train-labels-idx1-ubyte.gz",
        transforms=tforms,
    )
    sample_norms = np.array(
        [
            np.linalg.norm(mnist_train_dataset[idx][0])
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
    )
    compare_against = np.array(
        [
            5.369537,
            5.5454974,
            8.966858,
            7.547235,
            8.785921,
            7.848442,
            7.1654058,
            9.361828,
        ]
    )
    sample_labels = np.array(
        [
            mnist_train_dataset[idx][1]
            for idx in [1, 42, 1000, 2000, 3000, 4000, 5000, 5005]
        ]
    )
    compare_labels = np.array([0, 7, 0, 5, 9, 7, 7, 8])

    np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sample_labels, compare_labels)


# TODO: Speed up this test
@pytest.mark.slow
def test_dataloader_mnist():
    batch_size = 1
    mnist_train_dataset = ndl.data.MNISTDataset(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    mnist_train_dataloader = ndl.data.DataLoader(
        dataset=mnist_train_dataset, batch_size=batch_size, shuffle=False
    )

    for i, batch in enumerate(mnist_train_dataloader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        truth = mnist_train_dataset[i * batch_size : (i + 1) * batch_size]
        truth_x = truth[0] if truth[0].shape[0] > 1 else truth[0].reshape(-1)
        truth_y = truth[1] if truth[1].shape[0] > 1 else truth[1].reshape(-1)

        np.testing.assert_allclose(truth_x, batch_x.flatten())
        np.testing.assert_allclose(batch_y, truth_y)

    batch_size = 5
    mnist_test_dataset = ndl.data.MNISTDataset(
        "data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"
    )
    mnist_test_dataloader = ndl.data.DataLoader(
        dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False
    )

    for i, batch in enumerate(mnist_test_dataloader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        truth = mnist_test_dataset[i * batch_size : (i + 1) * batch_size]
        truth_x = truth[0]
        truth_y = truth[1]

        np.testing.assert_allclose(truth_x, batch_x)
        np.testing.assert_allclose(batch_y, truth_y)

    noshuf = ndl.data.DataLoader(
        dataset=mnist_test_dataset, batch_size=10, shuffle=False
    )
    shuf = ndl.data.DataLoader(dataset=mnist_test_dataset, batch_size=10, shuffle=True)
    diff = False
    for i, j in zip(shuf, noshuf):
        if i != j:
            diff = True
            break
    assert diff, "shuffling had no effect on the dataloader."


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
                (batch_size, 10, 10)
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


if __name__ == "__main__":
    test_flip_horizontal_hypothesis()
    test_random_crop_hypothesis()
    test_mnist_dataset()
    test_dataloader_mnist()
    test_dataloader_ndarray()
