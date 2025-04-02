import needle as ndl
import numpy as np
import pytest

from tests.utils import set_random_seeds

rng = np.random.default_rng(0)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_dataloader_batch(batch_size):
    arr = rng.standard_normal((100, 10, 10))
    train_dataset = ndl.data.NDArrayDataset(arr)
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    for i, batch in enumerate(train_dataloader):
        batch_x = batch[0].numpy()
        truth_x = train_dataset[i * batch_size : (i + 1) * batch_size][0].reshape((
            batch_size,
            10,
            10,
        ))
        np.testing.assert_allclose(truth_x, batch_x)


@pytest.mark.parametrize(
    "batch_size, expected_values, value_extractor",
    [
        (
            1,
            np.array([26.0, 86.0, 2.0, 55.0, 75.0, 93.0, 16.0, 73.0, 54.0, 95.0]),
            lambda batch: batch[0].numpy().item(),
        ),
        (
            10,
            np.array([
                207.12556,
                143.83324,
                193.41406,
                197.38034,
                103.75934,
                144.15616,
                195.77794,
                218.46281,
                180.46883,
                195.50447,
            ]),
            lambda batch: np.linalg.norm(batch[0].numpy()),
        ),
    ],
    ids=["single_item_shuffle", "batch_shuffle_norm"],
)
def test_dataloader_ndarray(batch_size, expected_values, value_extractor):
    """Test that DataLoader properly shuffles data and maintains expected values"""
    set_random_seeds(0)

    train_dataset = ndl.data.NDArrayDataset(np.arange(100))
    train_dataloader = iter(
        ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    )

    # Extract the first 10 values using the provided extractor function
    elements = np.array([value_extractor(next(train_dataloader)) for _ in range(10)])

    np.testing.assert_allclose(elements, expected_values, rtol=1e-6)
