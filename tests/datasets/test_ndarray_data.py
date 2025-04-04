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
            np.array([23.0, 8.0, 11.0, 7.0, 48.0, 13.0, 1.0, 91.0, 94.0, 54.0]),
            lambda batch: batch[0].numpy().item(),
        ),
        (
            10,
            np.array([
                152.54507,
                187.70189,
                122.62544,
                214.64389,
                165.87044,
                201.39514,
                212.88495,
                187.70456,
                168.95265,
                177.67386,
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
