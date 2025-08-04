import needle as ndl
import numpy as np
import pytest

rng = np.random.default_rng(0)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.skip("Skipping due to incomplete test implementation")
def test_dataloader_batch(batch_size):
    arr = rng.standard_normal((100, 10, 10))
    arr = ndl.NDArray(arr)
    y = rng.standard_normal((100, 10, 10))
    y = ndl.NDArray(y)

    train_dataset = ndl.data.NDArrayDataset(arr, y=y)
    _train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    #     for i, batch in enumerate(train_dataloader):
    #         batch_x = batch[0].numpy()
    #         start_idx = i * batch_size
    #         end_idx = min(start_idx + batch_size, len(train_dataset))
    #         truth_x = train_dataset[start_idx:end_idx][0]
    #         np.testing.assert_allclose(truth_x, batch_x)

    # @pytest.mark.parametrize(
    #     "batch_size, expected_values, value_extractor",
    #     [
    #         (
    #             1,
    #             np.array([49.0, 97.0, 53.0, 5.0, 33.0, 65.0, 62.0, 51.0, 38.0, 61.0]),
    #             lambda batch: batch[0].numpy().item(),
    #         ),
    #         (
    #             10,
    #             np.array([
    #                 177.67386,
    #                 168.95265,
    #                 187.70456,
    #                 212.88495,
    #                 201.39514,
    #                 165.87044,
    #                 214.64389,
    #                 122.62544,
    #                 187.70189,
    #                 152.54507,
    #             ]),
    #             lambda batch: np.linalg.norm(batch[0].numpy()),
    #         ),
    #     ],
    #     ids=["single_item_shuffle", "batch_shuffle_norm"],
    # )
    # def test_dataloader_ndarray(batch_size, expected_values, value_extractor):
    #     """Test DataLoader shuffling and value consistency"""
    #     set_random_seeds(0)

    x = ndl.NDArray(np.arange(100))
    y = ndl.NDArray(np.arange(100))
    train_dataset = ndl.data.NDArrayDataset(x, y=y)
    _train_dataloader = iter(
        ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    )


#     elements = np.array([value_extractor(next(train_dataloader)) for _ in range(10)])
#     np.testing.assert_allclose(elements, expected_values, rtol=1e-6)
