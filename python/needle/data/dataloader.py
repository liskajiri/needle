from __future__ import annotations

import random
from typing import TYPE_CHECKING

from needle.backend_selection import NDArray, default_device
from needle.tensor import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from needle.data.dataset import Dataset
    from needle.typing import AbstractBackend, BatchType


class DataLoader:
    """
    Data loader.
    Combines a dataset and a sampler, and provides an iterable over the given dataset.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """

    def __init__(
        self,
        dataset: Dataset[NDArray],
        batch_size: int = 1,
        device: AbstractBackend = default_device,
        shuffle: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if len(dataset) == 0:
            raise ValueError("dataset cannot be empty")

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device

    def __iter__(self) -> Iterator[BatchType]:
        """
        Returns an iterator over the dataset.

        Yields:
            Iterator[BatchType]: Batch of data.
        """
        if self.shuffle:
            indices = random.sample(range(len(self.dataset)), len(self.dataset))
        else:
            indices = list(range(len(self.dataset)))

        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            dataset_batch = self.dataset[batch_indices]
            yield tuple(Tensor(i, device=self.device) for i in dataset_batch)  # type: ignore

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
