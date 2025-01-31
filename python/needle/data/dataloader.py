from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from needle.backend_selection import array_api
from needle.tensor import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from needle.data.dataset import Dataset
    from needle.typing import BatchType


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
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if len(dataset) == 0:
            raise ValueError("dataset cannot be empty")

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[BatchType]:
        """
        Returns an iterator over the dataset.

        Yields:
            Iterator[BatchType]: Batch of data.
        """
        if self.shuffle:
            orders = np.random.permutation(len(self.dataset))
        else:
            orders = np.arange(len(self.dataset))

        ordering = array_api.array_split(
            orders, range(self.batch_size, len(self.dataset), self.batch_size)
        )
        for indices in ordering:
            yield tuple(Tensor(i) for i in self.dataset[indices])
