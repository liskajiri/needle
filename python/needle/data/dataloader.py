from collections.abc import Iterator
from typing import Self, cast

import numpy as np

from needle.backend_ndarray.ndarray import NDArray
from needle.data.dataset import Dataset
from needle.tensor import Tensor

BatchType = tuple[Tensor, ...]


class DataLoader(Iterator[BatchType]):
    r"""Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

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

        self.ordering: list[NDArray] = []
        self.index: int = 0

    def __iter__(self) -> Self:
        if self.shuffle:
            orders = np.random.permutation(len(self.dataset))
        else:
            orders = np.arange(len(self.dataset))

        self.ordering = np.array_split(
            orders,
            range(self.batch_size, len(self.dataset), self.batch_size),
        )
        # reset iteration start at every epoch
        self.index = 0

        return self

    def __next__(self) -> BatchType:
        if self.index >= len(self.ordering):
            raise StopIteration

        indices = self.ordering[self.index]
        self.index += 1

        return cast(BatchType, tuple(Tensor(i) for i in self.dataset[indices]))
