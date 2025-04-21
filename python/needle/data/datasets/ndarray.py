from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_selection import NDArray
from needle.data.dataset import Dataset

if TYPE_CHECKING:
    from needle.typing import IndexType


class NDArrayDataset(Dataset[NDArray]):
    def __init__(self, *array: NDArray) -> None:
        super().__init__()
        self.array = array

    def __len__(self) -> int:
        return self.array[0].shape[0]

    def __getitem__(self, i: IndexType) -> tuple[NDArray, ...]:
        return tuple([a[i] for a in self.array])
