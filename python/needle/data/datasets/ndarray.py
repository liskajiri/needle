from __future__ import annotations

from typing import TYPE_CHECKING

from needle.data.dataset import Dataset

if TYPE_CHECKING:
    from needle.backend_selection import NDArray


class NDArrayDataset(Dataset):
    def __init__(self, *array: NDArray) -> None:
        self.array = array

    def __len__(self) -> int:
        return self.array[0].shape[0]

    def __getitem__(self, i) -> tuple[NDArray, ...]:
        return tuple([a[i] for a in self.array])
