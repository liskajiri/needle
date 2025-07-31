from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_selection import NDArray
from needle.data.dataset import Dataset

if TYPE_CHECKING:
    from needle.typing import IndexType


class NDArrayDataset(Dataset[NDArray]):
    def __init__(self, x: NDArray, y: NDArray) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: IndexType) -> tuple[NDArray, NDArray]:
        return self.x[i], self.y[i]
