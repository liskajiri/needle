from typing import Any

from needle.data.dataset import Dataset


class NDArrayDataset(Dataset):
    def __init__(self, *arrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> Any:
        return tuple([a[i] for a in self.arrays])
