from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


class Dataset(Sequence):
    r"""
    An abstract class representing a `Dataset`.

    All subclasses should overwrite
    :meth:`__getitem__`, supporting fetching a data sample for a given key.
    Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: list | None = None) -> None:
        if transforms is None:
            transforms = []
        self.transforms: list = transforms

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x: T) -> T:
        if self.transforms is not None:
            # apply the transforms
            for transform in self.transforms:
                x = transform(x)
        return x
