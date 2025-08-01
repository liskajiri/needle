from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from needle.typing import IndexType


class Dataset[T](Sequence[T], ABC):
    """
    An abstract class representing a `Dataset`.

    Attributes:
        transforms: List of transform functions to apply to each item

    Example:
        >>> class NumberDataset(Dataset[float]):
        ...     def __init__(
        ...         self,
        ...         start: int,
        ...         end: int,
        ...         transforms: Sequence[Callable] | None = None,
        ...     ) -> None:
        ...         super().__init__(transforms=transforms)
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __getitem__(self, idx: int) -> float:
        ...         if idx >= len(self):
        ...             raise IndexError("Index out of range")
        ...         value = float(self.start + idx)
        ...         return self.apply_transforms(value)
        ...
        ...     def __len__(self) -> int:
        ...         return self.end - self.start
        >>> # Create dataset and test basic operations
        >>> ds = NumberDataset(0, 5)
        >>> len(ds)
        5
        >>> ds[0]  # Get first item
        0.0
        >>> ds[4]  # Get last item
        4.0
        >>> # Add transform to double values
        >>> ds_doubled = NumberDataset(0, 5, transforms=[lambda x: x * 2])
        >>> ds_doubled[0]
        0.0
        >>> ds_doubled[4]
        8.0
    """

    def __init__(self, transforms: Sequence[Callable[[T], T]] | None = None) -> None:
        self._transforms = tuple(transforms) if transforms else ()

    @abstractmethod
    def __getitem__(self, index: IndexType) -> T | tuple[T, int]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x: T) -> T:
        for transform in self._transforms:
            x = transform(x)
        return x
