from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from needle.backend_selection import NDArray, array_api, default_device
from needle.data.dataset import Dataset

if TYPE_CHECKING:
    from needle.needle_typing import AbstractBackend, IndexType


class SyntheticMNIST(Dataset[NDArray]):
    """Synthetic MNIST-like dataset for benchmarking.

    - Images are binary (0.0 or 1.0) with density proportional to label/(num_classes-1).
    - Image_shape is channel-first (C, H, W).
    - Returns items as (image: NDArray, label: NDArray[int]).

    >>> from needle.data.datasets.synthetic_mnist import SyntheticMNIST
    >>> ds = SyntheticMNIST(num_samples=4, num_classes=2, image_shape=(1, 4, 4), seed=7)
    >>> len(ds)
    4
    >>> img, lbl = ds[0]
    >>> img.shape
    (1, 4, 4)
    >>> lbl
    0.0
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 10,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        seed: int | None = None,
        transforms: Sequence | None = None,
        device: AbstractBackend = default_device,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        super().__init__(transforms=transforms)
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.image_shape = (
            int(image_shape[0]),
            int(image_shape[1]),
            int(image_shape[2]),
        )
        self.seed = seed
        self.device = device

        # Seed device RNG if supported
        if hasattr(self.device, "set_seed") and self.seed is not None:
            self.device.set_seed(self.seed)

        # Build evenly distributed labels (as Python ints)
        per_class = self.num_samples // self.num_classes
        remainder = self.num_samples % self.num_classes
        labels_list = [i for i in range(self.num_classes)] * per_class + list(
            range(remainder)
        )
        labels_list = labels_list[: self.num_samples]

        # Generate "images"
        images = [self._generate_image_for_label(int(lbl)) for lbl in labels_list]

        # Stack into backend NDArray with shape (N, C, H, W) and store labels
        self.x = array_api.stack(images).astype("float32")
        self.y = array_api.array(labels_list, dtype="int64")

    def _generate_image_for_label(self, label: int) -> NDArray:
        """Generate one "image" with given label.

        Returns shape (C, H, W) with dtype float32 and values 0.0 or 1.0.
        """
        C, H, W = self.image_shape

        base_density = label / (self.num_classes - 1)

        # Sample once per image (H, W), threshold by density, then replicate to channels
        mask_hw = self.device.rand((H, W)) <= base_density
        mask = array_api.broadcast_to(mask_hw, (C, H, W))

        return mask

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(self, index: IndexType) -> tuple[NDArray, NDArray]:
        img = self.x[index]
        lbl = self.y[index]
        return self.apply_transforms(img), lbl
