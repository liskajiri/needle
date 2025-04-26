from __future__ import annotations

from typing import TYPE_CHECKING

from needle.backend_selection import NDArray, array_api, default_device
from needle.data.dataset import Dataset

if TYPE_CHECKING:
    from needle.typing import AbstractBackend, IndexType


class ArtificialMNIST(Dataset[NDArray]):
    """An artificial dataset with configurable dimensions and number of classes.

    The dataset creates images where the label corresponds to pixel density:
    - Label 0: Image with all zeros
    - Label (num_classes-1): Image with all ones
    - Other labels: Images with proportional amounts of random ones

    The original MNIST-like configuration uses 10 classes (0-9).
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_dim: int = 28,
        num_classes: int = 10,
        transforms: None = None,
        device: AbstractBackend = default_device,
    ) -> None:
        """
        Create an artificial dataset with configurable properties.

        Args:
            train: If True, generates training data, else test data
            num_samples: Number of samples to generate
            image_dim: Size of images (image_dim x image_dim)
            num_classes: Number of different classes (density levels)
            **kwargs: Additional arguments passed to Dataset
        """
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        super().__init__(transforms=transforms)
        self.num_classes = num_classes
        self.image_dim = image_dim
        self.image_size = image_dim * image_dim
        self.device = device

        # Generate labels ensuring even distribution
        num_per_class = num_samples // num_classes
        remainder = num_samples % num_classes
        # TODO: convert to array
        self.y = [i for i in range(num_classes)] * num_per_class + list(
            range(remainder)
        )

        # Generate corresponding images
        images = []
        for label in self.y:
            image = ArtificialMNIST.generate_image(
                label, self.num_classes, self.image_dim, device
            )
            images.append(image)

        self.x = array_api.stack(tuple(images))
        self.x = self.x.reshape((-1, self.image_dim, self.image_dim, 1))

    def __getitem__(self, index: IndexType) -> tuple[NDArray, int]:
        if not isinstance(index, int):
            # TODO: remove once self.y is NDArray
            raise ValueError("Index must be an integer for now.")
        return self.apply_transforms(self.x[index]), self.y[index]

    def __len__(self) -> int:
        return len(self.y)

    @staticmethod
    def generate_image(
        label: int,
        num_classes: int,
        image_dim: int = 28,
        device: AbstractBackend = default_device,
    ) -> NDArray:
        """Generate an artificial image based on pixel density.

        Args:
            label: Integer from 0 to (num_classes-1) representing density level
                0 = all zeros
                num_classes-1 = all ones
                others = proportional random pixels
            num_classes: Number of different classes (density levels)
            image_dim: Size of image (image_dim x image_dim)

        Returns:
            image_dim x image_dim NDArray with values in [0.0, 1.0]
        """
        shape = (image_dim, image_dim)
        if label == 0:
            return array_api.zeros(shape, dtype="float32")
        elif label == num_classes - 1:
            return array_api.ones(shape, dtype="float32")
        else:
            # Calculate probability of pixel being 1 based on label
            prob = label / (num_classes - 1)
            random_vals = device.rand(shape) <= prob
            return array_api.array(random_vals, dtype="float32")
