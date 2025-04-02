from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from needle.backend_selection import NDArray
from needle.data.dataset import Dataset

if TYPE_CHECKING:
    from needle.typing import np_ndarray


class MNISTPaths:
    TRAIN_IMAGES = Path("data/mnist/train-images-idx3-ubyte.gz")
    TRAIN_LABELS = Path("data/mnist/train-labels-idx1-ubyte.gz")
    TEST_IMAGES = Path("data/mnist/t10k-images-idx3-ubyte.gz")
    TEST_LABELS = Path("data/mnist/t10k-labels-idx1-ubyte.gz")


class MNISTDataset(Dataset):
    IMAGE_DIM = 28
    IMAGE_SIZE = IMAGE_DIM * IMAGE_DIM

    def __init__(
        self,
        images: Path = MNISTPaths.TRAIN_IMAGES,
        labels: Path = MNISTPaths.TRAIN_LABELS,
        **kwargs,
    ) -> None:
        """
        Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            images (Path): Path to the gzipped images file in MNIST format.
                Defaults to MNISTPaths.TRAIN_IMAGES.
            labels (Path): Path to the gzipped labels file in MNIST format.
                Defaults to MNISTPaths.TRAIN_LABELS.
            *kwargs: Additional arguments passed to the Dataset parent class.

        Notes:
            The loaded data will be stored as:
            - self.X: NDArray containing the images, reshaped to (-1, 28, 28, 1).
              Values are normalized to range [0.0, 1.0].
            - self.y: numpy.ndarray[dtype=np.uint8] containing the labels (0-9).
        """

        super().__init__(**kwargs)

        self.X, self.y = MNISTDataset.parse_mnist(images, labels)
        self.X = (
            NDArray(self.X).compact().reshape((-1, self.IMAGE_DIM, self.IMAGE_DIM, 1))
        )
        # self.y = NDArray(self.y)

    def __getitem__(self, index: int | slice) -> tuple[NDArray, np_ndarray]:
        (x, y) = self.X[index], self.y[index]
        return self.apply_transforms(x), y

    def __len__(self) -> int:
        return self.X.shape[0]

    @staticmethod
    def parse_mnist(
        images_file: Path, labels_file: Path
    ) -> tuple[np_ndarray, np_ndarray]:
        # Read the images file
        with gzip.open(images_file, "rb") as image_file:
            image_file.read(16)  # Skip the header
            buffer = image_file.read()
            num_images = len(buffer) // MNISTDataset.IMAGE_SIZE
            # normalize to [0.0, 1.0]
            X = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
            X = X.reshape(num_images, MNISTDataset.IMAGE_SIZE)

        # Read the labels file
        with gzip.open(labels_file, "rb") as label_file:
            label_file.read(8)  # Skip the header
            buffer = label_file.read()
            y = np.frombuffer(buffer, dtype=np.uint8)
        return (X, y)
