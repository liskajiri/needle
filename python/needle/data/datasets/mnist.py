from __future__ import annotations

import array
import gzip
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from needle.backend_selection import NDArray
from needle.data.dataset import Dataset

if TYPE_CHECKING:
    from needle.typing import IndexType, np_ndarray


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

    def __getitem__(self, index: IndexType) -> tuple[NDArray, np_ndarray]:
        (x, y) = self.X[index], self.y[index]
        return self.apply_transforms(x), y

    def __len__(self) -> int:
        return self.X.shape[0]

    @staticmethod
    def parse_mnist(images_file: Path, labels_file: Path) -> tuple[NDArray, np_ndarray]:
        """Parse MNIST data without using numpy."""

        def parse_images(file: Path) -> NDArray:
            with gzip.open(file, "rb") as f:
                # Read header: magic number (4 bytes), number of images,
                # number of rows, number of columns
                magic, num_images, _rows, _cols = struct.unpack(">IIII", f.read(16))
                if magic != 2051:  # Magic number for images file
                    raise ValueError("Invalid magic number in images file")

                # Read image data
                image_data = array.array("B")
                image_data.frombytes(f.read())

                # Convert to float and normalize to [0.0, 1.0]
                X = NDArray(image_data) / 255.0
                return X.reshape((num_images, MNISTDataset.IMAGE_SIZE))

        def read_labels(file: Path) -> np_ndarray:
            import numpy as np

            with gzip.open(file, "rb") as f:
                # Read header: magic number (4 bytes), number of items
                magic, _num_labels = struct.unpack(">II", f.read(8))
                if magic != 2049:  # Magic number for labels file
                    raise ValueError("Invalid magic number in labels file")

                # Read label data
                label_data = array.array("B")
                label_data.frombytes(f.read())
                y = np.array(label_data, dtype=np.uint8)
                # y = NDArray(label_data)
                return y

        try:
            return parse_images(images_file), read_labels(labels_file)

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"MNIST files not found. "
                f"Please download and place them in {images_file.parent}"
            ) from e
