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
    IMAGE_SIZE = 28 * 28

    def __init__(
        self,
        images: Path,
        labels: Path,
        transforms: list | None = None,
    ):
        """
        Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format

        Returns
        -------
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                    data.  The dimensionality of the data should be
                    (num_examples x input_dim) where 'input_dim' is the full
                    dimension of the data, e.g., since MNIST images are 28x28, it
                    will be 784.  Values should be of type np.float32, and the data
                    should be normalized to have a minimum value of 0.0 and a
                    maximum value of 1.0. The normalization should be applied uniformly
                    across the whole dataset, _not_ individual images.

                y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.uint8 and
                    for MNIST will contain the values 0-9.

        """
        super().__init__(transforms)
        self.X, self.y = MNISTDataset.parse_mnist(images, labels)

        # TODO: should be (n, 784)
        # Fix tests afterwards
        self.X = NDArray(self.X).compact().reshape((-1, 28, 28, 1))
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
            X = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
            X = X.reshape(num_images, MNISTDataset.IMAGE_SIZE)

        # Read the labels file
        with gzip.open(labels_file, "rb") as label_file:
            label_file.read(8)  # Skip the header
            buffer = label_file.read()
            y = np.frombuffer(buffer, dtype=np.uint8)
        return (X, y)
