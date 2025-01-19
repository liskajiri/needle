import gzip

import numpy as np

from needle.data.dataset import Dataset


class MNISTDataset(Dataset):
    IMAGE_SIZE = 28 * 28

    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: list | None = None,
    ):
        """Read an images and labels file in MNIST format.  See this page:
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
        if transforms is None:
            transforms = []
        self.X, self.y = MNISTDataset.parse_mnist(image_filename, label_filename)

        # TODO: should be (n, 784)
        # Fix tests afterwards
        self.X = self.X.reshape(-1, 28, 28, 1)
        self.transforms = transforms

    def __getitem__(self, index: int) -> object:
        (x, y) = self.X[index], self.y[index]
        if self.transforms:
            return self.apply_transforms(x), y
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]

    @staticmethod
    def parse_mnist(
        image_filename: str, label_filename: str
    ) -> tuple[np.ndarray, np.ndarray]:
        # Read the images file
        with gzip.open(image_filename, "rb") as image_file:
            image_file.read(16)  # Skip the header
            buffer = image_file.read()
            num_images = len(buffer) // (
                MNISTDataset.IMAGE_SIZE
            )  # Each image is 28x28 pixels
            X = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
            X = X.reshape(num_images, MNISTDataset.IMAGE_SIZE)

        # Read the labels file
        with gzip.open(label_filename, "rb") as label_file:
            label_file.read(8)  # Skip the header
            buffer = label_file.read()
            y = np.frombuffer(buffer, dtype=np.uint8)
        return (X, y)
