import pickle
from pathlib import Path

import numpy as np

from needle.backend_ndarray.ndarray import NDArray
from needle.data.dataset import Dataset

CIFARPath = Path("data/cifar-10/cifar-10-batches-py")


class CIFAR10Dataset(Dataset):
    IMAGE_SHAPE = (3, 32, 32)
    LabelType = int

    def __init__(
        self,
        base_folder: Path = CIFARPath,
        train: bool = True,
        p: float | None = 0.5,
        transforms: list | None = None,
    ) -> None:
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        super().__init__(transforms)
        self.train = train

        X = []
        Y = []

        for file in base_folder.iterdir():
            if (file.name.startswith("data_batch") and train) or (
                file.name == "test_batch" and not train
            ):
                x, y = self._unpickle(file)
                X.extend(x.reshape(-1, *CIFAR10Dataset.IMAGE_SHAPE))
                Y.extend(y)
        self.X = np.stack(X)
        self.Y = np.array(Y)

    def __getitem__(self, index: int | np.ndarray) -> tuple[NDArray, LabelType]:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        x = self.apply_transforms(self.X[index])
        y = self.Y[index]

        expected_shape = (
            (index.size, *CIFAR10Dataset.IMAGE_SHAPE)
            if isinstance(index, np.ndarray)
            else CIFAR10Dataset.IMAGE_SHAPE
        )
        assert x.shape == expected_shape, f"Expected shape (3, 32, 32), got {x.shape}"

        return x, y

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.Y.shape[0]

    @staticmethod
    def _unpickle(file: Path) -> tuple[np.ndarray, list[int]]:
        file_data = Path.read_bytes(file)
        unpickled = pickle.loads(file_data, encoding="bytes")
        return unpickled[b"data"], unpickled[b"labels"]
