import pickle
from pathlib import Path

from needle.backend_ndarray.ndarray import NDArray
from needle.backend_selection import array_api
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

        new_shape = (-1, *CIFAR10Dataset.IMAGE_SHAPE)
        for i, file in enumerate(base_folder.iterdir()):
            if (file.name.startswith("data_batch") and train) or (
                file.name == "test_batch" and not train
            ):
                x, y = self._unpickle(file)
                x = array_api.array(x).reshape(new_shape)
                X.append(x)
                Y.extend(y)
        # X: (5, 10_000, 3, 32, 32) -> (50_000, 3, 32, 32)
        self.X = array_api.stack(X).reshape(new_shape)
        self.Y = array_api.array(Y)

    def __getitem__(self, index: int | tuple | NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        if isinstance(index, int) or len(index) == 1:
            new_shape = CIFAR10Dataset.IMAGE_SHAPE
        else:
            new_shape = (len(index), *CIFAR10Dataset.IMAGE_SHAPE)

        i = self.X[index].compact().reshape(new_shape)
        x = self.apply_transforms(i)
        y = self.Y[index]

        assert x.shape == new_shape, f"Expected shape {new_shape}, got {x.shape}"

        return x, y

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.Y.shape[0]

    @staticmethod
    def _unpickle(file: Path) -> tuple[NDArray, list[int]]:
        file_data = Path.read_bytes(file)
        unpickled = pickle.loads(file_data, encoding="bytes")
        return unpickled[b"data"], unpickled[b"labels"]
