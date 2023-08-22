import gzip
from typing import Iterable, List, Optional

import numpy as np

from needle.autograd import NDArray, Tensor


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: NDArray) -> NDArray:
        """
        Horizontally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        if np.random.rand() < self.p:
            return np.flip(img, axis=1)
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img: NDArray) -> NDArray:
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NDArray of clipped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        assert img.ndim == 3
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )

        H, W, C = img.shape

        H += 2 * self.padding
        W += 2 * self.padding

        img_padded = np.zeros((H, W, C))
        # copy img to the padded array
        img_padded[
            self.padding : H - self.padding, self.padding : W - self.padding, :
        ] = img

        return img_padded[
            self.padding + shift_x : H - self.padding + shift_x,
            self.padding + shift_y : W - self.padding + shift_y,
            :,
        ]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for transform in self.transforms:
                x = transform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        if self.shuffle:
            orders = np.random.permutation(len(self.dataset))
        else:
            orders = np.arange(len(self.dataset))

        self.ordering = np.array_split(
            orders,
            range(self.batch_size, len(self.dataset), self.batch_size),
        )

        return self

    def __next__(self) -> Iterable[Tensor]:
        if self.index >= len(self.ordering):
            raise StopIteration

        indices = self.ordering[self.index]
        self.index += 1

        return [Tensor(i) for i in self.dataset[indices]]


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = [],
    ):
        """Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format

        Returns:
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
        # Read the images file
        with gzip.open(image_filename, "rb") as image_file:
            image_file.read(16)  # Skip the header
            buffer = image_file.read()
            num_images = len(buffer) // (28 * 28)  # Each image is 28x28 pixels
            X = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
            X = X.reshape(num_images, 28 * 28)

        # Read the labels file
        with gzip.open(label_filename, "rb") as label_file:
            label_file.read(8)  # Skip the header
            buffer = label_file.read()
            y = np.frombuffer(buffer, dtype=np.uint8)

        self.X = X.reshape(-1, 28, 28, 1)
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index: int) -> object:
        (x, y) = self.X[index], self.y[index]
        return self.apply_transforms(x), y

    def __len__(self) -> int:
        return len(self.y)


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
