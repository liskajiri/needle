import numpy as np

from needle.backend_ndarray.ndarray import NDArray


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
        # TODO: Transformations without numpy
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
