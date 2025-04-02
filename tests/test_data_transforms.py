import needle as ndl
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from needle.backend_selection import NDArray

from tests.utils import set_random_seeds


@given(arrays(dtype=np.float32, shape=(5, 5, 5), elements=st.floats(0, 1)))
def test_flip_horizontal_hypothesis(a: np.ndarray):
    transform = ndl.data.RandomFlipHorizontal(p=1)

    b = np.flip(a, axis=1)

    ndl_a = ndl.NDArray(a)
    result = transform(ndl_a)

    np.testing.assert_allclose(result, b)

    # The transform should be reversible by applying it twice
    double_transform = transform(result)
    np.testing.assert_allclose(double_transform.numpy(), a)


def numpy_crop(img: NDArray, padding: int = 3) -> NDArray:
    """Zero pad and then randomly crop an image.

    Args:
            img: H x W x C NDArray of an image
    Return
        H x W x C NDArray of clipped image
    Note: generate the image shifted by shift_x, shift_y specified below

    """
    assert img.ndim == 3
    import random

    shift_x = random.randint(-padding, padding)
    shift_y = random.randint(-padding, padding)
    # shift_x, shift_y = np.random.randint(low=-padding, high=padding + 1, size=2)
    H, W, C = tuple(img.shape)

    H += 2 * padding
    W += 2 * padding

    img_padded = np.zeros((H, W, C))
    # copy img to the padded array
    img_padded[padding : H - padding, padding : W - padding, :] = img

    return img_padded[
        padding + shift_x : H - padding + shift_x,
        padding + shift_y : W - padding + shift_y,
        :,
    ]


@given(
    arrays(
        dtype=np.float32,
        shape=array_shapes(min_dims=3, max_dims=3),
        elements=st.floats(0, 1),
    ),
    st.integers(1, 8),
)
def test_random_crop_hypothesis(a: np.ndarray, padding: int):
    set_random_seeds(0)

    transform = ndl.data.RandomCrop(padding)

    ndl_a = ndl.NDArray(a)
    ndl_result = transform(ndl_a)

    # set the same random values
    set_random_seeds(0)
    b = numpy_crop(a, padding)

    np.testing.assert_allclose(ndl_result.numpy(), b)
