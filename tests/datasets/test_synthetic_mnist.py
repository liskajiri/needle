import numpy as np
from needle.backend_selection import array_api
from needle.data.datasets.synthetic_mnist import SyntheticMNIST


def test_synthetic_mnist_shapes():
    ds = SyntheticMNIST(num_samples=20, num_classes=4, image_shape=(1, 8, 8), seed=1)
    assert len(ds) == 20
    img, label = ds[0]
    # image is channel-first (C, H, W)
    assert img.shape == (1, 8, 8)
    assert label.item() == 0


def test_synthetic_mnist_determinism():
    ds1 = SyntheticMNIST(num_samples=10, num_classes=3, image_shape=(1, 4, 4), seed=123)
    ds2 = SyntheticMNIST(num_samples=10, num_classes=3, image_shape=(1, 4, 4), seed=123)
    for i in range(10):
        img1, lbl1 = ds1[i]
        img2, lbl2 = ds2[i]
        np.testing.assert_array_equal(img1, img2)
        np.testing.assert_array_equal(lbl1, lbl2)


def test_synthetic_mnist_class_count():
    num_classes = 5
    ds = SyntheticMNIST(
        num_samples=100, num_classes=num_classes, image_shape=(1, 2, 2), seed=0
    )
    labels = [ds[i][1].item() for i in range(len(ds))]
    # Should have at least one of each class
    for c in range(num_classes):
        assert c in labels, f"Class {c} missing"


def test_synthetic_mnist_value_ranges_and_density():
    num_classes = 5
    C, H, W = (1, 6, 6)
    ds = SyntheticMNIST(
        num_samples=num_classes, num_classes=num_classes, image_shape=(C, H, W), seed=42
    )
    for target_label in range(num_classes):
        found = False
        for x, y in [ds[i] for i in range(len(ds))]:
            if y.item() == target_label:
                found = True
                # values must be 0.0 or 1.0 and within [0,1]
                assert array_api.sum(x >= 0.0) == x.size
                assert array_api.sum(x <= 1.0) == x.size
                # check density
                density = array_api.sum(x).item() / float(x.size)
                expected = target_label / (num_classes - 1)
                assert abs(density - expected) <= 0.15
        assert found, f"Label {target_label} not found in dataset"


def test_invalid_num_classes():
    for nc in (1, 0, -2):
        try:
            SyntheticMNIST(num_samples=10, num_classes=nc)
            assert False, "Should have raised ValueError for invalid num_classes"
        except ValueError:
            pass
