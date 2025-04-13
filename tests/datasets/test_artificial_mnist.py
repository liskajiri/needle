"""Tests for the configurable artificial dataset implementation.

These tests verify the functionality of a dataset with configurable dimensions
and number of classes. Images are generated based on relative label values:
- Label 0: All zeros
- Label (num_classes-1): All ones
- Other labels: Random pixels with density proportional to label/(num_classes-1)

Test Configuration:
    IMAGE_DIMS: List of image dimensions to test. Each image is square (width=height)
    NUM_CLASSES: Different numbers of classes to test:
        - 2: Binary classification (zeros and ones)
        - 5: Small multi-class case
        - 10: MNIST-like case
"""

import numpy as np
import pytest
from needle.data.datasets.artificial_mnist import ArtificialMNIST, generate_image

# Test configuration
IMAGE_DIMS = [16, 28]
NUM_CLASSES = [5, 10]


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_generate_image_dimensions(image_dim, num_classes) -> None:
    """Test that generated images have correct dimensions."""
    img = generate_image(0, num_classes, image_dim)
    assert img.shape == (image_dim, image_dim)


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
@pytest.mark.parametrize(
    "label_type, expected_value",
    [
        ("min", 0.0),  # Test zeros (label 0)
        ("max", 1.0),  # Test ones (label num_classes-1)
    ],
)
def test_generate_image_extremes(
    image_dim, num_classes, label_type, expected_value
) -> None:
    """Test that extreme labels generate appropriate images (all zeros or all ones)."""
    label = 0 if label_type == "min" else num_classes - 1
    img = generate_image(label, num_classes, image_dim)
    np.testing.assert_allclose(
        img,
        expected_value,
        atol=0,
        err_msg=f"Label {label} should generate images with all {expected_value}s",
    )


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_generate_image_densities(image_dim, num_classes) -> None:
    """Test that intermediate labels generate correct pixel densities."""
    # Test a middle label for each num_classes
    middle_label = num_classes // 2
    img = generate_image(middle_label, num_classes, image_dim)
    density = np.mean(img)
    expected_density = middle_label / (num_classes - 1)

    np.testing.assert_allclose(density, expected_density, atol=0.1)


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("num_samples", [10, 100, 1000])
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_dataset_creation(image_dim, num_samples, num_classes) -> None:
    """Test basic dataset creation and properties."""
    dataset = ArtificialMNIST(
        num_samples=num_samples, image_dim=image_dim, num_classes=num_classes
    )
    assert len(dataset) == num_samples
    assert dataset.image_dim == image_dim
    assert dataset.image_size == image_dim * image_dim
    assert dataset.num_classes == num_classes


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_dataset_shapes(image_dim, num_classes) -> None:
    """Test shapes of images and labels from dataset."""
    dataset = ArtificialMNIST(
        num_samples=100, image_dim=image_dim, num_classes=num_classes
    )
    x, y = dataset[0]
    assert x.shape == (image_dim, image_dim, 1)
    assert y.dtype == np.uint8
    assert 0 <= y < num_classes, f"Label {y} should be in range [0, {num_classes})"


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("num_samples", [100, 1000])
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_dataset_label_distribution(image_dim, num_samples, num_classes) -> None:
    """Test that labels are evenly distributed."""
    expected_per_class = num_samples // num_classes

    dataset = ArtificialMNIST(
        num_samples=num_samples, image_dim=image_dim, num_classes=num_classes
    )
    unique, counts = np.unique(dataset.y, return_counts=True)

    assert len(unique) == num_classes, "All labels should be present"
    for label, count in zip(unique, counts):
        assert abs(count - expected_per_class) <= 1, (
            f"Expected ~{expected_per_class} samples for each class"
        )


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("batch_size", [10, 32])
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_dataset_value_ranges(image_dim, batch_size, num_classes) -> None:
    """Test that image values are in correct range [0,1]."""
    dataset = ArtificialMNIST(
        num_samples=batch_size, image_dim=image_dim, num_classes=num_classes
    )
    for i in range(len(dataset)):
        x, _ = dataset[i]
        assert x.shape == (image_dim, image_dim, 1)
        assert np.all(x >= 0.0), "Values should be non-negative"
        assert np.all(x <= 1.0), "Values should not exceed 1"


@pytest.mark.parametrize("image_dim", IMAGE_DIMS)
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_label_density_matching(image_dim, num_classes) -> None:
    """Test image densities match their labels."""
    dataset = ArtificialMNIST(
        num_samples=num_classes,
        image_dim=image_dim,
        num_classes=num_classes,
    )

    for target_label in range(num_classes):
        label_found = False
        for x, y in [dataset[i] for i in range(len(dataset))]:
            if y == target_label:
                label_found = True
                density = np.mean(x)
                if target_label == 0:
                    np.testing.assert_array_equal(
                        x, np.zeros_like(x), "Label 0 should be all zeros"
                    )
                elif target_label == num_classes - 1:
                    np.testing.assert_array_equal(
                        x, np.ones_like(x), f"Label {target_label} should be all ones"
                    )
                else:
                    expected = target_label / (num_classes - 1)
                    np.testing.assert_allclose(
                        density,
                        expected,
                        atol=0.1,
                        err_msg=f"Label {target_label} should have density ~{expected}",
                    )
                break
        np.testing.assert_(label_found, f"Label {target_label} not found in dataset")


@pytest.mark.parametrize("num_classes", [1, 0, -1])
def test_invalid_num_classes(num_classes) -> None:
    """Test that invalid num_classes values are rejected."""
    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        ArtificialMNIST(num_classes=num_classes)
