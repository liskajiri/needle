from typing import TYPE_CHECKING

from needle.backend_selection import default_device
from needle.tensor import Tensor
from needle.typing.utils import DType, Shape

if TYPE_CHECKING:
    from needle.backend_ndarray.device import AbstractBackend as Device


def rand(
    *shape: Shape,
    low: float = 0.0,
    high: float = 1.0,
    device: "Device" = default_device(),
    dtype: DType = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate random numbers uniform between low and high.

    Args:
        *shape: The shape of the output tensor
        low: Lower bound of uniform distribution
        high: Upper bound of uniform distribution
        device: "Device" to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random uniform values
    """
    array = device.rand(*shape) * (high - low) + low
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(
    *shape: Shape,
    mean: float = 0.0,
    std: float = 1.0,
    device: "Device" = default_device(),
    dtype: DType = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate random normal with specified mean and std deviation.

    Args:
        *shape: The shape of the output tensor
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
        device: "Device" to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random normal values
    """
    array = device.randn(*shape) * std + mean
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(
    *shape: Shape,
    c: float = 1.0,
    device: "Device" = default_device(),
    dtype: DType = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate constant Tensor.

    Args:
        *shape: The shape of the output tensor
        c: Constant value to fill tensor with
        device: "Device" to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with constant value
    """
    array = device.full(shape, c, dtype=dtype)
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(
    *shape: Shape,
    device: "Device" = default_device(),
    dtype: DType = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate all-ones Tensor.

    Args:
        *shape: The shape of the output tensor
        device: "Device" to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with ones
    """
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(
    *shape: Shape,
    device: "Device" = default_device(),
    dtype: DType = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate all-zeros Tensor.

    Args:
        *shape: The shape of the output tensor
        device: "Device" to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with zeros
    """
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(
    *shape: Shape,
    p: float = 0.5,
    device: "Device" = default_device(),
    dtype: DType = "bool",
    requires_grad: bool = False,
) -> Tensor:
    """Generate binary random Tensor.

    Args:
        *shape: The shape of the output tensor
        p: Probability of 1 in binary distribution
        device: "Device" to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Binary tensor with random values
    """
    array = device.rand(*shape) <= p
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(
    n: int,
    i: Tensor,
    device: "Device" = default_device(),
    dtype: DType = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate one-hot encoding Tensor.

    Args:
        n: Number of classes
        i: Indices tensor
        device: "Device" to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        One-hot encoded tensor
    """
    return Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(
    array: Tensor,
    *,
    device: "Device" = default_device(),
    requires_grad: bool = False,
) -> Tensor:
    """Generate tensor of zeros with same shape as input.

    Args:
        array: Template tensor
        device: "Device" to store the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor of zeros with same shape as input
    """
    return zeros(
        array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(
    array: Tensor,
    *,
    device: "Device" = default_device(),
    requires_grad: bool = False,
) -> Tensor:
    """Generate tensor of ones with same shape as input.

    Args:
        array: Template tensor
        device: "Device" to store the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor of ones with same shape as input
    """
    return ones(
        array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
