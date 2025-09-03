from __future__ import annotations

import math
from typing import TYPE_CHECKING

from needle.init.init_basic import rand, randn
from needle.nn.activations import ReLU

if TYPE_CHECKING:
    from typing import Literal

    from needle.needle_typing import Shape
    from needle.tensor import Tensor


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    gain: float = 1.0,
    **kwargs,
) -> Tensor:
    """Initialize weights using Xavier/Glorot uniform initialization.

    Args:
        fan_in: Number of input features
        fan_out: Number of output features
        gain: Scaling factor
        device: Device to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor initialized with Xavier uniform distribution
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive integers")

    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand((fan_in, fan_out), low=-a, high=a, **kwargs)


def xavier_normal(
    fan_in: int,
    fan_out: int,
    gain: float = 1.0,
    **kwargs,
) -> Tensor:
    """Initialize weights using Xavier/Glorot normal initialization.

    Args:
        fan_in: Number of input features
        fan_out: Number of output features
        gain: Scaling factor
        device: Device to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor initialized with Xavier normal distribution
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive integers")

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return randn((fan_in, fan_out), std=std, **kwargs)


def kaiming_uniform(
    fan_in: int = 1,
    fan_out: int = 1,
    shape: Shape | None = None,
    gain: float = math.sqrt(2.0),
    nonlinearity: ReLU | None = None,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    **kwargs,
) -> Tensor:
    """Initialize weights using He/Kaiming uniform initialization.

    Args:
        fan_in: Number of input features
        fan_out: Number of output features
        shape: Optional shape override
        gain: Optional gain factor
        nonlinearity: Type of non-linearity (only 'relu' supported)
        mode: Either 'fan_in' or 'fan_out'
        device: Device to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor initialized with Kaiming uniform distribution

    Raises:
        ValueError: If fan_in/fan_out invalid or unsupported nonlinearity
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive integers")
    if not nonlinearity:
        nonlinearity = ReLU()

    fan = fan_in if mode == "fan_in" else fan_out

    bound = gain * math.sqrt(3.0 / fan)

    if not shape:
        shape = (fan_in, fan_out)
    return rand(shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(
    fan_in: int,
    fan_out: int,
    gain: float = math.sqrt(2.0),
    nonlinearity: ReLU | None = None,
    mode: str = "fan_in",
    **kwargs,
) -> Tensor:
    """Initialize weights using He/Kaiming normal initialization.

    Args:
        fan_in: Number of input features
        fan_out: Number of output features
        gain: Optional gain factor
        nonlinearity: Type of non-linearity (only 'relu' supported)
        mode: Either 'fan_in' or 'fan_out'
        device: Device to store the tensor
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor initialized with Kaiming normal distribution

    Raises:
        ValueError: If fan_in/fan_out invalid or unsupported nonlinearity
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive integers")
    if not nonlinearity:
        nonlinearity = ReLU()

    fan = fan_in if mode == "fan_in" else fan_out

    std = gain / math.sqrt(fan)
    return randn((fan_in, fan_out), std=std, **kwargs)
