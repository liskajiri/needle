from .autograd import Tensor, cpu


def rand(
    *shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False
) -> Tensor:
    """Generate random numbers uniform between low and high"""
    device = cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(
    *shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False
) -> Tensor:
    """Generate random normal with specified mean and std deviation"""
    device = cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(
    *shape, c=1.0, device=None, dtype="float32", requires_grad=False
) -> Tensor:
    """Generate constant Tensor"""
    device = cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False) -> Tensor:
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False) -> Tensor:
    """Generate all-zeros Tensor"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def rand_binary(
    *shape, p=0.5, device=None, dtype="bool", requires_grad=False
) -> Tensor:
    """Generate binary random Tensor"""
    device = cpu() if device is None else device
    array = device.rand(*shape) <= p
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False) -> Tensor:
    """Generate one-hot encoding Tensor"""
    device = cpu() if device is None else device
    return Tensor(
        device.one_hot(n, i.numpy(), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs) -> Tensor:
    a = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs) -> Tensor:
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return randn(fan_in, fan_out, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out=1, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = (2**0.5) * (3 / fan_in) ** 0.5
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out=1, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    std = (2**0.5) / (fan_in**0.5)
    return randn(fan_in, fan_out, std=std, **kwargs)
