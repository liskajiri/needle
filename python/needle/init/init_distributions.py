from needle.autograd import Tensor
from needle.init.init_basic import rand, randn

__all__ = [
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
]


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
