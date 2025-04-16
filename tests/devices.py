import needle as ndl
import pytest

__all__ = ["all_devices"]

# _DEVICES = [
#     pytest.param(ndl.cpu(), id="cpu"),
#     pytest.param(
#         ndl.cuda(),
#         id="cuda",
#         marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"),
#     ),
# ]

_DEVICES = [pytest.param(ndl.cpu(), id="cpu")]
if ndl.cuda().enabled():
    _DEVICES.append(pytest.param(ndl.cuda(), id="cuda"))


def all_devices():
    """
    Return a pytest parametrize decorator for device testing
    """
    return pytest.mark.parametrize("device", _DEVICES)
