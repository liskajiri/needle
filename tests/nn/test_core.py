"""Tests for needle.nn.core module."""

from __future__ import annotations

import torch
from hypothesis import given
from needle.nn.core import Flatten, Identity, Residual, Sequential

from tests.devices import all_devices
from tests.hypothesis_strategies import single_array
from tests.utils import backward_forward, generic_op_test


@given(arr=single_array())
@backward_forward()
@all_devices()
def test_identity(arr, backward, device):
    torch_layer = torch.nn.Identity()

    def torch_op(x):
        return torch_layer(x)

    needle_layer = Identity()

    def ndl_op(x):
        return needle_layer(x)

    generic_op_test(ndl_op, torch_op, inputs=arr, backward=backward, device=device)


@given(arr=single_array(shape=(3, 3)))
@backward_forward()
@all_devices()
def test_flatten(arr, backward, device):
    torch_layer = torch.nn.Flatten()

    def torch_op(x):
        return torch_layer(x)

    needle_layer = Flatten()

    def ndl_op(x):
        return needle_layer(x)

    generic_op_test(ndl_op, torch_op, inputs=arr, backward=backward, device=device)


@given(arr=single_array(shape=(3, 3)))
@backward_forward()
@all_devices()
def test_sequential(arr, backward, device):
    torch_layer = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Identity())

    def torch_op(x):
        return torch_layer(x)

    needle_layer = Sequential(Flatten(), Identity())

    def ndl_op(x):
        return needle_layer(x)

    generic_op_test(ndl_op, torch_op, inputs=arr, backward=backward, device=device)


@given(arr=single_array())
@backward_forward()
@all_devices()
def test_residual(arr, backward, device):
    def torch_op(x):
        return x + x

    ndl_layer = Residual(Identity())

    def ndl_op(x):
        return ndl_layer(x)

    generic_op_test(ndl_op, torch_op, inputs=arr, backward=backward, device=device)
