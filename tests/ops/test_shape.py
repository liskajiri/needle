import needle.ops as ops
import pytest
import torch
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import (
    array_and_permutation,
    array_and_reshape_shape,
    broadcastable_arrays,
)
from tests.ops.utils import generic_op_test
from tests.utils import backward_forward


@given(inputs_and_axes=array_and_permutation())
@backward_forward()
@all_devices()
def test_transpose(inputs_and_axes, backward, device):
    arr, axes = inputs_and_axes
    generic_op_test(
        ndl_op=lambda x: ops.transpose(x, axes=axes),
        torch_op=lambda x: torch.permute(x, axes),
        inputs=(arr,),
        backward=backward,
        device=device,
    )


@given(inputs=array_and_reshape_shape())
@backward_forward()
@all_devices()
def test_reshape(inputs, backward, device):
    array, shape_to = inputs
    generic_op_test(
        ndl_op=lambda x: ops.reshape(x, shape_to),
        torch_op=lambda x: torch.reshape(x, shape_to),
        inputs=(array,),
        backward=backward,
        device=device,
    )


@given(inputs=broadcastable_arrays())
@backward_forward()
@all_devices()
@pytest.mark.xfail(reason="Broadcasting issues", strict=False)
def test_broadcast(inputs, backward, device):
    generic_op_test(
        ndl_op=lambda a, b: ops.broadcast_to(a, b.shape),
        torch_op=lambda a, b: torch.broadcast_to(a, b.shape),
        inputs=inputs,
        backward=backward,
        device=device,
    )
