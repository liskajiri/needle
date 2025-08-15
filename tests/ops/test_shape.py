import needle.ops as ops
import torch
from hypothesis import given
from hypothesis.extra.numpy import mutually_broadcastable_shapes

from tests.devices import all_devices
from tests.hypothesis_strategies import (
    array_and_permutation,
    array_and_reshape_shape,
)
from tests.utils import backward_forward, generic_op_test


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


@given(mbs=mutually_broadcastable_shapes(num_shapes=1))
@backward_forward()
@all_devices()
def test_broadcast_to(mbs, backward, device):
    src = mbs.input_shapes[0]
    dst = mbs.result_shape

    generic_op_test(
        ndl_op=lambda a, b: ops.broadcast_to(a, b.shape),
        torch_op=lambda a, b: torch.broadcast_to(a, b.shape),
        inputs=[src, dst],
        backward=backward,
        device=device,
        sum=True,
    )
