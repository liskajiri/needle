import needle.ops as ops
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.hypothesis_strategies import (
    array_and_axis,
    array_and_multiple_axes,
    single_array,
)
from tests.utils import backward_forward, generic_op_test


@given(inputs=array_and_axis())
@backward_forward()
@all_devices()
def test_stack(inputs, backward, device):
    arrs, axis = inputs

    generic_op_test(
        ndl_op=lambda *a: ops.stack(a, axis=axis),
        torch_op=lambda *a: torch.stack(a, dim=axis),
        inputs=[arrs],
        backward=backward,
        device=device,
    )


@given(arrs=single_array(), data=st.data())
@backward_forward()
@all_devices()
@pytest.mark.xfail(reason="Split has wrong shapes", strict=False)
def test_split(arrs, data, backward, device):
    # now pick a valid axis: 0 <= axis < ndim
    axis = data.draw(st.integers(min_value=0, max_value=arrs[0].ndim - 1))
    section_size = data.draw(st.integers(min_value=1, max_value=arrs[0].shape[axis]))

    generic_op_test(
        ndl_op=lambda a: ops.split(a, axis=axis, sections=section_size),
        torch_op=lambda a: torch.split(
            a, split_size_or_sections=section_size, dim=axis
        ),
        inputs=arrs,
        backward=backward,
        device=device,
    )


@given(inputs=array_and_axis())
@backward_forward()
@all_devices()
@pytest.mark.xfail(reason="Concatenate has wrong shapes", strict=False)
def test_concatenate(inputs, backward, device):
    arrs, axis = inputs

    generic_op_test(
        ndl_op=lambda a: ops.concatenate(a, axis=axis),
        torch_op=lambda *a: torch.concatenate(a, dim=axis),
        inputs=[arrs],
        backward=backward,
        device=device,
    )


@given(inputs=array_and_multiple_axes())
@backward_forward()
@all_devices()
def test_flip(inputs, backward, device):
    arr, axes = inputs
    generic_op_test(
        ndl_op=lambda a: ops.flip(a, axes=axes),
        torch_op=lambda a: torch.flip(a, dims=axes),
        inputs=(arr,),
        backward=backward,
        device=device,
    )


@given(inputs=single_array(), data=st.data())
@backward_forward()
@all_devices()
@pytest.mark.xfail(reason="Broadcasting issue", strict=False)
def test_get_item(inputs, data, backward, device):
    idx = data.draw(st.integers(min_value=0, max_value=inputs[0].shape[0] - 1))
    generic_op_test(
        ndl_op=lambda x: ops.get_item(x, idx),
        torch_op=lambda x: x[idx],
        inputs=inputs,
        backward=backward,
        device=device,
    )
