import needle as ndl
import needle.ops as ops
import numpy as np
import pytest
import torch
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import array_and_axis, single_array
from tests.ops.utils import generic_op_test
from tests.utils import backward_forward

rng = np.random.default_rng(0)


@given(inputs=single_array(shape=(2, 3)))
@backward_forward()
@all_devices()
def test_logsoftmax(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.logsoftmax,
        torch_op=torch.nn.functional.log_softmax,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@pytest.mark.parametrize(
    "tensor",
    [
        np.array([1.0, 2.0]),  # 1D input
        rng.standard_normal((2, 3, 4)),  # 3D input
    ],
    ids=["1d", "3d"],
)
def test_logsoftmax_invalid(tensor):
    with pytest.raises(AssertionError):
        ops.logsoftmax(ndl.Tensor(tensor))


@given(inputs=array_and_axis())
@backward_forward()
@all_devices()
@pytest.mark.xfail(reason="Broadcasting issues", strict=False)
def test_logsumexp(inputs, backward, device):
    arr, axis = inputs
    print(arr.shape, axis)
    generic_op_test(
        ndl_op=lambda x: ops.logsumexp(x, axes=axis),
        torch_op=lambda x: torch.logsumexp(x, dim=axis),
        inputs=[arr],
        backward=backward,
        device=device,
    )
