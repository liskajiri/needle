import needle.ops as ops
import torch
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import division_arrays, same_shape_arrays
from tests.ops.utils import generic_op_test
from tests.utils import backward_forward


@given(inputs=same_shape_arrays())
@backward_forward()
@all_devices()
def test_add(inputs, backward, device) -> None:
    generic_op_test(
        ndl_op=ops.add,
        torch_op=torch.add,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=same_shape_arrays())
@backward_forward()
@all_devices()
def test_multiply(inputs, backward, device) -> None:
    generic_op_test(
        ndl_op=ops.multiply,
        torch_op=torch.multiply,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=same_shape_arrays())
@backward_forward()
@all_devices()
def test_power(inputs, backward, device) -> None:
    generic_op_test(
        ndl_op=ops.power,
        torch_op=torch.pow,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=division_arrays())
@backward_forward()
@all_devices()
def test_divide(inputs, backward, device) -> None:
    generic_op_test(
        ndl_op=ops.divide,
        torch_op=torch.divide,
        inputs=inputs,
        backward=backward,
        device=device,
    )
