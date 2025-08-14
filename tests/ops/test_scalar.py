import needle.ops as ops
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import (
    float_strategy,
    safe_float_strategy,
    single_array,
)
from tests.ops.utils import generic_op_test
from tests.utils import backward_forward


@given(inputs=single_array(), scalar=float_strategy)
@backward_forward()
@all_devices()
def test_add_scalar(inputs, scalar, backward, device):
    generic_op_test(
        ndl_op=lambda x: ops.add_scalar(x, scalar),
        torch_op=lambda x: x + scalar,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=single_array(), scalar=float_strategy)
@backward_forward()
@all_devices()
def test_mul_scalar(inputs, scalar, backward, device):
    generic_op_test(
        ndl_op=lambda x: ops.mul_scalar(x, scalar),
        torch_op=lambda x: x * scalar,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=single_array(), scalar=safe_float_strategy)
@backward_forward()
@all_devices()
def test_power_scalar(inputs, scalar, backward, device):
    generic_op_test(
        ndl_op=lambda x: ops.power_scalar(x, scalar),
        torch_op=lambda x: x**scalar,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=single_array(), scalar=float_strategy)
@backward_forward()
@all_devices()
def test_divide_scalar(inputs, scalar, backward, device):
    generic_op_test(
        ndl_op=lambda x: ops.divide_scalar(x, scalar),
        torch_op=lambda x: x / scalar,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_neg_scalar(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.neg_scalar,
        torch_op=lambda x: -x,
        inputs=inputs,
        backward=backward,
        device=device,
    )
