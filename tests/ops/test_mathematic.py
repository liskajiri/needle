import needle.ops as ops
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.devices import all_devices
from tests.hypothesis_strategies import matmul_arrays, single_array
from tests.ops.utils import generic_op_test
from tests.utils import backward_forward

# ========== Summation


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_summation_all(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.summation,
        torch_op=torch.sum,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=single_array())
@all_devices()
def test_summation_axis(inputs, backward, device):
    generic_op_test(
        ndl_op=lambda x: ops.summation(x, axes=0),
        torch_op=lambda x: torch.sum(x, dim=0),
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_summation_axis_keepdims(inputs, backward, device):
    generic_op_test(
        ndl_op=lambda x: ops.summation(x, axes=0, keepdims=True),
        torch_op=lambda x: torch.sum(x, dim=0, keepdim=True),
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== Matmul


@given(inputs=matmul_arrays())
@backward_forward()
@all_devices()
def test_matmul(inputs, backward, device):
    # also tests batched matmul
    generic_op_test(
        ndl_op=ops.matmul,
        torch_op=torch.matmul,
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== Negate


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_negate(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.negate,
        torch_op=torch.neg,
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== Log


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_log(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.log,
        torch_op=torch.log,
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== Exp


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_exp(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.exp,
        torch_op=torch.exp,
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== ReLU


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_relu(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.relu,
        torch_op=torch.relu,
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== Sqrt


# sqrt negative not allowed
@given(inputs=single_array(elements=st.floats(min_value=0.0, max_value=10.0)))
@backward_forward()
@all_devices()
def test_sqrt(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.sqrt,
        torch_op=torch.sqrt,
        inputs=inputs,
        backward=backward,
        device=device,
    )


@given(inputs=single_array(elements=st.floats(max_value=-1)))
@backward_forward()
@all_devices()
def test_sqrt_negative_fail(inputs, backward, device):
    with pytest.raises(ValueError):
        generic_op_test(
            ndl_op=ops.sqrt,
            torch_op=torch.sqrt,
            inputs=inputs,
            backward=backward,
            device=device,
        )


# ========== Tanh


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_tanh(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.tanh,
        torch_op=torch.tanh,
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== Sigmoid


@given(inputs=single_array())
@backward_forward()
@all_devices()
def test_sigmoid(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.sigmoid,
        torch_op=torch.sigmoid,
        inputs=inputs,
        backward=backward,
        device=device,
    )


# ========== Mean


@given(inputs=single_array())
@backward_forward()
@all_devices()
@pytest.mark.xfail(reason="Known issue with broadcasting", strict=False)
def test_mean(inputs, backward, device):
    generic_op_test(
        ndl_op=ops.mean,
        torch_op=torch.mean,
        inputs=inputs,
        backward=backward,
        device=device,
    )
