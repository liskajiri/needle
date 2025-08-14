import needle.ops as ops
import torch
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import single_array
from tests.ops.utils import generic_op_test
from tests.utils import backward_forward


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
