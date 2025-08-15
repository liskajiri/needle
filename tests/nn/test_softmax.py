import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import needle as ndl
import numpy as np
import torch.nn.functional as F
from hypothesis import given

from tests.devices import all_devices
from tests.hypothesis_strategies import float_strategy
from tests.ops.utils import generic_op_test
from tests.utils import backward_forward

# Strategy to generate (rows, classes)
softmax_shape_strategy = st.tuples(
    # batch size
    st.integers(min_value=2, max_value=32),
    # num classes
    st.integers(min_value=2, max_value=32),
)


@given(shape=softmax_shape_strategy, Z=st.data())
@backward_forward()
@all_devices()
def test_softmax_loss(shape, Z, backward, device):
    rows, classes = shape

    logits = Z.draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(rows, classes),
            elements=float_strategy,
        )
    )

    labels = Z.draw(
        hnp.arrays(
            dtype=np.int64,
            shape=(rows,),
            elements=st.integers(min_value=0, max_value=classes - 1),
        )
    )

    generic_op_test(
        ndl_op=lambda logits, labels: ndl.nn.SoftmaxLoss()(logits, labels),
        torch_op=lambda logits, labels: F.cross_entropy(logits, labels.long()),
        inputs=(logits, labels),
        backward=backward,
        device=device,
    )
