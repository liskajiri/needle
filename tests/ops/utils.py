import logging

import numpy as np
import torch
from needle import Tensor

RTOL = 1e-5
ATOL = 1e-5


def generic_op_test(ndl_op, torch_op, inputs, backward, device) -> None:
    # Create Needle tensors
    ndl_inputs = [Tensor(arr, requires_grad=backward, device=device) for arr in inputs]
    ndl_out = ndl_op(*ndl_inputs)

    # Create Torch tensors
    torch_inputs = [
        torch.tensor(arr, dtype=torch.float32, requires_grad=backward) for arr in inputs
    ]
    torch_out = torch_op(*torch_inputs)

    # Forward check
    if not isinstance(ndl_out, Tensor):
        ndl_out = ndl_out[0]
        torch_out = torch_out[0]

    np.testing.assert_allclose(
        ndl_out.numpy(), torch_out.detach().numpy(), rtol=RTOL, atol=ATOL
    )

    assert ndl_out.device == device

    if backward:
        ndl_out.sum().backward()

        # check that the gradients are on the same device as the inputs
        for ndl_input in ndl_inputs:
            if not hasattr(ndl_input, "grad"):
                # This is because LogSoftmax does not produce gradients
                logging.error(f"Input tensor {ndl_input} has no gradient.")
                return
            else:
                assert ndl_input.grad.device == device

        ndl_grads = [t.grad.numpy() for t in ndl_inputs]

        # Backward Torch
        grad_torch = torch.autograd.grad(outputs=torch_out.sum(), inputs=torch_inputs)
        grad_torch = [g.detach().numpy() for g in grad_torch]

        # Gradient checks
        for g_ndl, g_torch in zip(ndl_grads, grad_torch):
            np.testing.assert_allclose(g_ndl, g_torch, rtol=RTOL, atol=ATOL)
