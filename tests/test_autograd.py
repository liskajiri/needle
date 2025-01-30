import random

import needle as ndl
import numdifftools as nd
import numpy as np
import pytest
import torch
from mnist_needle import loss_err, nn_epoch, softmax_loss
from mnist_numpy import parse_mnist
from needle.data.datasets.mnist import MNISTPaths

from tests.gradient_check import backward_check as gradient_check


##############################################################################
### TESTS for forward passes
def test_divide_forward():
    np.testing.assert_allclose(
        ndl.ops.divide(
            ndl.Tensor([[3.3, 4.35, 1.2], [2.45, 0.95, 2.55]]),
            ndl.Tensor([[4.6, 4.35, 4.8], [0.65, 0.7, 4.4]]),
        ).numpy(),
        np.array(
            [
                [0.717391304348, 1.0, 0.25],
                [3.769230769231, 1.357142857143, 0.579545454545],
            ]
        ),
    )


def test_divide_scalar_forward():
    np.testing.assert_allclose(
        ndl.ops.divide_scalar(ndl.Tensor([[1.7, 1.45]]), scalar=12).numpy(),
        np.array([[0.141666666667, 0.120833333333]]),
    )


# TODO: this module should only test the backward passes
# - move the forwards to separate tests/test_ops.py

# Also should be hypothesis tests with small ranges
TENSOR_MATMUL_CASES = [
    (
        [[4.95, 1.75, 0.25], [4.15, 4.25, 0.3], [0.3, 0.4, 2.1]],
        [[1.35, 2.2, 1.55], [3.85, 4.8, 2.6], [1.15, 0.85, 4.15]],
        [[13.7075, 19.5025, 13.26], [22.31, 29.785, 18.7275], [4.36, 4.365, 10.22]],
    ),
    (
        [[3.8, 0.05], [2.3, 3.35], [1.6, 2.6]],
        [[1.1, 3.5, 3.7], [0.05, 1.25, 1.0]],
        [[4.1825, 13.3625, 14.11], [2.6975, 12.2375, 11.86], [1.89, 8.85, 8.52]],
    ),
    (
        [
            [[4.0, 2.15], [1.25, 1.35], [0.75, 1.6]],
            [[2.9, 2.15], [3.3, 4.1], [2.5, 0.25]],
            [[2.9, 4.35], [1.2, 3.5], [3.55, 3.95]],
            [[2.55, 4.35], [4.25, 0.2], [3.95, 3.4]],
            [[2.2, 2.05], [0.95, 1.8], [2.7, 2.0]],
            [[0.45, 1.1], [3.15, 0.7], [2.9, 1.95]],
        ],
        [
            [[2.7, 4.05, 0.1], [1.75, 3.05, 2.3]],
            [[0.55, 4.1, 2.3], [4.45, 2.35, 2.55]],
            [[1.2, 3.95, 4.6], [4.2, 3.5, 3.35]],
            [[2.55, 4.4, 2.05], [2.4, 0.6, 4.65]],
            [[2.95, 0.8, 0.6], [0.45, 1.3, 0.75]],
            [[1.25, 2.1, 0.4], [0.85, 3.5, 3.7]],
        ],
        [
            [
                [14.5625, 22.7575, 5.345],
                [5.7375, 9.18, 3.23],
                [4.825, 7.9175, 3.755],
            ],
            [
                [11.1625, 16.9425, 12.1525],
                [20.06, 23.165, 18.045],
                [2.4875, 10.8375, 6.3875],
            ],
            [
                [21.75, 26.68, 27.9125],
                [16.14, 16.99, 17.245],
                [20.85, 27.8475, 29.5625],
            ],
            [
                [16.9425, 13.83, 25.455],
                [11.3175, 18.82, 9.6425],
                [18.2325, 19.42, 23.9075],
            ],
            [
                [7.4125, 4.425, 2.8575],
                [3.6125, 3.1, 1.92],
                [8.865, 4.76, 3.12],
            ],
            [
                [1.4975, 4.795, 4.25],
                [4.5325, 9.065, 3.85],
                [5.2825, 12.915, 8.375],
            ],
        ],
    ),
]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    TENSOR_MATMUL_CASES,
    ids=lambda case: f"{len(case[0])}@{len(case[1])}",
)
def test_matmul_from_tensor(a, b, expected):
    a = ndl.Tensor(a)
    b = ndl.Tensor(b)
    result = ndl.matmul(a, b)
    np.testing.assert_allclose(result.numpy(), np.array(expected), rtol=1e-6, atol=1e-6)


def test_summation_forward():
    np.testing.assert_allclose(
        ndl.ops.ops_mathematic.summation(
            ndl.Tensor(
                [
                    [2.2, 4.35, 1.4, 0.3, 2.65],
                    [1.0, 0.85, 2.75, 3.8, 1.55],
                    [3.2, 2.3, 3.45, 0.7, 0.0],
                ]
            )
        ).numpy(),
        np.array(30.5),
    )
    np.testing.assert_allclose(
        ndl.ops.ops_mathematic.summation(
            ndl.Tensor(
                [
                    [1.05, 2.55, 1.0],
                    [2.95, 3.7, 2.6],
                    [0.1, 4.1, 3.3],
                    [1.1, 3.4, 3.4],
                    [1.8, 4.55, 2.3],
                ]
            ),
            axes=1,
        ).numpy(),
        np.array([4.6, 9.25, 7.5, 7.9, 8.65]),
    )
    np.testing.assert_allclose(
        ndl.ops.ops_mathematic.summation(
            ndl.Tensor([[1.5, 3.85, 3.45], [1.35, 1.3, 0.65], [2.6, 4.55, 0.25]]),
            axes=0,
        ).numpy(),
        np.array([5.45, 9.7, 4.35]),
    )


def test_broadcast_to_forward():
    np.testing.assert_allclose(
        ndl.broadcast_to(ndl.Tensor([[1.85, 0.85, 0.6]]), shape=(3, 3, 3)).numpy(),
        np.array(
            [
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
            ]
        ),
    )


def test_reshape_forward():
    np.testing.assert_allclose(
        ndl.reshape(
            ndl.Tensor(
                [
                    [2.9, 2.0, 2.4],
                    [3.95, 3.95, 4.65],
                    [2.1, 2.5, 2.7],
                    [1.9, 4.85, 3.25],
                    [3.35, 3.45, 3.45],
                ]
            ),
            shape=(15,),
        ).numpy(),
        np.array(
            [
                2.9,
                2.0,
                2.4,
                3.95,
                3.95,
                4.65,
                2.1,
                2.5,
                2.7,
                1.9,
                4.85,
                3.25,
                3.35,
                3.45,
                3.45,
            ]
        ),
    )
    np.testing.assert_allclose(
        ndl.reshape(
            ndl.Tensor(
                [
                    [[4.1, 4.05, 1.35, 1.65], [3.65, 0.9, 0.65, 4.15]],
                    [[4.7, 1.4, 2.55, 4.8], [2.8, 1.75, 2.8, 0.6]],
                    [[3.75, 0.6, 0.0, 3.5], [0.15, 1.9, 4.75, 2.8]],
                ]
            ),
            shape=(2, 3, 4),
        ).numpy(),
        np.array(
            [
                [
                    [4.1, 4.05, 1.35, 1.65],
                    [3.65, 0.9, 0.65, 4.15],
                    [4.7, 1.4, 2.55, 4.8],
                ],
                [[2.8, 1.75, 2.8, 0.6], [3.75, 0.6, 0.0, 3.5], [0.15, 1.9, 4.75, 2.8]],
            ]
        ),
    )


def test_negate_forward():
    np.testing.assert_allclose(
        ndl.negate(ndl.Tensor([[1.45, 0.55]])).numpy(), np.array([[-1.45, -0.55]])
    )


def test_transpose_forward():
    np.testing.assert_allclose(
        ndl.transpose(ndl.Tensor([[[1.95]], [[2.7]], [[3.75]]]), axes=(1, 2)).numpy(),
        np.array([[[1.95]], [[2.7]], [[3.75]]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor([[[[0.95]]], [[[2.55]]], [[[0.45]]]]), axes=(2, 3)
        ).numpy(),
        np.array([[[[0.95]]], [[[2.55]]], [[[0.45]]]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor(
                [
                    [[[0.4, 0.05], [2.95, 1.3]], [[4.8, 1.2], [1.65, 3.1]]],
                    [[[1.45, 3.05], [2.25, 0.1]], [[0.45, 4.75], [1.5, 1.8]]],
                    [[[1.5, 4.65], [1.35, 2.7]], [[2.0, 1.65], [2.05, 1.2]]],
                ]
            )
        ).numpy(),
        np.array(
            [
                [[[0.4, 2.95], [0.05, 1.3]], [[4.8, 1.65], [1.2, 3.1]]],
                [[[1.45, 2.25], [3.05, 0.1]], [[0.45, 1.5], [4.75, 1.8]]],
                [[[1.5, 1.35], [4.65, 2.7]], [[2.0, 2.05], [1.65, 1.2]]],
            ]
        ),
    )
    np.testing.assert_allclose(
        ndl.transpose(ndl.Tensor([[[2.45]], [[3.5]], [[0.9]]]), axes=(0, 1)).numpy(),
        np.array([[[2.45], [3.5], [0.9]]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(ndl.Tensor([[4.4, 2.05], [1.85, 2.25], [0.15, 1.4]])).numpy(),
        np.array([[4.4, 1.85, 0.15], [2.05, 2.25, 1.4]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor([[0.05, 3.7, 1.35], [4.45, 3.25, 1.95], [2.45, 4.4, 4.5]])
        ).numpy(),
        np.array([[0.05, 4.45, 2.45], [3.7, 3.25, 4.4], [1.35, 1.95, 4.5]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor(
                [
                    [[0.55, 1.8, 0.2], [0.8, 2.75, 3.7], [0.95, 1.4, 0.8]],
                    [[0.75, 1.6, 1.35], [3.75, 4.0, 4.55], [1.85, 2.5, 4.8]],
                    [[0.2, 3.35, 3.4], [0.3, 4.85, 4.85], [4.35, 4.25, 3.05]],
                ]
            ),
            axes=(0, 1),
        ).numpy(),
        np.array(
            [
                [[0.55, 1.8, 0.2], [0.75, 1.6, 1.35], [0.2, 3.35, 3.4]],
                [[0.8, 2.75, 3.7], [3.75, 4.0, 4.55], [0.3, 4.85, 4.85]],
                [[0.95, 1.4, 0.8], [1.85, 2.5, 4.8], [4.35, 4.25, 3.05]],
            ]
        ),
    )


##############################################################################
### TESTS for backward passes


def test_divide_backward():
    gradient_check(
        ndl.ops.divide,
        ndl.Tensor(np.random.randn(5, 4)),
        ndl.Tensor(5 + np.random.randn(5, 4)),
    )


def test_divide_scalar_backward():
    gradient_check(
        ndl.ops.divide_scalar,
        ndl.Tensor(np.random.randn(5, 4)),
        scalar=random.random(),
    )


def test_matmul_simple_backward():
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(5, 4)),
        ndl.Tensor(np.random.randn(4, 5)),
    )


# TODO: forward pass
@pytest.mark.slow
def test_matmul_batched_backward():
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(6, 6, 5, 4)),
        ndl.Tensor(np.random.randn(6, 6, 4, 3)),
    )
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(1, 2, 1, 2)),
        ndl.Tensor(np.random.randn(2, 1)),
    )
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(6, 6, 5, 4)),
        ndl.Tensor(np.random.randn(4, 3)),
    )
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(5, 4)),
        ndl.Tensor(np.random.randn(6, 6, 4, 3)),
    )


def test_reshape_backward():
    gradient_check(ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(4, 5))


def test_negate_backward():
    gradient_check(ndl.negate, ndl.Tensor(np.random.randn(5, 4)))


def test_transpose_backward():
    gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
    gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(0, 1))


def test_broadcast_to_backward():
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(
        ndl.broadcast_to,
        ndl.Tensor(
            np.random.randn(
                1,
            )
        ),
        shape=(3, 3, 3),
    )
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(
        ndl.broadcast_to, ndl.Tensor(np.random.randn(5, 4, 1)), shape=(5, 4, 3)
    )
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(5)), shape=(1, 5))


def test_broadcast_to_backward_my():
    gradient_check(
        ndl.broadcast_to,
        ndl.Tensor(np.random.randn(2, 3, 1, 5)),
        shape=(2, 3, 4, 5),
    )


def test_summation_backward():
    gradient_check(
        ndl.ops.ops_mathematic.summation,
        ndl.Tensor(np.random.randn(5, 4)),
        axes=(1,),
    )
    gradient_check(
        ndl.ops.ops_mathematic.summation,
        ndl.Tensor(np.random.randn(5, 4)),
        axes=(0,),
    )
    gradient_check(
        ndl.ops.ops_mathematic.summation,
        ndl.Tensor(np.random.randn(5, 4)),
        axes=(0, 1),
    )
    gradient_check(
        ndl.ops.ops_mathematic.summation,
        ndl.Tensor(np.random.randn(5, 4, 1)),
        axes=(0, 1),
    )


##############################################################################
### TESTS for find_topo_sort


def test_topo_sort():
    # Test case 1
    a1, b1 = (
        ndl.Tensor(np.asarray([[0.88282157]])),
        ndl.Tensor(np.asarray([[0.90170084]])),
    )
    c1 = 3 * a1 * a1 + 4 * b1 * a1 - a1

    soln = np.array(
        [
            np.array([[0.88282157]]),
            np.array([[2.64846471]]),
            np.array([[2.33812177]]),
            np.array([[0.90170084]]),
            np.array([[3.60680336]]),
            np.array([[3.1841638]]),
            np.array([[5.52228558]]),
            np.array([[-0.88282157]]),
            np.array([[4.63946401]]),
        ]
    )

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c1])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 2
    a1, b1 = (
        ndl.Tensor(np.asarray([[0.20914675], [0.65264178]])),
        ndl.Tensor(np.asarray([[0.65394286, 0.08218317]])),
    )
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    soln = [
        np.array([[0.65394286, 0.08218317]]),
        np.array([[0.20914675], [0.65264178]]),
        np.array([[0.19040619]]),
        np.array([[1.53101102, 0.19240724]]),
        np.array([[0.44577898]]),
        np.array([[0.63618518]]),
        np.array([[1.90855553]]),
        np.array([[3.40855553]]),
    ]

    topo_order = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

    assert len(soln) == len(topo_order)
    # step through list as entries differ in length
    for t, s in zip(topo_order, soln, strict=False):
        np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    # Test case 3
    a = ndl.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = ndl.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = (a @ b + b - a) @ a

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([e])])

    soln = np.array(
        [
            np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
            np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
            np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
            np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
            np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
            np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
            np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]]),
        ]
    )

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)


##############################################################################
### TESTS for compute_gradient_of_variables


def test_compute_gradient_sum_matmul():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Initialize tensors
    A_data = np.random.randn(10, 9)
    B_data = np.random.randn(9, 8)
    C_data = np.random.randn(10, 8)

    # Needle implementation
    A_ndl = ndl.Tensor(A_data)
    B_ndl = ndl.Tensor(B_data)
    C_ndl = ndl.Tensor(C_data)

    out_ndl = ndl.ops.ops_mathematic.summation(
        (A_ndl @ B_ndl + C_ndl) * (A_ndl @ B_ndl), axes=None
    )
    out_ndl.backward()

    # PyTorch implementation
    A_torch = torch.tensor(A_data, requires_grad=True)
    B_torch = torch.tensor(B_data, requires_grad=True)
    C_torch = torch.tensor(C_data, requires_grad=True)

    out_torch = ((A_torch @ B_torch + C_torch) * (A_torch @ B_torch)).sum()
    out_torch.backward()

    # Compare results
    np.testing.assert_allclose(out_ndl.numpy(), out_torch.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(A_ndl.grad.numpy(), A_torch.grad.numpy(), rtol=1e-4)
    np.testing.assert_allclose(B_ndl.grad.numpy(), B_torch.grad.numpy(), rtol=1e-4)
    np.testing.assert_allclose(C_ndl.grad.numpy(), C_torch.grad.numpy(), rtol=1e-4)


TEST_CASES = [
    (
        "matmul_add_multiply",
        lambda A, B, C: ndl.ops.ops_mathematic.summation(
            (A @ B + C) * (A @ B), axes=None
        ),
        [(10, 9), (9, 8), (10, 8)],
        lambda A, B, C: ((A @ B + C) * (A @ B)).sum(),
    ),
    (
        "broadcast_multiply",
        lambda A, B: ndl.ops.ops_mathematic.summation(
            ndl.broadcast_to(A, shape=(10, 9)) * B, axes=None
        ),
        [(10, 1), (10, 9)],
        lambda A, B: (A.expand(10, 9) * B).sum(),
    ),
    (
        "reshape_matmul_divide",
        lambda A, B, C: ndl.ops.ops_mathematic.summation(
            ndl.reshape(A, shape=(10, 10)) @ B / 5 + C, axes=None
        ),
        [(100,), (10, 5), (10, 5)],
        lambda A, B, C: (A.view(10, 10) @ B / 5 + C).sum(),
    ),
]


@pytest.mark.parametrize(("test_id", "ndl_fn", "shapes", "torch_fn"), TEST_CASES)
def test_compute_gradients(test_id, ndl_fn, shapes, torch_fn):
    ndl_inputs = [ndl.Tensor(np.random.randn(*shape)) for shape in shapes]
    torch_inputs = [torch.tensor(x.numpy(), requires_grad=True) for x in ndl_inputs]

    gradient_check(
        ndl_fn, *ndl_inputs, torch_fn=torch_fn, torch_args=torch_inputs, backward=True
    )


def test_compute_gradient_of_gradient():
    # check gradient of gradient
    x2 = ndl.Tensor([6])  # type: ignore[no-untyped-call]
    x3 = ndl.Tensor([0])  # type: ignore[no-untyped-call]
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1


##############################################################################
### TESTS for softmax_loss


def test_softmax_loss_ndl():
    # test backward pass for log
    gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5, 4)))

    X, y = parse_mnist(
        MNISTPaths.TRAIN_IMAGES,
        MNISTPaths.TRAIN_LABELS,
    )
    np.random.seed(0)
    Z = ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = ndl.Tensor(y_one_hot)
    np.testing.assert_allclose(
        softmax_loss(Z, y).numpy(), 2.3025850, rtol=1e-6, atol=1e-6
    )
    Z = ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
    np.testing.assert_allclose(
        softmax_loss(Z, y).numpy(), 2.7291998, rtol=1e-6, atol=1e-6
    )

    # TODO:
    # # test softmax loss backward
    # Zsmall = ndl.Tensor(np.random.randn(16, 10).astype(np.float32))
    # ysmall = ndl.Tensor(y_one_hot[:16])
    # gradient_check(softmax_loss, Zsmall, ysmall, tol=0.01, backward=True)


def test_softmax_loss_ndl_random_array():
    import torch
    import torch.nn.functional as F

    Z_np = np.random.randn(16, 10).astype(np.float32)
    y_np = np.zeros((16, 10))
    y_np[np.arange(16), np.random.randint(0, 10, 16)] = 1

    # Needle implementation
    Z_ndl = ndl.Tensor(Z_np)
    y_ndl = ndl.Tensor(y_np)
    loss_ndl = softmax_loss(Z_ndl, y_ndl)
    loss_ndl.backward()

    # PyTorch implementation
    Z_torch = torch.tensor(Z_np, requires_grad=True)
    y_torch = torch.tensor(np.argmax(y_np, axis=1))
    loss_torch = F.cross_entropy(Z_torch, y_torch)
    loss_torch.backward()

    # Compare results
    np.testing.assert_allclose(loss_ndl.numpy(), loss_torch.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(Z_ndl.grad.numpy(), Z_torch.grad.numpy(), rtol=1e-5)


##############################################################################
### TESTS for nn_epoch


def test_relu():
    # test forward/backward pass for relu
    np.testing.assert_allclose(
        ndl.relu(
            ndl.Tensor(
                [
                    [-46.9, -48.8, -45.45, -49.0],
                    [-49.75, -48.75, -45.8, -49.25],
                    [-45.65, -45.25, -49.3, -47.65],
                ]
            )
        ).numpy(),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    gradient_check(ndl.relu, ndl.Tensor(np.random.randn(5, 4)))


@pytest.mark.slow
def test_nn_epoch_ndl():
    # test nn gradients
    np.random.seed(0)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)

    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    W1_0, W2_0 = W1.copy(), W2.copy()

    W1 = ndl.Tensor(W1)
    W2 = ndl.Tensor(W2)

    X_ = ndl.Tensor(X)
    y_one_hot = np.zeros((y.shape[0], 3))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)

    dW1 = nd.Gradient(
        lambda W1_: softmax_loss(
            ndl.relu(X_ @ ndl.Tensor(W1_).reshape((5, 10))) @ W2, y_
        ).numpy()[0]
    )(W1.numpy())
    dW2 = nd.Gradient(
        lambda W2_: softmax_loss(
            ndl.relu(X_ @ W1) @ ndl.Tensor(W2_).reshape((10, 3)), y_
        ).numpy()[0]
    )(W2.numpy())
    W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch_size=50)
    np.testing.assert_allclose(
        dW1.reshape(5, 10), W1_0 - W1.numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        dW2.reshape(10, 3), W2_0 - W2.numpy(), rtol=1e-4, atol=1e-4
    )

    # test full epoch
    X, y = parse_mnist(MNISTPaths.TRAIN_IMAGES, MNISTPaths.TRAIN_LABELS)
    np.random.seed(0)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch_size=100)
    np.testing.assert_allclose(
        np.linalg.norm(W1.numpy()), 28.437788, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(W2.numpy()), 10.455095, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        loss_err(ndl.relu(ndl.Tensor(X) @ W1) @ W2, y),
        (0.19770025, 0.06006667),
        rtol=1e-4,
        atol=1e-4,
    )


# def init_weights(input_dim, output_dim):
#     """Initialize weights with proper scaling."""
#     return ndl.Tensor(
#         np.random.randn(input_dim, output_dim).astype(np.float32)
#         / np.sqrt(output_dim)
#     )


# def one_hot_encode(y, num_classes):
#     """Convert labels to one-hot encoding."""
#     y_one_hot = np.zeros((y.shape[0], num_classes))
#     y_one_hot[np.arange(y.size), y] = 1
#     return ndl.Tensor(y_one_hot)


# import torch
# import torch.nn.functional as F


# @pytest.mark.slow
# def test_nn_against_pytorch():
#     """Compare needle implementation against PyTorch reference."""
#     np.random.seed(0)
#     torch.manual_seed(0)

#     # Setup dimensions
#     batch_size, input_dim, hidden_dim, num_classes = 50, 5, 10, 3

#     # Generate random data
#     X_np = np.random.randn(batch_size, input_dim).astype(np.float32)
#     y_np = np.random.randint(num_classes, size=(batch_size,)).astype(np.uint8)

#     # Needle setup
#     X_ndl = ndl.Tensor(X_np)
#     W1_ndl = init_weights(input_dim, hidden_dim)
#     W2_ndl = init_weights(hidden_dim, num_classes)

#     # PyTorch setup
#     X_torch = torch.tensor(X_np, requires_grad=True)
#     W1_torch = torch.tensor(W1_ndl.numpy(), requires_grad=True)
#     W2_torch = torch.tensor(W2_ndl.numpy(), requires_grad=True)
#     y_torch = torch.tensor(y_np)

#     # Forward pass - Needle
#     hidden_ndl = ndl.relu(X_ndl @ W1_ndl)
#     out_ndl = hidden_ndl @ W2_ndl
#     loss_ndl = softmax_loss(out_ndl, y_np)

#     # Forward pass - PyTorch
#     hidden_torch = F.relu(X_torch @ W1_torch)
#     out_torch = hidden_torch @ W2_torch
#     loss_torch = F.cross_entropy(out_torch, y_torch)

#     # Backward pass
#     loss_torch.backward()

#     # Compare losses
#     np.testing.assert_allclose(
#         loss_ndl.numpy(), loss_torch.detach().numpy(), rtol=1e-5, atol=1e-5
#     )

#     # Compare gradients
#     W1_ndl.backward()
#     W2_ndl.backward()
#     np.testing.assert_allclose(
#         W1_ndl.grad.numpy(), W1_torch.grad.numpy(), rtol=1e-5, atol=1e-5
#     )
#     np.testing.assert_allclose(
#         W2_ndl.grad.numpy(), W2_torch.grad.numpy(), rtol=1e-5, atol=1e-5
#     )
