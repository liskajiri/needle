import logging

import needle as ndl
import numpy as np
from needle import nn

logger = logging.getLogger(__name__)


def global_tensor_count() -> int:
    return ndl.autograd.value.Value._counter  # noqa: SLF001


def get_tensor(*shape, entropy=1) -> ndl.Tensor:
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(low, high, size=shape))


def learn_model_1d(feature_size, n_classes, _model, optimizer, epochs=1, **kwargs):
    np.random.seed(42)
    model = _model([])
    X = get_tensor(1024, feature_size).cached_data
    Y = get_int_tensor(1024, low=0, high=n_classes).cached_data.astype(np.uint8)
    m = X.shape[0]
    batch = 32

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)

    for _ in range(epochs):
        for _i, (X0, y0) in enumerate(
            zip(
                np.array_split(X, m // batch),
                np.array_split(Y, m // batch),
                strict=False,
            )
        ):
            opt.reset_grad()
            x, y = ndl.Tensor(X0, dtype="float32"), ndl.Tensor(y0)
            out = model(x)
            loss = loss_func(out, y)
            loss.backward()
            # Opt should not change gradients.
            grad_before = model.parameters()[0].grad.detach().cached_data
            opt.step()
            grad_after = model.parameters()[0].grad.detach().cached_data
            np.testing.assert_allclose(
                grad_before,
                grad_after,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Optim should not modify gradients in place",
            )

    return np.array(loss.cached_data)


def learn_model_1d_eval(feature_size, n_classes, _model, optimizer, epochs=1, **kwargs):
    np.random.seed(42)
    model = _model([])
    X = get_tensor(1024, feature_size).cached_data
    Y = get_int_tensor(1024, low=0, high=n_classes).cached_data.astype(np.uint8)
    m = X.shape[0]
    batch = 32

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)

    for _i, (X0, y0) in enumerate(
        zip(np.array_split(X, m // batch), np.array_split(Y, m // batch), strict=False)
    ):
        opt.reset_grad()
        x, y = ndl.Tensor(X0, dtype="float32"), ndl.Tensor(y0)
        out = model(x)
        loss = loss_func(out, y)
        loss.backward()
        opt.step()

    X_test = ndl.Tensor(get_tensor(batch, feature_size).cached_data)
    y_test = ndl.Tensor(
        get_int_tensor(batch, low=0, high=n_classes).cached_data.astype(np.uint8)
    )

    model.eval()

    return np.array(loss_func(model(X_test), y_test).cached_data)


def test_optim_sgd_vanilla_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.SGD,
            lr=0.01,
            momentum=0.0,
        ),
        np.array(3.207009),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_sgd_momentum_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.SGD,
            lr=0.01,
            momentum=0.9,
        ),
        np.array(3.311805),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_sgd_weight_decay_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.SGD,
            lr=0.01,
            momentum=0.0,
            weight_decay=0.01,
        ),
        np.array(3.202637),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_sgd_momentum_weight_decay_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.SGD,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.01,
        ),
        np.array(3.306993),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_sgd_layernorm_residual_1():
    nn.LayerNorm1d(8)
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(
                nn.Linear(64, 8),
                nn.ReLU(),
                nn.Residual(nn.Linear(8, 8)),
                nn.Linear(8, 16),
            ),
            ndl.optim.SGD,
            epochs=3,
            lr=0.01,
            weight_decay=0.001,
        ),
        np.array(2.852236),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_sgd_z_memory_check():
    # checks that not too many tensors are allocated for optimizers
    ndl.autograd.value.Value._counter = 0  # noqa: SLF001
    _a = (
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(
                nn.Linear(64, 8),
                nn.ReLU(),
                nn.Residual(nn.Linear(8, 8)),
                nn.Linear(8, 16),
            ),
            ndl.optim.SGD,
            epochs=3,
            lr=0.01,
            weight_decay=0.001,
        ),
    )

    error_tolerance = 500
    max_tensor_count = 387
    if global_tensor_count() > 0:
        logger.error("No tensors allocated")
    assert (
        max_tensor_count + error_tolerance >= global_tensor_count()
    ), f"""Allocated more tensors for SGD than needed,
        allocated {global_tensor_count()},
        but should be max {max_tensor_count}"""


def test_optim_adam_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.Adam,
            lr=0.001,
        ),
        np.array(3.703999),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_adam_weight_decay_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.Adam,
            lr=0.001,
            weight_decay=0.01,
        ),
        np.array(3.705134),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_adam_batchnorm_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 16)
            ),
            ndl.optim.Adam,
            lr=0.001,
            weight_decay=0.001,
        ),
        np.array(3.296256, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_adam_batchnorm_eval_mode_1():
    np.testing.assert_allclose(
        learn_model_1d_eval(
            64,
            16,
            lambda z: nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 16)
            ),
            ndl.optim.Adam,
            lr=0.001,
            weight_decay=0.001,
        ),
        np.array(3.192054, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_adam_layernorm_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.LayerNorm1d(32), nn.Linear(32, 16)
            ),
            ndl.optim.Adam,
            lr=0.01,
            weight_decay=0.01,
        ),
        np.array(2.82192, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_adam_weight_decay_bias_correction_1():
    np.testing.assert_allclose(
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.Adam,
            lr=0.001,
            weight_decay=0.01,
        ),
        np.array(3.705134),
        rtol=1e-5,
        atol=1e-5,
    )


def test_optim_adam_z_memory_check():
    # checks that not too many tensors are allocated for optimizers
    ndl.autograd.value.Value._counter = 0  # noqa: SLF001
    _a = (
        learn_model_1d(
            64,
            16,
            lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)),
            ndl.optim.Adam,
            lr=0.001,
            weight_decay=0.01,
        ),
    )
    max_tensor_count = 1132
    if global_tensor_count() > 0:
        logger.warning("No tensors allocated")
    assert (
        max_tensor_count >= global_tensor_count()
    ), f"""Allocated more tensors for Adam than needed,
        allocated {global_tensor_count()},
        but should be max {max_tensor_count}"""
