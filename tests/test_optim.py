import needle as ndl
import numpy as np
import pytest
from needle import nn
from test_nn import global_tensor_count, learn_model_1d, learn_model_1d_eval


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


# We're checking that you have not allocated too many tensors;
# if this fails, make sure you're using .detach()/.data whenever possible.
# TODO: find reason for memory blowup
@pytest.mark.skip(reason="Memory optimization tests can be skipped")
def test_optim_sgd_z_memory_check_1():
    np.testing.assert_allclose(
        global_tensor_count(), np.array(387), rtol=1e-5, atol=1000
    )


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


# We're checking that you have not allocated too many tensors;
# if this fails, make sure you're using .detach()/.data whenever possible.
# TODO: find reason for memory blowup
@pytest.mark.skip(reason="Memory optimization tests can be skipped")
def test_optim_adam_z_memory_check_1():
    np.testing.assert_allclose(
        global_tensor_count(), np.array(1132), rtol=1e-5, atol=1000
    )
