import logging

import needle as ndl
import numpy as np
import pytest
from needle import nn
from needle.data.datasets.synthetic_mnist import SyntheticMNIST

logger = logging.getLogger(__name__)


IMAGE_SHAPE = (1, 28, 28)
NUM_SAMPLES = 32
NUM_CLASSES = 10
SEED = 1
EPOCHS = 10
BATCH = NUM_SAMPLES // 2
HIDDEN_SIZE = 16


def _compute_loss_acc(dataloader, model):
    """Compute average loss and accuracy over dataloader (no grad)."""
    loss_func = nn.SoftmaxLoss()
    total_loss = 0.0
    total_n = 0
    total_correct = 0
    model.eval()

    for X, y in dataloader:
        out = model(X)
        loss = loss_func(out, y)

        b = X.shape[0]

        total_loss += float(np.array(loss.cached_data) * b)
        preds = np.array(out.cached_data).argmax(axis=1)

        total_correct += int((preds == np.array(y.cached_data)).sum())
        total_n += b

    if total_n == 0:
        return 0.0, 0.0

    return total_correct / total_n, total_loss / total_n


def _train_full(
    dataset,
    model_builder,
    optimizer_cls,
    epochs=EPOCHS,
    batch=BATCH,
    seed=42,
    **opt_kwargs,
):
    """Train model on SyntheticMNIST dataset."""

    np.random.seed(seed)
    dataloader = ndl.data.DataLoader(dataset=dataset, batch_size=batch, shuffle=False)
    model = model_builder()
    opt = optimizer_cls(model.parameters(), **opt_kwargs)

    init_acc, init_loss = _compute_loss_acc(dataloader, model)
    model.train()

    loss_fn = nn.SoftmaxLoss()

    for _ in range(epochs):
        for X, y in dataloader:
            opt.zero_grad()

            out = model(X)

            loss = loss_fn(out, y)
            loss.backward()

            opt.step()

    final_acc, final_loss = _compute_loss_acc(dataloader, model)
    return init_acc, init_loss, final_acc, final_loss


def make_builder(
    kind: str = "flat", hidden_size: int = HIDDEN_SIZE, num_classes: int = NUM_CLASSES
):
    """Return a zero-argument callable that builds the requested model variant.

    kind: "flat", "batchnorm", or "layernorm".
    """
    _c, h, w = IMAGE_SHAPE
    in_features = h * w

    if kind == "flat":
        return lambda: nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
    if kind == "batchnorm":
        return lambda: nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )
    if kind == "layernorm":
        return lambda: nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.LayerNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )
    raise ValueError(f"unknown builder kind: {kind}")


@pytest.mark.parametrize("optimizer", [ndl.optim.SGD, ndl.optim.Adam])
@pytest.mark.parametrize("lr", [0.1, 0.01])
@pytest.mark.parametrize("weight_decay", [0.1, 0.01])
@pytest.mark.parametrize("model_type", ["flat", "batchnorm", "layernorm"])
def test_optimizers(optimizer, lr, weight_decay, model_type):
    image_shape = IMAGE_SHAPE
    ds = SyntheticMNIST(
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        image_shape=image_shape,
        seed=SEED,
    )

    opt_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
    }

    model_builder = make_builder(model_type)

    init_acc, init_loss, final_acc, final_loss = _train_full(
        ds,
        model_builder,
        optimizer,
        epochs=EPOCHS,
        batch=BATCH,
        **opt_kwargs,
    )

    assert float(final_loss) <= float(init_loss), (
        "Did not reduce loss on SyntheticMNIST"
    )
    assert float(final_acc) >= float(init_acc), (
        "Did not improve accuracy on SyntheticMNIST"
    )


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_SGD_momentum(momentum):
    optimizer = ndl.optim.SGD

    image_shape = IMAGE_SHAPE
    ds = SyntheticMNIST(
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        image_shape=image_shape,
        seed=SEED,
    )

    opt_kwargs = {
        "momentum": momentum,
    }

    model_builder = make_builder("flat")

    init_acc, init_loss, final_acc, final_loss = _train_full(
        ds,
        model_builder,
        optimizer,
        epochs=EPOCHS,
        batch=BATCH,
        **opt_kwargs,
    )

    assert float(final_loss) <= float(init_loss), (
        "Did not reduce loss on SyntheticMNIST"
    )
    assert float(final_acc) >= float(init_acc), (
        "Did not improve accuracy on SyntheticMNIST"
    )


@pytest.mark.parametrize(
    "optimizer, count", [(ndl.optim.Adam, 10), (ndl.optim.SGD, 10)]
)
def test_optim_memory(optimizer, count):
    def global_tensor_count() -> int:
        return ndl.autograd.value.Value._counter  # noqa: SLF001

    # reset counters
    ndl.autograd.value.Value._counter = 0

    ds = SyntheticMNIST(num_samples=256, num_classes=10, image_shape=(1, 1, 64), seed=7)

    def model_builder():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

    _ = _train_full(ds, model_builder, optimizer)

    max_tensor_count = count
    if global_tensor_count() > 0:
        logger.warning("No tensors allocated")
    assert (
        max_tensor_count >= global_tensor_count()
    ), f"""Allocated more tensors for {optimizer.__name__} than needed,
        allocated {global_tensor_count()},
        but should be max {max_tensor_count}"""
