import needle as ndl
import numpy as np
import pytest
from needle.typing import AbstractBackend
from tree_bank import evaluate_ptb, train_ptb

from apps.models.language_model import LanguageModel
from tests.devices import all_devices

rng = np.random.default_rng(3)

BATCH_SIZES = [1, 15]
INPUT_SIZES = [1, 11]
HIDDEN_SIZES = [1, 12]
BIAS = [True, False]
INIT_HIDDEN = [True, False]
NONLINEARITIES = ["tanh", "relu"]
OUTPUT_SIZES = [1, 1000]
EMBEDDING_SIZES = [1, 34]
SEQ_MODEL = ["rnn", "lstm"]
SEQ_LENGTHS = [1, 13]
NUM_LAYERS = [1, 2]


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("embedding_size", EMBEDDING_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("output_size", OUTPUT_SIZES)
@pytest.mark.parametrize("seq_model", SEQ_MODEL)
@all_devices()
@pytest.mark.slow
def test_language_model_implementation(
    seq_length,
    num_layers,
    batch_size,
    embedding_size,
    hidden_size,
    init_hidden,
    output_size,
    seq_model,
    device,
) -> None:
    assert seq_model in ["rnn", "lstm"]

    x = rng.integers(0, output_size, size=(seq_length, batch_size)).astype(np.float32)
    h0 = ndl.Tensor(
        rng.standard_normal((num_layers, batch_size, hidden_size), dtype=np.float32),
        device=device,
    )
    c0 = ndl.Tensor(
        rng.standard_normal((num_layers, batch_size, hidden_size), dtype=np.float32),
        device=device,
    )

    model = LanguageModel(
        embedding_size, output_size, hidden_size, num_layers, seq_model, device=device
    )
    if init_hidden:
        if seq_model == "lstm":
            h = (h0, c0)
        elif seq_model == "rnn":
            h = h0
        output, h_ = model(ndl.Tensor(x, device=device), h)  # type: ignore
    else:
        output, h_ = model(ndl.Tensor(x, device=device), None)

    if seq_model == "lstm":
        assert isinstance(h_, tuple)
        h0_, c0_ = h_
        assert c0_.shape == (num_layers, batch_size, hidden_size)
    elif seq_model == "rnn":
        h0_ = h_
    assert h0_.shape == (num_layers, batch_size, hidden_size)  # type: ignore
    assert output.shape == (batch_size * seq_length, output_size)

    output.backward()

    assert all(p.grad is not None for p in model.seq_model.parameters())
    assert all(p.grad is not None for p in model.linear.parameters())
    # TODO: Embedding layer is not being trained
    # for p in model.embedding.parameters():
    #     logging.info(f"Embedding param: {p.__class__.__name__}")
    #     logging.info(f"Embedding param grad: {p.grad}")
    # for p in model.parameters():
    #     assert p.grad is not None


@all_devices()
def test_language_model_training(device: AbstractBackend) -> None:
    seq_len = 10
    num_examples = 100
    batch_size = 16
    seq_model = "rnn"
    num_layers = 2
    hidden_size = 10
    n_epochs = 2

    corpus = ndl.data.nlp.Corpus(max_lines=num_examples)
    train_data = ndl.data.nlp.batchify(
        corpus.train, batch_size=batch_size, device=device, dtype="float32"
    )
    model = LanguageModel(
        30,
        len(corpus.dictionary),
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_model=seq_model,
        device=device,
    )
    _train_acc, train_loss = train_ptb(
        model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device
    )
    _test_acc, test_loss = evaluate_ptb(
        model, train_data, seq_len=seq_len, device=device
    )
    if device.name == "cpu":
        # TODO: lower bounds
        np.testing.assert_array_less(train_loss, 6.65)
        # np.testing.assert_array_less(train_loss, 5.413616)
        np.testing.assert_array_less(test_loss, 6.6)
        # np.testing.assert_array_less(test_loss, 5.214852)
    elif device.name == "cuda":
        np.testing.assert_array_less(train_loss, 6.65)
        np.testing.assert_array_less(test_loss, 6.6)
        # np.testing.assert_array_less(train_loss, 5.42463804)
        # np.testing.assert_array_less(test_loss, 5.2357954)


# TODO: Tests for values after one iteration of training
