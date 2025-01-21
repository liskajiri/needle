import needle as ndl
import numpy as np
import pytest
from needle import backend_ndarray as nd

np.random.seed(2)


_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

BPTT = [3, 32]

TRAIN = [True, False]
BATCH_SIZES = [1, 15]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("bptt", BPTT)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ptb_dataset(batch_size, bptt, train, device):
    # TODO update with more tests?
    corpus = ndl.data.Corpus("data/ptb")
    if train:
        data = ndl.data.batchify(
            corpus.train, batch_size, device=device, dtype="float32"
        )
    else:
        data = ndl.data.batchify(
            corpus.test, batch_size, device=device, dtype="float32"
        )
    X, y = ndl.data.get_batch(data, np.random.randint(len(data)), bptt, device=device)
    assert X.shape == (bptt, batch_size)
    assert y.shape == (bptt * batch_size,)
    assert isinstance(X, ndl.Tensor)
    assert X.dtype == "float32"
    assert X.device == device
    assert isinstance(X.cached_data, nd.NDArray)
    ntokens = len(corpus.dictionary)
    assert ntokens == 10000
