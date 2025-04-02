from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from needle.backend_selection import NDArray, array_api, default_device
from needle.tensor import Tensor

if TYPE_CHECKING:
    from needle.typing import AbstractBackend, DType


class Dictionary:
    """Creates a dictionary from a list of words, mapping each word to a
    unique integer.

    Attributes:
        word2idx: dictionary mapping from a word to its unique ID
        idx2word: list of words in the dictionary, in the order they were added
            to the dictionary (i.e. each word only appears once in this list)
    """

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def add_word(self, word: str) -> int:
        """Adds a word to the dictionary if not present.

        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.

        Args:
            word (str): The word to add to the dictionary.

        Returns:
            int: The word's unique ID.
        """
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self) -> int:
        """Returns the number of unique words in the dictionary.

        Returns:
            int: The number of unique words in the dictionary.
        """
        return len(self.idx2word)


class Corpus:
    """Creates corpus from train and test txt files.

    Args:
        base_dir (Path) : Path to directory containing text files.
            Defaults to Path("data/tree_bank").
        max_lines (int): Maximum number of lines to read from each file.
            Defaults to -1 (all lines).

    Attributes:
        dictionary: Dictionary object containing all words from corpus.
        train: Tokenized training data.
        test: Tokenized test data.
    """

    TRAIN_FILE = "train.txt"
    TEST_FILE = "test.txt"
    END_OF_SENTENCE_TOKEN = "<eos>"

    def __init__(
        self, base_dir: Path = Path("data/tree_bank"), max_lines: int = -1
    ) -> None:
        self.dictionary = Dictionary()
        self.train = self.tokenize(base_dir.joinpath(self.TRAIN_FILE), max_lines)
        self.test = self.tokenize(base_dir.joinpath(self.TEST_FILE), max_lines)

    def tokenize(self, path: Path, max_lines: int = -1) -> list[int]:
        """Tokenizes a text file to a list of IDs.

        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs.
        When adding words to the dictionary '<eos>' is appended to the end of
        each line to properly account for the end of the sentence.

        Args:
            path (Path) : Path to text file.
            max_lines (int) : Maximum number of lines to read in.
                Defaults to -1 (all lines).

        Returns:
            list[int]: List of token ids.
        """
        with path.open(encoding="utf-8") as f:
            ids = []
            for i, line in enumerate(f):
                if max_lines != -1 and i >= max_lines:
                    break
                words = line.strip().split()
                ids += [self.dictionary.add_word(word) for word in words]
                ids.append(self.dictionary.add_word(self.END_OF_SENTENCE_TOKEN))
        return ids


def batchify(
    data: NDArray,
    batch_size: int,
    device: AbstractBackend = default_device,
    dtype: DType = "float32",
) -> NDArray:
    """Arranges the dataset into columns for batch processing.

    Starting from sequential data, batchify arranges the dataset into columns.
    These columns are treated as independent by the model, which means that the
    dependence of e.g. `g` on `f` cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.

    Args:
        data (NDArray) : The sequential data to be batched.
        batch_size (int): Size of each batch.
        device (AbstractBackend): Device to store the output array.
            Defaults to default_device.
        dtype (DType): Data type for the output array. Defaults to "float32".

    Returns:
        NDArray: The data as an NDArray of shape (num_batch, batch_size).

    Example:
        >>> import numpy as np
        >>> from needle.backend_selection import NDArray, array_api, default_device
        >>> alphabet = np.array([i for i in range(24)])  # 0-23
        >>> alphabet_ndarray = NDArray(alphabet, device=default_device)
        >>> result = batchify(alphabet_ndarray, 4)
        >>> # Check shape is correct (6 batches of 4 elements each)
        >>> result.shape
        (6, 4)
        >>> print(result)
        [[ 0.  6. 12. 18.]
         [ 1.  7. 13. 19.]
         [ 2.  8. 14. 20.]
         [ 3.  9. 15. 21.]
         [ 4. 10. 16. 22.]
         [ 5. 11. 17. 23.]]
    """
    n_batches = len(data) // batch_size
    data = data[: n_batches * batch_size]
    data = NDArray(data, device=device, dtype=dtype)

    # reshape the data to (batch_size, n_batches)
    return array_api.transpose(data.reshape((batch_size, n_batches)))


def get_batch(
    batches: NDArray,
    i: int,
    seq_len: int,
    device: AbstractBackend = default_device,
    dtype: DType = "float32",
) -> tuple[Tensor, Tensor]:
    """Subdivides the source data into chunks of length seq_len.

    The subdivision of data is not done along the batch dimension (i.e. dimension 1),
    since that was handled by the batchify function.
    The chunks are along dimension 0, corresponding to the seq_len
    dimension in the LSTM or RNN.

    Args:
        batches (NDArray) : NDArray returned from batchify function.
        i (int) : Starting index for the sequence.
        seq_len (int) : Sequence length for the chunks.
        device (AbstractBackend) : Device to store the output tensors.
            Defaults to default_device.
        dtype (DType) : Data type for the output tensors.
            Defaults to "float32".

    Returns:
        tuple(Tensor, Tensor):
        A tuple containing:
            data: Tensor of shape (seq_len, batch_size) with input sequences.
            target: Tensor of shape (seq_len*batch_size,) with target sequences.

    Example:
        >>> import numpy as np
        >>> from needle.backend_selection import NDArray, default_device
        >>> # Create a batched array similar to batchify example output
        >>> batched = np.array([[0, 6, 12, 18], [1, 7, 13, 19], [2, 8, 14, 20]])
        >>> batched_ndarray = NDArray(batched, device=default_device)
        >>> data, target = get_batch(batched_ndarray, i=0, seq_len=2)
        >>> data.shape
        (2, 4)
        >>> print(data)  # First two rows
        [[ 0.  6. 12. 18.]
         [ 1.  7. 13. 19.]]
        >>> target.shape
        (8,)
        >>> print(target)  # next two rows
        [ 1.  7. 13. 19.  2.  8. 14. 20.]
    """
    # Calculate sequence length, ensuring we don't go beyond available data
    seq_len = min(seq_len, batches.shape[0] - 1 - i)

    # current tokens
    data = Tensor(batches[i : i + seq_len], device=device, dtype=dtype)

    # next tokens
    target = Tensor(
        batches[i + 1 : i + 1 + seq_len].compact().flatten(), device=device, dtype=dtype
    )

    return data, target
