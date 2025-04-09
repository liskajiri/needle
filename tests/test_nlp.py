from needle.data import nlp


def test_dictionary_operations() -> None:
    """Test the Dictionary class operations."""
    dictionary = nlp.Dictionary()

    # Test adding words
    idx1 = dictionary.add_word("hello")
    assert idx1 == 0
    assert len(dictionary) == 1
    assert dictionary.word2idx["hello"] == 0
    assert dictionary.idx2word[0] == "hello"

    # Test adding the same word again (should return same index)
    idx2 = dictionary.add_word("hello")
    assert idx2 == idx1
    assert len(dictionary) == 1

    # Test adding multiple words
    idx3 = dictionary.add_word("world")
    assert idx3 == 1
    assert len(dictionary) == 2
    assert dictionary.word2idx["world"] == 1
    assert dictionary.idx2word[1] == "world"


def test_corpus_creation(tmp_path) -> None:
    """Test corpus creation with custom data."""
    # Create temporary train/test files
    corpus_dir = tmp_path / "tree_bank"
    corpus_dir.mkdir()

    train_file = corpus_dir / "train.txt"
    test_file = corpus_dir / "test.txt"

    train_file.write_text("hello world\nthis is a test")
    test_file.write_text("simple test\ndata for validation")

    # Create corpus with limited lines
    corpus = nlp.Corpus(base_dir=corpus_dir, max_lines=1)

    # Check dictionary and tokenized data
    assert (
        len(corpus.dictionary) == 4
    )  # "hello", "world", "<eos>", <unk>, from train + nothing from test
    assert len(corpus.train) == 3  # "hello", "world", "<eos>"
    assert len(corpus.test) == 3  # "simple", "test", "<eos>"

    # Check full corpus
    full_corpus = nlp.Corpus(base_dir=corpus_dir)
    assert len(full_corpus.dictionary) == 8  # All unique words + "<eos>"
    assert len(full_corpus.train) == 8  # 6 words + 2 "<eos>" tokens
    assert len(full_corpus.test) == 7  # 5 words + 2 "<eos>" tokens
