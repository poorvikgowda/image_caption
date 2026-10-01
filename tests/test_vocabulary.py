import os
import pytest
from src.data.vocabulary import Vocabulary

def test_vocabulary_building():
    sentences = [
        ["a", "dog", "runs"],
        ["a", "cat", "runs", "fast"],
        ["a", "dog", "sleeps"]
    ]
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocabulary(sentences)

    assert vocab.pad_idx == 0
    assert vocab.start_idx == 1
    assert vocab.end_idx == 2
    assert vocab.unk_idx == 3

    assert "a" in vocab.stoi
    assert "dog" in vocab.stoi
    assert "runs" in vocab.stoi
    assert "fast" not in vocab.stoi  # min_freq=2, 'fast' appears only once

def test_numericalization_and_decoding():
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary([["a", "red", "car"]])

    tokens = ["a", "red", "car"]
    num = vocab.numericalize(tokens, add_specials=True)
    assert num[0] == vocab.start_idx
    assert num[-1] == vocab.end_idx

    decoded = vocab.decode(num, remove_specials=True)
    assert decoded == ["a", "red", "car"]

def test_vocab_save_load(tmp_path):
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary([["hello", "world"]])

    save_path = os.path.join(tmp_path, "vocab.json")
    vocab.save(save_path)

    loaded_vocab = Vocabulary.load(save_path)
    assert len(loaded_vocab) == len(vocab)
    assert loaded_vocab.stoi["hello"] == vocab.stoi["hello"]
