"""Tests for the canonical global training-label vocabulary (ConSE closed-vocab)."""

from data.scripts.labels.build_global_label_mapping import global_label_vocabulary
from data.scripts.labels.canonical_labels import canonicalize


def test_global_vocab_is_canonical_sorted_and_deduped():
    vocab, counts = global_label_vocabulary()   # returns (vocabulary, per-label window counts)
    assert list(vocab) == sorted(vocab)
    assert set(counts) <= set(vocab), "counts key on labels outside the vocabulary"
    assert len(vocab) == len(set(vocab))
    # synonyms collapsed to their canonical form
    assert "lying" in vocab and "laying" not in vocab
    assert "cycling" in vocab and "bicycling" not in vocab
    assert "walking_upstairs" in vocab and "ascending_stairs" not in vocab
    assert "standing_up_from_sitting" in vocab and "sit_to_stand" not in vocab
    # every entry is already its own canonical form
    assert all(canonicalize(l) == l for l in vocab)
