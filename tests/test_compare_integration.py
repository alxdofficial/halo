"""Integration tests for the comparison line's data path (IMWUT handoff W6/W8).

These cover the seams that unit tests on synthetic corpora cannot reach: the bridge from the
sampler's recordings to positions in the real Phase-A dataset, and the zero-shot support draw that
decides which evaluation streams have a k=0 row at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from data.scripts.curate import deployment_policy
from data.scripts.curate.compatibility import are_compatible, stream_key


@dataclass
class _Stream:
    """Only the two fields the zero-shot support draw reads off an evaluation stream."""

    dataset: str
    stream: str


@pytest.fixture(scope="module")
def small_index():
    from training.tokenizer.pretrain_data import CorpusIndex

    return CorpusIndex(
        max_per_stream=40, seed=3, datasets=("dsads", "forth_trace"), alignment="native",
    )


def test_bridge_positions_address_the_real_dataset(small_index):
    """``window_index`` must be a position in the dataset the trainer builds, or batches misalign."""
    from training.compare.corpus import support_corpus_from_index

    corpus = support_corpus_from_index(small_index)
    assert len(corpus) > 0
    for recording in corpus.recordings[:200]:
        key = small_index.train[recording.window_index]
        ref = small_index.refs[key.stream_i]
        assert ref.dataset == recording.dataset
        assert ref.stream == recording.stream
        assert str(ref.subjects[key.window_i]) == recording.subject


def test_bridge_labels_match_the_corpus_vocabulary(small_index):
    from training.compare.corpus import support_corpus_from_index

    corpus = support_corpus_from_index(small_index)
    vocabulary = set(small_index.label_ids)
    assert {recording.label for recording in corpus.recordings} <= vocabulary


def test_bridge_keys_agree_with_the_stream_spec(small_index):
    from training.compare.corpus import support_corpus_from_index

    corpus = support_corpus_from_index(small_index)
    for index, (dataset, stream) in enumerate(corpus.stream_names):
        assert corpus.keys[index] == stream_key(dataset, stream)


def test_episodes_drawn_over_the_real_corpus_obey_the_rules(small_index):
    """The sampler's guarantees must survive contact with real subjects and executions."""
    import numpy as np

    from training.compare.corpus import support_corpus_from_index
    from training.compare.sampling import draw_batch

    corpus = support_corpus_from_index(small_index)
    episodes, telemetry = draw_batch(
        corpus, np.random.default_rng(0), batch_size=24, support_size=8, p_gt_present=0.5,
    )
    assert episodes
    for episode in episodes:
        query = corpus.recordings[episode.query]
        query_key = corpus.key_of(query)
        for index in episode.support:
            other = corpus.recordings[index]
            assert other.subject != query.subject
            assert other.execution != query.execution
            assert are_compatible(query_key, corpus.key_of(other))
    assert 0.0 <= telemetry["sampler/realised_gt_rate"] <= 1.0


@pytest.mark.parametrize("dataset,stream", [("upper_limb_use", "control_left_wrist")])
def test_streams_without_a_compatible_partner_are_reported_unsupported(dataset, stream):
    """The audit's finding, pinned so it cannot be silently 'fixed' by widening the relation.

    ``upper_limb_use`` mounts a research IMU on the wrist while every training wrist stream is a
    watch, so no training recording shares its configuration. The honest k=0 answer is that the
    deployed mechanism does not apply, not a different mechanism wearing its name.
    """
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=20, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    support, reason = HALOCompareAdapter()._zero_shot_support(
        _Stream(dataset, stream), state, ["walking", "reaching"],
    )
    assert support is None
    assert "acquisition configuration" in reason


def test_zero_shot_support_excludes_every_candidate_label():
    """The support set must not contain the answer, or k=0 is not zero-shot."""
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    candidates = ["walking", "sitting", "standing"]
    support, reason = HALOCompareAdapter()._zero_shot_support(
        _Stream("inclusivehar", "phone_waist"), state, candidates,
    )
    assert support is not None, reason
    banned = {c.replace("_", " ").lower() for c in candidates}
    for recording in support:
        assert recording.label.replace("_", " ").lower() not in banned


def test_zero_shot_support_is_configuration_compatible():
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    query_key = stream_key("inclusivehar", "phone_waist")
    support, reason = HALOCompareAdapter()._zero_shot_support(
        _Stream("inclusivehar", "phone_waist"), state, ["walking"],
    )
    assert support is not None, reason
    for recording in support:
        assert are_compatible(query_key, stream_key(recording.dataset, recording.stream))


def test_zero_shot_support_never_draws_from_an_evaluation_dataset():
    """Support comes from TRAINING only; drawing from an eval dataset would leak."""
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    support, _ = HALOCompareAdapter()._zero_shot_support(
        _Stream("inclusivehar", "phone_waist"), state, ["walking"],
    )
    training = set(deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS)
    for recording in support:
        assert recording.dataset in training


def test_evaluation_datasets_are_absent_from_the_training_corpus():
    """The whole comparison rests on this; assert it rather than trusting the roster."""
    evaluation = {
        "inclusivehar", "usc_had", "tnda_har", "ut_complex",
        "monipar", "spar", "upper_limb_use",
    }
    assert evaluation.isdisjoint(deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS)
