import numpy as np
import torch

from applications.motion_monitoring.task1 import (
    DetectionEpisode,
    DifferentiableSubsequenceMatcher,
    EmbeddingSequence,
    SyntheticDetectionDataset,
    collate_detection_episodes,
)
from applications.motion_monitoring.task1.matcher import (
    _decode_ranked_endpoints,
    _decode_ranked_endpoints_python,
    _dtw_tables,
    _dtw_tables_python,
)


def _sequence(values, valid=None):
    values = torch.as_tensor(values, dtype=torch.float32)
    intervals = torch.column_stack(
        [torch.arange(len(values)), torch.arange(1, len(values) + 1)]
    ).float()
    if valid is None:
        valid = torch.ones(len(values), dtype=torch.bool)
    return EmbeddingSequence(values, intervals, valid)


def test_alignment_masks_impossible_endpoints_padding_and_internal_gaps():
    reference = _sequence(torch.eye(4))
    query_values = torch.randn(9, 4)
    query_valid = torch.tensor([True, True, True, True, False, True, True, True, True])
    episode = DetectionEpisode(
        reference,
        _sequence(query_values, query_valid),
        torch.tensor([[1.0, 4.0]]),
    )
    batch = collate_detection_episodes([episode])

    output = DifferentiableSubsequenceMatcher(4)(batch)

    # Four reference patches require at least two query patches with bounded slope.
    assert output.endpoint_valid[0].tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    assert torch.isfinite(output.endpoint_logits).all()


def test_loss_guards_break_alignment_paths_instead_of_only_masking_endpoints():
    reference = _sequence(torch.eye(4))
    query = _sequence(torch.randn(6, 4))
    episode = DetectionEpisode(
        reference,
        query,
        torch.empty(0, 2),
        loss_valid=torch.tensor([True, True, False, True, True, True]),
    )

    output = DifferentiableSubsequenceMatcher(4)(
        collate_detection_episodes([episode])
    )

    assert output.endpoint_valid[0].tolist() == [False, True, False, False, True, True]


def test_all_trainable_matcher_parameters_receive_finite_nonzero_gradients():
    dataset = SyntheticDetectionDataset(
        8, feature_dim=8, query_patches=24, reference_patches=4, seed=3
    )
    batch = collate_detection_episodes([dataset[index] for index in range(8)])
    model = DifferentiableSubsequenceMatcher(8)

    output = model(batch)
    valid = batch.loss_valid & output.endpoint_valid
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.endpoint_logits[valid], batch.endpoint_targets[valid].float()
    )
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.any(parameter.grad != 0)


def test_alignment_loss_propagates_to_encoder_owned_input_embeddings():
    episode = SyntheticDetectionDataset(
        1, feature_dim=8, query_patches=24, reference_patches=4, seed=37
    )[0]
    reference = episode.reference.embeddings.detach().clone().requires_grad_()
    query = episode.query.embeddings.detach().clone().requires_grad_()
    differentiable_episode = DetectionEpisode(
        EmbeddingSequence(
            reference, episode.reference.intervals_sec, episode.reference.valid
        ),
        EmbeddingSequence(query, episode.query.intervals_sec, episode.query.valid),
        episode.targets_sec,
    )
    batch = collate_detection_episodes([differentiable_episode])
    output = DifferentiableSubsequenceMatcher(8)(batch)
    valid = batch.loss_valid & output.endpoint_valid

    torch.nn.functional.binary_cross_entropy_with_logits(
        output.endpoint_logits[valid], batch.endpoint_targets[valid].float()
    ).backward()

    assert reference.grad is not None and torch.any(reference.grad != 0)
    assert query.grad is not None and torch.any(query.grad != 0)


def test_projected_embeddings_remain_compatible_with_deployment_matcher():
    reference = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    query = torch.vstack([torch.tensor([[-1.0, 0.0]]).repeat(4, 1), reference])
    model = DifferentiableSubsequenceMatcher(2)
    intervals = np.column_stack([np.arange(len(query)), np.arange(1, len(query) + 1)])

    matches = model.detect(reference, query, intervals, score_threshold=0.01)

    assert [(match.start_patch, match.end_patch) for match in matches] == [(4, 7)]


def test_deployment_matcher_cannot_cross_invalid_query_gaps():
    reference = torch.eye(3)
    query = reference.clone()
    intervals = np.column_stack([np.arange(3), np.arange(1, 4)])
    model = DifferentiableSubsequenceMatcher(3)

    matches = model.detect(
        reference,
        query,
        intervals,
        score_threshold=0.1,
        query_valid=np.asarray([True, False, True]),
    )

    assert matches == []


def test_compiled_dtw_tables_exactly_match_reference_implementation():
    rng = np.random.default_rng(20260831)
    reference = rng.normal(size=(7, 11))
    query = rng.normal(size=(19, 11))
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    query /= np.linalg.norm(query, axis=1, keepdims=True)

    expected = _dtw_tables_python(reference, query, warp_penalty=0.05)
    observed = _dtw_tables(reference, query, warp_penalty=0.05)

    for expected_table, observed_table in zip(expected, observed, strict=True):
        np.testing.assert_array_equal(observed_table, expected_table)


def test_compiled_endpoint_decoding_exactly_matches_reference_implementation():
    rng = np.random.default_rng(19)
    reference = rng.normal(size=(7, 11))
    query = rng.normal(size=(31, 11))
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    accumulated, previous, path_lengths = _dtw_tables_python(
        reference, query, warp_penalty=0.05
    )
    endpoint_costs = accumulated[:, len(reference), 1:] / len(reference)
    endpoint_states = np.argmin(endpoint_costs, axis=0)
    endpoint_scores = endpoint_costs[
        endpoint_states, np.arange(endpoint_costs.shape[1])
    ]
    ranked = np.argsort(endpoint_scores, kind="stable")
    intervals = np.column_stack([np.arange(len(query)), np.arange(1, len(query) + 1)])
    arguments = (
        ranked,
        endpoint_states,
        endpoint_scores,
        previous,
        path_lengths,
        intervals,
        len(reference),
        0.3,
        None,
    )

    expected = _decode_ranked_endpoints_python(*arguments)
    observed = _decode_ranked_endpoints(*arguments)

    assert observed == expected
