import pytest
import torch

from applications.motion_monitoring.task3.consolidation import (
    recurrence_clusters,
    temporal_nms,
)
from applications.motion_monitoring.task3.metrics import (
    binary_auprc,
    binary_auroc,
    clustering_metrics,
    effective_rank,
    interval_discovery_metrics,
)


def test_temporal_nms_consolidates_multiscale_duplicates():
    starts = torch.tensor([0.0, 0.1, 4.0, 8.0])
    ends = torch.tensor([2.0, 2.1, 6.0, 10.0])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    kept = temporal_nms(starts, ends, scores, iou_threshold=0.5)

    assert kept.tolist() == [0, 2, 3]


def test_recurrence_graph_uses_mutual_edges_and_excludes_overlaps():
    starts = torch.tensor([0.0, 0.2, 5.0, 10.0, 15.0])
    ends = starts + 2.0
    affinity = torch.tensor(
        [
            [1.0, 0.99, 0.91, 0.90, 0.10],
            [0.99, 1.0, 0.20, 0.20, 0.10],
            [0.91, 0.20, 1.0, 0.92, 0.10],
            [0.90, 0.20, 0.92, 1.0, 0.10],
            [0.10, 0.10, 0.10, 0.10, 1.0],
        ]
    )

    clusters = recurrence_clusters(
        starts,
        ends,
        affinity,
        threshold=0.8,
        mutual_k=2,
        min_occurrences=2,
        overlap_iou=0.1,
    )

    assert len(clusters) == 1
    assert set(clusters[0].member_indices) == {0, 2, 3}
    assert clusters[0].medoid_index in {0, 2, 3}


def test_masked_pair_and_partition_metrics_are_correct():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1, 100.0])
    targets = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
    mask = torch.tensor([True, True, True, True, False])

    assert binary_auroc(scores, targets, mask) == 1.0
    assert binary_auprc(scores, targets, mask) == 1.0
    perfect = clustering_metrics(torch.tensor([7, 7, 9, 9]), torch.tensor([0, 0, 1, 1]))
    assert perfect["cluster/bcubed_f1"] == pytest.approx(1.0)
    assert perfect["cluster/pair_f1"] == pytest.approx(1.0)

    rank = effective_rank(torch.eye(4))
    assert 2.9 < rank <= 4.0


def test_interval_metrics_use_one_to_one_matching_and_real_recording_hours():
    predicted = torch.tensor([[0.0, 2.0], [0.1, 2.1], [8.0, 10.0]])
    target = torch.tensor([[0.0, 2.0], [4.0, 6.0]])

    metrics = interval_discovery_metrics(
        predicted, target, recording_duration_sec=3600.0, match_iou=0.5
    )

    assert metrics["discovery/occurrence_precision"] == pytest.approx(1 / 3)
    assert metrics["discovery/occurrence_recall"] == pytest.approx(1 / 2)
    assert metrics["discovery/count_absolute_error"] == 1.0
    assert metrics["discovery/false_occurrences_per_hour"] == 2.0
