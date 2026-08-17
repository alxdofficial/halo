import numpy as np
import torch

from training.evidence.eval_harnet_enrollment import support_predictions


def test_support_predictions_recover_separable_candidates():
    features = torch.tensor([
        [1.0, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 0.9],
        [0.8, 0.2], [0.2, 0.8],
    ])
    features = torch.nn.functional.normalize(features, dim=-1)
    predicted = support_predictions(
        features,
        support_rows=np.asarray([0, 1, 2, 3]),
        support_positions=np.asarray([0, 0, 1, 1]),
        query_rows=np.asarray([4, 5]),
        n_candidates=2,
        device=torch.device("cpu"),
    )
    for values in predicted.values():
        np.testing.assert_array_equal(values, np.asarray([0, 1]))


def test_support_predictions_requires_every_candidate():
    features = torch.eye(3)
    try:
        support_predictions(
            features,
            support_rows=np.asarray([0, 1]),
            support_positions=np.asarray([0, 0]),
            query_rows=np.asarray([2]),
            n_candidates=2,
            device=torch.device("cpu"),
        )
    except ValueError as error:
        assert "every candidate" in str(error)
    else:
        raise AssertionError("missing candidate support should fail")
