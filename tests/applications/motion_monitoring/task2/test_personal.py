import pytest
import torch

from applications.motion_monitoring.task2.personal import fit_personal_variation


def test_personal_fit_learns_correlated_expected_variation() -> None:
    along = torch.linspace(-2.0, 2.0, 25)
    accepted = torch.stack((along, along + 0.03 * torch.sin(3 * along)), dim=1)
    model = fit_personal_variation(
        accepted,
        measurement_floor=torch.tensor([0.05, 0.05]),
        feature_names=("phase_1", "phase_2"),
    )

    expected_direction = model.score(torch.tensor([[1.5, 1.5]])).joint_deviation
    unexpected_direction = model.score(torch.tensor([[1.5, -1.5]])).joint_deviation

    assert model.sample_count == 25
    assert not model.reference_limited
    assert 0.0 <= model.shrinkage <= 1.0
    assert unexpected_direction.item() > expected_direction.item()


def test_personal_fit_is_honest_with_too_few_references() -> None:
    model = fit_personal_variation(
        torch.tensor([[1.0, 2.0], [1.1, 2.1]]), measurement_floor=0.2
    )

    assert model.reference_limited
    assert model.shrinkage == 1.0
    assert torch.allclose(model.covariance, torch.eye(2, dtype=torch.float64))
    assert torch.isfinite(model.score(torch.tensor([1.2, 2.2])).joint_deviation)


def test_personal_fit_rejects_invalid_measurement_floor() -> None:
    with pytest.raises(ValueError, match="measurement_floor"):
        fit_personal_variation(torch.ones(3, 2), measurement_floor=torch.ones(3))

    with pytest.raises(ValueError, match="positive"):
        fit_personal_variation(torch.ones(3, 2), measurement_floor=0.0)
