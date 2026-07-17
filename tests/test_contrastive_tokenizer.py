import torch

from experiments.tokenizer_contrastive.data import (
    DomainBalancedBatchSampler,
    CoverageBalancedBatchSampler,
    WindowRecord,
    collate_windows,
)
from experiments.tokenizer_contrastive.losses import (
    contrastive_positive_mask,
    supervised_contrastive_loss,
)
from experiments.tokenizer_contrastive.views import PairedViewMaker, ViewConfig


def test_supervised_mask_never_treats_same_label_as_negative() -> None:
    labels = torch.tensor([0, 0, 1])

    mask = contrastive_positive_mask(labels, views=2, mode="supervised")

    assert mask.shape == (6, 6)
    assert mask[0, 1]
    assert mask[0, 2]
    assert mask[0, 3]
    assert not mask[0, 4]
    assert not mask.diagonal().any()


def test_contrastive_loss_rewards_same_class_alignment() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.0, 1.0]],
    ])
    bad = good.clone()
    bad[1] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    good_loss = supervised_contrastive_loss(good, labels)
    bad_loss = supervised_contrastive_loss(bad, labels)

    assert good_loss < bad_loss


def test_domain_balanced_sampler_has_requested_class_counts() -> None:
    records = []
    for label_id in range(3):
        for domain in ("a/phone", "b/watch"):
            for index in range(3):
                records.append(WindowRecord(
                    ref_index=0,
                    window_index=index,
                    label=f"label_{label_id}",
                    label_id=label_id,
                    domain=domain,
                    subject=f"s{index}",
                    split="train",
                ))
    sampler = DomainBalancedBatchSampler(
        records, classes_per_batch=2, samples_per_class=2, steps=1, seed=7
    )

    batch = next(iter(sampler))
    selected = [records[index] for index in batch]
    counts = {label: sum(record.label_id == label for record in selected) for label in range(3)}

    assert sorted(value for value in counts.values() if value) == [2, 2]
    for label in {record.label_id for record in selected}:
        assert len({record.domain for record in selected if record.label_id == label}) == 2


def test_rotation_views_preserve_triad_norms_and_cross_sensor_geometry() -> None:
    time = 40
    acc = torch.randn(time, 3)
    gyro = torch.randn(time, 3)
    data = torch.cat([acc, gyro], dim=1)
    maker = PairedViewMaker(ViewConfig(
        rotation_probability=1.0,
        jitter_probability=0.0,
        time_shift_probability=0.0,
    ))

    first, second = maker(
        data,
        ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        "present",
        60.0,
        "walking",
        "synthetic",
    )

    for view in (first, second):
        assert torch.allclose(view[:, :3].norm(dim=1), acc.norm(dim=1), atol=1e-5)
        assert torch.allclose(view[:, 3:].norm(dim=1), gyro.norm(dim=1), atol=1e-5)
        assert torch.allclose(
            (view[:, :3] * view[:, 3:]).sum(dim=1),
            (acc * gyro).sum(dim=1),
            atol=1e-5,
        )


def test_collate_pads_variable_windows_and_preserves_lengths() -> None:
    base = {
        "channel_mask": torch.ones(6, dtype=torch.bool),
        "rate_hz": 60.0,
        "label_id": 0,
        "label": "walking",
        "domain": "synthetic/phone",
        "subject": "s1",
        "window_index": 0,
    }
    short = {**base, "view_a": torch.ones(3, 6), "view_b": torch.ones(3, 6)}
    long = {
        **base,
        "view_a": torch.full((5, 6), 2.0),
        "view_b": torch.full((5, 6), 2.0),
        "window_index": 1,
    }

    batch = collate_windows([short, long])

    assert batch["view_a"].shape == (2, 5, 6)
    assert batch["window_length"].tolist() == [3, 5]
    assert torch.count_nonzero(batch["view_a"][0, 3:]) == 0


def test_coverage_sampler_consumes_every_record() -> None:
    records = []
    for label_id, count in enumerate((11, 5, 2)):
        for index in range(count):
            records.append(WindowRecord(
                ref_index=0,
                window_index=index,
                label=f"label_{label_id}",
                label_id=label_id,
                domain=f"domain_{index % 2}",
                subject=f"s{index}",
                split="train",
            ))
    sampler = CoverageBalancedBatchSampler(
        records, classes_per_batch=2, samples_per_class=2, seed=4
    )

    seen = {index: 0 for index in range(len(records))}
    for batch in sampler:
        for index in batch:
            seen[index] += 1

    assert all(count >= 1 for count in seen.values())
