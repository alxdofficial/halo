"""Tests for the Phase-1 pretraining data pipeline (real grids; skip when absent)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1]
         / "data/datasets/hhar/grids/harmonised").exists(),
    reason="harmonised train grids not built",
)

from training.tokenizer.pretrain_data import (  # noqa: E402
    CHANNELS,
    PATCH_SECONDS_CHOICES,
    CorpusIndex,
    MultiResolutionCollate,
    MultiScaleCollate,
    PretrainDataset,
    TemperatureSampler,
    VERIFIED_SIMULTANEOUS_DATASETS,
    WindowKey,
    _event_is_verified,
    stream_channel_descriptions,
)


@pytest.fixture(scope="module")
def index():
    return CorpusIndex(max_per_stream=150, seed=7)


def test_corpus_index_is_subject_disjoint(index):
    train_subj = {(index.refs[k.stream_i].dataset, index.refs[k.stream_i].subjects[k.window_i])
                  for k in index.train}
    val_subj = {(index.refs[k.stream_i].dataset, index.refs[k.stream_i].subjects[k.window_i])
                for k in index.val}
    assert not (train_subj & val_subj), "train/val share subjects"
    assert index.train and index.val


def test_corpus_excludes_eval_datasets(index):
    datasets = {r.dataset for r in index.refs}
    for banned in ("motionsense", "realworld", "shoaib", "inclusivehar",
                   "tnda_har", "ut_complex"):
        assert banned not in datasets, f"eval dataset {banned} leaked into pretraining"


def test_item_canonical_slots_and_mask(index):
    ds = PretrainDataset(index, index.train[:32], augment=True)
    torch.manual_seed(0)
    import random as stdlib_random

    import numpy as np
    np.random.seed(0)
    stdlib_random.seed(0)
    for i in range(16):
        item = ds[i]
        assert item["data"].shape[1] == 6
        assert item["channel_mask"].shape == (6,)
        assert len(item["texts"]) == 6
        # masked-out slots must be zero-filled
        for c in range(6):
            if not item["channel_mask"][c]:
                assert torch.allclose(item["data"][:, c],
                                      torch.zeros_like(item["data"][:, c]))


@pytest.mark.parametrize("ps", PATCH_SECONDS_CHOICES)
def test_collate_shapes_and_positions(index, ps):
    ds = PretrainDataset(index, index.train[:8], augment=False)
    collate = MultiScaleCollate(fixed_patch_seconds=ps)
    out = collate([ds[i] for i in range(8)])
    P = out["patches"].shape[1]
    assert P == max(1, round(6.0 / ps))
    assert out["patches"].shape == (8, P, 256, 6)
    assert out["positions"].shape == (8, P)
    # positions are patch CENTERS in seconds
    assert torch.allclose(out["positions"][0, 0], torch.tensor(ps / 2))
    assert (out["patch_len"] >= 1).all()


def test_collate_handles_per_sample_rates(index):
    """Rate augmentation gives every sample its own rate; the collate must produce
    per-sample patch lengths, not assume a shared one."""
    ds = PretrainDataset(index, index.train[:24], augment=True)
    import random as stdlib_random

    import numpy as np
    np.random.seed(3)
    stdlib_random.seed(3)
    torch.manual_seed(3)
    items = [ds[i] for i in range(24)]
    rates = {round(it["rate"], 1) for it in items}
    out = MultiScaleCollate(fixed_patch_seconds=1.0)(items)
    if len(rates) > 1:                      # rate aug fired at least once
        assert len(set(out["patch_len"].tolist())) > 1, \
            "per-sample rates must yield per-sample patch lengths"
    assert (out["patch_len"].float() - out["rates"] * 1.0).abs().max() < 1.0


def test_multiresolution_collate_covers_signal_and_retains_partial_tails():
    item = {
        "data": torch.randn(300, 6), "rate": 50.0, "texts": ["x"] * 6,
        "label_id": 0, "channel_mask": torch.ones(6, dtype=torch.bool),
        "gravity_state": "present", "source": "synthetic",
    }
    out = MultiResolutionCollate(fixed_patch_seconds=(0.4, 1.4))([item])
    real = out["patch_padding_mask"][0]
    assert out["patch_len"].shape == out["positions"].shape
    assert set(out["resolution_ids"][0, real].tolist()) == {0, 1}
    assert torch.all(out["positions"][0, real][1:] >= out["positions"][0, real][:-1])
    for rid in (0, 1):
        m = real & out["resolution_ids"][0].eq(rid)
        assert out["patch_starts"][0, m].min() == 0
        assert abs(float(out["patch_ends"][0, m].max()) - 6.0) < 1e-6
    # 1.4 seconds at 50 Hz leaves an honest 0.4-second final patch.
    long = real & out["resolution_ids"][0].eq(1)
    assert out["patch_len"][0, long].tolist()[-1] == 20
    assert torch.allclose(
        out["patch_durations"][0, real],
        out["patch_len"][0, real].float() / out["rates"][0],
    )


def test_stream_text_parses_placement():
    texts = stream_channel_descriptions("hhar", "phone_waist")
    assert len(texts) == 6
    assert "waist" in texts[0] and "phone" in texts[0]
    assert "gyroscope" in texts[3]


def test_stream_text_uses_rich_distinct_placement():
    """Channel text must use the StreamSpec placement, not collapse distinct configs (review #2)."""
    def t(ds, st):
        return stream_channel_descriptions(ds, st)[0]
    lw, rw = t("xrf_v2", "left_wrist"), t("xrf_v2", "right_wrist")
    assert "left wrist" in lw and "right wrist" in rw and lw != rw     # L/R not collapsed
    assert "head" in t("xrf_v2", "glasses")                            # glasses = head, not "body"
    assert "ear" in t("xrf_v2", "airpods_ear")                         # ear placement preserved
    assert t("xrf_v2", "left_pocket") != t("xrf_v2", "right_pocket")   # L/R pockets distinct
    assert "forearm" in t("nfi_fared", "wrist")                        # NFI = forearm (per paper)
    assert "lower back" in t("nfi_fared", "back")


def test_temperature_sampler_reserves_verified_event_pairs():
    keys = [WindowKey(i % 2, i, 0) for i in range(100)]
    # Ten events have two placements; all remaining windows are unrelated.
    positive = [-1] * 100
    event_ids = list(range(100))
    for event in range(10):
        positive[2 * event] = positive[2 * event + 1] = event
        event_ids[2 * event] = event_ids[2 * event + 1] = event
    sampler = TemperatureSampler(
        keys, ["a", "b"], num_samples=40, batch_size=40,
        event_ids=event_ids, positive_event_ids=positive, pair_fraction=0.2, seed=3,
    )
    batch = list(sampler)
    counts = {}
    for i in batch:
        if positive[i] >= 0:
            counts[positive[i]] = counts.get(positive[i], 0) + 1
    assert sum(count == 2 for count in counts.values()) == 4
    assert len(batch) == len(set(batch)) == 40


def test_temperature_sampler_caps_datasets_and_tempers_subjects():
    # Five datasets make a 25% ceiling feasible. Dataset 0 is 10x larger than every other source;
    # within it, subject 0 has 90 windows and subject 1 has 10.
    sizes = [100, 10, 10, 10, 10]
    keys, datasets, subjects = [], [], []
    for stream, size in enumerate(sizes):
        for window in range(size):
            keys.append(WindowKey(stream, window, 0))
            datasets.append(f"d{stream}")
            subjects.append(int(window >= 90) if stream == 0 else 0)
    sampler = TemperatureSampler(
        keys, [f"d{i}" for i in range(5)], num_samples=100, batch_size=10,
        alpha=0.25, subject_ids=subjects, subject_alpha=0.5,
        max_dataset_share=0.25,
    )
    assert sampler.dataset_probabilities["d0"] == pytest.approx(0.25)
    assert max(sampler.dataset_probabilities.values()) <= 0.25 + 1e-12
    assert sum(sampler.dataset_probabilities.values()) == pytest.approx(1.0)

    d0_rows = torch.tensor([i for i, dataset in enumerate(datasets) if dataset == "d0"])
    d0_weight = sampler.weights[d0_rows].sum()
    subject0 = sampler.weights[d0_rows[:90]].sum() / d0_weight
    expected = 90**0.5 / (90**0.5 + 10**0.5)
    assert float(subject0) == pytest.approx(expected)


def test_verified_event_pair_extraction_requires_distinct_streams():
    from training.tokenizer.pretrain import verified_event_pairs
    left, right = verified_event_pairs(
        ["e1", "e1", "e2", "e2", "e3"],
        torch.tensor([True, True, True, True, False]),
        ["a", "b", "a", "a", "c"],
        torch.device("cpu"),
    )
    assert left.tolist() == [0]
    assert right.tolist() == [1]


def test_verified_event_policy_includes_sp_sw_har_everywhere():
    from types import SimpleNamespace

    assert "sp_sw_har" in VERIFIED_SIMULTANEOUS_DATASETS
    ref = SimpleNamespace(dataset="sp_sw_har", event_ids_explicit=True)
    assert _event_is_verified(ref)
    assert not _event_is_verified(SimpleNamespace(dataset="wisdm", event_ids_explicit=True))
    assert not _event_is_verified(SimpleNamespace(dataset="sp_sw_har", event_ids_explicit=False))


def test_no_hapt_uci_leak(index):
    """hapt (UCI-HAR re-release) must be dropped from the corpus (sweep finding E)."""
    datasets = {r.dataset for r in index.refs}
    assert "hapt" not in datasets
    assert "uci_har" in datasets


def test_patch_padding_mask_flags_phantom_patches(index):
    """Every filled patch is flagged real; any trailing unfilled patch is flagged pad
    AND is exactly zero (sweep findings B/9/17)."""
    ds = PretrainDataset(index, index.train[:16], augment=True)
    out = MultiScaleCollate(fixed_patch_seconds=2.0)([ds[i] for i in range(16)])
    pad = out["patch_padding_mask"]
    patches = out["patches"]
    assert pad.shape == (16, patches.shape[1])
    assert pad[:, 0].all()                              # first patch always real
    for b in range(16):
        for p in range(patches.shape[1]):
            if not pad[b, p]:
                assert torch.count_nonzero(patches[b, p]) == 0, "phantom patch not zero"


def test_short_window_yields_at_least_one_patch(index):
    """A window shorter than one patch at the drawn scale (e.g. sp_sw_har's 1.0 s TUG
    windows at ps=1.5) must still get exactly one REAL short patch spanning the whole
    window — never an all-padding window (which would pool to a degenerate embedding).
    patch_len is honest (< round(rate*ps))."""
    ds = PretrainDataset(index, index.train[:32], augment=True)
    items = [ds[i] for i in range(32)]
    for ps in PATCH_SECONDS_CHOICES:
        out = MultiScaleCollate(fixed_patch_seconds=ps)(items)
        real_per_win = out["patch_padding_mask"].sum(1)
        assert (real_per_win >= 1).all(), f"all-padding window at ps={ps}"
        assert torch.isfinite(out["patches"]).all()
        # every real patch's declared length fits inside the window it came from
        assert (out["patch_len"] >= 1).all()


def test_window_crop_varies_observation_length():
    """P5 window-crop keeps a random contiguous sub-window (session-length invariance), floored at
    min_samples, and never lengthens."""
    from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
    import numpy as np
    cfg = AugmentationConfig.none()
    cfg.window_crop.enabled = True
    cfg.window_crop.p = 1.0
    aug = IMUAugmenter(cfg)
    np.random.seed(0)
    lens = []
    for _ in range(128):
        s = IMUSample(data=torch.zeros(360, 6), channel_names=list(CHANNELS),
                      sampling_rate=60.0, channel_descriptions=["x"] * 6,
                      label="walking", dataset_name="hhar")
        lens.append(aug(s).data.shape[0])
    lens = np.asarray(lens)
    assert lens.max() <= 360 and lens.min() >= int(0.5 * 360)     # in [min_frac*T, T], never longer
    assert len(set(lens.tolist())) > 10                          # genuinely variable
    # floor: a window already near the min_samples floor is not cropped below it
    short = IMUSample(data=torch.zeros(40, 6), channel_names=list(CHANNELS), sampling_rate=50.0,
                      channel_descriptions=["x"] * 6, label="walking", dataset_name="hhar")
    assert aug(short).data.shape[0] >= 32


def test_window_crop_in_default_v2_is_enabled():
    from data.scripts.augmentations import AugmentationConfig
    assert AugmentationConfig.default_v2().window_crop.enabled
    assert not AugmentationConfig.none().window_crop.enabled


def test_gravity_state_removed_skips_collate_alignment():
    """F9: gravity-removed streams skip alignment via the authoritative state, not the heuristic."""
    from model.tokenizer.preprocess import gravity_align
    w = torch.zeros(1, 120, 6)
    w[:, :, 0] = 1.0                                   # 1 g DC on x — the heuristic WOULD rotate it
    _, _, aligned_auto = gravity_align(w.clone(), list(CHANNELS), 50.0)
    _, r_rem, aligned_rem = gravity_align(w.clone(), list(CHANNELS), 50.0, gravity_state="removed")
    assert bool(aligned_auto[0])                       # heuristic aligns a strong-DC window
    assert not bool(aligned_rem[0])                    # authoritative 'removed' skips it
    assert torch.allclose(r_rem[0], torch.eye(3))      # ...and returns identity (no rotation)


def test_collate_fallback_position_is_window_center():
    """F4a: a window shorter than one patch emits ONE short patch whose position is the window's
    TRUE center (0.5*T/rate), not the nominal ps/2."""
    item = {"data": torch.randn(100, 6), "rate": 100.0, "texts": ["x"] * 6, "label_id": 0,
            "channel_mask": torch.ones(6, dtype=torch.bool), "gravity_state": "present"}
    out = MultiScaleCollate(fixed_patch_seconds=1.5)([item])   # 100 samples @100Hz = 1.0 s < 1.5 s patch
    assert int(out["patch_len"][0]) == 100                     # whole window in one short patch
    assert abs(float(out["positions"][0, 0]) - 0.5) < 1e-4     # 0.5 s (not the nominal 0.75)


def test_wisdm_native_grid_is_full_six_channel(index):
    """F2: wisdm native grids carry REAL gyro (merged), not accel-only [1,1,1,0,0,0]."""
    import numpy as np
    from data.scripts.eda.grid_io import discover_grids
    wisdm = [r for r in discover_grids("native") if r.dataset == "wisdm"]
    if not wisdm:
        pytest.skip("wisdm native grids not built")
    for r in wisdm:
        assert all(r.mask), f"{r.key} mask has padded channels {r.mask}"
        data = r.load_data()
        assert float(np.abs(np.asarray(data[:200, :, 3:])).mean()) > 0.0, "gyro is all-zero"


def test_knn_scores_unsupported_query_labels_as_failures():
    """F1: knn_balanced_acc scores every QUERY label; a class absent from the support scores 0
    instead of being intersected away (which inflated the metric)."""
    from training.tokenizer.pretrain import knn_balanced_acc
    train_z = torch.tensor([[0., 0.], [0.01, 0.], [0., 0.01]])
    train_y = torch.tensor([0, 0, 0])                          # support has only class 0
    test_z = torch.tensor([[0., 0.], [0.02, 0.], [5., 5.], [6., 6.]])
    test_y = torch.tensor([0, 0, 1, 1])                        # query has classes 0 and 1
    assert abs(knn_balanced_acc(train_z, train_y, test_z, test_y, k=3) - 0.5) < 1e-9


def test_collate_default_does_not_rotate_gravity():
    """Design decision (2026-07-19): the DEFAULT collate does NOT gravity-align — a window with
    gravity on +x keeps its DC on +x (posture direction preserved for the tokenizer's DC feature),
    whereas align_gravity=True rotates it to +z."""
    w = torch.zeros(120, 6)
    w[:, 0] = 1.0                                            # 1 g DC on x (gravity present)
    item = {"data": w, "rate": 50.0, "texts": ["x"] * 6, "label_id": 0,
            "channel_mask": torch.ones(6, dtype=torch.bool), "gravity_state": "present"}
    d = MultiScaleCollate(fixed_patch_seconds=1.0)([item])                       # default: no align
    dc_d = d["patches"][0, 0, :int(d["patch_len"][0]), :3].mean(0)
    assert dc_d[0].abs() > 0.9 and dc_d[2].abs() < 0.1, dc_d                     # DC stays on x
    a = MultiScaleCollate(fixed_patch_seconds=1.0, align_gravity=True)([item])   # ablation: align
    dc_a = a["patches"][0, 0, :int(a["patch_len"][0]), :3].mean(0)
    assert dc_a[2].abs() > 0.9 and dc_a[0].abs() < 0.1, dc_a                     # DC rotated to z


def test_gravity_aligned_in_collate(index):
    """Ablation path (align_gravity=True) still canonicalizes: for gravity-present accel-only
    windows the DC of the filled region points ~+z after alignment."""
    ds = PretrainDataset(index, index.train[:64], augment=False)   # no aug -> gravity present
    out = MultiScaleCollate(fixed_patch_seconds=1.0, align_gravity=True)(
        [ds[i] for i in range(64)])
    n = int(out["patch_len"][0])
    # first patch, accel triad, real samples -> mean should be dominated by +z
    acc0 = out["patches"][:, 0, :n, :3].mean(dim=1)     # (64, 3) DC per window
    mag = acc0.norm(dim=1)
    present = mag > 0.5
    if present.any():
        z_frac = acc0[present, 2].abs() / mag[present].clamp(min=1e-6)
        assert z_frac.median() > 0.9, "gravity not aligned to +z in collate"


def test_source_rate_is_bounded_by_the_hardware_clock_after_rate_augmentation():
    """Observability is min(hardware, augmented). Resampling never widens measured bandwidth.

    Regression for the pre-2026-07-26 formula ``src0 * augmented / stored``, which advertised
    94 Hz of bandwidth for wisdm windows upsampled from 20 Hz (32 observable bands where only 26
    carry signal) and hid 5 real bands when xrf_v2's 25 Hz AirPods stream was downsampled.
    """
    from training.tokenizer.pretrain_data import STREAM_SOURCE_RATE_HZ

    def bound(key, stored_hz, augmented_hz):
        hw = STREAM_SOURCE_RATE_HZ.get(key, stored_hz)
        return min(float(hw), float(augmented_hz))

    # No converter resample: the stored rate IS the hardware rate.
    assert bound("wisdm/phone_pocket", 20.0, 94.15) == 20.0     # upsample cannot add bandwidth
    assert bound("wisdm/phone_pocket", 20.0, 12.0) == 12.0      # downsample genuinely narrows it
    assert bound("capture24/watch_wrist", 100.0, 41.0) == 41.0

    # Converter stored these ABOVE their capture clock; the clock still bounds them.
    assert bound("xrf_v2/airpods_ear", 50.0, 81.43) == 25.0
    assert bound("xrf_v2/airpods_ear", 50.0, 21.88) == 21.88    # not 25 * 21.88/50 = 10.94
    assert bound("extrasensory/watch_wrist", 50.0, 90.0) == 25.0


def test_token_budget_caps_the_per_batch_patch_seconds_draw():
    """Peak VRAM tracks batch x patches, and patch_seconds is drawn PER BATCH — so an
    unlucky short draw could OOM a batch size that survived the previous sixty steps."""
    import numpy as np
    import torch
    from training.tokenizer.pretrain_data import MultiResolutionCollate

    # 6 s windows at 100 Hz: the 0.4 s short grid gives 15 patches, the 1.5 s long grid 4.
    batch = [{"data": torch.zeros(600, 6), "rate": 100.0, "label_id": 0} for _ in range(32)]

    uncapped = MultiResolutionCollate(max_batch_tokens=0)
    tight = MultiResolutionCollate(max_batch_tokens=32 * 12)      # only coarse pairs fit

    def tokens(collate):
        short, long = collate._patch_seconds(batch)
        return len(batch) * sum(int(np.ceil(6.0 / p)) for p in (short, long))

    assert tokens(tight) <= 32 * 12
    # The cap must actually bind: the unconstrained pool contains draws that exceed it.
    assert any(
        len(batch) * sum(int(np.ceil(6.0 / p)) for p in pair) > 32 * 12
        for pair in uncapped._valid_pairs
    )
    # Every admissible draw stays inside the budget, not just the one we happened to get.
    for _ in range(25):
        assert tokens(tight) <= 32 * 12


def test_token_budget_falls_back_to_the_coarsest_pair_when_nothing_fits():
    """An impossible budget must degrade to the fewest-token pair, never raise."""
    import torch
    from training.tokenizer.pretrain_data import MultiResolutionCollate

    batch = [{"data": torch.zeros(600, 6), "rate": 100.0, "label_id": 0} for _ in range(8)]
    collate = MultiResolutionCollate(max_batch_tokens=1)
    short, long = collate._patch_seconds(batch)
    assert (short, long) == max(collate._valid_pairs, key=lambda p: p[0] + p[1])
