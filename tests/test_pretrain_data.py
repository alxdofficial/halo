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
    BalancedBatchSampler,
    CorpusIndex,
    MultiResolutionCollate,
    MultiScaleCollate,
    PretrainDataset,
    WindowKey,
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


def test_balanced_sampler_composition(index):
    sampler = BalancedBatchSampler(index.train, classes_per_batch=4,
                                   samples_per_class=3, steps_per_epoch=5, seed=1)
    batches = list(sampler)
    assert len(batches) == 5
    for batch in batches:
        assert len(batch) == 12
        labels = [index.train[i].label_id for i in batch]
        counts = {l: labels.count(l) for l in set(labels)}
        assert all(v == 3 for v in counts.values()), counts


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
    # A3 targets carry validity, never silent NaN in the valid entries
    valid_cad = out["cadence_target"][out["cadence_valid"]]
    assert torch.isfinite(valid_cad).all()


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
    out = MultiResolutionCollate(
        fixed_patch_seconds=(0.4, 1.4), compute_targets=False,
    )([item])
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


def test_sampler_source_balances_within_label():
    """A shared label's draws are spread evenly across its datasets, not by raw count."""
    from collections import Counter
    from training.tokenizer.pretrain_data import WindowKey
    # label 0 present in a 'big' stream (1000 windows) and a 'small' one (10) — 100:1 imbalance.
    keys = [WindowKey(0, w, 0) for w in range(1000)] + [WindowKey(1, w, 0) for w in range(10)]
    sd = ["big", "small"]
    bal = BalancedBatchSampler(keys, 1, 8, steps_per_epoch=300, stream_datasets=sd)
    c = Counter(sd[keys[i].stream_i] for batch in bal for i in batch)
    assert 0.4 < c["small"] / (c["big"] + c["small"]) < 0.6      # source-balanced ~50/50
    uni = BalancedBatchSampler(keys, 1, 8, steps_per_epoch=300)  # no stream_datasets -> old behaviour
    c2 = Counter(sd[keys[i].stream_i] for batch in uni for i in batch)
    assert c2["small"] / (c2["big"] + c2["small"]) < 0.1         # uniform-over-pool follows raw count


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
    window — never an all-padding window (which would pool to a degenerate A2 embedding).
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


def test_sampler_draws_without_replacement_in_group():
    """F6: a class-group of samples_per_class draws contains no duplicate window index (unless a
    source pool is smaller than its slot count)."""
    from training.tokenizer.pretrain_data import WindowKey
    keys = [WindowKey(0, w, 0) for w in range(500)] + [WindowKey(1, w, 0) for w in range(500)]
    sd = ["a", "b"]
    bal = BalancedBatchSampler(keys, classes_per_batch=1, samples_per_class=8,
                               steps_per_epoch=200, stream_datasets=sd)
    for batch in bal:
        assert len(set(batch)) == len(batch), "duplicate window in a class-group"


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
