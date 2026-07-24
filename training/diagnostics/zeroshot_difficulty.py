"""Zero-shot difficulty attribution harness for a frozen HALO tokenizer encoder.

READ-ONLY MEASUREMENT. Loads a frozen encoder + eval grids (no training) and decomposes
*why* zero-shot HAR is hard by attributing the performance drop to specific heterogeneity
axes, and — the key part — distinguishing whether each axis breaks the ENCODER
(representation) or the BRIDGE (label-transfer), quantified by distribution shift (MMD).

For each axis we build a ``matched`` window set and a ``shifted`` window set that differ
ONLY in that axis, then push both through ONE shared measurement function:

  * Encoder retention = kNN_BA(shifted) / kNN_BA(matched)          (subject-disjoint kNN)
  * Bridge   retention = ZS_macroF1(shifted) / ZS_macroF1(matched)  (ConSE text-cosine probe)
  * MMD               = RBF-kernel MMD (median-heuristic bandwidth) matched-vs-shifted embs
  * Verdict           = encoder-limited | bridge-limited | both | robust

Axes: rate, channel, orientation, gravity, placement, subject, label-novelty, plus a
compound (placement+orientation+rate) additivity check.

FAIL-LOUD METHODOLOGY GUARDS (see the per-axis docstrings for the exact contracts):
  * subject axis     — the shifted (held-out) query subjects are asserted DISJOINT from the
                       support subjects, so the probe never sees the "unseen" people. A support
                       that shared a single subject with the shifted query used to make the axis
                       trivially "robust ~1.0"; that is now impossible (it raises / marks INVALID).
  * label-novelty    — labels absent from Phase-A training are scored ZERO-SHOT: their labeled
                       examples are EXCLUDED from the ConSE probe fit and they are reached only
                       through the SBERT label-text prototype. An assertion forbids any novel-label
                       example from entering the probe support.
  * placement        — pairing is DATA-DETERMINED, never assumed. If the two placement grids are
                       row-aligned (identical per-window subject+label arrays, e.g. xrf_v2's
                       video-aligned body IMU), the matched vs shifted query are the SAME rows
                       (genuine same-subject-same-instant, device the only difference). Otherwise
                       the axis is relabelled an "unpaired cross-device distribution shift" and the
                       weaker control (A-support and B-query share subject + label set) is asserted.
Any axis whose control cannot be met with the available data emits an explicit
``verdict="AXIS INVALID"`` row with an ``invalid_reason`` instead of a misleading retention value.
The exact per-window transform axes (rate/channel/orientation/gravity) are unaffected.

Every metric primitive is REUSED from the training code:
  training.tokenizer.eval_transfer  -> build_encoder, encode_dataset, knn_balanced_acc
  training.tokenizer.pretrain       -> conse_probe_predict (ConSE text-cosine ZS head)
  training.tokenizer.pretrain_data  -> stream_channel_descriptions, _stream_gravity_state
  data.scripts.eda.grid_io          -> discover_grids
  data.scripts.augmentations        -> _random_so3 (SO(3) rotation), butter gravity removal

Run (real assets, small subset):
  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.diagnostics.zeroshot_difficulty \
      --max-windows 150 --eval-streams motionsense:phone_front_pocket realworld:phone_waist

Smoke without real assets (self-builds tiny synthetic grids + random encoder, then deletes):
  ... -m training.diagnostics.zeroshot_difficulty --synthetic --max-windows 60

NOT for headline numbers. Absolute values depend on the subset; the RETENTION RATIOS and
verdicts are the signal. Do NOT commit outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sps

from data.scripts.eda.grid_io import GridRef, discover_grids
from data.scripts.augmentations import _random_so3
from training.tokenizer.eval_transfer import (build_encoder, encode_dataset,
                                              knn_balanced_acc)
from training.tokenizer.pretrain import conse_probe_predict
from training.tokenizer.pretrain_data import (CHANNELS, stream_channel_descriptions,
                                              _stream_gravity_state)

DEFAULT_CKPT = Path("training/tokenizer/outputs/pretrain_native/best.pt")
# Held-out eval datasets (never in TRAIN_DATASETS) for the per-window transform axes.
DEFAULT_EVAL_STREAMS = (
    ("motionsense", "phone_front_pocket"),
    ("realworld", "phone_waist"),
    ("shoaib", "phone_right_pocket"),
    ("inclusivehar", "phone_waist"),
)
# Multi-placement dataset for the placement axis. xrf_v2's body IMU streams are cut from ONE
# video-aligned recording at the SAME sample index per placement, so their grids are row-aligned
# (window i is the same subject+instant across placements) -> genuine same-subject-same-instant
# pairing. Any (dataset, A, B) works; pairing is auto-detected from the grids at run time, and if
# the two grids are NOT row-aligned the axis degrades to an honest "unpaired cross-device" shift.
# Both defaults are body IMU with gravity present, so the pair isolates placement alone.
DEFAULT_PLACEMENT = ("xrf_v2", "left_wrist", "left_pocket")
KNN_K = 5
VERDICT_T = 0.9   # retention below this = "low"


# ======================================================================================
# Window carrier + subset sampling
# ======================================================================================
@dataclass
class WindowSet:
    """A bundle of raw windows + the config metadata the encoder needs to embed them."""
    data: np.ndarray                # (N, T, 6) float
    labels: np.ndarray              # (N,) str
    subjects: np.ndarray            # (N,) str
    texts: list                     # 6 per-channel description strings
    rate: float
    gravity_state: object           # 'present' | 'removed' | None
    channel_mask: tuple             # (6,) bool
    dataset: str
    stream: str
    tag: str = ""


def _stratified_subset(ref: GridRef, max_windows: int, seed: int) -> np.ndarray:
    """Deterministic per-label-balanced subset of window indices (covers rare labels/subjects)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(ref.labels)
    n = len(labels)
    if max_windows is None or n <= max_windows:
        return np.arange(n)
    uniq = sorted(set(labels.tolist()))
    per = max(1, max_windows // len(uniq))
    chosen: list[int] = []
    for lab in uniq:
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        chosen.extend(int(i) for i in idx[:per])
    if len(chosen) < max_windows:                       # top up to the cap
        rest = np.setdiff1d(np.arange(n), np.asarray(chosen, dtype=np.int64))
        rng.shuffle(rest)
        chosen.extend(int(i) for i in rest[: max_windows - len(chosen)])
    return np.sort(np.asarray(chosen[:max_windows], dtype=np.int64))


def _build_windowset(ref: GridRef, idx: np.ndarray) -> WindowSet:
    data = np.asarray(ref.load_data()[idx], dtype=np.float32)
    return WindowSet(
        data=data,
        labels=np.asarray(ref.labels)[idx],
        subjects=np.asarray(ref.subjects)[idx],
        texts=stream_channel_descriptions(ref.dataset, ref.stream),
        rate=float(ref.rate_hz),
        gravity_state=_stream_gravity_state(ref.dataset, ref.stream),
        channel_mask=tuple(ref.mask),
        dataset=ref.dataset,
        stream=ref.stream,
        tag=f"{ref.dataset}/{ref.stream}",
    )


def load_windowset(ref: GridRef, max_windows: int, seed: int) -> WindowSet:
    return _build_windowset(ref, _stratified_subset(ref, max_windows, seed))


def load_placement_pair(ref_a: GridRef, ref_b: GridRef, max_windows: int, seed: int):
    """Load two placement grids for the placement axis. If the FULL grids are ROW-ALIGNED (equal
    length and identical per-window subject+label arrays — as for a video-aligned multi-placement
    capture like xrf_v2), apply ONE shared stratified subset to BOTH so the returned windows stay
    paired (row i is the same subject at the same instant in A and B). Otherwise subset each grid
    independently, and the placement axis degrades to the honest unpaired cross-device control.
    Returns (wa, wb, row_aligned)."""
    aligned = (len(ref_a.labels) == len(ref_b.labels)
               and ref_a.labels == ref_b.labels and ref_a.subjects == ref_b.subjects)
    if aligned:
        idx = _stratified_subset(ref_a, max_windows, seed)
        return _build_windowset(ref_a, idx), _build_windowset(ref_b, idx), True
    return (load_windowset(ref_a, max_windows, seed),
            load_windowset(ref_b, max_windows, seed), False)


# ======================================================================================
# Per-axis window transforms (each returns a NEW WindowSet differing in ONE axis)
# ======================================================================================
def shift_rate(ws: WindowSet, target_hz: float) -> WindowSet:
    """Anti-aliased polyphase resample of the SAME windows to a lower rate (exact control)."""
    native = ws.rate
    tgt = target_hz if target_hz < native else native / 2.0
    frac = (np.array([tgt / native])).item()
    from fractions import Fraction
    f = Fraction(frac).limit_denominator(50)
    up, down = f.numerator, f.denominator
    rs = sps.resample_poly(ws.data, up, down, axis=1).astype(np.float32)
    actual = native * up / down
    out = _clone(ws, tag=ws.tag + f"|rate{actual:.0f}")
    out.data = rs
    out.rate = float(actual)
    return out


def shift_channel(ws: WindowSet) -> WindowSet:
    """6-ch -> 3-ch: zero the gyro triad AND turn its channel_mask off (acc-only deployment)."""
    data = ws.data.copy()
    data[:, :, 3:] = 0.0
    out = _clone(ws, tag=ws.tag + "|acc_only")
    out.data = data
    out.channel_mask = tuple(list(ws.channel_mask[:3]) + [False, False, False])
    return out


def shift_orientation(ws: WindowSet, seed: int) -> WindowSet:
    """One uniform-random SO(3) rotation PER WINDOW applied jointly to the acc and gyro triads
    (shared R = one rigid body frame; gravity rotates with the accel, as in rotation_3d)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = ws.data.copy()
    acc = torch.from_numpy(data[:, :, :3])
    gyr = torch.from_numpy(data[:, :, 3:])
    for i in range(data.shape[0]):
        R = _random_so3().to(acc.dtype)                  # (3,3)
        data[i, :, :3] = torch.einsum("ij,tj->ti", R, acc[i]).numpy()
        data[i, :, 3:] = torch.einsum("ij,tj->ti", R, gyr[i]).numpy()
    out = _clone(ws, tag=ws.tag + "|so3")
    out.data = data
    return out


def shift_gravity(ws: WindowSet) -> WindowSet:
    """Remove the gravity/DC component from the accelerometer (low-pass butter subtract),
    manufacturing the iOS userAcceleration representation. Only the acc triad carries gravity."""
    sr = ws.rate
    wn = 0.4 / (sr / 2.0)
    data = ws.data.copy()
    if 0.0 < wn < 1.0 and data.shape[1] > 12:
        b, a = sps.butter(2, wn, btype="low")
        for c in range(3):                               # acc channels only
            grav = sps.filtfilt(b, a, data[:, :, c], axis=1)
            data[:, :, c] = data[:, :, c] - grav
    else:                                                # rate too low for the filter -> DC subtract
        data[:, :, :3] -= data[:, :, :3].mean(axis=1, keepdims=True)
    out = _clone(ws, tag=ws.tag + "|grav_removed")
    out.data = data.astype(np.float32)
    out.gravity_state = "removed"
    out.texts = [_mark_grav_removed(t) if i < 3 else t for i, t in enumerate(ws.texts)]
    return out


def _mark_grav_removed(desc: str) -> str:
    d = desc.replace("; includes gravity", "").rstrip(" ;")
    return d + " (gravity removed)"


def _clone(ws: WindowSet, tag: str) -> WindowSet:
    return WindowSet(data=ws.data, labels=ws.labels, subjects=ws.subjects, texts=list(ws.texts),
                     rate=ws.rate, gravity_state=ws.gravity_state, channel_mask=ws.channel_mask,
                     dataset=ws.dataset, stream=ws.stream, tag=tag)


# ======================================================================================
# Encoding + metric primitives (all REUSED from training code)
# ======================================================================================
@torch.no_grad()
def embed(enc, ws: WindowSet, device) -> torch.Tensor:
    """(N, d) pooled embeddings via the shared eval_transfer.encode_dataset path (CPU tensor)."""
    return encode_dataset(enc, ws.data, ws.texts, device, ws.rate,
                          gravity_state=ws.gravity_state, channel_mask=ws.channel_mask,
                          dataset=ws.dataset, stream=ws.stream)


def build_label_protos(enc, id2label: dict, device) -> torch.Tensor:
    """(L, 384) L2-normalized frozen-SBERT prototype per label id — same recipe as
    pretrain.label_text_prototypes, but for an arbitrary label list."""
    prompts = [f"a person {id2label[i].replace('_', ' ')}" for i in range(len(id2label))]
    emb, mask = enc.text_encoder.encode(prompts, device=torch.device("cpu"))
    m = mask.unsqueeze(-1).float()
    proto = (emb * m).sum(1) / m.sum(1).clamp(min=1.0)
    return F.normalize(proto, dim=1)


def macro_f1(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Macro-averaged F1 over the labels present in true ∪ pred."""
    labels = sorted(set(true.tolist()) | set(pred.tolist()))
    f1s = []
    for c in labels:
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def rbf_mmd(X: torch.Tensor, Y: torch.Tensor, max_n: int = 400) -> float:
    """RBF-kernel MMD (square-rooted) with a median-heuristic bandwidth over the pooled sample."""
    X = X[:max_n].float()
    Y = Y[:max_n].float()
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    d = torch.pdist(torch.cat([X, Y], 0))
    sigma = d.median().clamp(min=1e-6)
    g = 1.0 / (2.0 * sigma * sigma)
    kxx = torch.exp(-g * torch.cdist(X, X) ** 2)
    kyy = torch.exp(-g * torch.cdist(Y, Y) ** 2)
    kxy = torch.exp(-g * torch.cdist(X, Y) ** 2)
    m, n = len(X), len(Y)
    mmd2 = ((kxx.sum() - kxx.diag().sum()) / (m * (m - 1))
            + (kyy.sum() - kyy.diag().sum()) / (n * (n - 1))
            - 2.0 * kxy.mean())
    return float(torch.sqrt(mmd2.clamp(min=0.0)))


def _ids(labels: np.ndarray, lab2id: dict) -> torch.Tensor:
    return torch.tensor([lab2id[str(l)] for l in labels], dtype=torch.long)


def score_pair(z_sup, y_sup, z_q, y_q, protos):
    """One (support -> query) scoring: (kNN balanced acc, ConSE macro-F1). y_* are id tensors."""
    ba = knn_balanced_acc(z_sup, y_sup.tolist(), z_q, y_q.tolist(), k=KNN_K)
    pred = conse_probe_predict(z_sup, y_sup, z_q, y_q, protos)
    f1 = macro_f1(pred, y_q)
    return ba, f1


# ======================================================================================
# THE shared per-axis measurement function
# ======================================================================================
def measure(name, enc, protos, lab2id, device, *,
            z_sup, y_sup, zq_m, yq_m, zq_s, yq_s, z_mmd_m, z_mmd_s,
            verdict_t: float = VERDICT_T) -> dict:
    """Decompose one axis. Support embeddings come from the MATCHED config (the encoder's
    comfort zone); the matched query and shifted query are scored against that SAME support so
    the shift is isolated as a query-time perturbation. Returns one result row."""
    ba_m, f1_m = score_pair(z_sup, y_sup, zq_m, yq_m, protos)
    ba_s, f1_s = score_pair(z_sup, y_sup, zq_s, yq_s, protos)
    enc_ret = ba_s / ba_m if ba_m and ba_m == ba_m and ba_m > 0 else float("nan")
    br_ret = f1_s / f1_m if f1_m and f1_m == f1_m and f1_m > 0 else float("nan")
    mmd = rbf_mmd(z_mmd_m, z_mmd_s)
    enc_low = enc_ret == enc_ret and enc_ret < verdict_t
    br_low = br_ret == br_ret and br_ret < verdict_t
    verdict = ("both" if enc_low and br_low else
               "encoder-limited" if enc_low else
               "bridge-limited" if br_low else "robust")
    return dict(axis=name, invalid=False, mmd=_r(mmd),
                knn_ba_matched=_r(ba_m), knn_ba_shifted=_r(ba_s), enc_retention=_r(enc_ret),
                zs_f1_matched=_r(f1_m), zs_f1_shifted=_r(f1_s), bridge_retention=_r(br_ret),
                verdict=verdict, n_support=int(len(y_sup)),
                n_query_matched=int(len(yq_m)), n_query_shifted=int(len(yq_s)))


def _invalid_row(axis: str, reason: str, **extra) -> dict:
    """A methodologically-invalid axis: emit an explicit AXIS INVALID marker with the reason
    instead of a misleading retention value. All metric fields are None so it is excluded from the
    attribution ranking and printed as INVALID rather than silently scoring as 'robust'."""
    print(f"  [AXIS INVALID] {axis}: {reason}", flush=True)
    row = dict(axis=axis, invalid=True, invalid_reason=reason, mmd=None,
               knn_ba_matched=None, knn_ba_shifted=None, enc_retention=None,
               zs_f1_matched=None, zs_f1_shifted=None, bridge_retention=None,
               verdict="AXIS INVALID", n_support=0, n_query_matched=0, n_query_shifted=0)
    row.update(extra)
    return row


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), nd)


def subject_split(subjects: np.ndarray, seed: int):
    """Deterministic subject-disjoint 50/50 split -> (support_subjects, held_subjects)."""
    subj = sorted(set(subjects.tolist()))
    np.random.default_rng(seed).shuffle(subj)
    hold = set(subj[: max(1, len(subj) // 2)])
    return set(subj) - hold, hold


# ======================================================================================
# Axis drivers
# ======================================================================================
def _local_labmap(*label_arrays):
    labs = sorted(set(str(l) for arr in label_arrays for l in arr))
    lab2id = {l: i for i, l in enumerate(labs)}
    id2lab = {i: l for l, i in lab2id.items()}
    return lab2id, id2lab


def run_aligned_axis(name, enc, base: WindowSet, shifted: WindowSet, device, seed):
    """Axes where matched & shifted are the SAME windows (rate/channel/orientation/gravity)."""
    lab2id, id2lab = _local_labmap(base.labels, shifted.labels)
    protos = build_label_protos(enc, id2lab, device)
    z_m = embed(enc, base, device)
    z_s = embed(enc, shifted, device)
    sup_subj, held_subj = subject_split(base.subjects, seed)
    sup = np.array([s in sup_subj for s in base.subjects])
    qry = ~sup
    y_all = _ids(base.labels, lab2id)
    return measure(name, enc, protos, lab2id, device,
                   z_sup=z_m[sup], y_sup=y_all[sup],
                   zq_m=z_m[qry], yq_m=y_all[qry],
                   zq_s=z_s[qry], yq_s=y_all[qry],
                   z_mmd_m=z_m, z_mmd_s=z_s)


def run_subject_axis(name, enc, ws: WindowSet, device, seed):
    """Subject axis (FINDING 4 fix): ONE support drawn only from SEEN subjects; the shift is
    whether the query subjects are in that support.

      * support       = SEEN subjects (half of each seen subject's windows)
      * matched query = the OTHER half of the SEEN subjects' windows (same people as the support)
      * shifted query = ALL windows of the HELD-OUT (unseen) subjects, DISJOINT from the support

    The previous version fit the shifted-subject probe on a random split whose support already
    contained the held-out subjects (empirically 100% overlap), so no subject shift was isolated
    and the axis read a meaningless "robust ~1.0". Here the support-subject set is asserted to be
    disjoint from the shifted-query-subject set (fail loud / mark INVALID otherwise), and every
    matched-query subject is asserted to appear in the support (a genuine "seen subject" control)."""
    subjects = np.asarray(ws.subjects)
    seen_subj, unseen_subj = subject_split(subjects, seed)
    if not seen_subj or not unseen_subj:
        return _invalid_row(name, f"cannot form a seen/unseen subject split from "
                            f"{len(set(subjects.tolist()))} subject(s) on {ws.tag}")
    lab2id, id2lab = _local_labmap(ws.labels)
    protos = build_label_protos(enc, id2lab, device)
    z = embed(enc, ws, device)
    y = _ids(ws.labels, lab2id)

    # Within SEEN subjects, split each subject's windows ~50/50 into support vs matched-query so
    # every matched-query subject is guaranteed to also appear in the support (seen condition).
    rng = np.random.default_rng(seed)
    sup_idx, mq_idx = [], []
    for subj in sorted(seen_subj):
        idx = np.where(subjects == subj)[0]
        rng.shuffle(idx)
        if len(idx) == 1:                                # single window -> keep the subject "seen"
            sup_idx.append(int(idx[0]))
        else:
            h = len(idx) // 2
            sup_idx.extend(int(i) for i in idx[:h])
            mq_idx.extend(int(i) for i in idx[h:])
    sq_idx = np.where(np.isin(subjects, list(unseen_subj)))[0]
    sup_idx = np.asarray(sorted(sup_idx), dtype=np.int64)
    mq_idx = np.asarray(sorted(mq_idx), dtype=np.int64)

    if len(sup_idx) == 0 or len(mq_idx) == 0 or len(sq_idx) == 0:
        return _invalid_row(name, f"empty support/matched/shifted partition on {ws.tag} "
                            f"(|sup|={len(sup_idx)} |matched|={len(mq_idx)} |shifted|={len(sq_idx)})")

    sup_subjects = set(subjects[sup_idx].tolist())
    mq_subjects = set(subjects[mq_idx].tolist())
    sq_subjects = set(subjects[sq_idx].tolist())
    # FINDING 4 assertion: the held-out (shifted) query subjects must NOT be in the support.
    leak = sup_subjects & sq_subjects
    if leak:
        raise AssertionError(f"[subject axis/{ws.tag}] support and shifted-query share subjects "
                             f"{sorted(leak)} — this is NOT a held-out-subject test.")
    # matched query must be drawn from subjects the support has actually seen.
    unseen_in_support = mq_subjects - sup_subjects
    if unseen_in_support:
        raise AssertionError(f"[subject axis/{ws.tag}] matched-query subjects {sorted(unseen_in_support)} "
                             f"are absent from the support — the matched condition is not 'seen subject'.")
    # metrics require some shared labels between support and each query
    if not (set(y[sup_idx].tolist()) & set(y[sq_idx].tolist())):
        return _invalid_row(name, f"support and shifted-query share no labels on {ws.tag}")

    row = measure(name, enc, protos, lab2id, device,
                  z_sup=z[sup_idx], y_sup=y[sup_idx],
                  zq_m=z[mq_idx], yq_m=y[mq_idx],
                  zq_s=z[sq_idx], yq_s=y[sq_idx],
                  z_mmd_m=z[sup_idx], z_mmd_s=z[sq_idx])
    row.update(n_support_subjects=len(sup_subjects), n_shifted_subjects=len(sq_subjects),
               subjects_disjoint=True)
    return row


def _rows_aligned(wa: WindowSet, wb: WindowSet) -> bool:
    """True iff A and B grids are row-aligned: same length and element-wise identical per-window
    subject AND label arrays. For a video-aligned multi-placement capture (e.g. xrf_v2) this holds
    because every placement is cut from ONE recording at the same sample index, so window i is the
    SAME subject at the SAME instant across placements — a genuine same-instant join key."""
    return (len(wa.labels) == len(wb.labels)
            and np.array_equal(np.asarray(wa.labels), np.asarray(wb.labels))
            and np.array_equal(np.asarray(wa.subjects), np.asarray(wb.subjects)))


def run_placement_axis(name, enc, wa: WindowSet, wb: WindowSet, device, seed):
    """Placement axis (FINDING 6 fix). Pairing is DATA-DETERMINED, never assumed:

    PAIRED (row-aligned grids, e.g. xrf_v2 body IMU): the matched and shifted query are the SAME
      rows (identical held-out-subject indices), so they are genuinely the same subject at the same
      instant with the same label and the DEVICE PLACEMENT is the only difference. Support = A on a
      DISJOINT set of (seen) subjects. Asserts row-alignment, matched==shifted pairing, and
      support/query subject-disjointness.

    UNPAIRED (grids not row-aligned, e.g. WISDM phone vs watch — different lengths, ~7% same-subject
      / ~4% same-label at equal index): there is NO instant-level join, so the axis is RELABELLED an
      "unpaired cross-device distribution shift". The weaker but valid control asserted here is that
      the A-support and the B-query share the same subject set and the same label set (device is the
      systematic difference; instant pairing is explicitly NOT claimed)."""
    lab2id, id2lab = _local_labmap(wa.labels, wb.labels)
    protos = build_label_protos(enc, id2lab, device)
    z_a = embed(enc, wa, device)
    z_b = embed(enc, wb, device)
    ya = _ids(wa.labels, lab2id)
    yb = _ids(wb.labels, lab2id)
    sa = np.asarray(wa.subjects)
    sb = np.asarray(wb.subjects)

    if _rows_aligned(wa, wb):
        # ---- genuine same-subject-same-instant pairing ----
        seen_subj, held_subj = subject_split(sa, seed)
        if not seen_subj or not held_subj:
            return _invalid_row(name, f"cannot form seen/held subject split on {wa.tag} (paired)")
        a_sup = np.isin(sa, list(seen_subj))
        held = np.isin(sa, list(held_subj))              # SAME rows drive matched (A) and shifted (B)
        if not a_sup.any() or not held.any():
            return _invalid_row(name, f"empty support/held partition on {wa.tag} (paired)")
        # assertions: pairing is real and support is subject-disjoint from the query
        assert np.array_equal(ya[held], yb[held]), \
            f"[placement/{wa.tag}] paired rows disagree on label — alignment broken"
        assert np.array_equal(sa[held], sb[held]), \
            f"[placement/{wa.tag}] paired rows disagree on subject — alignment broken"
        leak = set(sa[a_sup].tolist()) & set(sa[held].tolist())
        assert not leak, f"[placement/{wa.tag}] support and query share subjects {sorted(leak)}"
        if not (set(ya[a_sup].tolist()) & set(ya[held].tolist())):
            return _invalid_row(name, f"support and query share no labels on {wa.tag} (paired)")
        row = measure(name, enc, protos, lab2id, device,
                      z_sup=z_a[a_sup], y_sup=ya[a_sup],
                      zq_m=z_a[held], yq_m=ya[held],
                      zq_s=z_b[held], yq_s=yb[held],
                      z_mmd_m=z_a[held], z_mmd_s=z_b[held])
        row.update(pairing="same-subject-same-instant (row-aligned grids)", paired=True,
                   placement_a=wa.tag, placement_b=wb.tag,
                   n_paired_query=int(held.sum()), n_support_subjects=len(seen_subj))
        return row

    # ---- no instant-level join: honest unpaired cross-device distribution shift ----
    name = "placement_unpaired"
    shared_subj = sorted(set(sa.tolist()) & set(sb.tolist()))
    shared_lab = sorted(set(wa.labels) & set(wb.labels))
    if not shared_subj or not shared_lab:
        return _invalid_row(name, f"{wa.tag} vs {wb.tag} share no subjects/labels — no cross-device "
                            f"control possible", pairing="none", paired=False)
    a_keep = np.isin(sa, shared_subj) & np.isin(np.asarray(wa.labels), shared_lab)
    b_keep = np.isin(sb, shared_subj) & np.isin(np.asarray(wb.labels), shared_lab)
    # per-subject split of A into support/matched so EVERY shared subject stays in the A-support
    a_idx = np.where(a_keep)[0]
    rng = np.random.default_rng(seed)
    a_sup_idx, a_mq_idx = [], []
    for subj in shared_subj:
        idx = a_idx[sa[a_idx] == subj]
        rng.shuffle(idx)
        if len(idx) == 1:
            a_sup_idx.append(int(idx[0]))
        elif len(idx) >= 2:
            h = len(idx) // 2
            a_sup_idx.extend(int(i) for i in idx[:h])
            a_mq_idx.extend(int(i) for i in idx[h:])
    a_sup_idx = np.asarray(sorted(a_sup_idx), dtype=np.int64)
    a_mq_idx = np.asarray(sorted(a_mq_idx), dtype=np.int64)
    b_q_idx = np.where(b_keep)[0]
    if len(a_sup_idx) == 0 or len(a_mq_idx) == 0 or len(b_q_idx) == 0:
        return _invalid_row(name, f"empty partition for the unpaired control on {wa.tag}/{wb.tag}",
                            pairing="unpaired", paired=False)
    # FINDING 6 weaker control: A-support and B-query must share subject set and label set.
    sup_subjects = set(sa[a_sup_idx].tolist())
    b_subjects = set(sb[b_q_idx].tolist())
    if sup_subjects != b_subjects:
        return _invalid_row(name, f"A-support subjects {sorted(sup_subjects)} != B-query subjects "
                            f"{sorted(b_subjects)} — cross-device control not met", pairing="unpaired",
                            paired=False)
    sup_labels = set(ya[a_sup_idx].tolist())
    b_labels = set(yb[b_q_idx].tolist())
    if not (sup_labels & b_labels):
        return _invalid_row(name, f"A-support and B-query share no labels on {wa.tag}/{wb.tag}",
                            pairing="unpaired", paired=False)
    row = measure(name, enc, protos, lab2id, device,
                  z_sup=z_a[a_sup_idx], y_sup=ya[a_sup_idx],
                  zq_m=z_a[a_mq_idx], yq_m=ya[a_mq_idx],
                  zq_s=z_b[b_q_idx], yq_s=yb[b_q_idx],
                  z_mmd_m=z_a[a_keep], z_mmd_s=z_b[b_keep])
    row.update(pairing="UNPAIRED cross-device distribution shift (no instant-level join; subject+"
               "label sets matched, instants NOT paired)", paired=False,
               placement_a=wa.tag, placement_b=wb.tag,
               n_shared_subjects=len(shared_subj), n_shared_labels=len(shared_lab))
    return row


def _probe_support_mask(subjects, seen_subj, y, novel_ids) -> np.ndarray:
    """Bool mask selecting the ConSE probe-fit support: SEEN-subject windows whose label is NOT
    novel. Novel labels are excluded from the fit so they are scored genuinely zero-shot (FINDING 5).
    Isolated as a helper so the fail-loud assertion in ``run_label_novelty`` has a testable seam."""
    return np.isin(subjects, list(seen_subj)) & ~np.isin(np.asarray(y), list(novel_ids))


def run_label_novelty(enc, eval_sets: list[WindowSet], train_labels: set, device, seed):
    """Semantic reachability horizon (FINDING 5 fix): per eval label, ZS accuracy vs (a) text-cosine
    distance to the nearest TRAINING label prototype, (b) a coarse/fine granularity flag.

    Labels absent from Phase-A training (``novel=True``) are scored GENUINELY ZERO-SHOT: their
    labeled examples are EXCLUDED from the ConSE probe fit, so they are reachable only through their
    SBERT label-text prototype. The previous version fit the probe on support examples of EVERY eval
    label — including the "novel" ones — which measured supervised cross-subject transfer, not
    zero-shot semantic reachability. An assertion here forbids any novel-label example from entering
    the probe support. (Non-novel/seen labels keep their support examples — that IS the seen
    condition.) Returns the per-label scatter (list), NOT a scalar."""
    tl = sorted(train_labels)
    tl_id2lab = {i: l for i, l in enumerate(tl)}
    train_protos = build_label_protos(enc, tl_id2lab, device) if tl else None    # (Ltrain, 384)
    scatter = []
    for ws in eval_sets:
        lab2id, id2lab = _local_labmap(ws.labels)
        protos = build_label_protos(enc, id2lab, device)
        z = embed(enc, ws, device)
        y = _ids(ws.labels, lab2id)
        subjects = np.asarray(ws.subjects)
        seen_subj, held_subj = subject_split(subjects, seed)
        novel_ids = {lid for lid, lab in id2lab.items() if lab not in train_labels}

        # Probe support = SEEN-subject windows of NON-novel labels only (novel labels are held out
        # of the fit entirely -> genuine zero-shot). Query = held-out-subject windows (all labels).
        sup = _probe_support_mask(subjects, seen_subj, y.numpy(), novel_ids)
        qry = np.isin(subjects, list(held_subj))
        # FINDING 5 assertion: no novel-label example may enter the probe-fitting support.
        assert not (set(y[torch.from_numpy(sup)].tolist()) & novel_ids), \
            f"[label-novelty/{ws.tag}] a novel-label example leaked into the probe support"
        if int(sup.sum()) == 0 or int(qry.sum()) == 0:
            print(f"  [warn] label-novelty on {ws.tag}: empty probe support or query "
                  f"(|sup|={int(sup.sum())} |qry|={int(qry.sum())}) — skipped", flush=True)
            continue

        z_sup = z[torch.from_numpy(sup)]
        y_sup = y[torch.from_numpy(sup)]
        z_qry = z[torch.from_numpy(qry)]
        yq = y[torch.from_numpy(qry)]
        pred = conse_probe_predict(z_sup, y_sup, z_qry, yq, protos)
        for lid, lab in id2lab.items():
            mask = yq == lid
            if int(mask.sum()) == 0:
                continue
            is_novel = lid in novel_ids
            zs_acc = float((pred[mask] == lid).float().mean())
            # text distance = 1 - max cosine to any TRAINING label prototype (clamp fp noise >1)
            if train_protos is not None:
                lp = build_label_protos(enc, {0: lab}, device)[0]
                text_dist = round(float(1.0 - (train_protos @ lp).max().clamp(max=1.0)), 4)
            else:
                text_dist = None
            granularity = "fine" if ("_" in lab or " " in lab.strip()) else "coarse"
            scatter.append(dict(label=lab, dataset=ws.dataset, stream=ws.stream,
                                text_dist=text_dist,
                                granularity=granularity,
                                novel=bool(is_novel),
                                scored_zero_shot=bool(is_novel),   # novel => no support examples used
                                zs_acc=round(zs_acc, 4),
                                n_query=int(mask.sum())))
    scatter.sort(key=lambda d: (d["text_dist"] is None, d["text_dist"] or 0.0))
    return scatter


def run_compound(enc, wa: WindowSet, wb: WindowSet, device, seed, rate_hz: float):
    """Apply placement+orientation+rate together (≈ full cross-dataset) and compare the compound
    ZS-F1 drop to the SUM of the single-axis drops (additive vs super-additive). All measured on
    the SAME base (placement A support/matched query) for comparability."""
    lab2id, id2lab = _local_labmap(wa.labels, wb.labels)
    protos = build_label_protos(enc, id2lab, device)
    sup_subj, held_subj = subject_split(wa.subjects, seed)
    a_sup = np.array([s in sup_subj for s in wa.subjects])
    a_qry = np.array([s in held_subj for s in wa.subjects])
    b_qry = np.array([s in held_subj for s in wb.subjects])
    ya, yb = _ids(wa.labels, lab2id), _ids(wb.labels, lab2id)

    z_a = embed(enc, wa, device)
    z_sup, y_sup = z_a[a_sup], ya[a_sup]
    _, f1_matched = score_pair(z_sup, y_sup, z_a[a_qry], ya[a_qry], protos)

    def f1_of(ws_shift, ids, mask):
        z = embed(enc, ws_shift, device)
        _, f1 = score_pair(z_sup, y_sup, z[mask], ids[mask], protos)
        return f1

    # single-axis shifts, each relative to matched A
    f1_place = f1_of(wb, yb, b_qry)
    f1_rot = f1_of(shift_orientation(wa, seed), ya, a_qry)
    f1_rate = f1_of(shift_rate(wa, rate_hz), ya, a_qry)
    # compound: placement B + rotation + downsample (applied to B, queried on held subjects)
    wb_compound = shift_rate(shift_orientation(wb, seed), rate_hz)
    f1_compound = f1_of(wb_compound, yb, b_qry)

    d_place = f1_matched - f1_place
    d_rot = f1_matched - f1_rot
    d_rate = f1_matched - f1_rate
    d_sum = d_place + d_rot + d_rate
    d_compound = f1_matched - f1_compound
    return dict(f1_matched=_r(f1_matched), f1_compound=_r(f1_compound),
                drop_placement=_r(d_place), drop_orientation=_r(d_rot), drop_rate=_r(d_rate),
                drop_sum_of_singles=_r(d_sum), drop_compound=_r(d_compound),
                ratio_compound_over_sum=_r(d_compound / d_sum if d_sum > 1e-9 else float("nan")),
                mode=("super-additive" if d_sum > 1e-9 and d_compound > d_sum + 0.02 else
                      "sub-additive" if d_sum > 1e-9 and d_compound < d_sum - 0.02 else
                      "additive"))


# ======================================================================================
# Synthetic fallback (only if the real checkpoint / grids are absent)
# ======================================================================================
def build_synthetic(tmp: Path, seed: int):
    """Write tiny synthetic native grids + return a random-init encoder. NUMBERS MEANINGLESS —
    this only proves every axis + the output path run end-to-end without real assets."""
    from model.tokenizer.encoder import SetTokenizerEncoder
    from training.tokenizer.pretrain_data import DFT_SIZE
    rng = np.random.default_rng(seed)
    ds_root = tmp / "datasets"
    subjects = [f"s{i}" for i in range(8)]
    labels = ["walking", "sitting", "standing", "walking_upstairs", "jogging", "eating_soup"]

    def write_grid(dataset, stream, rate, n, motif_seed):
        d = ds_root / dataset / "grids" / "native" / stream
        d.mkdir(parents=True, exist_ok=True)
        T = int(rate * 6.0)
        subs, labs, wins = [], [], []
        mr = np.random.default_rng(motif_seed)
        for i in range(n):
            lab = labels[i % len(labels)]
            t = np.arange(T) / rate
            freq = 1.0 + labels.index(lab)
            acc = np.stack([np.sin(2 * np.pi * freq * t) + (0 if k else 9.8)  # z carries gravity DC
                            for k in range(3)], axis=-1)
            gyr = 0.3 * np.cos(2 * np.pi * freq * t)[:, None] * np.ones((1, 3))
            w = np.concatenate([acc, gyr], axis=-1) + 0.05 * mr.standard_normal((T, 6))
            wins.append(w.astype(np.float32))
            subs.append(subjects[i % len(subjects)])
            labs.append(lab)
        np.save(d / "data.npy", np.stack(wins))
        np.save(d / "mask.npy", np.ones(6, bool))
        (d / "meta.json").write_text(json.dumps(dict(
            dataset=dataset, stream_id=stream, alignment="native", rate_hz=float(rate),
            channels=list(CHANNELS), labels=labs, subjects=subs)))

    # multi-placement dataset (A/B share subjects+labels) + a held-out transform dataset
    write_grid("synthplace", "loc_a", 50.0, 96, 1)
    write_grid("synthplace", "loc_b", 50.0, 96, 2)
    write_grid("syntheval", "phone_waist", 50.0, 96, 3)

    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, dft_size=DFT_SIZE, frontend="fixed",
                              text_conditioning="per_channel").eval()
    return enc, ds_root, dict(eval=[("syntheval", "phone_waist")],
                              placement=("synthplace", "loc_a", "loc_b"),
                              train_labels={"walking", "sitting", "standing"})


# ======================================================================================
# Main
# ======================================================================================
def parse_stream_arg(s: str):
    ds, st = s.split(":", 1)
    return (ds, st)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-windows", type=int, default=400, help="per-stream subset cap")
    ap.add_argument("--eval-streams", nargs="+", type=parse_stream_arg, default=None,
                    help="dataset:stream held-out streams for the per-window transform axes")
    ap.add_argument("--placement", nargs=3, metavar=("DATASET", "A", "B"), default=None,
                    help="multi-placement dataset + two streams (same-instant pairing is "
                         "auto-detected from row-alignment; else axis = unpaired cross-device)")
    ap.add_argument("--rate-hz", type=float, default=20.0, help="target rate for the rate axis")
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="force the synthetic runs-only fallback (no real assets needed)")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tmpdir = None
    synthetic = args.synthetic or not args.checkpoint.exists()
    if synthetic:
        tmpdir = Path(tempfile.mkdtemp(prefix="zsdiff_synth_"))
        enc, ds_root, plan = build_synthetic(tmpdir, args.seed)
        enc = enc.to(device)
        refs = {(r.dataset, r.stream): r for r in discover_grids("native", datasets_dir=ds_root)}
        eval_streams = plan["eval"]
        placement = plan["placement"]
        train_labels = plan["train_labels"]
        provenance = "SYNTHETIC (random encoder + tiny synthetic grids) — RUNS-ONLY, NUMBERS MEANINGLESS"
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        enc = build_encoder(ckpt, device)
        refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
        eval_streams = args.eval_streams or list(DEFAULT_EVAL_STREAMS)
        placement = tuple(args.placement) if args.placement else DEFAULT_PLACEMENT
        train_labels = set(ckpt.get("label_ids", {}).keys())
        provenance = (f"REAL checkpoint {args.checkpoint} (val_ba {ckpt.get('val_ba'):.3f}, "
                      f"step {ckpt.get('step')}) + real native grids")

    eval_streams = [tuple(s) for s in eval_streams if tuple(s) in refs]
    if not eval_streams:
        raise SystemExit("no requested eval streams found among the discovered grids")
    print(f"provenance: {provenance}", flush=True)
    print(f"device={device}  max_windows={args.max_windows}  eval_streams={eval_streams}  "
          f"placement={placement}", flush=True)

    eval_sets = [load_windowset(refs[s], args.max_windows, args.seed) for s in eval_streams]

    # ---- per-window transform axes (aggregate across eval streams) ----
    axis_rows = []
    transform_axes = {
        "rate": lambda ws: shift_rate(ws, args.rate_hz),
        "channel": lambda ws: shift_channel(ws),
        "orientation": lambda ws: shift_orientation(ws, args.seed),
        "gravity": lambda ws: shift_gravity(ws),
    }
    per_stream = {}
    for axis, fn in transform_axes.items():
        rows = []
        for ws in eval_sets:
            try:
                rows.append(run_aligned_axis(axis, enc, ws, fn(ws), device, args.seed))
            except Exception as e:                       # keep the harness robust per-stream
                print(f"  [warn] {axis} on {ws.tag} failed: {e}", flush=True)
        per_stream[axis] = rows
        axis_rows.append(_aggregate(axis, rows))

    # ---- subject axis (per eval stream, aggregate) ----
    subj_rows = []
    for ws in eval_sets:
        try:
            subj_rows.append(run_subject_axis("subject", enc, ws, device, args.seed))
        except Exception as e:
            print(f"  [warn] subject on {ws.tag} failed: {e}", flush=True)
    per_stream["subject"] = subj_rows
    axis_rows.append(_aggregate("subject", subj_rows))

    # ---- placement axis (single multi-placement dataset; pairing auto-detected from the grids) ----
    placement_row = None
    pa = (placement[0], placement[1])
    pb = (placement[0], placement[2])
    wa = wb = None
    if pa in refs and pb in refs:
        wa, wb, row_aligned = load_placement_pair(refs[pa], refs[pb], args.max_windows, args.seed)
        print(f"  placement pair {pa[0]}: {placement[1]} vs {placement[2]} -> "
              f"row_aligned={row_aligned} "
              f"({'genuine same-subject-same-instant' if row_aligned else 'UNPAIRED cross-device'})",
              flush=True)
        placement_row = run_placement_axis("placement", enc, wa, wb, device, args.seed)
        axis_rows.append(placement_row)
    else:
        print(f"  [warn] placement streams {pa}/{pb} not both present — axis skipped", flush=True)

    # ---- label-novelty scatter ----
    scatter = run_label_novelty(enc, eval_sets, train_labels, device, args.seed)

    # ---- compound additivity (on the placement dataset) ----
    compound = None
    if wa is not None and wb is not None:
        compound = run_compound(enc, wa, wb, device, args.seed, args.rate_hz)

    # ---- attribution summary ----
    ranked_bridge = sorted([r for r in axis_rows if r["bridge_retention"] is not None],
                           key=lambda r: r["bridge_retention"])
    ranked_mmd = sorted([r for r in axis_rows if r["mmd"] is not None],
                        key=lambda r: -r["mmd"])
    headline = []
    if ranked_bridge:
        w = ranked_bridge[0]
        headline.append(f"Largest bridge-retention drop: {w['axis']} "
                        f"(bridge_ret={w['bridge_retention']}, enc_ret={w['enc_retention']}, "
                        f"verdict={w['verdict']}).")
    if ranked_mmd:
        m = ranked_mmd[0]
        headline.append(f"Largest distribution shift (MMD): {m['axis']} (MMD={m['mmd']}).")
    if ranked_bridge:
        w = ranked_bridge[0]
        headline.append(f"=> zero-shot HAR is most {w['verdict'].split('-')[0].upper()}-limited "
                        f"on the {w['axis']} axis.")

    result = dict(provenance=provenance, device=str(device), seed=args.seed,
                  max_windows=args.max_windows, eval_streams=[list(s) for s in eval_streams],
                  placement=list(placement), verdict_threshold=VERDICT_T,
                  axes=axis_rows, per_stream=per_stream,
                  label_novelty=scatter, compound=compound, headline=headline)

    out = args.out or (args.checkpoint.parent / "zeroshot_difficulty.json"
                       if not synthetic else tmpdir / "zeroshot_difficulty.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print_report(result, out)

    if tmpdir is not None:
        # persist the JSON to cwd before deleting the synthetic scratch tree
        final = Path("zeroshot_difficulty_synth.json")
        final.write_text(json.dumps(result, indent=2))
        print(f"\n(synthetic scratch grids deleted; JSON copied to {final.resolve()})")
        shutil.rmtree(tmpdir, ignore_errors=True)


def _aggregate(axis: str, rows: list[dict]) -> dict:
    """Average the per-stream metric rows into one axis row (retention from averaged values).
    INVALID per-stream rows are excluded; if none remain valid the whole axis is marked INVALID
    rather than silently averaging to a misleading value."""
    if not rows:
        return dict(axis=axis, invalid=False, mmd=None, knn_ba_matched=None, knn_ba_shifted=None,
                    enc_retention=None, zs_f1_matched=None, zs_f1_shifted=None,
                    bridge_retention=None, verdict="n/a", n_streams=0)
    valid = [r for r in rows if not r.get("invalid")]
    if not valid:
        reasons = "; ".join(sorted({r.get("invalid_reason", "?") for r in rows}))
        return _invalid_row(axis, f"all {len(rows)} per-stream row(s) invalid: {reasons}",
                            n_streams=len(rows))

    def mean(key):
        vals = [r[key] for r in valid if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    ba_m, ba_s = mean("knn_ba_matched"), mean("knn_ba_shifted")
    f1_m, f1_s = mean("zs_f1_matched"), mean("zs_f1_shifted")
    enc_ret = ba_s / ba_m if ba_m else None
    br_ret = f1_s / f1_m if f1_m else None
    enc_low = enc_ret is not None and enc_ret < VERDICT_T
    br_low = br_ret is not None and br_ret < VERDICT_T
    verdict = ("both" if enc_low and br_low else "encoder-limited" if enc_low else
               "bridge-limited" if br_low else "robust")
    return dict(axis=axis, invalid=False, mmd=_r(mean("mmd")),
                knn_ba_matched=_r(ba_m), knn_ba_shifted=_r(ba_s), enc_retention=_r(enc_ret),
                zs_f1_matched=_r(f1_m), zs_f1_shifted=_r(f1_s), bridge_retention=_r(br_ret),
                verdict=verdict, n_streams=len(valid), n_streams_invalid=len(rows) - len(valid))


def print_report(result: dict, out: Path) -> None:
    print("\n" + "=" * 100)
    print("ZERO-SHOT DIFFICULTY ATTRIBUTION")
    print("=" * 100)
    hdr = (f"{'axis':<13}{'MMD':>8}{'kNN_m':>8}{'kNN_s':>8}{'enc_ret':>9}"
           f"{'F1_m':>8}{'F1_s':>8}{'br_ret':>8}  verdict")
    print(hdr)
    print("-" * 100)

    def cell(v, w=8):
        return (f"{'--':>{w}}" if v is None else f"{v:>{w}.3f}")
    for r in result["axes"]:
        if r.get("invalid"):
            print(f"{r['axis']:<13}{'--':>8}{'--':>8}{'--':>8}{'--':>9}{'--':>8}{'--':>8}{'--':>8}"
                  f"  AXIS INVALID: {r.get('invalid_reason', '?')}")
            continue
        print(f"{r['axis']:<13}{cell(r['mmd'])}{cell(r['knn_ba_matched'])}{cell(r['knn_ba_shifted'])}"
              f"{cell(r['enc_retention'],9)}{cell(r['zs_f1_matched'])}{cell(r['zs_f1_shifted'])}"
              f"{cell(r['bridge_retention'])}  {r['verdict']}")
        if r.get("pairing"):
            print(f"{'':<13}   pairing: {r['pairing']}")
    print("-" * 100)

    print("\nLABEL-NOVELTY SCATTER (semantic reachability horizon; sorted by text distance;"
          " novel labels scored ZERO-SHOT = no support examples in the probe fit):")
    print(f"  {'label':<22}{'dataset':<14}{'text_dist':>10}{'gran':>7}{'novel':>7}{'zeroshot':>9}{'zs_acc':>8}")
    for s in result["label_novelty"]:
        td = "--" if s["text_dist"] is None else f"{s['text_dist']:.3f}"
        print(f"  {s['label']:<22}{s['dataset']:<14}{td:>10}{s['granularity']:>7}"
              f"{str(s['novel']):>7}{str(s.get('scored_zero_shot', False)):>9}{s['zs_acc']:>8.3f}")

    c = result.get("compound")
    if c:
        print("\nCOMPOUND vs SUM-OF-SINGLES (placement + orientation + rate):")
        print(f"  single drops: placement={c['drop_placement']} orientation={c['drop_orientation']} "
              f"rate={c['drop_rate']}  -> sum={c['drop_sum_of_singles']}")
        print(f"  compound drop={c['drop_compound']}  ratio(compound/sum)="
              f"{c['ratio_compound_over_sum']}  => {c['mode'].upper()}")

    print("\nATTRIBUTION HEADLINE:")
    for line in result["headline"]:
        print(f"  {line}")
    print(f"\n-> JSON written to {out}")


if __name__ == "__main__":
    main()
