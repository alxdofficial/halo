"""Pipeline A Phase-1 data pipeline (pretraining corpus + sampler + collate).

Design decisions carried in from the gates:
  * Corpus = the 12 TRAIN datasets' **native** grids (eval sets never touched): native sampling
    RATE (no 60 Hz resample) + canonical labels + 6-ch pad+mask. The filterbank tokenizer is
    rate-invariant, so HALO trains on the corpus's REAL native rates (20/50/100 Hz) instead of a
    homogenized 60 Hz base — the 60 Hz "harmonised" grids are the layout-locked baselines' crutch,
    not HALO's. Source-balanced sampling (no per-stream cap) spreads each activity across configs.
  * Subject-disjoint train/val split per dataset.
  * The default label-free recipe uses source-temperature sampling. Label-balanced batches remain
    available only for the historical label-SupCon control.
  * The FULL augmentation stack (data/scripts/augmentations.py default_v2), with
    per-worker reseeding of BOTH np.random and stdlib random (the CrossHAR lesson —
    the augmenter draws from both RNGs).
  * Multi-scale patch duration: ONE patch_seconds draw per BATCH (the token-count
    axis, §5.2.1); sampling RATE varies per SAMPLE (the filterbank takes (B,) rates
    and (B,) patch lengths — batches need bucketing only by patch_seconds; this
    CORRECTS the earlier "bucket by (rate, patch_seconds)" note, which applied to
    the legacy experiment encoder, not the ported filterbank).
  * A3 grounding targets computed per sample on the FINAL augmented view (the §5.2.3
    rule) with validity masks; primitives are rate/rotation/gain-invariant so most
    augs are free, and the non-analytic ones (PCHIP time-warp) are simply recomputed.
  * Channel identity/config = per-stream TEXT (placement parsed from the stream id),
    with the text augmentations (paraphrase/dropout) supplying variety; absent
    channels (pad+mask grids) carry a channel_mask, never fake text confidence.
"""

from __future__ import annotations

import random as stdlib_random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
from data.scripts.eda.grid_io import GridRef, discover_grids
from model.tokenizer.preprocess import gravity_align
from model.tokenizer.primitives import cadence, eigen_ratios

# ----------------------------------------------------------------------------------------------
# Corpus configuration
# ----------------------------------------------------------------------------------------------
# hapt DROPPED: the sweep confirmed it is the UCI-HAR re-release — same 30 subjects /
# recordings (per-window NCC 0.98 vs uci_har), so keeping both leaks near-duplicate val
# windows into train across the pair. uci_har is the canonical windowed release; keep it.
TRAIN_DATASETS = (
    "uci_har", "hhar", "pamap2", "wisdm", "kuhar", "unimib_shar",
    "mhealth", "capture24", "sp_sw_har", "nfi_fared", "harmes", "xrf_v2",
)
# Fully wired scale sources, opt-in through pretrain.py --datasets. NHANES is
# intentionally absent from label probes and Phase B because it has no activity
# annotations; ExtraSensory is labelled but remains opt-in so data-scale results
# cannot be confused with the frozen 12-source technique comparison.
OPTIONAL_PHASE_A_DATASETS = ("extrasensory", "nhanes", "hmog")
PHASE_A_ONLY_DATASETS = frozenset({"nhanes"})
UNLABELED_LABEL = "__unlabeled__"
MAX_PER_STREAM = 20_000          # legacy optional balanced-corpus cap; PER-STREAM (so
                                 # wisdm's 2 streams get 2x — kept for baseline parity,
                                 # phone+watch ARE distinct configs; audited 2026-07-18)
WINDOW_SECONDS = 6.0
VAL_SUBJECT_FRACTION = 0.10      # subject-disjoint val within the train datasets
GRAVITY_AUG_P = 0.15             # DROP from default_v2's 0.5: the audit found gravity
                                 # removed on 52% of windows, killing the M0 gravity-align /
                                 # DC-tilt features on half the corpus. 0.15 keeps the
                                 # iOS-userAcceleration robustness without dominating.
MIN_WINDOWS_TO_ANCHOR = 8        # balanced/SupCon control: labels below this can't form real positives
                                 # (would be duplicate-oversampled); excluded from anchoring
# per-batch multi-scale draw -> T = 6s / ps in {12, 8, 6, 4}. 2.0s (T=3) DROPPED: T=3 is
# too coarse for masked prediction, and it's where native-short windows (uci_har 2.56s,
# unimib falls) collapse to a single un-maskable patch (objective-health audit 2026-07-18).
PATCH_SECONDS_CHOICES = (0.5, 0.75, 1.0, 1.5)
SHORT_PATCH_SECONDS_CHOICES = (0.4, 0.5, 0.6, 0.7, 0.8)
LONG_PATCH_SECONDS_CHOICES = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
MIN_RESOLUTION_RATIO = 1.75
MIN_TAIL_FRACTION = 0.25          # F7: drop a resolution's tail patch if it covers < this fraction of
                                 # a full patch (and it isn't the only patch) — avoids degenerate
                                 # 1-sample tokens whose duration is clamped to the embedding floor.
VAL_RESOLUTION_PAIR = (0.5, 1.5)
# Hard ceiling on the per-batch TOKEN count (batch x patches). Peak VRAM tracks tokens, not
# windows, and patch_seconds is drawn PER BATCH — so without this, memory is a random variable:
# measured P swings 12->22 at fixed batch, and batch 64 ran 60 steps before an unlucky 0.4 s
# draw OOMed it. 6144 admits every draw at batch 256 (worst case 22 patches -> 5,632) while
# forcing batch 512 onto the coarser pairs it can actually afford. Set 0 to disable.
MAX_BATCH_TOKENS = 6144
DFT_SIZE = 256                   # must cover max NATIVE rate (100 Hz) x max patch (1.5 s) = 150;
                                 # the rate aug caps at 100 Hz too, so 256 keeps ample headroom
# Streams whose SOURCE (acquisition) rate differs from the rate the grid is stored at, because a
# converter resampled them onto the dataset-wide grid rate. Upsampling cannot create information, so
# the filterbank must take its Nyquist/observability bound from the ACQUISITION rate while the DFT
# bin->frequency mapping keeps using the true array rate. Without this, xrf_v2's AirPods stream
# (captured at 25 Hz, stored at 50 Hz) was advertised as having all 32 bands observable when only
# ~27 carry real signal — the rest is interpolation artifact. Authority: each dataset's convert.py.
STREAM_SOURCE_RATE_HZ = {
    "xrf_v2/airpods_ear": 25.0,      # AirPods Pro ear IMU @25 Hz, upsampled 25->50 in convert.py
    "extrasensory/watch_wrist": 25.0, # Pebble acquisition clock; converter stores at 50 Hz
    # Phone clocks vary (~30-200 Hz). Thirty Hz is the conservative observed
    # floor, so bands above 15 Hz are never claimed as physically observable.
    "extrasensory/phone_pocket": 30.0,
    "extrasensory/phone_hand": 30.0,
}
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
SEED = 20260718

PLACEMENT_WORDS = {
    "waist": "the waist", "wrist": "the wrist", "pocket": "a trouser pocket",
    "hip": "the hip", "belt": "the belt", "thigh": "the thigh",
    "back": "the lower back", "ankle": "the ankle", "chest": "the chest",
}


_DEVICE_WORDS = {"phone": "phone", "watch": "watch", "watch_proxy": "phone",
                 "device": "wearable device"}


def stream_channel_descriptions(dataset: str, stream: str) -> list[str]:
    """Per-channel base text for a deployment stream — HALO's configuration-conditioning input.

    Uses the StreamSpec's curated `placement` + `device_profile` from the deployment policy (e.g.
    "smart glasses on the head" / "the left wrist" / "an earbud in the ear"), which distinguishes
    left vs right, head vs ear vs pocket, etc. Falls back to stream-id tokens only if no spec exists.
    """
    try:
        from data.scripts.curate.deployment_policy import get_stream_spec
        spec = get_stream_spec(dataset, stream)
        place = spec.placement if spec.placement.startswith(("the ", "a ", "an ", "smart")) \
            else f"the {spec.placement}"
        device = _DEVICE_WORDS.get(spec.device_profile, spec.device_profile.replace("_", " "))
        gravity_removed = (spec.gravity_state == "removed")
    except (KeyError, ValueError, ImportError):
        tokens = stream.lower().split("_")
        device = "phone" if "phone" in tokens else ("watch" if "watch" in tokens else "device")
        place = next((PLACEMENT_WORDS[w] for w in tokens if w in PLACEMENT_WORDS), "the body")
        gravity_removed = False
    where = f"{place} ({device})"
    # Gravity state is a real acquisition-config axis: accelerometer from gravity-removed streams
    # (kuhar, xrf_v2/airpods_ear) has |DC|~0 vs ~1 g for gravity-present streams. Only the
    # accelerometer carries it (the gyroscope is unaffected). The clause mirrors the sibling
    # deployment_policy.channel_description() and is read back as AUTHORITATIVE by the gravity
    # augmentation's _gravity_present() / _mark_gravity_removed() (so it also gates rotation/
    # gravity-removal correctly on native gravity-removed streams, not the misfiring heuristic).
    grav = "; gravity removed" if gravity_removed else "; includes gravity"
    return ([f"accelerometer {a}-axis worn at {where}{grav}" for a in "xyz"]
            + [f"gyroscope {a}-axis worn at {where}" for a in "xyz"])


# Fixed intra-sensor channel ROLE text (AXIS ONLY — never modality/placement/device). One entry per
# CHANNELS slot; constant across the whole corpus. Modality (accel vs gyro) is NOT a role fact — it
# moved to the per-sensor text below, because accel and gyro are now modelled as two distinct
# modality-level SENSORS (docs/design/TEXT_CONDITIONING.md). Role is the purely positional x/y/z axis.
# NOTE: axis is encoded as TEXT ("x"/"y"/"z") so it flows through the same frozen-LM pooler as every
# other text source. A tiny LEARNED 3-way role embedding would be cleaner (only three axes; SBERT-
# embedding single letters is wasteful) but that is a model-side change to FactoredChannelTextFusion —
# kept as text here to avoid over-engineering the data path.
_CHANNEL_ROLE_TEXT = {
    "acc_x": "x", "acc_y": "y", "acc_z": "z",
    "gyro_x": "x", "gyro_y": "y", "gyro_z": "z",
}


def stream_sensor_texts(
    dataset: str,
    stream: str,
    *,
    gravity_removed: bool | None = None,
    has_accel: bool = True,
    has_gyro: bool = True,
) -> tuple[list[str], list[str], list[int]]:
    """Factored config text for a stream (docs/design/TEXT_CONDITIONING.md).

    Accel and gyro are modelled as two distinct modality-level SENSORS, so a stream factors as:
      * ``role_texts``  — one string per CHANNELS slot, AXIS ONLY ("x"/"y"/"z"); corpus-constant.
      * ``sensor_texts``— one string per PRESENT modality: ``[accel_sensor, gyro_sensor]`` when the
                          stream carries both, each with device + modality + placement (the gravity
                          convention rides on the ACCEL sensor only — the gyroscope has no gravity
                          component). An accel-only stream (capture24, unimib_shar) emits just
                          ``[accel_sensor]`` — no phantom gyroscope; the ragged sensor count is padded
                          by encode_texts_factored.
      * ``sensor_id``   — length-6 map: accel channels -> the accel sensor's index, gyro channels ->
                          the gyro sensor's index. Absent-modality channels are ``channel_mask``-masked
                          and pool to nothing, so their id only needs to be a valid index.

    Placement/device/modality live ONLY in the sensor text and axis ONLY in the role text, so no config
    fact is injected twice when the two are summed (the compounding hazard of §6).
    """
    role_texts = [_CHANNEL_ROLE_TEXT[c] for c in CHANNELS]
    try:
        from data.scripts.curate.deployment_policy import get_stream_spec
        spec = get_stream_spec(dataset, stream)
        place = spec.placement if spec.placement.startswith(("the ", "a ", "an ", "smart")) \
            else f"the {spec.placement}"
        device = _DEVICE_WORDS.get(spec.device_profile, spec.device_profile.replace("_", " "))
        stream_gravity_removed = (spec.gravity_state == "removed")
    except (KeyError, ValueError, ImportError):
        tokens = stream.lower().split("_")
        device = "phone" if "phone" in tokens else ("watch" if "watch" in tokens else "device")
        place = next((PLACEMENT_WORDS[w] for w in tokens if w in PLACEMENT_WORDS), "the body")
        stream_gravity_removed = False
    removed = stream_gravity_removed if gravity_removed is None else bool(gravity_removed)
    grav = "gravity removed" if removed else "includes gravity"
    accel_sensor = f"a {device} accelerometer on {place}; {grav}"
    gyro_sensor = f"a {device} gyroscope on {place}"
    # Emit only the modalities actually present. Absent-modality channels are channel_mask-masked, so
    # their sensor_id just needs to stay a valid index into sensor_texts.
    sensor_texts: list[str] = []
    accel_id = gyro_id = 0
    if has_accel:
        accel_id = len(sensor_texts)
        sensor_texts.append(accel_sensor)
    if has_gyro:
        gyro_id = len(sensor_texts)
        sensor_texts.append(gyro_sensor)
    if not sensor_texts:                     # defensive: neither modality flagged present
        sensor_texts.append(accel_sensor)
        accel_id = gyro_id = 0
    sensor_id = [accel_id if c.startswith("acc") else gyro_id for c in CHANNELS]
    return role_texts, sensor_texts, sensor_id


_GRAVITY_STATE_CACHE: dict[tuple[str, str], str | None] = {}


def _stream_gravity_state(dataset: str, stream: str) -> str | None:
    """Authoritative per-stream gravity state ('present'/'removed') from the deployment policy,
    cached. Fed to the collate's gravity_align so gravity-removed streams (kuhar, xrf/airpods_ear)
    skip alignment instead of trusting the magnitude heuristic (which misfires on ~3% of them)."""
    key = (dataset, stream)
    if key not in _GRAVITY_STATE_CACHE:
        try:
            from data.scripts.curate.deployment_policy import get_stream_spec
            _GRAVITY_STATE_CACHE[key] = get_stream_spec(dataset, stream).gravity_state
        except (KeyError, ValueError, ImportError):
            _GRAVITY_STATE_CACHE[key] = None
    return _GRAVITY_STATE_CACHE[key]


# ----------------------------------------------------------------------------------------------
# Corpus index
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowKey:
    stream_i: int
    window_i: int
    label_id: int


class CorpusIndex:
    """Discover, balance, split, and label the pretraining corpus (lazy windows)."""

    def __init__(self, max_per_stream: int | None = None, seed: int = SEED,
                 datasets: Sequence[str] = TRAIN_DATASETS, alignment: str = "native"):
        # HALO trains on the "native" grids: native sampling RATE (no 60 Hz resample — the filterbank
        # is rate-invariant, so real rates beat a homogenized base + synthetic rate aug) with the
        # canonical labels + 6-ch pad+mask layout this loader expects. The 60 Hz "harmonised" grids
        # remain the layout-locked baselines' source; they are not used here.
        self.alignment = alignment
        self.max_per_stream = max_per_stream   # retained for the checkpoint corpus fingerprint (F5)
        self.seed = seed
        self.datasets = tuple(datasets)
        self.refs: list[GridRef] = [
            r for r in discover_grids(alignment) if r.dataset in set(datasets)
        ]
        if not self.refs:
            raise FileNotFoundError(f"no {alignment} train grids found — build grids first "
                                    f"(python -m data.scripts.build_grids --alignment {alignment})")
        # A zero-window placeholder is not a usable dataset. Treat it as missing
        # so a failed/empty optional conversion cannot satisfy an explicit roster.
        materialized_datasets = {ref.dataset for ref in self.refs if ref.n_windows > 0}
        missing_datasets = sorted(set(datasets) - materialized_datasets)
        if missing_datasets:
            raise FileNotFoundError(
                f"requested {alignment} datasets have no grids: {missing_datasets}. "
                "Build them explicitly or remove them from --datasets; silently training on a "
                "partial roster would make the run unattributable."
            )
        self.stream_datasets = [r.dataset for r in self.refs]   # stream_i -> dataset (for the sampler)
        rng = np.random.default_rng(seed)

        # Windows the plausibility scan rejected as physically impossible (accel/gyro beyond any
        # consumer full-scale range). Cached by data.scripts.scan_implausible so indexing stays lazy.
        from data.scripts.scan_implausible import load as _load_implausible
        self.implausible = _load_implausible(alignment)
        # Byte-identical repeated windows — a device re-emitting a stale buffer, not motion
        # (ExtraSensory's Pebble does this for hours at a time). Merged into the same drop set:
        # both are "this window is not an observation", and CorpusIndex applies one filter.
        from data.scripts.scan_duplicates import load as _load_duplicates
        self.duplicates = _load_duplicates(alignment)
        self.excluded = {
            key: self.implausible.get(key, set()) | self.duplicates.get(key, set())
            for key in set(self.implausible) | set(self.duplicates)
        }

        # Subject-disjoint split per dataset, chosen to COVER AS MANY LABELS as the budget allows.
        # A purely random 10% draw left whole labels with zero val windows (e.g. `sleeping`: 15,100
        # train / 0 val; `table_tennis`: 216/0), so val_knn_ba / val_conse_ba / best.pt selection
        # silently scored 91 of 93 labels while the code claimed all of them. Greedy set-cover over
        # subjects fixes that without touching disjointness (a subject is still wholly train or val).
        val_subjects: set[tuple[str, str]] = set()
        by_dataset: dict[str, set[str]] = {}
        subj_labels: dict[tuple[str, str], set[str]] = {}
        for ref in self.refs:
            by_dataset.setdefault(ref.dataset, set()).update(ref.subjects)
            for w in range(ref.n_windows):
                subj_labels.setdefault((ref.dataset, ref.subjects[w]), set()).add(ref.labels[w])
        for dataset, subjects in sorted(by_dataset.items()):
            if dataset in PHASE_A_ONLY_DATASETS:
                continue
            ordered = sorted(subjects)
            rng.shuffle(ordered)
            n_val = max(1, int(round(len(ordered) * VAL_SUBJECT_FRACTION)))
            need = set().union(*(subj_labels.get((dataset, s), set()) for s in ordered))
            picked: list[str] = []
            # greedy: repeatedly take the subject covering the most still-uncovered labels
            while len(picked) < n_val and need:
                best = max((s for s in ordered if s not in picked),
                           key=lambda s: (len(subj_labels.get((dataset, s), set()) & need), s),
                           default=None)
                if best is None or not (subj_labels.get((dataset, best), set()) & need):
                    break
                picked.append(best)
                need -= subj_labels.get((dataset, best), set())
            for s in ordered:                       # fill any remaining budget in shuffled order
                if len(picked) >= n_val:
                    break
                if s not in picked:
                    picked.append(s)
            val_subjects.update((dataset, s) for s in picked)

        # balanced selection + label map (train labels only)
        label_ids: dict[str, int] = {}
        self.train: list[WindowKey] = []
        self.val: list[WindowKey] = []
        self.n_implausible_dropped = 0
        self.n_duplicate_dropped = 0
        for stream_i, ref in enumerate(self.refs):
            n = ref.n_windows
            bad = self.implausible.get(ref.key, set())
            dup = self.duplicates.get(ref.key, set())
            chosen = (np.arange(n) if max_per_stream is None or n <= max_per_stream
                      else rng.choice(n, size=max_per_stream, replace=False))
            for w in np.sort(chosen):
                if int(w) in bad:                    # physically impossible window — drop, never clip
                    self.n_implausible_dropped += 1
                    continue
                if int(w) in dup:                    # stale re-emitted buffer, not an observation
                    self.n_duplicate_dropped += 1
                    continue
                label = ref.labels[int(w)]
                if label not in label_ids:
                    label_ids[label] = len(label_ids)
                key = WindowKey(stream_i, int(w), label_ids[label])
                if (ref.dataset, ref.subjects[int(w)]) in val_subjects:
                    self.val.append(key)
                else:
                    self.train.append(key)
        self.label_ids = label_ids
        # Label-free physical-event identities. New grids persist session-id + local-window identity;
        # only datasets whose publications/releases establish simultaneous streams are eligible for
        # cross-placement positives. Old grids have stream-local fallback ids and therefore cannot
        # silently manufacture positives from equal lengths/labels/subjects.
        verified_simultaneous = {"xrf_v2", "nfi_fared"}
        ev: dict[str, int] = {}
        self.train_event_ids: list[int] = []
        self.train_event_is_verified: list[bool] = []
        for k in self.train:
            ref = self.refs[k.stream_i]
            verified = ref.dataset in verified_simultaneous and ref.event_ids_explicit
            key = ref.event_ids[k.window_i] if verified else f"{ref.key}:{k.window_i}"
            self.train_event_ids.append(ev.setdefault(key, len(ev)))
            self.train_event_is_verified.append(verified)
        counts: dict[int, int] = {}
        for event_id, verified in zip(self.train_event_ids, self.train_event_is_verified):
            if verified:
                counts[event_id] = counts.get(event_id, 0) + 1
        self.train_positive_event_ids = [
            event_id if verified and counts.get(event_id, 0) > 1 else -1
            for event_id, verified in zip(self.train_event_ids, self.train_event_is_verified)
        ]
        subject_ids: dict[tuple[str, str], int] = {}
        self.train_subject_ids = []
        for key in self.train:
            ref = self.refs[key.stream_i]
            subject = (ref.dataset, str(ref.subjects[key.window_i]))
            self.train_subject_ids.append(
                subject_ids.setdefault(subject, len(subject_ids))
            )
        self.row_aligned_datasets = sorted({
            self.refs[k.stream_i].dataset
            for k, event_id in zip(self.train, self.train_positive_event_ids)
            if event_id >= 0
        })
        # Which labels the VAL split can actually score. best.pt is selected on val_knn_ba, so a
        # label with zero val windows is silently unscored — surface it instead of claiming
        # "all classes are scored" (audit F1).
        val_labels = {k.label_id for k in self.val}
        metric_labels = {
            label: label_id for label, label_id in label_ids.items()
            if label != UNLABELED_LABEL
        }
        self.val_missing_labels = sorted(
            label for label, label_id in metric_labels.items() if label_id not in val_labels
        )
        self.n_val_labels = len(val_labels)
        self.n_semantic_labels = len(metric_labels)
        # Shuffle val so any truncated eval subset (embed() caps at val_max_windows) is a
        # representative cross-dataset sample — index.val is otherwise stream-ordered, so a
        # 2k cap saw only capture24 (alphabetically first) = 8 of 56 labels.
        rng.shuffle(self.val)

    def summary(self) -> str:
        extra = ""
        if getattr(self, "n_implausible_dropped", 0):
            extra += f" · {self.n_implausible_dropped} implausible dropped"
        if getattr(self, "n_duplicate_dropped", 0):
            extra += f" · {self.n_duplicate_dropped} duplicate dropped"
        if getattr(self, "val_missing_labels", None):
            extra += f" · val scores {self.n_val_labels}/{self.n_semantic_labels} labels " \
                     f"(missing: {', '.join(self.val_missing_labels[:4])})"
        elif hasattr(self, "n_val_labels"):
            extra += f" · val scores ALL {self.n_val_labels} labels"
        return (f"{len(self.refs)} streams · {len(self.train)} train / {len(self.val)} val "
                f"windows · {len(self.label_ids)} labels{extra}")


# ----------------------------------------------------------------------------------------------
# Dataset + augmentation
# ----------------------------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """Reseed np.random AND stdlib random per worker (the augmenter uses both)."""
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + worker_id)
    stdlib_random.seed(seed + worker_id + 1)


class PretrainDataset(Dataset):
    """One item = one augmented window: variable (T', 6) data + rate + texts + label.

    ``two_view`` (augmentation-agreement A2): also emit an independently augmented second view
    window under ``item["view_b"]`` (its own signal augmentation, rate, channel_mask and
    augmentation-consistent channel text). The collate patchifies it into the ``*_b`` keys."""

    def __init__(self, index: CorpusIndex, keys: list[WindowKey],
                 augment: bool = True, two_view: bool = False):
        self.index = index
        self.keys = keys
        self.two_view = two_view
        cfg = AugmentationConfig.default_v2() if augment else AugmentationConfig.none()
        if augment:
            cfg.gravity.p = GRAVITY_AUG_P     # audit: 0.5 killed gravity on half the corpus
            cfg.label_text.enabled = False    # A2 uses label IDs, not text — this aug is
                                              # computed-then-discarded in pretraining (label
                                              # text is a Pipeline-B concern)
        self.augmenter = IMUAugmenter(cfg)
        self._data_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def _grid(self, stream_i: int) -> np.ndarray:
        if stream_i not in self._data_cache:
            self._data_cache[stream_i] = self.index.refs[stream_i].load_data()
        return self._data_cache[stream_i]

    def _augment_to_slots(self, ref, key: WindowKey, base_texts: list[str], slot: dict,
                          shared_config_seed: int | None = None) -> dict:
        """Load a FRESH copy of the raw window, run one augmentation pass, and scatter the
        survivors back into the canonical 6-slot layout. Each call draws independently from the
        RNG, so two calls on the same key give two independent positive views."""
        window = torch.tensor(
            np.asarray(self._grid(key.stream_i)[key.window_i], dtype=np.float32)
        )
        role_texts, sensor_texts, sensor_id = stream_sensor_texts(
            ref.dataset, ref.stream,
            has_accel=bool(any(ref.mask[:3])), has_gyro=bool(any(ref.mask[3:])),
        )
        sample = IMUSample(
            data=window,
            channel_names=list(CHANNELS),
            sampling_rate=ref.rate_hz,
            channel_descriptions=list(base_texts),
            label=ref.labels[key.window_i],
            dataset_name=ref.dataset,
            channel_mask=[bool(m) for m in ref.mask],   # real vs zero-padded channels (F10b)
            role_descriptions=role_texts,
            sensor_descriptions=sensor_texts,
            sensor_id=sensor_id,
            gravity_state=_stream_gravity_state(ref.dataset, ref.stream),
        )
        sample = self.augmenter(sample, shared_config_seed=shared_config_seed)

        # channel_dropout REMOVES channels from the tensor (e.g. gyro drop -> (T',3)).
        # Scatter survivors back into the canonical 6-slot layout and mask the rest —
        # the same pad+mask contract the grids use.
        data6 = sample.data.new_zeros(sample.data.shape[0], len(CHANNELS))
        mask6 = torch.zeros(len(CHANNELS), dtype=torch.bool)
        texts6 = list(base_texts)
        for j, name in enumerate(sample.channel_names):
            i = slot[name]
            data6[:, i] = sample.data[:, j]
            mask6[i] = bool(ref.mask[i])
            texts6[i] = sample.channel_descriptions[j]    # keep augmented text
        # Enforce the pad+mask contract: augmentations (jitter etc.) write noise into
        # grid-masked zero-filled channels — a masked slot must stay exactly zero.
        data6[:, ~mask6] = 0.0
        # Scatter the fully augmented role text back to canonical slots just like the signal. Sensor
        # text is already sensor-level and carries any gravity/phrase/dropout augmentation.
        role_texts6 = [_CHANNEL_ROLE_TEXT[c] for c in CHANNELS]
        sensor_id6 = torch.zeros(len(CHANNELS), dtype=torch.long)
        for j, name in enumerate(sample.channel_names):
            i = slot[name]
            if sample.role_descriptions is not None:
                role_texts6[i] = sample.role_descriptions[j]
            if sample.sensor_id is not None:
                sensor_id6[i] = int(sample.sensor_id[j])
        # Acquisition rate = the widest band the HARDWARE ever measured, which no later processing
        # can widen. Two stages can only ever narrow it:
        #   * the converter (STREAM_SOURCE_RATE_HZ records streams stored above their capture clock),
        #   * the rate augmentation, which is a real anti-aliased resample_poly.
        # So the bound is min(hardware, augmented). Scaling the hardware rate by the augmentation
        # ratio (the previous formula) is wrong in BOTH directions: upsampling wisdm 20 -> 94 Hz
        # advertised 94 Hz of measured bandwidth (32 observable bands where only 26 carry signal),
        # and downsampling xrf_v2's 25 Hz AirPods stream to 21.9 Hz reported 10.9 Hz, hiding 5 real
        # bands. This feeds the encoder tokens, the fixed A1 target and a1_feature_valid, so an
        # over-claim teaches the filterbank that interpolation artifacts are measurable signal.
        hardware_rate = STREAM_SOURCE_RATE_HZ.get(f"{ref.dataset}/{ref.stream}", float(ref.rate_hz))
        source_rate = min(float(hardware_rate), float(sample.sampling_rate))
        return {
            "data": data6,                                # (T', 6) canonical slots
            "rate": float(sample.sampling_rate),
            "source_rate": source_rate,
            "texts": texts6,
            # Factored text (docs/design/TEXT_CONDITIONING.md §4b): carried per view so the A2
            # second view gets its OWN independently-augmented role/sensor text. label_id is
            # view-independent and lives in __getitem__ below.
            "role_texts": role_texts6,
            "sensor_texts": list(sample.sensor_descriptions or sensor_texts),
            "sensor_id": sensor_id6,
            "channel_mask": mask6,
            "gravity_state": sample.gravity_state,
        }

    def __getitem__(self, i: int) -> dict:
        key = self.keys[i]
        ref = self.index.refs[key.stream_i]
        base_texts = stream_channel_descriptions(ref.dataset, ref.stream)
        slot = {c: k for k, c in enumerate(CHANNELS)}
        # One CONFIG-text-dropout decision per window, shared by both positive views. The signal
        # augmentations still draw independently (that is what makes the pair a positive), but the
        # views never disagree on WHETHER the acquisition config was described — otherwise NT-Xent
        # would train embed(config) == embed(no config), i.e. to ignore the config text, which is
        # the opposite of the config-conditional thesis. See SensorTextDropoutCfg.shared_across_views.
        cfg_seed = stdlib_random.randrange(2 ** 31)
        view = self._augment_to_slots(ref, key, base_texts, slot, shared_config_seed=cfg_seed)
        item = {
            **view,                                       # data/rate/texts/role_texts/sensor_texts/sensor_id/channel_mask/gravity_state
            "label_id": key.label_id,
            "source": ref.dataset,                        # for per-source telemetry
            "stream": ref.key,
            "window_index": key.window_i,
            "subject": f"{ref.dataset}:{ref.subjects[key.window_i]}",
            "event_id": ref.event_ids[key.window_i],
            "event_verified": bool(
                ref.event_ids_explicit and ref.dataset in {"xrf_v2", "nfi_fared"}
            ),
        }
        if self.two_view:
            # Second view: signal augmentation INDEPENDENT, config-text dropout SHARED (same seed).
            item["view_b"] = {
                **self._augment_to_slots(ref, key, base_texts, slot, shared_config_seed=cfg_seed),
                "label_id": key.label_id,
                "source": ref.dataset,
            }
        return item


class BalancedBatchSampler(Sampler[list[int]]):
    """classes_per_batch x samples_per_class over the train keys — guarantees SupCon
    positives. Classes with too few windows are excluded from anchoring (rare labels are
    oversampled with replacement instead of dropped).

    When ``stream_datasets`` is given, the per-class draw is additionally **source-balanced**:
    a label's `samples_per_class` slots are spread evenly across the datasets/configs that
    carry that label (round-robin over datasets, then a window drawn within each), instead of
    uniformly over the pooled windows. This stops a large single-config dataset (e.g. capture24,
    acc-only wrist free-living) from dominating a shared activity's window pool, so each activity
    is seen across many acquisition configs — the point of the config-generalization corpus. It
    also makes SupCon positives cross-config. Without it, behaviour is the old uniform-over-pool."""

    def __init__(self, keys: list[WindowKey], classes_per_batch: int,
                 samples_per_class: int, steps_per_epoch: int, seed: int = SEED,
                 min_windows: int = MIN_WINDOWS_TO_ANCHOR, *,
                 stream_datasets: list[str] | None = None):
        by_label: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            by_label.setdefault(k.label_id, []).append(i)
        # Exclude labels too small to form real positives — anchoring an 8-slot batch
        # with a 2-window class means 4x duplicated windows (degenerate SupCon positives).
        self.excluded = sorted(l for l, w in by_label.items() if len(w) < min_windows)
        kept = {l: w for l, w in by_label.items() if len(w) >= min_windows}
        if self.excluded:
            print(f"[sampler] excluded {len(self.excluded)} labels below "
                  f"{min_windows} windows from anchoring: {self.excluded}")
        # Per label, group key indices by their source dataset (for source-balanced draws).
        # stream_datasets is None -> a single pooled group per label (old uniform behaviour).
        self.by_label_ds: dict[int, list[np.ndarray]] = {}
        for l, idxs in kept.items():
            if stream_datasets is None:
                self.by_label_ds[l] = [np.asarray(idxs)]
            else:
                per_ds: dict[str, list[int]] = {}
                for i in idxs:
                    per_ds.setdefault(stream_datasets[keys[i].stream_i], []).append(i)
                self.by_label_ds[l] = [np.asarray(v) for v in per_ds.values()]
        self.labels = list(self.by_label_ds)
        self.classes_per_batch = min(classes_per_batch, len(self.labels))
        self.samples_per_class = samples_per_class
        self.steps = steps_per_epoch
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps):
            classes = rng.choice(len(self.labels), size=self.classes_per_batch,
                                 replace=False)
            batch: list[int] = []
            for c in classes:
                pools = self.by_label_ds[self.labels[int(c)]]   # one array per source dataset
                order = rng.permutation(len(pools))             # round-robin datasets, shuffled
                # Assign each of samples_per_class slots to a source round-robin, then draw the
                # slots for each source WITHOUT replacement (F6): no duplicate window indices in a
                # class-group unless a source pool is smaller than its assigned slot count (a rare
                # tiny pool), which keeps SupCon positives distinct.
                counts: dict[int, int] = {}
                for s in range(self.samples_per_class):
                    src = int(order[s % len(pools)])
                    counts[src] = counts.get(src, 0) + 1
                for src, cnt in counts.items():
                    pool = pools[src]
                    picks = rng.choice(len(pool), size=cnt, replace=cnt > len(pool))
                    batch.extend(int(pool[p]) for p in picks)
            yield batch


def _capped_probabilities(
    scores: dict[str, float],
    max_share: float | None,
) -> dict[str, float]:
    """Normalize positive source scores while enforcing a feasible probability ceiling."""
    if not scores or any(value <= 0 for value in scores.values()):
        raise ValueError("source scores must be non-empty and strictly positive")
    total = float(sum(scores.values()))
    probabilities = {key: float(value) / total for key, value in scores.items()}
    if max_share is None:
        return probabilities
    if not 0.0 < float(max_share) <= 1.0:
        raise ValueError("max_dataset_share must be in (0, 1]")

    # Fewer than 1/max_share datasets makes the requested ceiling mathematically impossible.
    ceiling = max(float(max_share), 1.0 / len(probabilities))
    fixed: dict[str, float] = {}
    remaining = dict(probabilities)
    while remaining:
        budget = 1.0 - sum(fixed.values())
        remainder_total = sum(remaining.values())
        proposed = {
            key: budget * value / remainder_total
            for key, value in remaining.items()
        }
        over = [key for key, value in proposed.items() if value > ceiling + 1e-12]
        if not over:
            fixed.update(proposed)
            break
        for key in over:
            fixed[key] = ceiling
            del remaining[key]
    return fixed


class TemperatureSampler(Sampler[int]):
    """Hierarchical label-free corpus sampler (NO activity-class balancing).

    Dataset mass starts proportional to ``n_dataset ** alpha`` and can be explicitly capped.
    When subject IDs are supplied, each dataset's mass is then split as
    ``P(subject | dataset) ∝ n_subject ** subject_alpha`` before drawing a window uniformly
    from that subject.
      * alpha = 1  -> proportional to dataset size (large corpora dominate),
      * alpha = 0  -> uniform per source (every dataset equally likely),
      * alpha = 0.25 + a 0.25 cap is the Phase-A recipe.
    Draws ``num_samples`` indices WITH replacement via ``torch.multinomial``; with a DataLoader
    ``batch_size`` and ``drop_last=True`` that yields ``num_samples // batch_size`` full batches.
    Unlike ``BalancedBatchSampler`` there is no per-class structure; activity labels are unused."""

    def __init__(self, keys: list[WindowKey], stream_datasets: list[str], num_samples: int,
                 alpha: float = 0.5, seed: int = SEED, batch_size: int = 0,
                 event_ids: Sequence[int] | None = None,
                 positive_event_ids: Sequence[int] | None = None,
                 pair_fraction: float = 0.0,
                 subject_ids: Sequence[int] | None = None,
                 subject_alpha: float = 1.0,
                 max_dataset_share: float | None = None):
        from collections import Counter
        datasets = [stream_datasets[k.stream_i] for k in keys]
        counts = Counter(datasets)
        if float(alpha) < 0:
            raise ValueError("alpha must be nonnegative")
        if not 0.0 <= float(subject_alpha) <= 1.0:
            raise ValueError("subject_alpha must be in [0, 1]")
        if subject_ids is not None and len(subject_ids) != len(keys):
            raise ValueError("subject_ids must align 1:1 with keys")
        self.dataset_probabilities = _capped_probabilities(
            {dataset: count ** float(alpha) for dataset, count in counts.items()},
            max_dataset_share,
        )

        if subject_ids is None:
            row_weights = [
                self.dataset_probabilities[dataset] / counts[dataset]
                for dataset in datasets
            ]
        else:
            subject_counts = Counter(zip(datasets, subject_ids))
            subject_scores: dict[str, float] = {}
            for (dataset, _subject), count in subject_counts.items():
                subject_scores[dataset] = (
                    subject_scores.get(dataset, 0.0) + count ** float(subject_alpha)
                )
            row_weights = []
            for dataset, subject in zip(datasets, subject_ids):
                count = subject_counts[(dataset, subject)]
                within_dataset = (
                    count ** (float(subject_alpha) - 1.0)
                    / subject_scores[dataset]
                )
                row_weights.append(self.dataset_probabilities[dataset] * within_dataset)
        self.weights = torch.tensor(row_weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.alpha = float(alpha)
        self.subject_alpha = float(subject_alpha)
        self.max_dataset_share = max_dataset_share
        self.seed = int(seed)
        # When >0, draw each batch of this size WITHOUT replacement so a batch never contains an exact
        # duplicate window (a false negative for the SimCLR control and wasted diversity for VICReg);
        # still holds ACROSS batches (F11). 0 => legacy per-window with-replacement draw.
        self.batch_size = int(batch_size)
        # Per-window EVENT id; two windows sharing one id are simultaneous recordings of the same
        # physical event (different placements) and must not co-occur unless explicitly paired.
        self.event_ids = None if event_ids is None else np.asarray(event_ids)
        self.positive_event_ids = (
            None if positive_event_ids is None else np.asarray(positive_event_ids)
        )
        self.pair_fraction = float(pair_fraction)
        if not 0.0 <= self.pair_fraction < 1.0:
            raise ValueError("pair_fraction must be in [0, 1)")
        if self.positive_event_ids is not None and len(self.positive_event_ids) != len(keys):
            raise ValueError("positive_event_ids must align 1:1 with keys")
        self.positive_event_members: dict[int, list[int]] = {}
        if self.positive_event_ids is not None:
            for i, event_id in enumerate(self.positive_event_ids.tolist()):
                if event_id >= 0:
                    self.positive_event_members.setdefault(int(event_id), []).append(i)
            self.positive_event_members = {
                event_id: members for event_id, members in self.positive_event_members.items()
                if len(members) >= 2
            }
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        # Advance the seed per epoch so the calibration pass and each training epoch draw
        # different windows (mirrors BalancedBatchSampler's per-__iter__ epoch bump).
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        bs = self.batch_size
        if 1 < bs <= self.weights.numel():
            # Each batch drawn WITHOUT replacement (no exact within-batch duplicate = no NT-Xent
            # false negative), WITH replacement across batches. num_samples is a multiple of bs.
            for _ in range(self.num_samples // bs):
                take: list[int] = []
                seen_windows: set[int] = set()
                seen_events: set[int] = set()

                # Reserve a controlled fraction of the batch for verified simultaneous positives.
                # Each selected event contributes exactly two placements; remaining draws avoid that
                # event so every event yields one unambiguous pair.
                n_pairs = min(
                    int(round(bs * self.pair_fraction / 2.0)),
                    len(self.positive_event_members),
                )
                if n_pairs:
                    event_keys = sorted(self.positive_event_members)
                    chosen = torch.randperm(len(event_keys), generator=gen)[:n_pairs].tolist()
                    for j in chosen:
                        event_id = event_keys[j]
                        members = self.positive_event_members[event_id]
                        order = torch.randperm(len(members), generator=gen)[:2].tolist()
                        for pos in order:
                            i = members[pos]
                            take.append(i)
                            seen_windows.add(i)
                        seen_events.add(event_id)

                # Oversample once, then keep unique windows and unique non-positive events. This
                # preserves the old event hygiene when pair_fraction=0 while allowing only the
                # deliberately reserved positive pairs when pair_fraction>0.
                draw_n = min(max(4 * bs, bs), self.weights.numel())
                candidates = torch.multinomial(
                    self.weights, draw_n, replacement=False, generator=gen
                ).tolist()
                for i in candidates:
                    if len(take) >= bs:
                        break
                    if i in seen_windows:
                        continue
                    event_id = int(self.event_ids[i]) if self.event_ids is not None else i
                    positive_id = (
                        int(self.positive_event_ids[i])
                        if self.positive_event_ids is not None else -1
                    )
                    grouping_id = positive_id if positive_id >= 0 else event_id
                    if grouping_id in seen_events:
                        continue
                    take.append(i)
                    seen_windows.add(i)
                    seen_events.add(grouping_id)
                if len(take) < bs:
                    # The corpus has vastly more events than one batch; reaching this fallback means
                    # an unusually tiny synthetic test corpus. Preserve batch size without duplicating
                    # a window, even if distinct-event hygiene must relax.
                    for i in candidates:
                        if len(take) >= bs:
                            break
                        if i not in seen_windows:
                            take.append(i)
                            seen_windows.add(i)
                if len(take) != bs:
                    raise RuntimeError(
                        f"could only draw {len(take)}/{bs} unique windows for a temperature batch"
                    )
                order = torch.randperm(bs, generator=gen).tolist()
                yield from [take[i] for i in order]
        else:
            idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=gen)
            yield from idx.tolist()


# ----------------------------------------------------------------------------------------------
# Multi-scale collate
# ----------------------------------------------------------------------------------------------
class MultiScaleCollate:
    """Draw ONE patch_seconds per batch; patchify each sample at its OWN rate.

    Per sample: compute A3 targets on the raw augmented window (rotation-invariant) -> patchify.
    Trailing patches the (possibly rate-shortened / cropped) window can't fill are flagged in
    `patch_padding_mask`, never treated as real.

    **Gravity is NOT aligned by default (design decision, 2026-07-19).** The tokenizer's signed-DC
    feature already exposes the gravity DIRECTION per channel, so posture (stand/sit/lie, which differ
    only in that direction) stays readable by the model; and the `rotation_3d` augmentation teaches
    pose/mount-rotation robustness. Canonicalizing pitch/roll to +z did the opposite of both — it
    flattened every posture's DC to (0,0,1) and cancelled most of `rotation_3d`. `align_gravity=True`
    is kept only for the align-vs-no-align ablation. NOTE: eval/inference must match this (no align).

    Output: patches (B, P, S, 6) zero-padded · patch_len (B,) · rates (B,) ·
    positions (B, P) s · channel_mask (B, 6) · patch_padding_mask (B, P) True=real ·
    texts · labels · A3 targets (validity-masked).
    """

    def __init__(self, dft_size: int = DFT_SIZE,
                 patch_choices: Sequence[float] = PATCH_SECONDS_CHOICES,
                 fixed_patch_seconds: float | None = None, seed: int = SEED,
                 align_gravity: bool = False, compute_targets: bool = True,
                 two_view: bool = False):
        self.dft_size = dft_size
        self.patch_choices = tuple(patch_choices)
        self.fixed = fixed_patch_seconds
        self.align_gravity = align_gravity
        # A3 grounding targets (cadence + eigen) cost an FFT + PCA per window. They are ONLY used
        # by the training loss; validation/embedding never reads them, so val loaders pass
        # compute_targets=False to skip that per-window DSP (the val-speed fix — 2026-07-19).
        self.compute_targets = compute_targets
        # A2: also patchify item["view_b"] (the second augmented view) into `*_b` keys.
        self.two_view = two_view
        self.seed = seed

    def _patch_seconds(self, batch: list[dict]) -> float:
        if self.fixed is not None:
            return self.fixed
        # Seed from the batch content (label ids) so the draw is deterministic yet
        # DIFFERENT across batches/workers — a single cloned rng repeats across workers.
        key = hash(tuple(item["label_id"] for item in batch)) ^ self.seed
        return float(np.random.default_rng(key & 0xFFFFFFFF).choice(self.patch_choices))

    def __call__(self, batch: list[dict]) -> dict:
        ps = self._patch_seconds(batch)
        out = self._collate_impl(batch, ps, self.compute_targets)
        if self.two_view and batch and "view_b" in batch[0]:
            # Second positive view: same patch_seconds; A1/A3 use only view A.
            out_b = self._collate_impl([item["view_b"] for item in batch], ps, compute_targets=False)
            for k in ("patches", "patch_len", "rates", "source_rates", "positions", "texts",
                      "role_texts", "sensor_texts", "sensor_id",
                      "channel_mask", "patch_padding_mask"):
                out[f"{k}_b"] = out_b[k]
        return out

    def _collate_impl(self, batch: list[dict], ps: float, compute_targets: bool) -> dict:
        P = max(1, int(round(WINDOW_SECONDS / ps)))
        B = len(batch)
        patches = torch.zeros(B, P, self.dft_size, len(CHANNELS))
        patch_len = torch.zeros(B, dtype=torch.long)
        patch_pad = torch.zeros(B, P, dtype=torch.bool)     # True = real patch
        rates = torch.zeros(B)
        source_rates = torch.zeros(B)
        cadence_t = torch.zeros(B)
        cadence_v = torch.zeros(B, dtype=torch.bool)
        eigen_t = torch.full((B, 4, 3), float("nan"))
        eigen_v = torch.zeros(B, dtype=torch.bool)
        # Per-sample patch-CENTER positions (seconds). Default = the nominal patch grid; the
        # usable==0 fallback below overrides row b's single patch to the window's TRUE center (F4a).
        positions = (torch.arange(P).float() * ps + ps / 2).unsqueeze(0).repeat(B, 1)

        for b, item in enumerate(batch):
            data, rate = item["data"], item["rate"]
            n = max(1, int(round(rate * ps)))
            if n > self.dft_size:
                raise ValueError(f"patch length {n} exceeds dft_size {self.dft_size}")
            # A3 targets on the RAW augmented view (rotation-invariant, so pre-align).
            # Call cadence + eigen DIRECTLY on the accel triad (canonical slots 0:3) — the
            # full compute_primitives would also gravity-align internally and compute 4
            # unused families (grav-energy/coherence/shape/dc_tilt); that redundant work +
            # the double align is the second-agent efficiency finding. Skipped entirely when
            # compute_targets is False (val/embedding loaders don't use A3).
            if compute_targets:
                acc = data[:, :3].unsqueeze(0)                           # (1, T, 3)
                cad, eig = cadence(acc, rate), eigen_ratios(acc, rate)
                cadence_t[b] = cad.values[0, 0].nan_to_num(0.0)
                cadence_v[b] = cad.valid[0]
                eigen_t[b] = eig.values[0]
                eigen_v[b] = eig.valid[0]
            # Gravity-align the whole window on its true length (one R per window). Pass the
            # AUTHORITATIVE gravity_state so gravity-removed streams skip alignment (F9).
            if self.align_gravity:
                data, _, _ = gravity_align(data.unsqueeze(0), list(CHANNELS), rate,
                                           gravity_state=item.get("gravity_state"))
                data = data[0]
            per_patch = n
            usable = min(P, data.shape[0] // n)
            if usable == 0 and data.shape[0] > 0:
                # Window shorter than one patch at this scale (e.g. sp_sw_har's 1.0 s TUG
                # windows in a ps=1.5 batch). Emit ONE short patch spanning the whole
                # window rather than an all-padding window (which yields a degenerate
                # pooled embedding that poisons A2). patch_len is honest (< n); the
                # filterbank flags the under-resolved bands via its resolution mask.
                per_patch, usable = data.shape[0], 1
                positions[b, 0] = 0.5 * per_patch / rate   # true center of the short patch (F4a)
                patches[b, 0, :per_patch] = data[:per_patch]
            else:
                for p in range(usable):
                    patches[b, p, :n] = data[p * n:(p + 1) * n]
                # Recover the discarded tail (data.shape[0] % n samples) with ONE extra patch.
                if usable < P and data.shape[0] > usable * n:
                    # End-anchor one full-length tail patch. It may overlap the prior patch, which is
                    # valid because the filterbank tokenizes every patch independently.
                    patches[b, usable, :n] = data[-n:]
                    positions[b, usable] = (data.shape[0] - 0.5 * n) / rate
                    usable += 1
            patch_pad[b, :usable] = True
            patch_len[b] = per_patch
            rates[b] = rate
            source_rates[b] = float(item.get("source_rate", rate))
        return {
            "patches": patches,
            "patch_len": patch_len,
            "rates": rates,
            "source_rates": source_rates,
            "positions": positions,
            "patch_seconds": ps,
            "texts": [item["texts"] for item in batch],
            # Factored text conditioning (docs/design/TEXT_CONDITIONING.md §4b), read ONLY by the
            # factored path. Tolerant of manual items (eval/tests) that omit them — those keep the
            # legacy per_channel path where these are unused (None), so the default is unaffected.
            "role_texts": [item.get("role_texts") for item in batch],
            "sensor_texts": [item.get("sensor_texts") for item in batch],
            "sensor_id": (torch.stack([item["sensor_id"] for item in batch])
                          if "sensor_id" in batch[0] else None),
            "labels": torch.tensor([item["label_id"] for item in batch]),
            "sources": [item.get("source", "?") for item in batch],   # per-window dataset (telemetry)
            "streams": [item.get("stream", "?") for item in batch],
            "window_indices": torch.tensor([item.get("window_index", -1) for item in batch]),
            "subjects": [item.get("subject", "?") for item in batch],
            "event_ids": [item.get("event_id") for item in batch],
            "event_verified": torch.tensor(
                [bool(item.get("event_verified", False)) for item in batch]
            ),
            "channel_mask": torch.stack([item["channel_mask"] for item in batch]),
            "patch_padding_mask": patch_pad,
            "cadence_target": cadence_t,
            "cadence_valid": cadence_v,
            "eigen_target": eigen_t,
            "eigen_valid": eigen_v,
        }


class MultiResolutionCollate:
    """Present one randomly drawn short and long patch grid in the same token sequence.

    Unlike ``MultiScaleCollate``, this collate retains partial tail patches and emits a true
    length for every token. Tokens are sorted by physical center time, with ``resolution_ids``
    distinguishing short (0) from long (1) supports. Signal augmentation has already happened
    once in ``PretrainDataset``; both grids therefore describe exactly the same augmented view.
    """

    def __init__(
        self,
        dft_size: int = DFT_SIZE,
        short_choices: Sequence[float] = SHORT_PATCH_SECONDS_CHOICES,
        long_choices: Sequence[float] = LONG_PATCH_SECONDS_CHOICES,
        fixed_patch_seconds: tuple[float, float] | None = None,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        min_resolution_ratio: float = MIN_RESOLUTION_RATIO,
        seed: int = SEED,
        align_gravity: bool = False,
        compute_targets: bool = True,
        two_view: bool = False,
    ):
        self.dft_size = int(dft_size)
        self.short_choices = tuple(float(x) for x in short_choices)
        self.long_choices = tuple(float(x) for x in long_choices)
        self.fixed = fixed_patch_seconds
        self.max_batch_tokens = int(max_batch_tokens)
        self.min_resolution_ratio = float(min_resolution_ratio)
        self.seed = int(seed)
        self.align_gravity = bool(align_gravity)
        self.compute_targets = bool(compute_targets)
        # A2: also patchify item["view_b"] (the second augmented view) into `*_b` keys.
        self.two_view = bool(two_view)
        self._valid_pairs = tuple(
            (short, long)
            for short in self.short_choices
            for long in self.long_choices
            if long >= self.min_resolution_ratio * short
        )
        if self.fixed is not None:
            short, long = map(float, self.fixed)
            if long < self.min_resolution_ratio * short:
                raise ValueError("fixed resolution pair does not satisfy min_resolution_ratio")
            self.fixed = (short, long)
        elif not self._valid_pairs:
            raise ValueError("no short/long duration pair satisfies min_resolution_ratio")

    def _patch_seconds(self, batch: list[dict]) -> tuple[float, float]:
        if self.fixed is not None:
            return self.fixed
        key = hash(tuple(item["label_id"] for item in batch)) ^ self.seed
        rng = np.random.default_rng(key & 0xFFFFFFFF)
        pairs = self._valid_pairs
        if self.max_batch_tokens:
            # Peak VRAM scales with the TOKEN count B x P, and P is a function of the
            # patch_seconds drawn for THIS batch: short draws tile the window more finely.
            # Measured, P swings 12 -> 22 at a fixed batch size, so memory was a random
            # variable and an unlucky draw could OOM a batch size that had survived the
            # previous sixty steps. Restrict the draw to pairs whose token count fits the
            # budget; if none do, take the coarsest pair (fewest tokens) rather than fail.
            n = len(batch)
            max_seconds = max(
                float(item["data"].shape[0]) / max(float(item["rate"]), 1e-9) for item in batch
            )
            def _tokens(pair: tuple[float, float]) -> int:
                return n * sum(int(np.ceil(max_seconds / p)) for p in pair)
            affordable = [p for p in pairs if _tokens(p) <= self.max_batch_tokens]
            pairs = affordable or [max(pairs, key=lambda p: p[0] + p[1])]
        return pairs[int(rng.integers(len(pairs)))]

    def __call__(self, batch: list[dict]) -> dict:
        pair = self._patch_seconds(batch)
        out = self._collate_impl(batch, pair, self.compute_targets)
        if self.two_view and batch and "view_b" in batch[0]:
            # Second positive view: same resolution pair; A1/A3 use only view A.
            out_b = self._collate_impl([item["view_b"] for item in batch], pair,
                                       compute_targets=False)
            for k in ("patches", "patch_len", "rates", "source_rates", "positions", "patch_durations",
                      "resolution_ids", "texts", "role_texts", "sensor_texts", "sensor_id",
                      "channel_mask", "patch_padding_mask"):
                out[f"{k}_b"] = out_b[k]
        return out

    def _collate_impl(self, batch: list[dict], pair: tuple[float, float],
                      compute_targets: bool) -> dict:
        B = len(batch)
        rates = torch.zeros(B)
        source_rates = torch.zeros(B)
        channel_mask = torch.stack([item["channel_mask"] for item in batch])
        cadence_t = torch.zeros(B)
        cadence_v = torch.zeros(B, dtype=torch.bool)
        eigen_t = torch.full((B, 4, 3), float("nan"))
        eigen_v = torch.zeros(B, dtype=torch.bool)
        all_entries: list[list[tuple]] = []

        for b, item in enumerate(batch):
            data, rate = item["data"], float(item["rate"])
            rates[b] = rate
            source_rates[b] = float(item.get("source_rate", rate))
            if compute_targets:
                acc = data[:, :3].unsqueeze(0)
                cad, eig = cadence(acc, rate), eigen_ratios(acc, rate)
                cadence_t[b] = cad.values[0, 0].nan_to_num(0.0)
                cadence_v[b] = cad.valid[0]
                eigen_t[b] = eig.values[0]
                eigen_v[b] = eig.valid[0]
            if self.align_gravity:
                data, _, _ = gravity_align(data.unsqueeze(0), list(CHANNELS), rate,
                                           gravity_state=item.get("gravity_state"))
                data = data[0]

            entries = []
            for resolution_id, duration in enumerate(pair):
                nominal_n = max(1, int(round(rate * duration)))
                if nominal_n > self.dft_size:
                    raise ValueError(
                        f"patch length {nominal_n} exceeds dft_size {self.dft_size}"
                    )
                min_tail = max(4, int(MIN_TAIL_FRACTION * nominal_n))   # F7 tail floor
                res_start = len(entries)
                for start in range(0, data.shape[0], nominal_n):
                    end = min(start + nominal_n, data.shape[0])
                    n = end - start
                    if n <= 0:
                        continue
                    # F7: a sub-fraction TAIL patch (e.g. UniMiB's 1-sample remainder) is a degenerate
                    # token — its ~0.02 s duration is clamped to the duration-embedding floor and it
                    # still joins self-attention. Drop it rather than feed a physically meaningless
                    # token; but never drop the ONLY patch of a resolution (a window shorter than one
                    # patch keeps its single partial). Duration-weighting (F1) handles the rest.
                    if n < min_tail and len(entries) > res_start:
                        continue
                    start_s, end_s = start / rate, end / rate
                    entries.append((
                        0.5 * (start_s + end_s), resolution_id, start_s, end_s,
                        n / rate, n, data[start:end],
                    ))
            # Physical-time order keeps causal/windowed attention meaningful. Short tokens
            # precede long tokens only when their centers are exactly equal.
            entries.sort(key=lambda x: (x[0], x[1]))
            all_entries.append(entries)

        P = max((len(entries) for entries in all_entries), default=1)
        patches = torch.zeros(B, P, self.dft_size, len(CHANNELS))
        patch_len = torch.zeros(B, P, dtype=torch.long)
        patch_pad = torch.zeros(B, P, dtype=torch.bool)
        positions = torch.zeros(B, P)
        patch_durations = torch.zeros(B, P)
        patch_starts = torch.zeros(B, P)
        patch_ends = torch.zeros(B, P)
        resolution_ids = torch.full((B, P), -1, dtype=torch.long)

        for b, entries in enumerate(all_entries):
            for p, (center, rid, start, end, duration, n, values) in enumerate(entries):
                patches[b, p, :n] = values
                patch_len[b, p] = n
                patch_pad[b, p] = True
                positions[b, p] = center
                patch_durations[b, p] = duration
                patch_starts[b, p] = start
                patch_ends[b, p] = end
                resolution_ids[b, p] = rid

        return {
            "patches": patches,
            "patch_len": patch_len,
            "rates": rates,
            "source_rates": source_rates,
            "positions": positions,
            "patch_durations": patch_durations,
            "patch_starts": patch_starts,
            "patch_ends": patch_ends,
            "resolution_ids": resolution_ids,
            "patch_seconds": pair,
            "texts": [item["texts"] for item in batch],
            # Factored text conditioning (docs/design/TEXT_CONDITIONING.md §4b), read ONLY by the
            # factored path. Tolerant of manual items (eval/tests) that omit them — those keep the
            # legacy per_channel path where these are unused (None), so the default is unaffected.
            "role_texts": [item.get("role_texts") for item in batch],
            "sensor_texts": [item.get("sensor_texts") for item in batch],
            "sensor_id": (torch.stack([item["sensor_id"] for item in batch])
                          if "sensor_id" in batch[0] else None),
            "labels": torch.tensor([item["label_id"] for item in batch]),
            "sources": [item.get("source", "?") for item in batch],
            "streams": [item.get("stream", "?") for item in batch],
            "window_indices": torch.tensor([item.get("window_index", -1) for item in batch]),
            "subjects": [item.get("subject", "?") for item in batch],
            "event_ids": [item.get("event_id") for item in batch],
            "event_verified": torch.tensor(
                [bool(item.get("event_verified", False)) for item in batch]
            ),
            "channel_mask": channel_mask,
            "patch_padding_mask": patch_pad,
            "cadence_target": cadence_t,
            "cadence_valid": cadence_v,
            "eigen_target": eigen_t,
            "eigen_valid": eigen_v,
        }
