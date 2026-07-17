"""Pipeline A Phase-1 data pipeline (pretraining corpus + sampler + collate).

Design decisions carried in from the gates:
  * Corpus = the 9 TRAIN datasets' harmonised grids (eval sets never touched),
    balanced at MAX_PER_STREAM per stream — the same balanced-corpus recipe the
    baselines were trained on (apples-to-apples).
  * Subject-disjoint train/val split per dataset.
  * Label-balanced batches (classes x samples) so A2 SupCon always has positives.
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
from model.tokenizer.primitives import compute_primitives

# ----------------------------------------------------------------------------------------------
# Corpus configuration
# ----------------------------------------------------------------------------------------------
# hapt DROPPED: the sweep confirmed it is the UCI-HAR re-release — same 30 subjects /
# recordings (per-window NCC 0.98 vs uci_har), so keeping both leaks near-duplicate val
# windows into train across the pair. uci_har is the canonical windowed release; keep it.
TRAIN_DATASETS = (
    "uci_har", "hhar", "pamap2", "wisdm", "kuhar", "unimib_shar",
    "mhealth", "capture24",
)
MAX_PER_STREAM = 20_000          # the balanced-corpus cap (107k-scale); PER-STREAM (so
                                 # wisdm's 2 streams get 2x — kept for baseline parity,
                                 # phone+watch ARE distinct configs; audited 2026-07-18)
WINDOW_SECONDS = 6.0
VAL_SUBJECT_FRACTION = 0.10      # subject-disjoint val within the train datasets
GRAVITY_AUG_P = 0.15             # DROP from default_v2's 0.5: the audit found gravity
                                 # removed on 52% of windows, killing the M0 gravity-align /
                                 # DC-tilt features on half the corpus. 0.15 keeps the
                                 # iOS-userAcceleration robustness without dominating.
MIN_WINDOWS_TO_ANCHOR = 8        # labels below this can't form real SupCon positives
                                 # (would be duplicate-oversampled); excluded from anchoring
PATCH_SECONDS_CHOICES = (0.5, 0.75, 1.0, 1.5, 2.0)   # per-batch multi-scale draw
DFT_SIZE = 256                   # must cover max rate (100 Hz) x max patch (2 s) = 200
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
SEED = 20260718

PLACEMENT_WORDS = {
    "waist": "the waist", "wrist": "the wrist", "pocket": "a trouser pocket",
    "hip": "the hip", "belt": "the belt", "thigh": "the thigh",
    "back": "the lower back", "ankle": "the ankle", "chest": "the chest",
}


def stream_channel_descriptions(dataset: str, stream: str) -> list[str]:
    """Per-channel base text from the stream id (e.g. 'phone_waist')."""
    tokens = stream.lower().split("_")
    device = "phone" if "phone" in tokens else ("watch" if "watch" in tokens else "device")
    place = next((PLACEMENT_WORDS[w] for w in tokens if w in PLACEMENT_WORDS), "the body")
    where = f"{place} ({device})"
    return ([f"accelerometer {a}-axis worn at {where}" for a in "xyz"]
            + [f"gyroscope {a}-axis worn at {where}" for a in "xyz"])


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

    def __init__(self, max_per_stream: int = MAX_PER_STREAM, seed: int = SEED,
                 datasets: Sequence[str] = TRAIN_DATASETS):
        self.refs: list[GridRef] = [
            r for r in discover_grids("harmonised") if r.dataset in set(datasets)
        ]
        if not self.refs:
            raise FileNotFoundError("no harmonised train grids found — build grids first")
        rng = np.random.default_rng(seed)

        # subject-disjoint split per dataset
        val_subjects: set[tuple[str, str]] = set()
        by_dataset: dict[str, set[str]] = {}
        for ref in self.refs:
            by_dataset.setdefault(ref.dataset, set()).update(ref.subjects)
        for dataset, subjects in sorted(by_dataset.items()):
            ordered = sorted(subjects)
            rng.shuffle(ordered)
            n_val = max(1, int(round(len(ordered) * VAL_SUBJECT_FRACTION)))
            val_subjects.update((dataset, s) for s in ordered[:n_val])

        # balanced selection + label map (train labels only)
        label_ids: dict[str, int] = {}
        self.train: list[WindowKey] = []
        self.val: list[WindowKey] = []
        for stream_i, ref in enumerate(self.refs):
            n = ref.n_windows
            chosen = (rng.choice(n, size=max_per_stream, replace=False)
                      if n > max_per_stream else np.arange(n))
            for w in np.sort(chosen):
                label = ref.labels[int(w)]
                if label not in label_ids:
                    label_ids[label] = len(label_ids)
                key = WindowKey(stream_i, int(w), label_ids[label])
                if (ref.dataset, ref.subjects[int(w)]) in val_subjects:
                    self.val.append(key)
                else:
                    self.train.append(key)
        self.label_ids = label_ids
        # Shuffle val so any truncated eval subset (embed() caps at val_max_windows) is a
        # representative cross-dataset sample — index.val is otherwise stream-ordered, so a
        # 2k cap saw only capture24 (alphabetically first) = 8 of 56 labels.
        rng.shuffle(self.val)

    def summary(self) -> str:
        return (f"{len(self.refs)} streams · {len(self.train)} train / {len(self.val)} val "
                f"windows · {len(self.label_ids)} labels")


# ----------------------------------------------------------------------------------------------
# Dataset + augmentation
# ----------------------------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """Reseed np.random AND stdlib random per worker (the augmenter uses both)."""
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + worker_id)
    stdlib_random.seed(seed + worker_id + 1)


class PretrainDataset(Dataset):
    """One item = one augmented window: variable (T', 6) data + rate + texts + label."""

    def __init__(self, index: CorpusIndex, keys: list[WindowKey],
                 augment: bool = True):
        self.index = index
        self.keys = keys
        cfg = AugmentationConfig.default_v2() if augment else AugmentationConfig.none()
        if augment:
            cfg.gravity.p = GRAVITY_AUG_P     # audit: 0.5 killed gravity on half the corpus
        self.augmenter = IMUAugmenter(cfg)
        self._data_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def _grid(self, stream_i: int) -> np.ndarray:
        if stream_i not in self._data_cache:
            self._data_cache[stream_i] = self.index.refs[stream_i].load_data()
        return self._data_cache[stream_i]

    def __getitem__(self, i: int) -> dict:
        key = self.keys[i]
        ref = self.index.refs[key.stream_i]
        window = torch.tensor(
            np.asarray(self._grid(key.stream_i)[key.window_i], dtype=np.float32)
        )
        sample = IMUSample(
            data=window,
            channel_names=list(CHANNELS),
            sampling_rate=ref.rate_hz,
            channel_descriptions=stream_channel_descriptions(ref.dataset, ref.stream),
            label=ref.labels[key.window_i],
            dataset_name=ref.dataset,
        )
        sample = self.augmenter(sample)

        # channel_dropout REMOVES channels from the tensor (e.g. gyro drop -> (T',3)).
        # Scatter survivors back into the canonical 6-slot layout and mask the rest —
        # the same pad+mask contract the grids use.
        base_texts = stream_channel_descriptions(ref.dataset, ref.stream)
        data6 = sample.data.new_zeros(sample.data.shape[0], len(CHANNELS))
        mask6 = torch.zeros(len(CHANNELS), dtype=torch.bool)
        texts6 = list(base_texts)
        slot = {c: i for i, c in enumerate(CHANNELS)}
        for j, name in enumerate(sample.channel_names):
            i = slot[name]
            data6[:, i] = sample.data[:, j]
            mask6[i] = bool(ref.mask[i])
            texts6[i] = sample.channel_descriptions[j]    # keep augmented text
        # Enforce the pad+mask contract: augmentations (jitter etc.) write noise into
        # grid-masked zero-filled channels — a masked slot must stay exactly zero.
        data6[:, ~mask6] = 0.0
        return {
            "data": data6,                                # (T', 6) canonical slots
            "rate": float(sample.sampling_rate),
            "texts": texts6,
            "label_id": key.label_id,
            "channel_mask": mask6,
        }


class BalancedBatchSampler(Sampler[list[int]]):
    """classes_per_batch x samples_per_class over the train keys — guarantees SupCon
    positives. Classes with too few windows are excluded from anchoring (still seen
    as negatives via other classes' batches is NOT possible here, so rare labels are
    simply oversampled with replacement instead of dropped)."""

    def __init__(self, keys: list[WindowKey], classes_per_batch: int,
                 samples_per_class: int, steps_per_epoch: int, seed: int = SEED,
                 min_windows: int = MIN_WINDOWS_TO_ANCHOR):
        by_label: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            by_label.setdefault(k.label_id, []).append(i)
        # Exclude labels too small to form real positives — anchoring an 8-slot batch
        # with a 2-window class means 4x duplicated windows (degenerate SupCon positives).
        self.excluded = sorted(l for l, w in by_label.items() if len(w) < min_windows)
        self.by_label = {l: w for l, w in by_label.items() if len(w) >= min_windows}
        if self.excluded:
            print(f"[sampler] excluded {len(self.excluded)} labels below "
                  f"{min_windows} windows from anchoring: {self.excluded}")
        self.labels = list(self.by_label)
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
                pool = self.by_label[self.labels[int(c)]]
                take = rng.choice(len(pool), size=self.samples_per_class,
                                  replace=len(pool) < self.samples_per_class)
                batch.extend(pool[int(t)] for t in take)
            yield batch


# ----------------------------------------------------------------------------------------------
# Multi-scale collate
# ----------------------------------------------------------------------------------------------
class MultiScaleCollate:
    """Draw ONE patch_seconds per batch; patchify each sample at its OWN rate.

    Per sample, in order: compute A3 targets on the raw augmented window (dc_tilt needs
    the un-aligned frame) -> **gravity-align the whole window on its REAL length** (one
    rotation per window; NOT per zero-padded patch — the sweep found the padded-patch
    estimate is diluted to a no-op) -> patchify. Trailing patches that the (possibly
    rate-shortened) window can't fill are flagged in `patch_padding_mask`, never treated
    as real.

    Output: patches (B, P, S, 6) zero-padded · patch_len (B,) · rates (B,) ·
    positions (B, P) s · channel_mask (B, 6) · patch_padding_mask (B, P) True=real ·
    texts · labels · A3 targets (validity-masked).
    """

    def __init__(self, dft_size: int = DFT_SIZE,
                 patch_choices: Sequence[float] = PATCH_SECONDS_CHOICES,
                 fixed_patch_seconds: float | None = None, seed: int = SEED,
                 align_gravity: bool = True):
        self.dft_size = dft_size
        self.patch_choices = tuple(patch_choices)
        self.fixed = fixed_patch_seconds
        self.align_gravity = align_gravity
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
        P = max(1, int(round(WINDOW_SECONDS / ps)))
        B = len(batch)
        patches = torch.zeros(B, P, self.dft_size, len(CHANNELS))
        patch_len = torch.zeros(B, dtype=torch.long)
        patch_pad = torch.zeros(B, P, dtype=torch.bool)     # True = real patch
        rates = torch.zeros(B)
        cadence_t = torch.zeros(B)
        cadence_v = torch.zeros(B, dtype=torch.bool)
        eigen_t = torch.full((B, 4, 3), float("nan"))
        eigen_v = torch.zeros(B, dtype=torch.bool)

        for b, item in enumerate(batch):
            data, rate = item["data"], item["rate"]
            n = max(1, int(round(rate * ps)))
            if n > self.dft_size:
                raise ValueError(f"patch length {n} exceeds dft_size {self.dft_size}")
            # A3 targets on the RAW augmented view (rotation-invariant; dc_tilt pre-align)
            prims = compute_primitives(data.unsqueeze(0), CHANNELS, rate)
            cadence_t[b] = prims["cadence"].values[0, 0].nan_to_num(0.0)
            cadence_v[b] = prims["cadence"].valid[0]
            eigen_t[b] = prims["eigen_ratios"].values[0]
            eigen_v[b] = prims["eigen_ratios"].valid[0]
            # Gravity-align the whole window on its true length (one R per window)
            if self.align_gravity:
                data, _, _ = gravity_align(data.unsqueeze(0), list(CHANNELS), rate)
                data = data[0]
            usable = min(P, data.shape[0] // n)
            for p in range(usable):
                patches[b, p, :n] = data[p * n:(p + 1) * n]
            patch_pad[b, :usable] = True
            patch_len[b] = n
            rates[b] = rate

        # .contiguous(): expand() aliases memory and pin_memory refuses aliased tensors
        positions = (torch.arange(P).float() * ps + ps / 2).unsqueeze(0) \
            .expand(B, P).contiguous()
        return {
            "patches": patches,
            "patch_len": patch_len,
            "rates": rates,
            "positions": positions,
            "patch_seconds": ps,
            "texts": [item["texts"] for item in batch],
            "labels": torch.tensor([item["label_id"] for item in batch]),
            "channel_mask": torch.stack([item["channel_mask"] for item in batch]),
            "patch_padding_mask": patch_pad,
            "cadence_target": cadence_t,
            "cadence_valid": cadence_v,
            "eigen_target": eigen_t,
            "eigen_valid": eigen_v,
        }
