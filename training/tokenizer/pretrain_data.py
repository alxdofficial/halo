"""Pipeline A Phase-1 data pipeline (pretraining corpus + sampler + collate).

Design decisions carried in from the gates:
  * Corpus = the 12 TRAIN datasets' **native** grids (eval sets never touched): native sampling
    RATE (no 60 Hz resample) + canonical labels + 6-ch pad+mask. The filterbank tokenizer is
    rate-invariant, so HALO trains on the corpus's REAL native rates (20/50/100 Hz) instead of a
    homogenized 60 Hz base — the 60 Hz "harmonised" grids are the layout-locked baselines' crutch,
    not HALO's. Source-balanced sampling (no per-stream cap) spreads each activity across configs.
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
# per-batch multi-scale draw -> T = 6s / ps in {12, 8, 6, 4}. 2.0s (T=3) DROPPED: T=3 is
# too coarse for masked prediction, and it's where native-short windows (uci_har 2.56s,
# unimib falls) collapse to a single un-maskable patch (objective-health audit 2026-07-18).
PATCH_SECONDS_CHOICES = (0.5, 0.75, 1.0, 1.5)
DFT_SIZE = 256                   # must cover max NATIVE rate (100 Hz) x max patch (1.5 s) = 150;
                                 # the rate aug caps at 100 Hz too, so 256 keeps ample headroom
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
        self.refs: list[GridRef] = [
            r for r in discover_grids(alignment) if r.dataset in set(datasets)
        ]
        if not self.refs:
            raise FileNotFoundError(f"no {alignment} train grids found — build grids first "
                                    f"(python -m data.scripts.build_grids --alignment {alignment})")
        self.stream_datasets = [r.dataset for r in self.refs]   # stream_i -> dataset (for the sampler)
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
            chosen = (np.arange(n) if max_per_stream is None or n <= max_per_stream
                      else rng.choice(n, size=max_per_stream, replace=False))
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
            channel_mask=[bool(m) for m in ref.mask],   # real vs zero-padded channels (F10b)
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
            "gravity_state": _stream_gravity_state(ref.dataset, ref.stream),
        }


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
                 align_gravity: bool = False):
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
            # the double align is the second-agent efficiency finding.
            acc = data[:, :3].unsqueeze(0)                               # (1, T, 3)
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
                # F4b: recover the discarded tail (data.shape[0] % n samples) with ONE extra patch
                # anchored to the END of the window, when a patch slot is free. Uniform length n
                # (it overlaps the previous patch); its position is that end-anchored patch's true
                # center. Floors used to silently drop up to one patch of real signal per window.
                if usable < P and data.shape[0] > usable * n:
                    patches[b, usable, :n] = data[-n:]
                    positions[b, usable] = (data.shape[0] - 0.5 * n) / rate
                    usable += 1
            patch_pad[b, :usable] = True
            patch_len[b] = per_patch
            rates[b] = rate
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
