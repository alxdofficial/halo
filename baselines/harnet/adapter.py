"""harnet / ssl-wearables adapter (ConSE tier).

A FROZEN OxWearables ``harnet5`` ResNet trunk (30 Hz, 3-ch accelerometer, g-units
WITH gravity, no other normalization) + a linear-softmax head fit on OUR 59-way
global training vocabulary (``data/labels/global_labels.json``). The base bridges
that softmax to the target dataset's labels with ConSE, so harnet — which has no
text tower — is scored zero-shot exactly like the other closed-vocab baselines.

Ported from the working legacy adapter
(``legacy_code/.../baselines/ssl_wearables.py`` + ``evaluate_ssl_wearables.py``).
Two things are carried over verbatim, one is fixed:

  * INPUT CONTRACT (verified in BASELINES.md + the cached hubconf): harnet5 wants
    ``(N, 3, 150)`` = 5 s @ 30 Hz, accelerometer only, g WITH gravity. Each grid
    window (native rate/length, in g) is resampled to 30 Hz and center-crop/wrap-
    padded to 150 samples. harnet10 (300 samples / 10 s) is NOT used: our eval
    grids are <=6 s and would need >=4 s of padding that breaks the 30 Hz kernel
    timing — the same reason the legacy adapter ran harnet5.

  * The released weights load ONLY into ``feature_extractor``; the EvaClassifier
    head is randomly initialized. We freeze the trunk and fit the head on the
    training corpus (the 9 global-vocab train datasets).

  * LEAKAGE FIX (the legacy had a naive random window split): the head-fit model
    selection uses a SUBJECT-DISJOINT split of the training corpus — no source
    subject appears in both the head-train and the head-selection fold, so the
    early-stopping signal cannot be inflated by within-subject correlation. The
    eval targets are a separate held-out cohort, so there is no target leakage at
    all (that guarantee is structural in ZS-XD).

Gravity guard: harnet's contract requires gravity RETAINED. Training datasets
whose accelerometer has gravity removed (detected as a near-zero DC magnitude,
e.g. kuhar) are excluded from the head-fit LOUDLY — feeding gravity-removed
signal to a gravity-expecting trunk is off-distribution. The same check drives
:meth:`is_incompatible` so a gravity-removed EVAL dataset is a disclosed N/A cell
rather than a silently invalid number.

The fitted head is cached to ``baselines/harnet/harnet5_conse_head.pt``
(``*.pt`` is gitignored) with a label-vocab stamp; setup is instant on re-run and
re-fits automatically if the global vocabulary changes.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

from baselines.base import ConSEAdapter, InputContract, global_labels, register
from data.scripts.labels.canonical_labels import canonicalize
from eval import data as eval_data
from eval import scoring

# --- model / window contract (harnet5) ---
HARNET_NAME = "harnet5"
TARGET_HZ = 30
TARGET_LEN = 150          # harnet5 native input: 5 s @ 30 Hz -> (N, 3, 150)
FEAT_DIM = 512            # harnet5 trunk output width
ACC_CHANNELS = ("acc_x", "acc_y", "acc_z")

# torch.hub source pin (mirrors the legacy reproducibility pin).
SSL_HUB_REPO = "OxWearables/ssl-wearables"
SSL_HUB_TAG = "v1.0.0"

# Training corpus = the global-vocab train datasets (data.scripts.labels.build_global_label_mapping).
TRAIN_DATASETS = [
    "uci_har", "hhar", "pamap2", "wisdm", "kuhar", "unimib_shar", "hapt", "mhealth", "capture24",
]

# Gravity guard: a gravity-retaining accel window has |mean vector| ~ 1 g; a
# gravity-removed (linear-accel) one ~ 0 g. Below this threshold => incompatible.
GRAVITY_MIN_G = 0.5

# head-fit hyperparameters (parity with the legacy ConSE head-fit).
FIT_EPOCHS = 100
FIT_BATCH = 512
FIT_LR = 1e-3
FIT_SEED = 3431
EMBED_BATCH = 512

_HERE = Path(__file__).resolve().parent
_HEAD_CACHE = _HERE / "harnet5_conse_head.pt"


# =============================================================================
# torch.hub model + frozen-trunk feature extraction
# =============================================================================

def _hub_ref() -> str:
    return f"{SSL_HUB_REPO}:{SSL_HUB_TAG}"


def _hub_dir() -> Path:
    return Path(torch.hub.get_dir()) / _hub_ref().replace("/", "_").replace(":", "_")


def _load_harnet(num_classes: int, device) -> nn.Module:
    """Pretrained harnet5 (frozen trunk + fresh EvaClassifier head of `num_classes`)."""
    hubdir = _hub_dir()
    if hubdir.exists():
        model = torch.hub.load(str(hubdir), HARNET_NAME, class_num=num_classes,
                               pretrained=True, source="local")
    else:  # offline cache miss -> fetch once from GitHub
        model = torch.hub.load(SSL_HUB_REPO, HARNET_NAME, class_num=num_classes,
                               pretrained=True, source="github", trust_repo=True)
    model.to(device)
    for p in model.feature_extractor.parameters():
        p.requires_grad_(False)
    model.train(False)
    return model


def _to_30hz_150(windows: np.ndarray, rate_hz: float) -> np.ndarray:
    """(N, T, 3) at `rate_hz` -> (N, 3, 150) at 30 Hz for harnet5.

    Resample to 30 Hz (polyphase, anti-aliased), then center-crop to 150 samples
    if longer or wrap-pad if shorter (wrap matches harnet's circular conv padding
    and preserves the gravity DC; only the <5 s train sets ever need padding — the
    <=6 s eval grids are always cropped).
    """
    frac = Fraction(int(round(TARGET_HZ)), int(round(rate_hz))).limit_denominator(1000)
    y = resample_poly(windows.astype(np.float64), frac.numerator, frac.denominator, axis=1)
    L = y.shape[1]
    if L > TARGET_LEN:
        off = (L - TARGET_LEN) // 2
        y = y[:, off:off + TARGET_LEN, :]
    elif L < TARGET_LEN:
        total = TARGET_LEN - L
        left = total // 2
        y = np.pad(y, ((0, 0), (left, total - left), (0, 0)), mode="wrap")
    return np.transpose(y, (0, 2, 1)).astype(np.float32)   # (N, 3, 150)


def _select_accel(windows: np.ndarray, channels: List[str]) -> np.ndarray:
    """Pick the 3 accelerometer channels in x,y,z order."""
    try:
        idx = [channels.index(c) for c in ACC_CHANNELS]
    except ValueError as e:
        raise ValueError(
            f"harnet needs channels {ACC_CHANNELS}; grid has {channels}") from e
    return windows[:, :, idx]


@torch.no_grad()
def _extract_feats(model: nn.Module, x_n3l: np.ndarray, device) -> np.ndarray:
    """(N, 3, 150) -> (N, FEAT_DIM) frozen-trunk features."""
    feats = []
    for s in range(0, len(x_n3l), EMBED_BATCH):
        b = torch.from_numpy(x_n3l[s:s + EMBED_BATCH]).float().to(device)
        f = model.feature_extractor(b)              # (B, C, 1)
        feats.append(f.flatten(1).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# =============================================================================
# training-corpus loading + gravity guard
# =============================================================================

def _load_grid(dataset: str, stream: str):
    """Read a grid directly (train datasets have no eval_labels.json, so the
    eval loader cannot open them). Returns (windows, labels, subjects, channels, rate)."""
    gdir = eval_data.DATASETS_DIR / dataset / "grids" / "non_harmonised" / stream
    windows = np.load(gdir / "data.npy")
    meta = json.loads((gdir / "meta.json").read_text())
    return (windows, list(meta["labels"]), list(map(str, meta["subjects"])),
            list(meta["channels"]), float(meta["rate_hz"]))


def _gravity_dc(windows: np.ndarray, channels: List[str]) -> float:
    """Median magnitude of the per-window mean accel vector (g). ~1 = gravity
    retained, ~0 = gravity removed. Sampled for speed."""
    acc = _select_accel(windows[:2000], channels)
    return float(np.median(np.linalg.norm(acc.mean(axis=1), axis=1)))


# =============================================================================
# adapter
# =============================================================================

@register
class HarnetAdapter(ConSEAdapter):
    """ssl-wearables / harnet5, ConSE tier (accel-only, 30 Hz, g with gravity)."""

    name = "harnet"
    contract = InputContract(channels=list(ACC_CHANNELS), rate_hz=TARGET_HZ,
                             window_sec=TARGET_LEN / TARGET_HZ)

    # ---- gravity compatibility (disclosed N/A instead of a bad number) --------
    def is_incompatible(self, dataset: str):
        streams = eval_data.list_streams(dataset)
        if not streams:
            return None
        try:
            windows, _, _, channels, _ = _load_grid(dataset, streams[0])
        except FileNotFoundError:
            return None
        dc = _gravity_dc(windows, channels)
        if dc < GRAVITY_MIN_G:
            return (f"gravity-removed accelerometer (median |DC|={dc:.3f} g); harnet "
                    "requires gravity retained")
        return None

    # ---- setup: load frozen trunk + fit (or load cached) ConSE head -----------
    def setup(self, device):
        vocab = global_labels()
        model = _load_harnet(len(vocab), device)

        cached = self._load_cached_head(vocab, device)
        if cached is not None:
            model.classifier.load_state_dict(cached)
            model.train(False)
            return {"model": model}

        self._fit_head(model, vocab, device)
        model.train(False)
        return {"model": model}

    def _load_cached_head(self, vocab, device):
        if not _HEAD_CACHE.exists():
            return None
        blob = torch.load(str(_HEAD_CACHE), map_location=device, weights_only=True)
        if list(blob.get("labels", [])) != list(vocab):
            return None   # global vocab changed -> re-fit
        return blob["head"]

    def _fit_head(self, model, vocab, device):
        """Fit the EvaClassifier head on frozen harnet features over the training
        corpus, selecting the best epoch on a SUBJECT-DISJOINT held-out fold."""
        label_to_idx = {l: i for i, l in enumerate(vocab)}
        # Head-fit is compute-heavy (frozen features over ~250k windows); use the
        # GPU for it when present, regardless of the (cpu) eval device.
        fit_device = torch.device("cuda" if torch.cuda.is_available() else device)
        model.to(fit_device)

        feats, labs, subjs, used, skipped = [], [], [], [], []
        for ds in TRAIN_DATASETS:
            for stream in eval_data.list_streams(ds):
                windows, raw_labels, subjects, channels, rate = _load_grid(ds, stream)
                dc = _gravity_dc(windows, channels)
                if dc < GRAVITY_MIN_G:
                    skipped.append(f"{ds}/{stream} (|DC|={dc:.3f}g)")
                    continue
                gl = np.array([label_to_idx.get(canonicalize(l), -1) for l in raw_labels])
                keep = gl >= 0                       # drop labels outside the global vocab
                if not keep.any():
                    continue
                x = _to_30hz_150(_select_accel(windows[keep], channels), rate)
                feats.append(_extract_feats(model, x, fit_device))
                labs.append(gl[keep])
                subjs.append(np.array([f"{ds}:{s}" for s in np.asarray(subjects)[keep]]))
                used.append(f"{ds}/{stream}")
        print(f"[harnet] head-fit corpus: {used}")
        if skipped:
            print(f"[harnet] EXCLUDED (gravity-removed): {skipped}")

        X = np.concatenate(feats, 0)
        Y = np.concatenate(labs, 0)
        S = np.concatenate(subjs, 0)

        # SUBJECT-DISJOINT split (leakage fix): fit on the train fold, select the
        # best epoch on the disjoint val fold. (test fold unused here.)
        ti, vi, _ = scoring.subject_disjoint_split(S, seed=FIT_SEED)
        Xt = torch.from_numpy(X[ti]).float()
        Yt = torch.from_numpy(Y[ti]).long()
        Xv = torch.from_numpy(X[vi]).float().to(fit_device)
        Yv = Y[vi]

        head = model.classifier.to(fit_device)
        for p in head.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(head.parameters(), lr=FIT_LR)
        crit = nn.CrossEntropyLoss()
        rng = np.random.RandomState(FIT_SEED)
        n = len(Xt)
        best_acc, best_sd = -1.0, None
        for _ in range(FIT_EPOCHS):
            head.train()
            perm = rng.permutation(n)
            for s in range(0, n, FIT_BATCH):
                bi = perm[s:s + FIT_BATCH]
                xb = Xt[bi].to(fit_device)
                yb = Yt[bi].to(fit_device)
                opt.zero_grad()
                loss = crit(head(xb), yb)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                va = float((head(Xv).argmax(1).cpu().numpy() == Yv).mean())
            if va > best_acc:
                best_acc = va
                best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if best_sd is not None:
            head.load_state_dict(best_sd)

        n_val_subj = len(set(S[vi]))
        print(f"[harnet] fitted head: val_acc={best_acc:.3f} over {n_val_subj} held-out subjects")
        torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                    "labels": list(vocab)}, str(_HEAD_CACHE))
        model.to(device)

    # ---- window_probs: (N, 59) softmax over the global vocab -------------------
    def window_probs(self, stream, state, device) -> np.ndarray:
        model = state["model"]
        x = _to_30hz_150(_select_accel(stream.windows, stream.channels), stream.rate_hz)
        probs = []
        with torch.no_grad():
            for s in range(0, len(x), EMBED_BATCH):
                b = torch.from_numpy(x[s:s + EMBED_BATCH]).float().to(device)
                probs.append(F.softmax(model(b), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)
