"""LiMU-BERT adapter (ConSE tier).

LiMU-BERT is a self-supervised IMU method (masked reconstruction, ~62K params) —
it ships NO released weights, so its backbone is pretrained ON OUR CORPUS by
``baselines/limubert/train.py`` (see that file for the full-run command). This
adapter loads that self-pretrained backbone and, exactly like the harnet adapter,
fits a softmax head over OUR 59-way global training vocabulary for the zero-shot
ConSE bridge.

Ported from the working legacy adapter + ``evaluate_limubert.py``. Carried over:

  * INPUT CONTRACT (verified in BASELINES.md): 6-ch acc+gyro, 20 Hz, 120 samples
    (6 s); accelerometer ÷ 9.8 -> g (gravity RETAINED), gyro raw, + the model's
    internal LayerNorm. Grid windows (native rate/length, g) are mapped to that
    contract by :mod:`baselines.limubert.prep`; the ÷9.8 is applied here.
  * ARCHITECTURE (base_v1): a 4-layer (weight-shared), 72-d Transformer encoder.
    The module below reproduces the upstream ``LIMUBertModel4Pretrain``
    state_dict layout exactly so the self-pretrained checkpoint loads strict.

LEAKAGE-SAFE HEAD-FIT (mirrors harnet): the head is fit on FROZEN backbone
features (mean-pooled sequence embedding) over the 9 training datasets, selecting
the best epoch on a SUBJECT-DISJOINT held-out fold. The fitted head is cached to
``baselines/limubert/limubert_conse_head.pt`` (gitignored) with a vocab stamp; it
re-fits automatically if the global vocabulary changes.

Backbone checkpoint: ``baselines/limubert/limubert_backbone.pt`` (gitignored),
produced by ``train.py``. If absent, :meth:`setup` fails loud with the command.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.base import ConSEAdapter, InputContract, global_labels, register
from baselines.limubert import prep
from data.scripts.labels.canonical_labels import canonicalize
from eval import scoring

# --- contract / architecture (base_v1) ---
FEATURE_NUM = 6
HIDDEN = 72
HIDDEN_FF = 144
N_LAYERS = 4
N_HEADS = 4
SEQ_LEN = 120
FEAT_DIM = HIDDEN
ACC_NORM = 9.8            # accel ÷ 9.8 -> g, gravity retained

# --- head-fit hyperparameters (parity with the harnet ConSE head-fit) ---
FIT_EPOCHS = 100
FIT_BATCH = 512
FIT_LR = 1e-3
FIT_SEED = 3431
EMBED_BATCH = 512
HEAD_FIT_MAX_PER_STREAM = 20000

_HERE = Path(__file__).resolve().parent
_BACKBONE_CKPT = _HERE / "limubert_backbone.pt"
_HEAD_CACHE = _HERE / "limubert_conse_head.pt"


def _backbone_fp() -> str:
    """Content hash of the pretrained backbone — stamped into the head cache so a re-pretrained
    backbone invalidates a head that was fit on the old backbone's feature space."""
    import hashlib
    return hashlib.sha256(_BACKBONE_CKPT.read_bytes()).hexdigest() if _BACKBONE_CKPT.exists() else ""


# =============================================================================
# Backbone (reproduces upstream LIMUBertModel4Pretrain state_dict layout)
# =============================================================================

def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _split_last(x, shape):
    shape = list(shape)
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -int(np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def _merge_last(x, n_dims):
    s = x.size()
    return x.view(*s[:-n_dims], -1)


class _LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - u) / torch.sqrt(s + self.eps) + self.beta


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FEATURE_NUM, HIDDEN)
        self.pos_embed = nn.Embedding(SEQ_LEN, HIDDEN)
        self.norm = _LayerNorm(HIDDEN)

    def forward(self, x):
        pos = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), x.size(1))
        e = self.norm(self.lin(x))
        return self.norm(e + self.pos_embed(pos))


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(HIDDEN, HIDDEN)
        self.proj_k = nn.Linear(HIDDEN, HIDDEN)
        self.proj_v = nn.Linear(HIDDEN, HIDDEN)
        self.n_heads = N_HEADS

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (_split_last(t, (self.n_heads, -1)).transpose(1, 2) for t in (q, k, v))
        scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)), dim=-1)
        h = (scores @ v).transpose(1, 2).contiguous()
        return _merge_last(h, 2)


class _PWFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, HIDDEN_FF)
        self.fc2 = nn.Linear(HIDDEN_FF, HIDDEN)

    def forward(self, x):
        return self.fc2(_gelu(self.fc1(x)))


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = _Embeddings()
        self.n_layers = N_LAYERS
        self.attn = _Attention()
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.norm1 = _LayerNorm(HIDDEN)
        self.pwff = _PWFF()
        self.norm2 = _LayerNorm(HIDDEN)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class _Backbone(nn.Module):
    """Encoder-only wrapper matching the upstream LIMUBertModel4Pretrain
    checkpoint (fc/linear/norm/decoder retained for a strict load; forward
    returns the encoder sequence output like ``output_embed=True``)."""
    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()
        self.fc = nn.Linear(HIDDEN, HIDDEN)
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self.norm = _LayerNorm(HIDDEN)
        self.decoder = nn.Linear(HIDDEN, FEATURE_NUM)

    def forward(self, x):
        return self.transformer(x)          # (N, 120, 72)


def _normalize(x6: np.ndarray) -> np.ndarray:
    """LiMU-BERT normalization: accel ÷ 9.8 -> g (gravity retained), gyro raw."""
    out = x6.copy()
    out[:, :, :3] = out[:, :, :3] / ACC_NORM
    return out


@torch.no_grad()
def _features(backbone: nn.Module, x6: np.ndarray, device) -> np.ndarray:
    """(N, 120, 6) native g -> (N, 72) mean-pooled frozen-backbone features."""
    xin = _normalize(x6)
    feats = []
    for s in range(0, len(xin), EMBED_BATCH):
        b = torch.from_numpy(xin[s:s + EMBED_BATCH]).float().to(device)
        feats.append(backbone(b).mean(dim=1).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# =============================================================================
# adapter
# =============================================================================

@register
class LiMUBERTAdapter(ConSEAdapter):
    """LiMU-BERT (masked-reconstruction Transformer), ConSE tier."""

    name = "limubert"
    contract = InputContract(channels=list(prep.SIX_CHANNELS), rate_hz=prep.TARGET_HZ,
                             window_sec=prep.TARGET_LEN / prep.TARGET_HZ)

    def setup(self, device):
        vocab = global_labels()
        backbone = self._load_backbone(device)

        head = nn.Linear(FEAT_DIM, len(vocab)).to(device)
        cached = self._load_cached_head(vocab, device)
        if cached is not None:
            head.load_state_dict(cached)
        else:
            self._fit_head(backbone, head, vocab, device)
        head.eval()
        return {"backbone": backbone, "head": head}

    def _load_backbone(self, device) -> nn.Module:
        if not _BACKBONE_CKPT.exists():
            raise FileNotFoundError(
                f"LiMU-BERT backbone checkpoint missing at {_BACKBONE_CKPT}. "
                "Self-pretrain it on our corpus first:\n"
                "  python -m baselines.limubert.train")
        backbone = _Backbone().to(device)
        sd = torch.load(str(_BACKBONE_CKPT), map_location=device, weights_only=True)
        backbone.load_state_dict(sd)
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        return backbone

    def _load_cached_head(self, vocab, device):
        if not _HEAD_CACHE.exists():
            return None
        blob = torch.load(str(_HEAD_CACHE), map_location=device, weights_only=True)
        if list(blob.get("labels", [])) != list(vocab):
            return None
        # Re-fit if the backbone changed (e.g. after the deferred full pretrain overwrites it).
        if blob.get("backbone_fp") != _backbone_fp():
            return None
        return blob["head"]

    def _fit_head(self, backbone, head, vocab, device):
        """Fit the softmax head on frozen backbone features over the training
        corpus, selecting the best epoch on a SUBJECT-DISJOINT held-out fold."""
        label_to_idx = {l: i for i, l in enumerate(vocab)}
        fit_device = torch.device("cuda" if torch.cuda.is_available() else device)
        backbone.to(fit_device)

        feats, labs, subjs, used = [], [], [], []
        for ds, stream, x6, raw_labels, subjects in prep.iter_train_streams(
                max_per_stream=HEAD_FIT_MAX_PER_STREAM, seed=FIT_SEED):
            gl = np.array([label_to_idx.get(canonicalize(l), -1) for l in raw_labels])
            keep = gl >= 0
            if not keep.any():
                continue
            feats.append(_features(backbone, x6[keep], fit_device))
            labs.append(gl[keep])
            subjs.append(np.array([f"{ds}:{s}" for s in subjects[keep]]))
            used.append(f"{ds}/{stream}")
        print(f"[limubert] head-fit corpus: {used}")

        X = np.concatenate(feats, 0)
        Y = np.concatenate(labs, 0)
        S = np.concatenate(subjs, 0)

        ti, vi, _ = scoring.subject_disjoint_split(S, seed=FIT_SEED)
        Xt = torch.from_numpy(X[ti]).float()
        Yt = torch.from_numpy(Y[ti]).long()
        Xv = torch.from_numpy(X[vi]).float().to(fit_device)
        Yv = Y[vi]

        head.to(fit_device)
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
                opt.zero_grad()
                loss = crit(head(Xt[bi].to(fit_device)), Yt[bi].to(fit_device))
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
        print(f"[limubert] fitted head: val_acc={best_acc:.3f} over {n_val_subj} held-out subjects")
        torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                    "labels": list(vocab), "backbone_fp": _backbone_fp()}, str(_HEAD_CACHE))
        backbone.to(device)
        head.to(device)

    def window_probs(self, stream, state, device) -> np.ndarray:
        backbone, head = state["backbone"], state["head"]
        x6 = prep.grid_to_contract(stream.windows, stream.channels, stream.rate_hz)
        feats = _features(backbone, x6, device)
        probs = []
        with torch.no_grad():
            for s in range(0, len(feats), EMBED_BATCH):
                b = torch.from_numpy(feats[s:s + EMBED_BATCH]).float().to(device)
                probs.append(F.softmax(head(b), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)
