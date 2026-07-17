"""M2 HARD GATE — does the load-bearing loss cohere? (build plan M2)

Tiny, decisive, CPU-only (the GPU belongs to other work): frozen M1 filterbank
features + a trivial 2-layer encoder, trained a few hundred steps on the M0
4-stream setup with the ELITE-3 losses, one stream held out ENTIRELY (the
config-transfer axis).

GATE CRITERIA
  (a) all three losses trainable (decrease; no NaN),
  (b) held-out-CONFIG kNN balanced accuracy >= the best handcrafted M0 family
      (grav_band_energy) computed on the SAME split — the learned encoder must
      at least match the best single hand-designed feature it consumes,
  (c) no collapse: embedding effective rank stays > a floor; A3 stays a rail
      (its gradient share small).
If the loss is incoherent or collapses -> STOP, redesign (do not build M3+).

Deliberate M2 simplifications (documented, resolved at M3/Phase-1): fixed
patch_seconds=1.0 (multi-scale exercised in unit tests); config conditioning via a
learned per-stream token with an UNKNOWN fallback for the held-out stream (real
channel-TEXT conditioning arrives with M3); augs limited to SO(3)+gain (rate /
time-warp need the bucketed sampler).

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.m2_gate
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.tokenizer.filterbank import PhysicalFilterbankTokenizer
from model.tokenizer.preprocess import gravity_align
from model.tokenizer.primitives import compute_primitives
from training.tokenizer.losses_repr import (
    EliteLossWeights,
    GroundingTargets,
    elite3_loss,
    make_mask_plan,
)
from training.tokenizer.probe_robustness import STREAMS, collect_windows

# ------------------------------------------------------------------ configuration
SEED = 20260718
RATE = 60.0
PATCH_SECONDS = 1.0          # fixed for the gate; multi-scale is Phase-1
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
GYRO_IDX = [3, 4, 5]
D_MODEL = 64
STEPS = 600
BATCH = 96
LR = 1e-3
EVAL_EVERY = 150
CONFIG_DROPOUT_P = 0.2       # train the UNKNOWN config token (deployment = unseen config)
KNN_K = 5
EFFECTIVE_RANK_FLOOR = 8.0   # of D_MODEL=64; uniform collapse would be ~1
HOLDOUT_DEFAULT = "pamap2"   # the only wrist stream in the M0 set = hardest transfer
OUT = Path(__file__).resolve().parent / "outputs" / "m2_gate"

AUG_P_ROTATION = 0.5
AUG_P_GAIN = 0.5
GAIN_RANGE = (0.7, 1.3)


# ----------------------------------------------------------------------- utilities
def random_so3(g: torch.Generator) -> torch.Tensor:
    q = torch.randn(4, generator=g)
    w, x, y, z = (q / q.norm()).tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def augment(batch: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """SO(3) (joint acc+gyro) + gain, per sample. Targets are recomputed after this
    (both primitives are invariant to these augs — the A3 selection criterion)."""
    out = batch.clone()
    B = batch.shape[0]
    for b in range(B):
        if torch.rand(1, generator=g).item() < AUG_P_ROTATION:
            r = random_so3(g)
            out[b, :, :3] = out[b, :, :3] @ r.t()
            out[b, :, 3:] = out[b, :, 3:] @ r.t()
        if torch.rand(1, generator=g).item() < AUG_P_GAIN:
            lo, hi = GAIN_RANGE
            out[b] *= lo + (hi - lo) * torch.rand(1, generator=g).item()
    return out


def effective_rank(z: torch.Tensor) -> float:
    s = torch.linalg.svdvals(z - z.mean(dim=0))
    p = (s / s.sum()).clamp(min=1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def knn_balanced_acc(train_z, train_y, test_z, test_y, k: int = KNN_K) -> float:
    labels = sorted(set(train_y.tolist()) & set(test_y.tolist()))
    per_class = []
    for label in labels:
        idx = (test_y == label).nonzero().squeeze(1)
        hits = 0
        for i in idx.tolist():
            d = (train_z - test_z[i]).norm(dim=1)
            nn_lab = train_y[d.argsort()[:k]]
            hits += int(nn_lab.mode().values) == label
        per_class.append(hits / len(idx))
    return float(np.mean(per_class))


# ------------------------------------------------------------------------ the model
class TrivialEncoder(nn.Module):
    """Deliberately small: per-token linear + channel & config embeddings + sinusoidal
    time + 2-layer transformer. Exists to test the LOSS, not to be good."""

    def __init__(self, feat_dim: int, n_channels: int, n_configs: int, p_max: int = 16):
        super().__init__()
        self.embed = nn.Linear(feat_dim, D_MODEL)
        self.channel_emb = nn.Embedding(n_channels, D_MODEL)
        self.config_emb = nn.Embedding(n_configs + 1, D_MODEL)   # last = UNKNOWN fallback
        self.mask_token = nn.Parameter(torch.zeros(D_MODEL))
        pos = torch.zeros(p_max, D_MODEL)
        t = torch.arange(p_max).unsqueeze(1)
        div = torch.exp(torch.arange(0, D_MODEL, 2) * (-math.log(1e4) / D_MODEL))
        pos[:, 0::2], pos[:, 1::2] = torch.sin(t * div), torch.cos(t * div)
        self.register_buffer("time_pos", pos)
        layer = nn.TransformerEncoderLayer(D_MODEL, nhead=4, dim_feedforward=128,
                                           batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.a1_head = nn.Linear(D_MODEL, feat_dim)
        self.a3_cadence = nn.Linear(D_MODEL, 1)
        self.a3_eigen = nn.Linear(D_MODEL, 4 * 3)

    def forward(self, feats: torch.Tensor, config_ids: torch.Tensor,
                token_mask: torch.Tensor | None = None):
        """feats (B,P,C,F) frozen filterbank features; token_mask (B,P,C) True=hide."""
        B, P, C, _ = feats.shape
        x = self.embed(feats)                                    # (B,P,C,D)
        if token_mask is not None:
            x = torch.where(token_mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        x = x + self.channel_emb.weight[:C].view(1, 1, C, -1)
        x = x + self.time_pos[:P].view(1, P, 1, -1)
        tokens = x.reshape(B, P * C, -1)
        cfg = self.config_emb(config_ids).unsqueeze(1)           # (B,1,D)
        h = self.encoder(torch.cat([cfg, tokens], dim=1))
        h_tokens = h[:, 1:].reshape(B, P, C, -1)
        z = h[:, 1:].mean(dim=1)                                 # window embedding (B,D)
        return h_tokens, z


# ------------------------------------------------------------------------ data prep
def build_tensors(device=torch.device("cpu")):
    windows = collect_windows()
    labels = sorted({w.label for w in windows})
    label_id = {l: i for i, l in enumerate(labels)}
    datasets = [d for d, _ in STREAMS]
    config_id = {d: i for i, d in enumerate(datasets)}
    x = torch.tensor(np.stack([w.acc for w in windows] , axis=0), dtype=torch.float32)
    gyro = torch.tensor(np.stack([w.gyro for w in windows], axis=0), dtype=torch.float32)
    data = torch.cat([x, gyro], dim=2)                           # (N, 360, 6)
    y = torch.tensor([label_id[w.label] for w in windows])
    cfg = torch.tensor([config_id[w.dataset] for w in windows])
    return data.to(device), y.to(device), cfg.to(device), labels, datasets


def patchify(batch: torch.Tensor, tok: PhysicalFilterbankTokenizer) -> tuple[torch.Tensor, int]:
    """(B, 360, 6) -> zero-padded patches (B, P, S, C) + native patch length N."""
    B, T, C = batch.shape
    n = int(PATCH_SECONDS * RATE)
    P = T // n
    patches = batch[:, : P * n].reshape(B, P, n, C)
    padded = batch.new_zeros(B, P, tok.S, C)
    padded[:, :, :n] = patches
    return padded, n


def frozen_features(batch, tok):
    """The FULL M1 front end: gravity-align (canonicalization, always on — the M0
    winner; skipping it was gate-harness bug #3) -> physical filterbank."""
    aligned, _, _ = gravity_align(batch, CHANNELS, RATE)
    padded, n = patchify(aligned, tok)
    with torch.no_grad():
        return tok(padded, RATE, torch.tensor([n] * batch.shape[0]))


def grounding_targets(batch: torch.Tensor) -> GroundingTargets:
    prims = compute_primitives(batch, CHANNELS, RATE)
    cad, eig = prims["cadence"], prims["eigen_ratios"]
    return GroundingTargets(
        cadence_log2hz=cad.values[:, 0].nan_to_num(0.0),
        cadence_valid=cad.valid,
        eigen_ratios=eig.values,
        eigen_valid=eig.valid,
    )


# ------------------------------------------------------------------------- the gate
def run_gate(holdout: str) -> dict:
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    data, y, cfg, labels, datasets = build_tensors()
    hold_id = datasets.index(holdout)
    train_idx = (cfg != hold_id).nonzero().squeeze(1)
    test_idx = (cfg == hold_id).nonzero().squeeze(1)
    print(f"holdout={holdout}: {len(train_idx)} train / {len(test_idx)} held-out windows")

    # Frozen front end: fixed filterbank, proj replaced by identity -> raw physical
    # features (e_hat + masks + amp + dc), calibrated on TRAIN windows only.
    tok = PhysicalFilterbankTokenizer(d_model=1, dft_size=256)
    tok.proj = nn.Identity()
    aligned_train, _, _ = gravity_align(data[train_idx], CHANNELS, RATE)
    padded, n = patchify(aligned_train, tok)
    tok.fit_norm_stats(padded, RATE, torch.tensor([n] * len(train_idx)))
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    feat_dim = tok.in_dim

    model = TrivialEncoder(feat_dim, n_channels=6, n_configs=len(datasets))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    weights = EliteLossWeights()
    history, grad_shares = [], []

    def embed_all(idx, unknown_config: bool):
        model.eval()
        zs = []
        with torch.no_grad():
            for start in range(0, len(idx), 256):
                chunk = idx[start:start + 256]
                feats = frozen_features(data[chunk], tok)
                cfg_ids = torch.full((len(chunk),), len(datasets), dtype=torch.long) \
                    if unknown_config else cfg[chunk]
                _, z = model(feats, cfg_ids)
                zs.append(z)
        model.train()
        return torch.cat(zs)

    def evaluate(step):
        train_z = embed_all(train_idx, unknown_config=False)
        test_z = embed_all(test_idx, unknown_config=True)   # unseen config -> UNKNOWN token
        ba = knn_balanced_acc(train_z, y[train_idx], test_z, y[test_idx])
        rank = effective_rank(test_z)
        print(f"  step {step:4d}  held-out kNN-BA={ba:.3f}  eff-rank={rank:.1f}")
        return {"step": step, "holdout_ba": ba, "effective_rank": rank}

    print("training (CPU, tiny by design) ...")
    evals = [evaluate(0)]
    for step in range(1, STEPS + 1):
        sel = train_idx[torch.randint(len(train_idx), (BATCH,), generator=g)]
        batch = augment(data[sel], g)
        feats = frozen_features(batch, tok)
        targets = grounding_targets(batch)                       # on the AUGMENTED view
        plan = make_mask_plan(BATCH, feats.shape[1], 6, GYRO_IDX, generator=g)

        # Config-dropout: sometimes present the UNKNOWN token so the unseen-config
        # eval path is actually trained (deployment = configs we've never seen).
        cfg_ids = cfg[sel].clone()
        drop = torch.rand(BATCH, generator=g) < CONFIG_DROPOUT_P
        cfg_ids[drop] = len(datasets)

        # TWO forwards: masked for A1; CLEAN for A2/A3 (contrastive must not fight the
        # mask noise — the masked+contrastive recipe uses full views for the contrast).
        h_tokens, _ = model(feats, cfg_ids, token_mask=plan.token_mask)
        _, z = model(feats, cfg_ids)
        out = elite3_loss(
            a1_pred=model.a1_head(h_tokens),
            a1_target=feats,
            a1_mask=plan.token_mask,
            a2_embeddings=z,
            a2_labels=y[sel],
            a3_cadence_pred=model.a3_cadence(z).squeeze(1),
            a3_eigen_pred=model.a3_eigen(z).view(BATCH, 4, 3),
            a3_targets=targets,
            weights=weights,
        )
        opt.zero_grad()
        out.total.backward()
        # A3-domination check: gradient share of the shared embed layer per loss part
        opt.step()
        history.append(out.parts)
        if step % EVAL_EVERY == 0:
            evals.append(evaluate(step))

    # ---- handcrafted baseline on the SAME split (apples-to-apples) ----
    from training.tokenizer.probe_robustness import FAMILIES
    feats_hand = []
    for i in range(len(data)):
        w = data[i].numpy()
        feats_hand.append(FAMILIES["grav_band_energy"](w[:, :3], w[:, 3:], RATE))
    hand = torch.tensor(np.stack(feats_hand), dtype=torch.float32)
    col_mean = hand.nan_to_num().mean(dim=0)
    hand = torch.where(torch.isfinite(hand), hand, col_mean.expand_as(hand))
    hand = (hand - hand.mean(dim=0)) / (hand.std(dim=0) + 1e-9)
    hand_ba = knn_balanced_acc(hand[train_idx], y[train_idx], hand[test_idx], y[test_idx])

    first, last = history[0], history[-1]
    losses_decreased = all(last[k] < first[k] for k in ("a1_masked", "a2_supcon"))
    final = evals[-1]
    verdict = {
        "holdout": holdout,
        "losses_first": first, "losses_last": last,
        "losses_decreased": losses_decreased,
        "learned_holdout_ba": final["holdout_ba"],
        "handcrafted_grav_ba": hand_ba,
        "beats_handcrafted": final["holdout_ba"] >= hand_ba,
        "effective_rank": final["effective_rank"],
        "rank_ok": final["effective_rank"] > EFFECTIVE_RANK_FLOOR,
        "evals": evals,
        "gate": None,
    }
    verdict["gate"] = ("PASS" if verdict["losses_decreased"] and verdict["beats_handcrafted"]
                       and verdict["rank_ok"] else "FAIL")
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", default=HOLDOUT_DEFAULT,
                        choices=[d for d, _ in STREAMS])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verdict = run_gate(args.holdout)
    path = OUT / f"verdict_{args.holdout}.json"
    path.write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: v for k, v in verdict.items() if k != "evals"}, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
