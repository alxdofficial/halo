"""Pipeline A Phase-1 pretraining — FULLY SELF-SUPERVISED by default.

The default recipe uses NO labels in the training loss:
  * A1 masked-latent recon (masked forward; feature targets from the CALIBRATED frozen
    filterbank) — the world-model rail.                     (M2 lesson 1)
  * A2 = self-supervised SimCLR NT-Xent between TWO independent augmentations of each
    window (a2_mode='simclr'). Labels are used ONLY for the val kNN/ConSE metric.
  * TF-C = time-frequency consistency: NT-Xent pulling the FREQUENCY view (the main encoder's
    pooled output, reused from A2) toward a TIME view (an auxiliary TimeEncoder over the raw
    samples) of the SAME window. Augmentation-free, ON by default alongside SimCLR; summed into
    the total (weight cfg.tfc_weight). --tfc-weight 0 (or --a2-mode supcon) skips it. The
    TimeEncoder is discarded at inference.
  * All three (A1 + SimCLR + TF-C) are SUMMED into one loss — no separate ablation arms.
  * A3 grounding is OFF by default (a3_weight=0); heads + DSP re-enable with --a3-weight 0.1.
  * The corpus sampler is the per-window 'temperature' sampler (P(dataset) ∝ n^alpha),
    NOT class-balanced.
The ORIGINAL label-supervised recipe is fully selectable for the do-no-harm ablation:
  --a2-mode supcon --a3-weight 0.1 --sampler balanced (A1 + label-SupCon A2 + A3 rail).

Other invariants:
  * Config conditioning is channel TEXT; the text-dropout/paraphrase augs supply the
    "unseen description" robustness.                        (M2 lesson 2, upgraded by M3)
  * Gravity alignment is disabled by default; signed DC preserves posture while SO(3)
    augmentation supplies orientation robustness.           (2026-07-19 decision)
  * The encoder's inner filterbank norm is CALIBRATED before training.  (M3 lesson)

Model selection: subject-disjoint val kNN balanced accuracy (macro), not loss.
Checkpoints carry config + label map + filterbank norm stats + git provenance.

Run (CPU smoke):   .../python -m training.tokenizer.pretrain --steps 20 --smoke
Run (real, GPU):   .../python -m training.tokenizer.pretrain --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.tokenizer.encoder import SetTokenizerEncoder
from model.tokenizer.filterbank import PhysicalFilterbankTokenizer
from model.tokenizer.preprocess import gravity_align
from training.tokenizer.losses_repr import (
    EliteLossWeights,
    GroundingTargets,
    elite3_loss,
    make_mask_plan,
    make_multiresolution_mask_plan,
    masked_latent_per_window,
    nt_xent,
)
from training.tokenizer.time_encoder import TimeEncoder
from training.tokenizer.pretrain_data import (
    CHANNELS,
    DFT_SIZE,
    LONG_PATCH_SECONDS_CHOICES,
    MIN_RESOLUTION_RATIO,
    SHORT_PATCH_SECONDS_CHOICES,
    VAL_RESOLUTION_PAIR,
    BalancedBatchSampler,
    CorpusIndex,
    MultiResolutionCollate,
    MultiScaleCollate,
    PretrainDataset,
    TRAIN_DATASETS,
    TemperatureSampler,
    _seed_worker,
)

GYRO_IDX = [3, 4, 5]
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "pretrain"


@dataclass
class PretrainConfig:
    # d256/6L/~7.3M: clears all three consumer floors (tokenizer rep, A1/A2/A3 heads,
    # evidence-engine multi-vector memory), data-appropriate for the ~307k-window native corpus
    # (306,666 train / 38,167 val, 93 labels; well below the shortcut-prone 20M legacy scale).
    # NOTE: the 20k-step budget dates from the old ~96k-window corpus — re-justify/re-tune it for
    # the ~3.2x larger native corpus using the repaired val metric before the real run (F11).
    # the frozen encoder's d sets the memory-bank vector width. (User-approved 2026-07-18.)
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    arm: str = "fixed"                    # headline preset: fixed | learnable
    frontend: str = "fixed"               # independently switchable for attribution
    # Config-text conditioning (docs/design/TEXT_CONDITIONING.md §4b). 'per_channel' (default) is the
    # legacy one-description-per-channel path; 'factored' splits it into per-channel ROLE text +
    # per-sensor IDENTITY text. Default MUST stay 'per_channel' (do-no-harm). asdict(cfg) serializes
    # both into the checkpoint config, so eval/reconstruction picks up the arm automatically.
    text_conditioning: str = "per_channel"
    gate_bias_init: float = -2.0          # factored fusion identity-gate bias at init (sigma~=0.12)
    # NB: the `pretrain` CLI defaults multiresolution=True (the diagnostic-confirmed winner); this
    # dataclass default stays False so direct constructors (e.g. grad_check) get the single-res encoder.
    multiresolution: bool = False
    frontend_lr_scale: float = 0.1         # physical adaptation moves slower than the encoder
    frontend_reg_weight: float = 1e-3
    center_shift_fraction: float = 0.45
    bandwidth_factor_max: float = 1.5
    compression_gain_max: float = 2.0
    filter_shape_min: float = 1.5
    filter_shape_max: float = 2.5
    adaptive_gate_init: float = 0.1
    duration_gate_init: float = 0.1
    short_patch_choices: tuple[float, ...] = SHORT_PATCH_SECONDS_CHOICES
    long_patch_choices: tuple[float, ...] = LONG_PATCH_SECONDS_CHOICES
    min_resolution_ratio: float = MIN_RESOLUTION_RATIO
    val_resolution_pair: tuple[float, float] = VAL_RESOLUTION_PAIR
    classes_per_batch: int = 64           # 64x8 = batch 512 (profiled 2026-07-19: throughput knee on
    samples_per_class: int = 8            # the 4090; 64 of 93 labels/step keeps SupCon negatives rich,
                                          # positive-group size unchanged at 8; peak 9.5 GB, safe)
    steps: int = 30_000                   # 50 epochs of the 305k corpus at batch 512 (596 steps/epoch)
    lr: float = 4.2e-4                    # 3e-4 x sqrt(2): sqrt-scaling for the 256->512 batch doubling
    weight_decay: float = 0.05
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    # A3 (physical-primitive grounding) weight. NEW DEFAULT 0.0 — A3 is OFF in the fully
    # self-supervised recipe (grounding rail dropped). Re-enable the rail with --a3-weight 0.1;
    # the heads + grounding_loss + target DSP all stay intact and switch back on together.
    a3_weight: float = 0.0
    # A1 (masked-latent recon) weight. A1's target is the FIXED filterbank's features, so it is
    # home-field for the filterbank arm (audit #7). For an objective-NEUTRAL tokenizer comparison set
    # a1_weight=0 -> train on A2 + A3 (grounding) only, both frontend-agnostic.
    a1_weight: float = 1.0
    # A2 window-level contrastive mode. NEW DEFAULT 'simclr' = self-supervised NT-Xent over two
    # independent augmentations (no labels in the loss). 'supcon' = the label-SupCon ablation
    # (reproduced with --a2-mode supcon --a3-weight 0.1 --sampler balanced).
    a2_mode: str = "simclr"
    simclr_temperature: float = 0.1       # NT-Xent temperature (SimCLR default)
    # TF-C-INSPIRED cross-domain consistency: a THIRD self-supervised term alongside A1 + A2, ON with
    # the SSL recipe (a2_mode='simclr'). It pulls the frequency-domain embedding (the main encoder's
    # pooled output) toward a time-domain embedding (an auxiliary rate/position-aware TimeEncoder over
    # the raw samples) of the SAME window; NT-Xent, augmentation-free. NOTE this is NOT faithful TF-C
    # (Zhang et al. 2022): there is no time/time contrastive term and the time encoder is discarded at
    # inference — it is a one-directional cross-domain REGULARIZER (see time_encoder.py). Default weight
    # 0.25, NOT 1.0: at weight 1 the raw NT-Xent magnitude (~13) dwarfed A1 (~2) and SimCLR (~4) and
    # drove training; 0.25 makes the three objectives comparable at init. The {0, 0.1, 0.25, 1} weight
    # ablation is still a required pre-retrain GPU experiment (audit F5). tfc_weight==0 cleanly skips the
    # TimeEncoder forward + the term; a2_mode='supcon' also skips it — TF-C is SSL-only.
    tfc_weight: float = 0.25
    tfc_temperature: float = 0.1          # NT-Xent temperature for the time<->freq contrast
    # Corpus sampler. NEW DEFAULT 'temperature' = per-window draw with P(dataset) ∝ n^alpha, NO
    # class balancing. 'balanced' = the label-balanced BalancedBatchSampler (needed for supcon).
    sampler: str = "temperature"
    sampler_alpha: float = 0.5            # 1=proportional, 0=uniform-per-source, 0.5=middle
    batch_size: int = 512                 # batch for the temperature sampler (balanced uses
                                          # classes_per_batch * samples_per_class instead)
    calib_batches: int = 50               # frontend norm calibration pass
    val_every: int = 1_000
    val_per_label: int = 40               # kNN val: windows PER LABEL (stratified, all classes scored)
    knn_k: int = 5
    num_workers: int = 12                 # profiled: nw=4 was data-bound (~35% GPU idle); 12 of 24
                                          # cores makes the step compute-bound (2026-07-19)
    seed: int = 20260718                  # MODEL seed: weight init, augmentation, batch order (varies per replicate)
    # DATA seed: the subject train/val split. Held FIXED across arms AND replicates so every run — and
    # the metric harness — sees the SAME subject-disjoint split (audit 2026-07-23 #1: the eval used the
    # default split regardless of --seed, so seed!=default leaked ~19 train subjects into metric-val).
    data_seed: int = 20260718
    train_datasets: tuple | None = None   # None = full TRAIN_DATASETS; set for the ablation subset
    max_per_stream: int | None = None     # None = use ALL windows; the sampler is source-balanced
    device: str = "cpu"


class PipelineAModel(nn.Module):
    def __init__(self, cfg: PretrainConfig, a1_target_dim: int):
        super().__init__()
        self.encoder = SetTokenizerEncoder(
            d_model=cfg.d_model, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout, dft_size=DFT_SIZE,
            frontend=cfg.frontend,                # 'fixed' (default) | 'learnable'
            text_conditioning=cfg.text_conditioning,  # 'per_channel' (default) | 'factored'
            gate_bias_init=cfg.gate_bias_init,
            use_duration_embedding=cfg.multiresolution,
            duration_min_seconds=min(cfg.short_patch_choices),
            duration_max_seconds=max(cfg.long_patch_choices),
            duration_gate_init=cfg.duration_gate_init,
            rope_min_period=0.4 if cfg.multiresolution else 0.5,
            center_shift_fraction=cfg.center_shift_fraction,
            bandwidth_factor_max=cfg.bandwidth_factor_max,
            compression_gain_max=cfg.compression_gain_max,
            filter_shape_min=cfg.filter_shape_min,
            filter_shape_max=cfg.filter_shape_max,
            adaptive_gate_init=cfg.adaptive_gate_init,
        )
        self.a1_head = nn.Linear(cfg.d_model, a1_target_dim)
        self.a2_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 128),
        )
        self.a3_cadence = nn.Linear(cfg.d_model, 1)
        self.a3_eigen = nn.Linear(cfg.d_model, 4 * 3)
        # --- TF-C rail (auxiliary; discarded at inference) ------------------------------------
        # A compact TIME-domain encoder over the raw samples, plus two SEPARATE projection heads
        # (kept independent of a2_proj so SimCLR and TF-C do not share a bottleneck): tfc_proj
        # projects the FREQUENCY view (reused encoder pooled output), tfc_proj_time the TIME view.
        self.time_encoder = TimeEncoder(cfg.d_model, n_channels=len(CHANNELS))
        self.tfc_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 128),
        )
        self.tfc_proj_time = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 128),
        )


def git_commit() -> str:
    """Short HEAD, suffixed '-dirty' when the working tree has uncommitted changes, so a checkpoint
    honestly records that its source was not a clean commit (F5 — converters/loader are often dirty)."""
    try:
        repo = Path(__file__).resolve().parents[2]
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=repo).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                              text=True, timeout=5, cwd=repo).stdout.strip()
        return f"{head}-dirty" if dirty else (head or "unknown")
    except Exception:
        return "unknown"


def corpus_fingerprint(index) -> str:
    """Stable 16-hex signature of the assembled TRAINING corpus (per-stream dataset/rate/shape +
    label vocab + the cap/seed/selected-window counts that determine WHICH windows were drawn),
    stored in the checkpoint so it records the exact corpus that produced the weights (F5 — grid
    meta.json carries no raw fingerprint; audit #10: two subset runs with different cap/seed must
    NOT collide, so the sampling knobs and realised split sizes are part of the identity)."""
    import hashlib
    sig = [f"{r.dataset}/{r.stream}:{r.rate_hz}:{tuple(r.shape)}:{len(set(r.labels))}"
           for r in sorted(index.refs, key=lambda r: (r.dataset, r.stream))]
    sig.append("labels=" + ",".join(sorted(index.label_ids)))
    sig.append(f"cap={getattr(index, 'max_per_stream', None)}:seed={getattr(index, 'seed', None)}"
               f":ntrain={len(index.train)}:nval={len(index.val)}")
    return hashlib.sha256("|".join(sig).encode()).hexdigest()[:16]


def align_batch(batch: dict) -> dict:
    """DEPRECATED no-op. Gravity alignment now happens PER WINDOW on real-length data
    inside MultiScaleCollate (the sweep found aligning the zero-padded patch buffer here
    was diluted to a ~96% no-op and rotated each patch independently). Kept as a pass-
    through so older call sites don't break; do not add logic here."""
    return batch


def knn_balanced_acc(train_z, train_y, test_z, test_y, k: int) -> float:
    # Score EVERY query label (F1 fix). A query class absent from the support scores 0 — kNN
    # retrieves other-class neighbours — instead of being dropped from the metric. The old
    # `set(train_y) & set(test_y)` intersection silently omitted unsupported query classes,
    # inflating the number and making best.pt selection depend on which classes the random
    # support cap happened to include. Vectorized (cdist+topk+mode) — was a per-query Python loop.
    labels = sorted(set(test_y.tolist()))
    if not labels:
        return float("nan")
    d = torch.cdist(test_z.float(), train_z.float())            # (Nq, Ns) euclidean
    nn_lab = train_y[d.topk(min(k, d.shape[1]), largest=False).indices]   # (Nq, k)
    pred = nn_lab.mode(dim=1).values                            # majority (ties -> smallest id)
    per_class = [float((pred[test_y == label] == label).float().mean())
                 for label in labels if (test_y == label).any()]
    return float(np.mean(per_class)) if per_class else float("nan")


def balanced_acc(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Macro-averaged per-class recall (balanced accuracy) from hard predictions."""
    per_class = []
    for label in sorted(set(true.tolist())):
        m = true == label
        if m.any():
            per_class.append(float((pred[m] == label).float().mean()))
    return float(np.mean(per_class)) if per_class else float("nan")


@torch.no_grad()
def label_text_prototypes(model: PipelineAModel, label_ids: dict) -> torch.Tensor:
    """(L, 384) L2-normalized frozen-LM embedding of each label's name, indexed BY label id.

    Turns "brushing_teeth" -> "a person brushing teeth" -> mean-pooled MiniLM vector. These are
    the class prototypes for the ConSE-style text-cosine probe: the same frozen text tower the
    encoder already uses, so the probe measures whether the sensor representation aligns to label
    SEMANTICS (the downstream zero-shot target), not just cluster purity like kNN."""
    id2str = {i: s for s, i in label_ids.items()}
    prompts = [f"a person {id2str[i].replace('_', ' ')}" for i in range(len(id2str))]
    emb, mask = model.encoder.text_encoder.encode(prompts, device=torch.device("cpu"))  # (L,S,384)
    m = mask.unsqueeze(-1).float()
    proto = (emb * m).sum(1) / m.sum(1).clamp(min=1.0)
    return torch.nn.functional.normalize(proto, dim=1)


def _l2n(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def conse_probe_predict(train_z, train_y, val_z, val_y, protos,
                        ridge_lambda: float = 1.0) -> torch.Tensor:
    """CRUDE-but-comparable zero-shot head: ridge-fit a linear map sensor_emb -> label-text space
    on the TRAIN support (this IS ConSE's semantic projection), then cosine-match each val window
    to the candidate labels' text prototypes (candidates = the val label set). Returns predicted
    label ids (aligned to val_z rows). Fit fresh each val, no calibration — a live proxy for the
    downstream ZS protocol."""
    Zt, Zv = _l2n(train_z.float()), _l2n(val_z.float())
    T = protos[train_y]                                         # (N, 384) target text vectors
    d = Zt.shape[1]
    W = torch.linalg.solve(Zt.t() @ Zt + ridge_lambda * torch.eye(d), Zt.t() @ T)   # (d, 384)
    proj = _l2n(Zv @ W)                                          # val projected into text space
    cand = torch.tensor(sorted(set(val_y.tolist())))
    sims = proj @ _l2n(protos[cand]).t()                        # (Nval, Ncand) cosine
    return cand[sims.argmax(dim=1)]


@torch.no_grad()
def embed_stratified(model: PipelineAModel, loader: DataLoader, device, per_label: int,
                     target_labels: set | None = None):
    """Embed up to ``per_label`` windows PER LABEL (deterministic — the loader must be
    shuffle=False / pre-shuffled). Guarantees every label is represented so the kNN metric covers
    all classes and best.pt selection is stable — replaces the old 'first max_windows in loader
    order' cap that silently missed rare labels and overshot by a batch (F1). When ``target_labels``
    is given (the support over the huge train set), stops early once every target label is saturated."""
    from collections import Counter
    model.eval()
    zs, ys, srcs = [], [], []
    counts: Counter = Counter()
    done: set = set()
    for batch in loader:
        lab = batch["labels"]
        take = [j for j, l in enumerate(lab.tolist()) if counts[l] < per_label]
        if take:
            # factored: channel_texts carries ROLE text and the sensor identity is passed alongside;
            # per_channel (default): unchanged (sensor_texts/sensor_id stay None -> forward defaults).
            factored = model.encoder.text_conditioning == "factored"
            texts = batch["role_texts"] if factored else batch["texts"]
            plen = batch["patch_len"]
            out = model.encoder(
                batch["patches"].to(device), batch["rates"].to(device),
                plen.to(device), texts,
                batch["positions"].to(device),
                patch_durations=(batch["patch_durations"].to(device)
                                 if "patch_durations" in batch else None),
                resolution_ids=(batch["resolution_ids"].to(device)
                                if "resolution_ids" in batch else None),
                channel_mask=batch["channel_mask"].to(device),
                patch_padding_mask=batch["patch_padding_mask"].to(device),
                sensor_texts=(batch["sensor_texts"] if factored else None),
                sensor_id=(batch["sensor_id"].to(device) if factored else None),
            )
            pooled = out["pooled"].cpu()
            zs.append(pooled[take])
            ys.append(lab[take])
            srcs.extend(batch["sources"][j] for j in take)      # per-window source (telemetry)
            for l in lab[take].tolist():
                counts[l] += 1
                if counts[l] >= per_label:
                    done.add(l)
        if target_labels is not None and target_labels <= done:
            break
    model.train()
    return torch.cat(zs), torch.cat(ys), srcs


def module_grad_norms(model) -> dict:
    """Per-module gradient L2 norm (call AFTER unscale_, BEFORE clip → real, un-clipped scale).
    A cheap reduction — computed only on log steps, so no hot-loop cost. Diagnoses vanish/explode
    per component (encoder vs each head)."""
    def _gn(params) -> float:
        sq = sum(float(p.grad.detach().pow(2).sum()) for p in params if p.grad is not None)
        return sq ** 0.5

    mods = (("encoder", model.encoder), ("a1", model.a1_head), ("a2", model.a2_proj),
            ("a3_cad", model.a3_cadence), ("a3_eig", model.a3_eigen),
            ("time_encoder", model.time_encoder))
    out = {f"grad/{name}": _gn(mod.parameters()) for name, mod in mods}
    # TF-C projection heads (both freq- and time-view heads) under one key so their health is
    # visible alongside grad/time_encoder — is the TF-C rail alive, and is any objective drowning?
    out["grad/tfc_proj"] = _gn(list(model.tfc_proj.parameters())
                               + list(model.tfc_proj_time.parameters()))
    return out


def per_source_mean(values: torch.Tensor, sources: list) -> dict:
    """Group a (B,) tensor of per-window values by source dataset (NaN-safe) for telemetry."""
    agg: dict = {}
    for s, v in zip(sources, values.tolist()):
        if v == v:                                              # skip NaN
            agg.setdefault(s, []).append(v)
    return {s: round(float(np.mean(vs)), 4) for s, vs in agg.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny corpus + tiny model for a fast CPU end-to-end check")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing non-empty output dir (default: refuse)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="warm-resume from a checkpoint (restore encoder/heads/opt/sched/scaler/"
                             "RNG + step and continue the remaining steps)")
    parser.add_argument("--arm", choices=("fixed", "learnable"), default="fixed",
                        help="frontend preset. DEFAULT 'fixed' = fixed physical filterbank + "
                             "multiresolution (the winning Phase-A config; see "
                             "docs/design/LEARNABLE_TOKENIZER_ARM.md). 'learnable' swaps in the "
                             "constrained adaptive frontend — a documented negative result, kept opt-in.")
    parser.add_argument("--frontend", choices=("fixed", "learnable"), default=None,
                        help="tokenizer arm. 'fixed' = physical-Hz constant-Q filterbank (default); "
                             "'learnable' = the constrained-adaptive filterbank (documented negative "
                             "result, kept opt-in; docs/design/LEARNABLE_TOKENIZER_ARM.md).")
    parser.add_argument("--text-conditioning", choices=("per_channel", "factored"), default=None,
                        help="config-text conditioning (docs/design/TEXT_CONDITIONING.md §4b). "
                             "'per_channel' (default) = one description per channel; 'factored' = "
                             "per-channel ROLE text + per-sensor IDENTITY text. None keeps the "
                             "config default (per_channel).")
    parser.add_argument("--multiresolution", action=argparse.BooleanOptionalAction, default=None,
                        help="override multiresolution (default ON); --no-multiresolution is the "
                             "single-resolution ablation")
    parser.add_argument("--a1-weight", type=float, default=None,
                        help="scale the A1 masked-recon loss. Set 0 for an objective-NEUTRAL tokenizer "
                             "comparison because A1's target is the fixed filterbank.")
    parser.add_argument("--a2-mode", choices=("simclr", "supcon"), default=None,
                        help="A2 contrastive mode. DEFAULT 'simclr' = self-supervised NT-Xent over two "
                             "augmented views (no labels). 'supcon' = the label-SupCon ablation.")
    parser.add_argument("--a3-weight", type=float, default=None,
                        help="scale the A3 grounding rail. DEFAULT 0.0 (off); pass 0.1 to re-enable it.")
    parser.add_argument("--simclr-temperature", type=float, default=None,
                        help="NT-Xent temperature for --a2-mode simclr (default 0.1).")
    parser.add_argument("--tfc-weight", type=float, default=None,
                        help="scale the TF-C (time-frequency consistency) term, summed alongside "
                             "A1 + SimCLR. DEFAULT 1.0 (ON with the SSL recipe). 0 cleanly skips the "
                             "TimeEncoder forward + TF-C loss (a hygiene knob, not a separate arm); "
                             "TF-C is also skipped under --a2-mode supcon.")
    parser.add_argument("--tfc-temperature", type=float, default=None,
                        help="NT-Xent temperature for the TF-C time<->freq contrast (default 0.1).")
    parser.add_argument("--sampler", choices=("temperature", "balanced"), default=None,
                        help="corpus sampler. DEFAULT 'temperature' (per-window, P(dataset) ∝ n^alpha, "
                             "no class balancing). 'balanced' = label-balanced batches (needed for supcon).")
    parser.add_argument("--sampler-alpha", type=float, default=None,
                        help="temperature-sampler exponent: 1=proportional, 0=uniform-per-source, "
                             "0.5=middle (default).")
    parser.add_argument("--batch", type=int, default=None,
                        help="batch size for the temperature sampler (default 512). The balanced "
                             "sampler ignores this and uses classes_per_batch * samples_per_class.")
    parser.add_argument("--subset", action="store_true",
                        help="train on the tokenizer-ablation 3-rate-core subset (5 datasets, xrf_v2 "
                             "held out) instead of the full corpus. See ablation_subset.py.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="explicit train dataset list (overrides --subset and the full corpus).")
    parser.add_argument("--max-per-stream", type=int, default=None,
                        help="per-stream window cap (default: None=all; --subset defaults to the "
                             "ablation DEFAULT_CAP so train and metric-eval share one corpus).")
    parser.add_argument("--seed", type=int, default=None,
                        help="MODEL seed (init/augmentation/batch order). Vary this across replicates.")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="DATA seed = the subject train/val split. Keep FIXED across all arms and "
                             "replicates so the split (and the metric harness) stays identical (#1).")
    args = parser.parse_args()

    cfg = PretrainConfig(
        device=args.device,
        arm=args.arm,
        frontend="learnable" if args.arm == "learnable" else "fixed",
        multiresolution=True,          # new Phase-A default: multiresolution ON (diagnostic-confirmed
                                       # winner, 0.835 held-out transfer); --no-multiresolution to ablate
        text_conditioning="factored",  # PAPER default (F8): factored role+sensor conditioning is the
                                       # committed arm; --text-conditioning per_channel is the ablation.
                                       # (The dataclass default stays per_channel for direct/test ctors.)
    )
    if args.frontend is not None:
        cfg.frontend = args.frontend
    if args.text_conditioning is not None:
        cfg.text_conditioning = args.text_conditioning
    if args.multiresolution is not None:
        cfg.multiresolution = args.multiresolution
    if args.a1_weight is not None:
        cfg.a1_weight = args.a1_weight
    if args.a2_mode is not None:
        cfg.a2_mode = args.a2_mode
    if args.a3_weight is not None:
        cfg.a3_weight = args.a3_weight
    if args.simclr_temperature is not None:
        cfg.simclr_temperature = args.simclr_temperature
    if args.tfc_weight is not None:
        cfg.tfc_weight = args.tfc_weight
    if args.tfc_temperature is not None:
        cfg.tfc_temperature = args.tfc_temperature
    if args.sampler is not None:
        cfg.sampler = args.sampler
    if args.sampler_alpha is not None:
        cfg.sampler_alpha = args.sampler_alpha
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_seed is not None:
        cfg.data_seed = args.data_seed
    if args.datasets is not None:
        cfg.train_datasets = tuple(args.datasets)
    elif args.subset:
        from training.tokenizer.ablation_subset import SUBSET_TRAIN_DATASETS, DEFAULT_CAP
        cfg.train_datasets = SUBSET_TRAIN_DATASETS
        # Apply the SAME per-stream cap the metric harness uses (build_subset_index(cap=DEFAULT_CAP)),
        # so TRAIN and EVAL share one corpus definition (audit 2026-07-23 #3: --subset previously left
        # max_per_stream=None -> trained on ~94k windows while metrics used the 10k cap).
        if args.max_per_stream is None:
            cfg.max_per_stream = DEFAULT_CAP
    if args.max_per_stream is not None:
        cfg.max_per_stream = args.max_per_stream
    if args.steps:
        cfg.steps = args.steps
    if args.smoke:
        cfg = PretrainConfig(
            d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
            classes_per_batch=8, samples_per_class=4, batch_size=32, steps=args.steps or 10,
            warmup_steps=2, calib_batches=3, val_every=max(args.steps or 10, 5),
            val_per_label=10, num_workers=0, max_per_stream=200,
            device=args.device, arm=cfg.arm, frontend=cfg.frontend,
            multiresolution=cfg.multiresolution,
            # carry the fine-grained overrides the smoke reconstruction used to DROP (audit #6):
            a1_weight=cfg.a1_weight, seed=cfg.seed, data_seed=cfg.data_seed,
            train_datasets=cfg.train_datasets,
            text_conditioning=cfg.text_conditioning, gate_bias_init=cfg.gate_bias_init,
            a2_mode=cfg.a2_mode, a3_weight=cfg.a3_weight,
            simclr_temperature=cfg.simclr_temperature, sampler=cfg.sampler,
            sampler_alpha=cfg.sampler_alpha,
            tfc_weight=cfg.tfc_weight, tfc_temperature=cfg.tfc_temperature,
        )
    device = torch.device(cfg.device)
    if device.type == "cuda":
        # TF32 for the fp32 regions autocast doesn't cover (filterbank einsum/proj, val ridge
        # solve). Free + zero-risk on Ampere+; the transformer already runs fp16 under autocast.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # F5: never silently append to / overwrite a prior run. A stale checkpoint or log in the out dir
    # means a fresh run would mix results and append to the old log — refuse unless --force/--smoke.
    # --resume KEEPS the dir (that is where the checkpoint we continue from lives).
    stale = list(args.out.glob("*.pt")) + list(args.out.glob("log.jsonl"))
    if stale and not args.resume:
        if args.force or args.smoke:
            for p in stale:
                p.unlink()
        else:
            raise SystemExit(f"output dir {args.out} already contains {[p.name for p in stale]}; "
                             f"choose a fresh --out or pass --force to overwrite (or --resume).")
    print(f"arm={cfg.arm} frontend={cfg.frontend} multiresolution={cfg.multiresolution}", flush=True)
    # NB: run_config.json is written AFTER resume validation (below), so a rejected --resume can't
    # overwrite the metadata with the bad attempted config (audit 2026-07-23 #6).

    # ------------------------------------------------------------------ data
    # DATA seed (fixed subject split), NOT the model seed, so the split is identical across replicates
    # and reconstructable by the metric harness (#1).
    index = CorpusIndex(max_per_stream=cfg.max_per_stream, seed=cfg.data_seed,
                        datasets=cfg.train_datasets or TRAIN_DATASETS)
    print(f"corpus: {index.summary()}  (datasets={sorted(cfg.train_datasets or TRAIN_DATASETS)})",
          flush=True)
    two_view = cfg.a2_mode == "simclr"            # SimCLR needs a second augmented view per window
    train_compute_targets = cfg.a3_weight > 0     # A3 off (default) -> skip the per-window A3 DSP
    train_ds = PretrainDataset(index, index.train, augment=True, two_view=two_view)
    val_ds = PretrainDataset(index, index.val, augment=False)
    train_collate = (MultiResolutionCollate(
        short_choices=cfg.short_patch_choices, long_choices=cfg.long_patch_choices,
        min_resolution_ratio=cfg.min_resolution_ratio, seed=cfg.seed,
        compute_targets=train_compute_targets, two_view=two_view,
    ) if cfg.multiresolution
                     else MultiScaleCollate(seed=cfg.seed, compute_targets=train_compute_targets,
                                            two_view=two_view))
    loader_kwargs = dict(
        collate_fn=train_collate, num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda")
    if cfg.sampler == "balanced":
        # Label-balanced batches (the supcon path): classes_per_batch x samples_per_class.
        train_loader = DataLoader(
            train_ds,
            batch_sampler=BalancedBatchSampler(index.train, cfg.classes_per_batch,
                                               cfg.samples_per_class, cfg.steps, cfg.seed,
                                               stream_datasets=index.stream_datasets),
            **loader_kwargs)
    else:
        # Temperature sampler (default): per-window draw, P(dataset) ∝ n^alpha, no class balancing.
        train_loader = DataLoader(
            train_ds,
            sampler=TemperatureSampler(index.train, index.stream_datasets,
                                       num_samples=cfg.steps * cfg.batch_size,
                                       alpha=cfg.sampler_alpha, seed=cfg.seed,
                                       batch_size=cfg.batch_size),   # within-batch no-replacement (F11)
            batch_size=cfg.batch_size, drop_last=True, **loader_kwargs)
    # val: no aug, fixed 1.0 s patches, plain order. compute_targets=False skips the per-window
    # A3 DSP (unused by embedding), and parallel persistent workers cut the collate time — together
    # these take a val from ~9.5 min to seconds (the val-speed fix; val ran 5x the train cost).
    val_workers = min(6, cfg.num_workers)
    val_collate = (
        MultiResolutionCollate(fixed_patch_seconds=cfg.val_resolution_pair,
                               min_resolution_ratio=cfg.min_resolution_ratio,
                               compute_targets=False)
        if cfg.multiresolution else
        MultiScaleCollate(fixed_patch_seconds=1.0, compute_targets=False)
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=val_collate,
                            num_workers=val_workers, persistent_workers=val_workers > 0,
                            pin_memory=device.type == "cuda")
    # Fixed generator so the kNN SUPPORT bank is deterministic across evals AND identical between the
    # two arms (audit 2026-07-23 #5: shuffle=True with no generator drew a different support set per
    # arm/eval, so matched training seeds did NOT give matched validation). Seeded from DATA seed (the
    # split), not the model seed, and RESET before each val (#4) so the support is identical at every
    # evaluation and across replicates — not just across arms.
    train_eval_gen = torch.Generator().manual_seed(cfg.data_seed)
    train_eval_loader = DataLoader(
        PretrainDataset(index, index.train, augment=False), batch_size=256,
        shuffle=True, collate_fn=val_collate, generator=train_eval_gen,
        num_workers=val_workers, persistent_workers=val_workers > 0,
        pin_memory=device.type == "cuda", worker_init_fn=_seed_worker,
    )

    def _cycle(loader):
        while True:
            yield from loader

    # ---------------------------------------------------- A1 target tokenizer (only if A1 is on)
    # A1's target is the fixed filterbank's per-channel features. When a1_weight=0, building and
    # calibrating it for calib_batches is avoidable work; the a1_head stays but receives zero weight.
    compute_a1 = cfg.a1_weight > 0
    if compute_a1:
        target_tok = PhysicalFilterbankTokenizer(d_model=1, dft_size=DFT_SIZE)
        target_tok.proj = nn.Identity()
        print(f"calibrating filterbank norm on {cfg.calib_batches} batches ...", flush=True)
        target_tok.reset_norm_accumulator()
        calib_iter = _cycle(train_loader)     # robust if calib_batches > sampler steps (smoke)
        for _ in range(cfg.calib_batches):
            batch = next(calib_iter)          # gravity-aligned + patch-masked in the collate
            target_tok.accumulate_norm_stats(
                batch["patches"], batch["rates"], batch["patch_len"],
                patch_mask=batch["patch_padding_mask"], channel_mask=batch["channel_mask"])
        target_tok.finalize_norm_stats()
        target_tok.eval()
        for p in target_tok.parameters():
            p.requires_grad_(False)
        # A1 predicts only the SIGNAL-content dims (band energies + amplitude + dc); rate-metadata
        # masks dropped (they were ~81% of the target norm — audit 2026-07-18).
        signal_idx = torch.tensor(target_tok.signal_feature_indices(), device=device)
        a1_target_dim = len(signal_idx)
    else:
        target_tok, signal_idx, a1_target_dim = None, None, 1   # dormant a1_head

    # ------------------------------------------------------------------ model
    model = PipelineAModel(cfg, a1_target_dim=a1_target_dim).to(device)
    # Calibrate the ENCODER's frontend with its OWN accumulate/finalize (per-band + signed-DC stats),
    # over the same data as the A1 target.
    fe = model.encoder.filterbank
    fe.reset_norm_accumulator()
    fe_iter = _cycle(train_loader)
    for _ in range(cfg.calib_batches):
        b = next(fe_iter)
        fe.accumulate_norm_stats(
            b["patches"].to(device), b["rates"].to(device), b["patch_len"].to(device),
            patch_mask=b["patch_padding_mask"].to(device), channel_mask=b["channel_mask"].to(device))
    fe.finalize_norm_stats()
    if compute_a1:
        target_tok = target_tok.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {n_params / 1e6:.2f}M trainable params · device={device}", flush=True)

    # Label-text prototypes for the live ConSE-style zero-shot probe (built once, frozen LM).
    label_protos = label_text_prototypes(model, index.label_ids)   # (L, 384) cpu, normalized

    adaptive_names = {
        "encoder.filterbank._center_offsets", "encoder.filterbank._bandwidth_logits",
        "encoder.filterbank._compression_logits", "encoder.filterbank._shape_logit",
        "encoder.filterbank._adaptive_gate_logit",
    }
    adaptive_params, base_params = [], []
    for name, parameter in model.named_parameters():
        if name in adaptive_names:
            adaptive_params.append(parameter)
        else:
            base_params.append(parameter)
    param_groups = [{"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    if adaptive_params:
        # Explicit physical regularization replaces AdamW's logit-space decay. In particular,
        # weight decay would pull the residual gate logit toward zero, i.e. gate=0.5, not fixed=0.
        param_groups.append({"params": adaptive_params, "lr": cfg.lr * cfg.frontend_lr_scale,
                             "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(cfg.warmup_steps, 1), 1.0)
        * 0.5 * (1 + np.cos(np.pi * min(s / cfg.steps, 1.0))),
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    weights = EliteLossWeights(a1_masked=cfg.a1_weight, a3_grounding=cfg.a3_weight)
    log_path = args.out / "log.jsonl"
    best_ba = -1.0
    t0 = time.time()

    def checkpoint(name: str, step: int, val_ba: float):
        import random as _stdrandom
        torch.save({
            "encoder": model.encoder.state_dict(),
            "heads": {k: v.state_dict() for k, v in
                      (("a1", model.a1_head), ("a2", model.a2_proj),
                       ("a3_cadence", model.a3_cadence), ("a3_eigen", model.a3_eigen),
                       # TF-C aux modules: saved only so a warm --resume stays consistent; the
                       # inference loaders read ckpt["encoder"] alone and never touch these.
                       ("time_encoder", model.time_encoder), ("tfc_proj", model.tfc_proj),
                       ("tfc_proj_time", model.tfc_proj_time))},
            "config": asdict(cfg),
            "label_ids": index.label_ids,
            "step": step, "val_ba": val_ba,
            "best_ba": max(best_ba, val_ba),   # running best so a resume can't overwrite a better best.pt (#6)
            "git": git_commit(),
            "corpus": index.summary(),
            "corpus_fingerprint": corpus_fingerprint(index),   # which corpus produced this (F5)
            # Full restart state so a killed run resumes without silently diverging (F5).
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {"torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": _stdrandom.getstate()},
        }, args.out / name)

    # F5b: warm-resume — restore weights + optimizer/scheduler/scaler/RNG + step, then continue the
    # REMAINING steps. Not bit-exact (the sampler re-draws a fresh epoch for the remaining steps), but
    # training continues correctly from the saved state rather than restarting from scratch.
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location=device, weights_only=False)
        saved_cfg = rk.get("config", {})
        # A faithful resume must reproduce the SAME optimization trajectory, so validate the ENTIRE
        # serialized config against the checkpoint — NOT a hand-listed subset. The old subset silently
        # accepted a mixed objective/sampler/seed resume (e.g. --a2-mode supcon --sampler balanced
        # --tfc-weight 0 --seed 999 on a SimCLR/TF-C run), which then overwrote run_config.json and
        # made the mixed-protocol run look pure. Only knobs that touch NEITHER the training trajectory
        # NOR the saved model's meaning may differ — runtime + eval-cadence. (steps stays checked: it
        # rescales the cosine LR schedule, so a faithful resume passes the same --steps.)
        _RESUME_RUNTIME_ONLY = {"device", "num_workers", "val_every", "val_per_label", "knn_k"}

        def _norm(v):
            return list(v) if isinstance(v, (list, tuple)) else v
        cur_cfg = asdict(cfg)
        for key in sorted(set(cur_cfg) | set(saved_cfg)):
            if key in _RESUME_RUNTIME_ONLY:
                continue
            saved = saved_cfg.get(key, cur_cfg.get(key))
            cur = cur_cfg.get(key)
            # train_datasets round-trips through asdict as a list; compare order-insensitively as sets
            if key == "train_datasets" and saved is not None and cur is not None:
                if set(saved) != set(cur):
                    raise ValueError(f"resume mismatch for {key}: checkpoint={saved!r}, requested={cur!r}")
            elif _norm(saved) != _norm(cur):
                raise ValueError(
                    f"resume configuration mismatch for {key}: checkpoint={saved!r}, requested={cur!r} "
                    f"— a resume must reproduce the run; only {sorted(_RESUME_RUNTIME_ONLY)} may differ.")
        saved_fp = rk.get("corpus_fingerprint")
        if saved_fp is not None and saved_fp != corpus_fingerprint(index):
            raise ValueError(
                f"resume corpus fingerprint mismatch: checkpoint={saved_fp}, "
                f"current={corpus_fingerprint(index)} — the corpus/cap/seed changed since the run started.")
        model.encoder.load_state_dict(rk["encoder"])
        _AUX_TFC = {"time_encoder", "tfc_proj", "tfc_proj_time"}
        for k, head in (("a1", model.a1_head), ("a2", model.a2_proj),
                        ("a3_cadence", model.a3_cadence), ("a3_eigen", model.a3_eigen),
                        ("time_encoder", model.time_encoder), ("tfc_proj", model.tfc_proj),
                        ("tfc_proj_time", model.tfc_proj_time)):
            if k not in rk["heads"]:           # aux TF-C modules absent in pre-TF-C checkpoints
                continue
            if k in _AUX_TFC:
                # The aux TF-C rail is training-only and discarded at inference, and its shape has
                # changed (the F2/F3 rate/position FiLM added parameters). A checkpoint predating
                # that must still be resumable, so load it leniently and re-init what is missing
                # rather than crashing the run on a module the encoder never uses.
                try:
                    head.load_state_dict(rk["heads"][k])
                except RuntimeError as exc:
                    print(f"[resume] aux TF-C module {k!r} predates the current architecture "
                          f"({str(exc).splitlines()[0]}); re-initialised.", flush=True)
            else:
                head.load_state_dict(rk["heads"][k])   # main heads stay STRICT
        opt.load_state_dict(rk["optimizer"])
        sched.load_state_dict(rk["scheduler"])
        scaler.load_state_dict(rk["scaler"])
        if "rng" in rk:
            import random as _sr
            torch.set_rng_state(rk["rng"]["torch"])
            np.random.set_state(rk["rng"]["numpy"])
            _sr.setstate(rk["rng"]["python"])
        start_step = int(rk["step"])
        # Restore the RUNNING best, not this checkpoint's own val_ba (#6): resuming from last.pt
        # (whose val_ba is the latest, not the best) must not let a later worse val overwrite best.pt.
        best_ba = float(rk.get("best_ba", rk["val_ba"]))
        # Draw FRESH windows for the remaining steps instead of REPLAYING the sampler prefix (audit F1):
        # advance the temperature sampler's epoch so the resumed run's training draw differs from the
        # interrupted run's. Not bit-exact (the design accepts a fresh epoch for the remaining steps),
        # but it no longer re-trains on the exact windows the prefix already saw.
        _samp = getattr(train_loader, "sampler", None)
        if isinstance(_samp, TemperatureSampler):
            _samp.epoch += start_step
        print(f"resumed from {args.resume} at step {start_step} (best_ba {best_ba:.3f})", flush=True)

    # Write run metadata only AFTER resume validation passed, so a rejected resume leaves it untouched (#6).
    (args.out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    def encode_clean_view_b(batch: dict) -> dict:
        """Encode the SECOND SimCLR view (the collate's ``*_b`` keys) — clean, no A1 mask.
        Only ['pooled'] is consumed (its a2_proj gives z_b). Tokenization runs fp32 (autocast
        off) like view A; the transformer runs under whatever autocast the caller is in. The
        conditioning path MATCHES view A (per_channel vs factored), so the encoder sees the same
        text mode — factored view B carries its OWN independently-augmented role/sensor text."""
        p_b = batch["patches_b"].to(device, non_blocking=True).float()
        r_b = batch["rates_b"].to(device)
        pl_b = batch["patch_len_b"].to(device)
        pos_b = batch["positions_b"].to(device)
        pdur_b = (batch["patch_durations_b"].to(device)
                  if "patch_durations_b" in batch else None)
        rids_b = (batch["resolution_ids_b"].to(device)
                  if "resolution_ids_b" in batch else None)
        cmask_b = batch["channel_mask_b"].to(device)
        ppad_b = batch["patch_padding_mask_b"].to(device)
        with torch.amp.autocast(device.type, enabled=False):
            tokens_b = model.encoder.tokenize(p_b, r_b, pl_b)
        if cfg.text_conditioning == "factored":
            te_b, tm_b, ste_b, stm_b = model.encoder.encode_texts_factored(
                batch["role_texts_b"], batch["sensor_texts_b"], device)
            sid_b = batch["sensor_id_b"].to(device)
        else:
            te_b, tm_b = model.encoder.encode_texts(batch["texts_b"], device)
            ste_b = stm_b = sid_b = None
        return model.encoder.encode(tokens_b, te_b, tm_b, pos_b,
                                    patch_durations=pdur_b, resolution_ids=rids_b,
                                    channel_mask=cmask_b, patch_padding_mask=ppad_b,
                                    sensor_text_embs=ste_b, sensor_text_masks=stm_b,
                                    sensor_id=sid_b)

    model.train()
    for step, batch in enumerate(train_loader, start=start_step + 1):
        patches = batch["patches"].to(device, non_blocking=True)   # NOT gravity-aligned (2026-07-19 design)
        rates = batch["rates"].to(device)
        patch_len = batch["patch_len"].to(device)
        positions = batch["positions"].to(device)
        patch_durations = (batch["patch_durations"].to(device)
                           if "patch_durations" in batch else None)
        resolution_ids = (batch["resolution_ids"].to(device)
                          if "resolution_ids" in batch else None)
        channel_mask = batch["channel_mask"].to(device)
        patch_pad = batch["patch_padding_mask"].to(device)
        labels = batch["labels"].to(device)
        B, P, _, C = patches.shape

        # A1's reconstruction target is the FILTERBANK's per-CHANNEL features; gate the whole A1 path
        # (incl. its second, masked encoder pass) on a1_weight so a1_weight=0 skips it entirely.
        compute_a1 = cfg.a1_weight > 0

        # validity-aware A1 mask: temporal block on real patches, drops on real channels.
        if compute_a1:
            if cfg.multiresolution:
                plan = make_multiresolution_mask_plan(
                    batch["patch_starts"].to(device), batch["patch_ends"].to(device),
                    resolution_ids, C, GYRO_IDX, channel_mask=channel_mask, valid_patches=patch_pad)
            else:
                plan = make_mask_plan(B, P, C, GYRO_IDX, device=device,
                                      valid_patches=patch_pad, channel_mask=channel_mask)
        targets = GroundingTargets(
            cadence_log2hz=batch["cadence_target"].to(device),
            cadence_valid=batch["cadence_valid"].to(device),
            eigen_ratios=batch["eigen_target"].to(device),
            eigen_valid=batch["eigen_valid"].to(device),
        )

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            # The filterbank DSP (rDFT + constant-Q einsum) runs in fp32 — fp16 has too little headroom
            # for the band-energy magnitudes (sweep finding 15). sensor_tokens keeps grad; A1 TARGET is no_grad.
            with torch.amp.autocast(device.type, enabled=False):
                if compute_a1:
                    with torch.no_grad():
                        a1_target = target_tok(patches.float(), rates, patch_len)[..., signal_idx]
                        o, _ = target_tok.masks(rates, patch_len)            # (B, K) Nyquist-observable
                        extra = o.new_ones(B, len(signal_idx) - o.shape[1])
                        a1_feature_valid = torch.cat([o, extra], dim=1).view(B, 1, 1, -1)
                sensor_tokens = model.encoder.tokenize(patches.float(), rates, patch_len)
                enc_channel_mask = channel_mask
                enc_texts = batch["texts"]
            # Config-text conditioning, built ONCE and reused by the clean and masked encode passes.
            # per_channel (default): per-channel descriptions -> (B,C,S,384); UNCHANGED from before.
            # factored: ROLE text -> text_embs/text_masks; the per-sensor IDENTITY carried separately
            # (sensor_text_embs/masks/id), summed inside the fusion (docs/design/TEXT_CONDITIONING.md).
            if cfg.text_conditioning == "factored":
                text_embs, text_masks, sensor_text_embs, sensor_text_masks = \
                    model.encoder.encode_texts_factored(
                        batch["role_texts"], batch["sensor_texts"], device)
                enc_sensor_id = batch["sensor_id"].to(device)
            else:
                text_embs, text_masks = model.encoder.encode_texts(enc_texts, device)
                sensor_text_embs = sensor_text_masks = enc_sensor_id = None
            clean = model.encoder.encode(sensor_tokens, text_embs, text_masks, positions,
                                         patch_durations=patch_durations,
                                         resolution_ids=resolution_ids,
                                         channel_mask=enc_channel_mask,
                                         patch_padding_mask=patch_pad,
                                         sensor_text_embs=sensor_text_embs,
                                         sensor_text_masks=sensor_text_masks,
                                         sensor_id=enc_sensor_id)
            z = model.a2_proj(clean["pooled"])
            # --- A1 masked pass (factored-aware), gated on a1_weight ---
            if compute_a1:
                masked = model.encoder.encode(sensor_tokens, text_embs, text_masks, positions,
                                              patch_durations=patch_durations,
                                              resolution_ids=resolution_ids,
                                              cross_resolution_attention=not cfg.multiresolution,
                                              token_mask=plan.token_mask,
                                              channel_mask=enc_channel_mask,
                                              patch_padding_mask=patch_pad,
                                              sensor_text_embs=sensor_text_embs,
                                              sensor_text_masks=sensor_text_masks,
                                              sensor_id=enc_sensor_id)
                a1_pred = model.a1_head(masked["tokens"])
                a1_loss_mask = plan.token_mask & enc_channel_mask.unsqueeze(1) & patch_pad.unsqueeze(2)
            else:
                # A1 neutralised: no masked pass; a zero-weight term (all-False mask -> 0 loss).
                a1_pred = model.a1_head(clean["tokens"].detach())
                a1_target = torch.zeros_like(a1_pred)
                a1_loss_mask = torch.zeros(clean["tokens"].shape[:3], dtype=torch.bool, device=device)
                a1_feature_valid = None
            # --- A2 (SimCLR) + TF-C, the SSL recipe ---
            tfc_loss = None
            if cfg.a2_mode == "simclr":
                # A2 = self-supervised NT-Xent between the two independently-augmented views.
                # Labels are NOT used in the loss (only later, in the val kNN/ConSE metric).
                z_b = model.a2_proj(encode_clean_view_b(batch)["pooled"])
                a2_loss, a2_key = nt_xent(z, z_b, cfg.simclr_temperature), "a2_simclr"
                # TF-C: pull the FREQUENCY view (REUSE view-A's already-computed pooled output —
                # no second main-encoder forward) toward the TIME view (auxiliary TimeEncoder over
                # the raw samples) of the SAME window; NT-Xent, augmentation-free. Separate heads
                # from a2_proj keep SimCLR and TF-C independent. tfc_weight==0 skips it entirely.
                if cfg.tfc_weight > 0:
                    z_freq = model.tfc_proj(clean["pooled"])
                    time_emb = model.time_encoder(patches.float(), patch_len, channel_mask,
                                                  patch_padding_mask=patch_pad,
                                                  patch_durations=patch_durations, positions=positions)
                    z_time = model.tfc_proj_time(time_emb)
                    tfc_loss = nt_xent(z_time, z_freq, cfg.tfc_temperature)
            else:
                a2_loss, a2_key = None, "a2_supcon"   # elite3_loss computes the label-SupCon
                # TF-C belongs to the SSL recipe; the old-recipe supcon ablation skips it.
            # A3 grounding rail: OFF by default (a3_weight=0). Skip the heads and pass the
            # (validity-False) targets so the term is exactly 0; --a3-weight 0.1 turns both the
            # heads and the collate's A3 DSP back on together.
            if cfg.a3_weight > 0:
                a3_cad = model.a3_cadence(clean["pooled"]).squeeze(1)
                a3_eig = model.a3_eigen(clean["pooled"]).view(B, 4, 3)
            else:
                a3_cad = clean["pooled"].new_zeros(B)
                a3_eig = clean["pooled"].new_zeros(B, 4, 3)
            out = elite3_loss(
                a1_pred=a1_pred, a1_target=a1_target,
                a1_mask=a1_loss_mask,
                a2_embeddings=z, a2_labels=labels,
                a3_cadence_pred=a3_cad,
                a3_eigen_pred=a3_eig,
                a3_targets=targets, weights=weights,
                a1_feature_valid=a1_feature_valid,
                a1_token_groups=resolution_ids,
                a1_token_durations=patch_durations,   # weight A1 by represented duration (F1)
                a2_loss=a2_loss, a2_key=a2_key,
            )
            # Total = a1 + simclr + tfc (A3 is off by default). TF-C scaled by cfg.tfc_weight;
            # the 'tfc' part is always logged (0.0 when skipped) so telemetry stays stable.
            if tfc_loss is not None:
                out.total = out.total + cfg.tfc_weight * tfc_loss
                out.parts["tfc"] = float(tfc_loss.detach())
            else:
                out.parts["tfc"] = 0.0
            frontend_reg = model.encoder.filterbank.adaptation_regularization()
            out.total = out.total + cfg.frontend_reg_weight * frontend_reg
            out.parts["frontend_reg"] = float(frontend_reg.detach())
            out.parts["frontend_reg_weighted"] = float(
                (cfg.frontend_reg_weight * frontend_reg).detach())

        do_log = step % 50 == 0 or step == 1
        opt.zero_grad(set_to_none=True)
        scaler.scale(out.total).backward()
        scaler.unscale_(opt)
        gnorms = module_grad_norms(model) if do_log else {}     # pre-clip per-module grad norms
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if do_log:
            lrs = sched.get_last_lr()
            rec = {"step": step, "lr": lrs[0],
                   "elapsed_s": round(time.time() - t0, 1),
                   "patch_seconds": batch["patch_seconds"],
                   "total": round(float(out.total.detach()), 4), **out.parts, **gnorms}
            if compute_a1:                                       # per-source A1 (diagnostic, off-graph)
                with torch.no_grad():
                    a1_pw = masked_latent_per_window(a1_pred.float(), a1_target, a1_loss_mask,
                                                     feature_valid=a1_feature_valid)
                rec["a1_by_source"] = per_source_mean(a1_pw, batch["sources"])
            if len(lrs) > 1:
                rec["lr_frontend"] = lrs[1]
            if model.encoder.filterbank.learnable:
                rec.update(model.encoder.filterbank.adaptation_summary())
            if model.encoder.use_duration_embedding:
                rec["duration/gate"] = float(torch.sigmoid(
                    model.encoder.duration_gate_logit.detach()))
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

        if step % cfg.val_every == 0 or step == cfg.steps:
            # Query = every val label (stratified, so all classes are scored); support = the same
            # labels drawn from the train set (early-stops once saturated). F1: covers all classes.
            val_z, val_y, val_src = embed_stratified(model, val_loader, device, cfg.val_per_label)
            train_eval_gen.manual_seed(cfg.data_seed)   # same support bank at every val + across arms (#4)
            train_z, train_y, _ = embed_stratified(model, train_eval_loader, device,
                                                   cfg.val_per_label, target_labels=set(val_y.tolist()))
            ba = knn_balanced_acc(train_z, train_y, val_z, val_y, cfg.knn_k)
            # ConSE-style text-cosine probe: ridge-map sensor->label-text space on the train
            # support, cosine-classify val against the label prototypes. A live proxy for the
            # downstream zero-shot metric (comparable to the ConSE baselines), fit fresh each val.
            conse_pred = conse_probe_predict(train_z, train_y, val_z, val_y, label_protos)
            conse_ba = balanced_acc(conse_pred, val_y)
            # per-source val BA — which datasets cluster (kNN) / align to text (conse) well
            vs = np.asarray(val_src)
            ba_by_src, conse_by_src = {}, {}
            for s in sorted(set(val_src)):
                mt = torch.from_numpy(vs == s)
                if int(mt.sum()) >= cfg.knn_k:
                    ba_by_src[s] = round(knn_balanced_acc(train_z, train_y, val_z[mt], val_y[mt],
                                                          cfg.knn_k), 4)
                    conse_by_src[s] = round(balanced_acc(conse_pred[mt], val_y[mt]), 4)
            rec = {"step": step, "val_knn_ba": ba, "val_conse_ba": round(conse_ba, 4),
                   "val_ba_by_source": ba_by_src, "val_conse_by_source": conse_by_src}
            if device.type == "cuda":
                # peak so far (train step + val embedding) — memory telemetry.
                rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            checkpoint("last.pt", step, ba)
            if ba > best_ba:
                best_ba = ba
                checkpoint("best.pt", step, ba)
        if step >= cfg.steps:
            break

    print(f"done: best val kNN-BA {best_ba:.3f} · checkpoints in {args.out}", flush=True)


if __name__ == "__main__":
    main()
