"""Pipeline A pretraining with two label-free objectives.

  * JEPA: a masked student predicts a clean EMA teacher's contextual tokens.
  * Relation: VICReg between two augmentations of every window, plus separately reduced
    scale-free agreement for verified simultaneous cross-placement recordings.

Labels are used only by the validation probes. The corpus sampler is hierarchical and label-free:
capped dataset-temperature mass, subject-temperature mass, then windows.

Other invariants:
  * Config conditioning is channel TEXT; the text-dropout/paraphrase augs supply the
    "unseen description" robustness.                        (M2 lesson 2, upgraded by M3)
  * Gravity alignment is disabled by default; signed DC preserves posture while SO(3)
    augmentation supplies orientation robustness.           (2026-07-19 decision)
  * The encoder's inner filterbank norm is CALIBRATED before training.  (M3 lesson)

Model selection: subject-disjoint val kNN balanced accuracy (macro), not loss.
Checkpoints carry config + label map + filterbank norm stats + provenance.

Run (CPU smoke):   .../python -m training.tokenizer.pretrain --steps 20 --smoke
Run (real, GPU):   .../python -m training.tokenizer.pretrain --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.losses_repr import (
    make_mask_plan,
    make_multiresolution_mask_plan,
    masked_ema_latent_loss,
    pair_contrast,
    phase_a_loss,
    relation_loss,
)
from training.tokenizer.pretrain_data import (
    DFT_SIZE,
    LONG_PATCH_SECONDS_CHOICES,
    MIN_RESOLUTION_RATIO,
    SHORT_PATCH_SECONDS_CHOICES,
    VAL_RESOLUTION_PAIR,
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
    # d256/6L sets the frozen encoder and evidence-memory vector width.
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    frontend: str = "fixed"               # fixed | constrained-learnable filterbank
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
    steps: int = 30_000                   # ~5 corpus passes at the live batch of 256 (6,025
                                          # steps/pass on the 1.54M-window corpus). The previous
                                          # '~10 passes at batch 512' note was written when
                                          # batch_size was 512 and was not updated by fd3ae4d.
    # LR / batch: fd3ae4d (2026-07-26) halved batch_size 512 -> 256 but left lr at the value that
    # had been sqrt-scaled UP *for* 512, so the live config runs 1.41x the LR its own rule
    # prescribes. Under that rule batch 256 wants 3e-4. Which value actually transfers better is
    # unmeasured, and LR is the knob the MTO reality-check literature (Xin et al., NeurIPS 2022)
    # found dominates loss-weight effects -- so LR is swept BEFORE any objective-weight sweep
    # rather than silently pinned here.
    lr: float = 4.2e-4                    # = 3e-4 x sqrt(2). See the LR / batch note above.
    weight_decay: float = 0.05
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    # Two fixed-weight objectives. ``jepa_weight=0`` gives the relation-only R0/R1 controls;
    # ``cross_placement_weight=0`` gives R0 without changing which ordinary windows are sampled.
    jepa_weight: float = 1.0
    relation_weight: float = 1.0
    jepa_ema_decay: float = 0.996
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_target_std: float = 1.0
    # Pair quota is a fraction of BATCH WINDOWS; two windows form one verified relation.
    cross_placement_weight: float = 0.1
    relation_pair_fraction: float = 0.2
    # Label-free hierarchical corpus sampler. Dataset mass is tempered and capped, then distributed
    # within each dataset as P(subject) ∝ n_subject^subject_alpha. This keeps Capture-24's useful
    # scale without letting one acc-only wrist corpus or its longest subjects define the encoder.
    sampler_alpha: float = 0.25
    sampler_max_dataset_share: float = 0.25
    sampler_subject_alpha: float = 0.5
    batch_size: int = 256                 # measured peak: 12.0 GiB on a 24 GB RTX 4090
    calib_batches: int = 50               # frontend norm calibration pass
    val_every: int = 1_000
    val_per_label: int = 40               # kNN val: windows PER LABEL (stratified, all classes scored)
    knn_k: int = 5
    # torch.compile the encoder's transformer. OFF by default: an isolated fixed-shape
    # microbenchmark showed 1.43x (13.57 -> 9.46 ms fwd+bwd at batch 16), but that did NOT
    # survive the real loop. Measured end-to-end, 200 steps at batch 256 on a 4090:
    #     eager     40.6 s   12.05 GiB
    #     compiled  85.3 s   10.94 GiB     <- 2.1x SLOWER
    # The loop draws patch_seconds PER BATCH, so P and patch_len change constantly and
    # dynamic=True pays guard evaluation and repeated specialisation on every new shape —
    # which a fixed-shape benchmark cannot see. It does save ~9% VRAM, so --compile is worth
    # trying if you are memory-bound rather than time-bound.
    compile_encoder: bool = False
    num_workers: int = 8                  # re-profiled 2026-07-26: the step is compute-bound well
                                          # before 12 — data is 2% of the step at nw=12 and 15% at
                                          # nw=6, so 8 is the free point and frees CPU + host RAM.
    seed: int = 20260718                  # MODEL seed: weight init, augmentation, batch order (varies per replicate)
    # DATA seed: the subject train/val split. Held FIXED across arms AND replicates so every run — and
    # the metric harness — sees the SAME subject-disjoint split (audit 2026-07-23 #1: the eval used the
    # default split regardless of --seed, so seed!=default leaked ~19 train subjects into metric-val).
    data_seed: int = 20260718
    train_datasets: tuple | None = None   # None = full TRAIN_DATASETS; set for the ablation subset
    max_per_stream: int | None = None     # None = use all windows; temperature sampling controls sources
    device: str = "cpu"


class PipelineAModel(nn.Module):
    def __init__(self, cfg: PretrainConfig):
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
        self.jepa_predictor = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.relation_projector = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 128),
        )


@torch.no_grad()
def update_ema_encoder(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """Update teacher parameters by EMA and copy non-parameter state exactly."""
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    student_params = dict(student.named_parameters())
    for name, target in teacher.named_parameters():
        target.mul_(float(decay)).add_(student_params[name], alpha=1.0 - float(decay))
    student_buffers = dict(student.named_buffers())
    for name, target in teacher.named_buffers():
        target.copy_(student_buffers[name])


def verified_event_pairs(event_ids: list, verified: torch.Tensor, streams: list,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """One deterministic cross-stream pair per verified simultaneous event in a batch."""
    groups: dict[str, list[int]] = {}
    for i, (event_id, ok) in enumerate(zip(event_ids, verified.tolist())):
        if ok and event_id is not None:
            groups.setdefault(str(event_id), []).append(i)
    left, right = [], []
    for event_id in sorted(groups):
        members = groups[event_id]
        pair = next(
            ((a, b) for pos, a in enumerate(members) for b in members[pos + 1:]
             if streams[a] != streams[b]),
            None,
        )
        if pair is not None:
            left.append(pair[0])
            right.append(pair[1])
    return (
        torch.tensor(left, device=device, dtype=torch.long),
        torch.tensor(right, device=device, dtype=torch.long),
    )


@torch.no_grad()
def representation_health(z: torch.Tensor) -> dict[str, float]:
    """Small-batch collapse diagnostics over a projector output."""
    x = z.detach().float()
    std = x.std(dim=0, unbiased=False)
    centered = x - x.mean(0)
    cov = centered.T @ centered / max(len(x) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    probs = eig / eig.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())
    offdiag = cov - torch.diag_embed(torch.diagonal(cov))
    return {
        "repr/min_std": float(std.min()),
        "repr/mean_std": float(std.mean()),
        "repr/effective_rank": float(effective_rank),
        "repr/mean_norm": float(x.norm(dim=1).mean()),
        "repr/cov_offdiag_abs_mean": float(offdiag.abs().mean()),
        "repr/cov_max_eigenvalue": float(eig.max()),
    }


def objective_encoder_grad_norm(loss: torch.Tensor, encoder: nn.Module) -> float:
    """Gradient norm contributed by one weighted objective to the shared encoder."""
    if not loss.requires_grad:
        return 0.0
    parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    total = sum(
        gradient.detach().float().square().sum()
        for gradient in gradients if gradient is not None
    )
    return float(total.sqrt()) if not isinstance(total, int) else 0.0


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

    import numpy as _np
    sig = []
    for r in sorted(index.refs, key=lambda r: (r.dataset, r.stream)):
        # CONTENT-sensitive (audit F5): shape+rate+label-COUNT alone collided for two corpora that
        # differed in their per-window label/subject assignment. Hash the actual per-window labels
        # and subjects (small, exact) plus a deterministic strided digest of the signal itself, so a
        # regenerated or re-ordered grid cannot silently reuse an earlier run's identity.
        lab = ",".join(map(str, r.labels)).encode()
        sub = ",".join(map(str, r.subjects)).encode()
        arr = _np.ascontiguousarray(r.load_data()[::97, ::13, :], dtype=_np.float32)
        sig.append(f"{r.dataset}/{r.stream}:{r.rate_hz}:{tuple(r.shape)}"
                   f":{hashlib.sha256(lab).hexdigest()[:12]}"
                   f":{hashlib.sha256(sub).hexdigest()[:12]}"
                   f":{hashlib.sha256(arr.tobytes()).hexdigest()[:12]}")
    sig.append("labels=" + ",".join(sorted(index.label_ids)))
    sig.append("implausible=" + str(getattr(index, "n_implausible_dropped", 0)))
    sig.append(f"cap={getattr(index, 'max_per_stream', None)}:seed={getattr(index, 'seed', None)}"
               f":ntrain={len(index.train)}:nval={len(index.val)}")
    return hashlib.sha256("|".join(sig).encode()).hexdigest()[:16]


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
                     target_labels: set | None = None, label_totals: dict | None = None):
    """Embed EXACTLY ``min(per_label, available)`` windows per label (deterministic — the loader
    must be shuffle=False / pre-shuffled). Guarantees every label is represented so the kNN metric
    covers all classes and best.pt selection is stable.

    Two things the earlier version got wrong, both measured on the live split:

    * The cap was evaluated for a whole batch BEFORE ``counts`` was updated, so a label sitting at
      39/40 admitted every one of its occurrences in that batch. Realised counts ran 3-79 per label
      on val and 27-89 on the train support, i.e. the kNN support bank was never balanced and the
      ConSE ridge fit was weighted toward over-represented labels. ``pending`` now closes the gap
      inside the batch, so the cap is exact.
    * The early exit required every target label to REACH ``per_label``. Labels with fewer than
      ``per_label`` windows in total can never reach it, so the loop always drained the whole
      loader — 186k val + 1.5M train rows per evaluation. ``label_totals`` (the label histogram of
      the keys backing this loader) makes the target ``min(per_label, total)``, which is reachable.
    """
    from collections import Counter
    model.eval()
    zs, ys, srcs = [], [], []
    counts: Counter = Counter()
    done: set = set()

    def _target_for(label: int) -> int:
        if label_totals is None:
            return per_label
        return min(per_label, int(label_totals.get(label, per_label)))

    wanted = set(target_labels) if target_labels is not None else None
    for batch in loader:
        lab = batch["labels"]
        take = []
        pending: Counter = Counter()
        for j, l in enumerate(lab.tolist()):
            if wanted is not None and l not in wanted:
                continue
            if counts[l] + pending[l] >= _target_for(l):
                continue
            pending[l] += 1
            take.append(j)
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
                if counts[l] >= _target_for(l):
                    done.add(l)
        # Exit as soon as every label we are still waiting on has hit its ACHIEVABLE target.
        if wanted is not None and wanted <= done:
            break
        if wanted is None and label_totals is not None and set(label_totals) <= done:
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

    mods = (("encoder", model.encoder), ("jepa_predictor", model.jepa_predictor),
            ("relation_projector", model.relation_projector))
    return {f"grad/{name}": _gn(mod.parameters()) for name, mod in mods}


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
    parser.add_argument("--frontend", choices=("fixed", "learnable"), default="fixed",
                        help="tokenizer arm. 'fixed' = physical-Hz constant-Q filterbank (default); "
                             "'learnable' = the constrained-adaptive filterbank arm "
                             "(docs/design/LEARNABLE_TOKENIZER_ARM.md).")
    parser.add_argument("--text-conditioning", choices=("per_channel", "factored"), default=None,
                        help="config-text conditioning (docs/design/TEXT_CONDITIONING.md §4b). "
                             "'per_channel' = one description per channel; 'factored' (the CLI "
                             "DEFAULT) = per-channel ROLE text + per-sensor IDENTITY text.")
    parser.add_argument("--multiresolution", action=argparse.BooleanOptionalAction, default=None,
                        help="override multiresolution (default ON); --no-multiresolution is the "
                             "single-resolution ablation")
    parser.add_argument("--jepa-weight", type=float, default=None,
                        help="masked contextual prediction weight; 0 selects relation-only R0/R1")
    parser.add_argument("--relation-weight", type=float, default=None,
                        help="unified relation objective weight (must be positive)")
    parser.add_argument("--jepa-ema-decay", type=float, default=None,
                        help="EMA teacher momentum (default 0.996).")
    parser.add_argument("--cross-placement-weight", type=float, default=None,
                        help="cross-placement subterm weight; 0 selects augmentation-only R0")
    parser.add_argument("--relation-pair-fraction", type=float, default=None,
                        help="batch-window fraction reserved for verified cross-placement pairs")
    parser.add_argument("--sampler-alpha", type=float, default=None,
                        help="temperature-sampler exponent: 1=proportional, 0=uniform-per-source, "
                             "0.25 is the default.")
    parser.add_argument("--sampler-max-dataset-share", type=float, default=None,
                        help="maximum ordinary-draw probability for one dataset (default 0.25; "
                             "raised automatically only when too few datasets make it infeasible).")
    parser.add_argument("--sampler-subject-alpha", type=float, default=None,
                        help="within-dataset subject-size exponent: 1=proportional, 0=uniform over "
                             "subjects, 0.5=square-root tempering (default).")
    parser.add_argument("--batch", type=int, default=None,
                        help="batch size for the temperature sampler (default 256)")
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
    parser.add_argument("--compile", dest="compile_encoder", action="store_true", default=None,
                        help="torch.compile the encoder. Default OFF: measured 2.1x SLOWER "
                             "end-to-end (per-batch shape churn) though ~9%% lighter on VRAM.")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="DATA seed = the subject train/val split. Keep FIXED across all arms and "
                             "replicates so the split (and the metric harness) stays identical (#1).")
    args = parser.parse_args()

    cfg = PretrainConfig(
        device=args.device,
        frontend=args.frontend,
        multiresolution=True,          # new Phase-A default: multiresolution ON (diagnostic-confirmed
                                       # winner, 0.835 held-out transfer); --no-multiresolution to ablate
        text_conditioning="factored",  # PAPER default (F8): factored role+sensor conditioning is the
                                       # committed arm; --text-conditioning per_channel is the ablation.
                                       # (The dataclass default stays per_channel for direct/test ctors.)
    )
    if args.text_conditioning is not None:
        cfg.text_conditioning = args.text_conditioning
    if args.multiresolution is not None:
        cfg.multiresolution = args.multiresolution
    if args.jepa_weight is not None:
        cfg.jepa_weight = args.jepa_weight
    if args.relation_weight is not None:
        cfg.relation_weight = args.relation_weight
    if args.jepa_ema_decay is not None:
        cfg.jepa_ema_decay = args.jepa_ema_decay
    if args.cross_placement_weight is not None:
        cfg.cross_placement_weight = args.cross_placement_weight
    if args.relation_pair_fraction is not None:
        cfg.relation_pair_fraction = args.relation_pair_fraction
    if args.sampler_alpha is not None:
        cfg.sampler_alpha = args.sampler_alpha
    if args.sampler_max_dataset_share is not None:
        cfg.sampler_max_dataset_share = args.sampler_max_dataset_share
    if args.sampler_subject_alpha is not None:
        cfg.sampler_subject_alpha = args.sampler_subject_alpha
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.seed is not None:
        cfg.seed = args.seed
    if args.compile_encoder is not None:
        cfg.compile_encoder = args.compile_encoder
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
    if args.steps is not None:
        cfg.steps = args.steps
    if args.smoke:
        # Preserve every experimental override and change only resource/budget knobs. Rebuilding
        # PretrainConfig field-by-field repeatedly dropped newly added objective settings, so a
        # passing smoke could exercise a different arm than the requested run.
        cfg = replace(
            cfg,
            d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
            batch_size=32,
            steps=args.steps if args.steps is not None else 10,
            warmup_steps=2, calib_batches=3,
            val_every=max(args.steps if args.steps is not None else 10, 5), val_per_label=10,
            num_workers=0, max_per_stream=200,
        )
    if cfg.sampler_alpha < 0:
        parser.error("--sampler-alpha must be nonnegative")
    if not 0 < cfg.sampler_max_dataset_share <= 1:
        parser.error("--sampler-max-dataset-share must be in (0,1]")
    if not 0 <= cfg.sampler_subject_alpha <= 1:
        parser.error("--sampler-subject-alpha must be in [0,1]")
    for field in ("jepa_weight", "cross_placement_weight", "frontend_reg_weight"):
        if float(getattr(cfg, field)) < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    if cfg.relation_weight <= 0:
        parser.error("--relation-weight must be positive")
    if cfg.steps <= 0 or cfg.batch_size <= 0:
        parser.error("--steps and --batch must be positive")
    if cfg.max_per_stream is not None and cfg.max_per_stream <= 0:
        parser.error("--max-per-stream must be positive when provided")
    if not 0 <= cfg.jepa_ema_decay < 1:
        parser.error("--jepa-ema-decay must be in [0,1)")
    if cfg.lr <= 0 or cfg.grad_clip <= 0:
        parser.error("learning rate and gradient clipping threshold must be positive")
    if not 0 <= cfg.relation_pair_fraction < 1:
        parser.error("--relation-pair-fraction must be in [0,1)")
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
    print(f"frontend={cfg.frontend} multiresolution={cfg.multiresolution}", flush=True)
    # NB: run_config.json is written AFTER resume validation (below), so a rejected --resume can't
    # overwrite the metadata with the bad attempted config (audit 2026-07-23 #6).

    # ------------------------------------------------------------------ data
    # DATA seed (fixed subject split), NOT the model seed, so the split is identical across replicates
    # and reconstructable by the metric harness (#1).
    index = CorpusIndex(max_per_stream=cfg.max_per_stream, seed=cfg.data_seed,
                        datasets=cfg.train_datasets or TRAIN_DATASETS)
    print(f"corpus: {index.summary()}  (datasets={sorted(cfg.train_datasets or TRAIN_DATASETS)})",
          flush=True)
    positive_events = {
        event_id for event_id in index.train_positive_event_ids if event_id >= 0
    }
    if cfg.cross_placement_weight > 0 and not positive_events:
        message = (
            "cross-placement relation learning is enabled but the corpus has no verified simultaneous "
            "events. Rebuild native grids with `python -m data.scripts.build_grids --alignment "
            "native` so explicit event_ids are persisted; refusing to silently train without the "
            "configured relation type. Pass --cross-placement-weight 0 for the R0 control."
        )
        if args.smoke:
            print(f"[smoke warning] {message}", flush=True)
        else:
            raise RuntimeError(message)
    train_ds = PretrainDataset(index, index.train, augment=True, two_view=True)
    def _stratified_subset(keys, per_label: int, seed: int, allowed_labels=None):
        """Deterministically pick min(per_label, available) keys per label.

        Preselecting is what makes evaluation cheap. Capping inside the embed loop cannot: the val
        keys are shuffled, so the last window of the rarest label sits near the end and the loop
        still drags all 186k val / 1.5M train rows through collate + encoder on every evaluation
        (measured: 99.8% of rows still scanned even with an exact achievable-target exit). Choosing
        the rows up front turns both loaders into ~3k-row passes.

        Sampled with a fixed RNG rather than taken in order: index.train is stream-ordered, so the
        first N windows of a label would all come from one dataset/subject. The fixed seed keeps the
        support bank identical at every val and across arms/replicates (audit #4/#5).
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for k in keys:
            if allowed_labels is None or k.label_id in allowed_labels:
                groups[k.label_id].append(k)
        rng = np.random.default_rng(seed)
        out = []
        for label in sorted(groups):
            g = groups[label]
            if len(g) > per_label:
                g = [g[int(i)] for i in rng.choice(len(g), size=per_label, replace=False)]
            out.extend(g)
        return out

    val_keys = _stratified_subset(index.val, cfg.val_per_label, cfg.data_seed)
    val_ds = PretrainDataset(index, val_keys, augment=False)
    train_collate = (MultiResolutionCollate(
        short_choices=cfg.short_patch_choices, long_choices=cfg.long_patch_choices,
        min_resolution_ratio=cfg.min_resolution_ratio, seed=cfg.seed,
        two_view=True,
    ) if cfg.multiresolution
                     else MultiScaleCollate(seed=cfg.seed, two_view=True))
    loader_kwargs = dict(
        collate_fn=train_collate, num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda")
    temperature_sampler = TemperatureSampler(
        index.train, index.stream_datasets,
        num_samples=cfg.steps * cfg.batch_size,
        alpha=cfg.sampler_alpha, seed=cfg.seed,
        batch_size=cfg.batch_size,
        event_ids=index.train_event_ids,
        positive_event_ids=index.train_positive_event_ids,
        pair_fraction=(cfg.relation_pair_fraction
                       if cfg.cross_placement_weight > 0 else 0.0),
        subject_ids=index.train_subject_ids,
        subject_alpha=cfg.sampler_subject_alpha,
        max_dataset_share=cfg.sampler_max_dataset_share,
    )
    shares = ", ".join(
        f"{dataset}={share:.1%}"
        for dataset, share in sorted(
            temperature_sampler.dataset_probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    print(
        f"sampler: dataset alpha={cfg.sampler_alpha:g}, "
        f"max_share={cfg.sampler_max_dataset_share:.1%}, "
        f"subject alpha={cfg.sampler_subject_alpha:g} ({shares})",
        flush=True,
    )
    train_loader = DataLoader(
        train_ds, sampler=temperature_sampler,
        batch_size=cfg.batch_size, drop_last=True, **loader_kwargs)
    # Validation uses deterministic fixed patch durations and no augmentation.
    val_workers = min(6, cfg.num_workers)
    val_collate = (
        MultiResolutionCollate(fixed_patch_seconds=cfg.val_resolution_pair,
                               min_resolution_ratio=cfg.min_resolution_ratio)
        if cfg.multiresolution else
        MultiScaleCollate(fixed_patch_seconds=1.0)
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=val_collate,
                            num_workers=val_workers, persistent_workers=val_workers > 0,
                            pin_memory=device.type == "cuda")
    # Fixed generator so the kNN SUPPORT bank is deterministic across evals AND identical between the
    # two arms (audit 2026-07-23 #5: shuffle=True with no generator drew a different support set per
    # arm/eval, so matched training seeds did NOT give matched validation). Seeded from DATA seed (the
    # split), not the model seed, and RESET before each val (#4) so the support is identical at every
    # evaluation and across replicates — not just across arms.
    # The support bank only ever needs the labels the val queries actually carry, and val is now a
    # fixed set, so both subsets can be chosen once here instead of re-scanned every evaluation.
    from collections import Counter as _Counter
    val_label_totals = dict(_Counter(k.label_id for k in val_keys))
    _val_label_set = set(val_label_totals)
    train_keys = _stratified_subset(index.train, cfg.val_per_label, cfg.data_seed,
                                    allowed_labels=_val_label_set)
    train_label_totals = dict(_Counter(k.label_id for k in train_keys))
    print(f"eval subsets: val {len(index.val):,} -> {len(val_keys):,} rows · "
          f"support {len(index.train):,} -> {len(train_keys):,} rows "
          f"({len(_val_label_set)} labels)", flush=True)

    train_eval_gen = torch.Generator().manual_seed(cfg.data_seed)
    train_eval_loader = DataLoader(
        PretrainDataset(index, train_keys, augment=False), batch_size=256,
        shuffle=False, collate_fn=val_collate, generator=train_eval_gen,
        num_workers=val_workers, persistent_workers=val_workers > 0,
        pin_memory=device.type == "cuda", worker_init_fn=_seed_worker,
    )

    def _cycle(loader):
        while True:
            yield from loader

    # ------------------------------------------------------------------ model
    model = PipelineAModel(cfg).to(device)
    # Calibrate the encoder frontend's per-band and signed-DC normalization.
    fe = model.encoder.filterbank
    print(f"calibrating filterbank norm on {cfg.calib_batches} batches ...", flush=True)
    fe.reset_norm_accumulator()
    fe_iter = _cycle(train_loader)
    for _ in range(cfg.calib_batches):
        b = next(fe_iter)
        fe.accumulate_norm_stats(
            b["patches"].to(device), b["rates"].to(device), b["patch_len"].to(device),
            patch_mask=b["patch_padding_mask"].to(device),
            channel_mask=b["channel_mask"].to(device),
            source_rate_hz=b.get("source_rates", b["rates"]).to(device))
    fe.finalize_norm_stats()

    jepa_teacher = None
    if cfg.jepa_weight > 0:
        jepa_teacher = copy.deepcopy(model.encoder).to(device).eval()
        for parameter in jepa_teacher.parameters():
            parameter.requires_grad_(False)

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
    log_path = args.out / "log.jsonl"
    best_ba = -1.0
    t0 = time.time()

    def checkpoint(name: str, step: int, val_ba: float):
        import random as _stdrandom
        torch.save({
            "encoder": model.encoder.state_dict(),
            "heads": {k: v.state_dict() for k, v in
                      (("jepa_predictor", model.jepa_predictor),
                       ("relation_projector", model.relation_projector))},
            "config": asdict(cfg),
            "jepa_teacher": (jepa_teacher.state_dict() if jepa_teacher is not None else None),
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
        # serialized config against the checkpoint — NOT a hand-listed subset. Only knobs that touch
        # neither the training trajectory
        # NOR the saved model's meaning may differ — runtime + eval-cadence. (steps stays checked: it
        # rescales the cosine LR schedule, so a faithful resume passes the same --steps.)
        _RESUME_RUNTIME_ONLY = {"device", "num_workers", "val_every", "val_per_label", "knn_k"}

        def _norm(v):
            return list(v) if isinstance(v, (list, tuple)) else v
        cur_cfg = asdict(cfg)
        missing_trajectory = sorted(
            set(cur_cfg) - set(saved_cfg) - _RESUME_RUNTIME_ONLY
        )
        if missing_trajectory:
            raise ValueError(
                "resume checkpoint predates trajectory-defining configuration fields "
                f"{missing_trajectory}; start a new attributable run instead of silently applying "
                "today's defaults to an older optimization trajectory"
            )
        for key in sorted(set(cur_cfg) | set(saved_cfg)):
            if key in _RESUME_RUNTIME_ONLY:
                continue
            saved = saved_cfg.get(key)
            cur = cur_cfg.get(key)
            # train_datasets round-trips through asdict as a list; compare order-insensitively as sets
            if key == "train_datasets" and saved is not None and cur is not None:
                if set(saved) != set(cur):
                    raise ValueError(f"resume mismatch for {key}: checkpoint={saved!r}, requested={cur!r}")
            elif _norm(saved) != _norm(cur):
                raise ValueError(
                    f"resume configuration mismatch for {key}: checkpoint={saved!r}, requested={cur!r} "
                    f"— a resume must reproduce the run; only {sorted(_RESUME_RUNTIME_ONLY)} may differ.")
        saved_step = int(rk["step"])
        saved_fp = rk.get("corpus_fingerprint")
        if saved_fp is not None and saved_fp != corpus_fingerprint(index):
            raise ValueError(
                f"resume corpus fingerprint mismatch: checkpoint={saved_fp}, "
                f"current={corpus_fingerprint(index)} — the corpus/cap/seed changed since the run started.")
        model.encoder.load_state_dict(rk["encoder"])
        for key, head in (("jepa_predictor", model.jepa_predictor),
                          ("relation_projector", model.relation_projector)):
            if key not in rk.get("heads", {}):
                raise ValueError(
                    f"resume checkpoint predates the consolidated Phase-A objective: missing {key!r}"
                )
            head.load_state_dict(rk["heads"][key])
        if jepa_teacher is not None:
            if rk.get("jepa_teacher") is not None:
                jepa_teacher.load_state_dict(rk["jepa_teacher"])
            else:
                raise ValueError("resume checkpoint is missing the JEPA teacher state")
        opt.load_state_dict(rk["optimizer"])
        sched.load_state_dict(rk["scheduler"])
        scaler.load_state_dict(rk["scaler"])
        if "rng" in rk:
            import random as _sr
            torch.set_rng_state(rk["rng"]["torch"])
            np.random.set_state(rk["rng"]["numpy"])
            _sr.setstate(rk["rng"]["python"])
        start_step = saved_step
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
        """Encode the SECOND augmentation-positive view (the collate's ``*_b`` keys), clean/no mask.
        Only ['pooled'] is consumed by the relation projector. Tokenization runs fp32 (autocast
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
            tokens_b = model.encoder.tokenize(
                p_b, r_b, pl_b,
                source_rate_hz=batch.get("source_rates_b", r_b).to(device))
        if cfg.text_conditioning == "factored":
            te_b, tm_b, ste_b, stm_b = model.encoder.encode_texts_factored(
                batch["role_texts_b"], batch["sensor_texts_b"], device)
            sid_b = batch["sensor_id_b"].to(device)
        else:
            te_b, tm_b = model.encoder.encode_texts(batch["texts_b"], device)
            ste_b = stm_b = sid_b = None
        return encode_fn(tokens_b, te_b, tm_b, pos_b,
                                    patch_durations=pdur_b, resolution_ids=rids_b,
                                    channel_mask=cmask_b, patch_padding_mask=ppad_b,
                                    sensor_text_embs=ste_b, sensor_text_masks=stm_b,
                                    sensor_id=sid_b)

    # One compiled callable reused by the clean / masked / view-B passes. The EMA teacher stays
    # eager: it is no-grad, cheaper, and its weights change every step, so compiling it buys little.
    encode_fn = model.encoder.encode
    if cfg.compile_encoder and device.type == "cuda":
        # Objective gradient telemetry re-walks the graph with retain_graph=True; disable Inductor's
        # donated-buffer optimization so the diagnostic backward remains valid.
        torch._functorch.config.donated_buffer = False
        encode_fn = torch.compile(model.encoder.encode, dynamic=True)
        print("torch.compile: encoder.encode (dynamic, donated_buffer off) — step 1 pays the compile",
              flush=True)

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
        B, P, _, C = patches.shape

        if cfg.jepa_weight > 0:
            if cfg.multiresolution:
                plan = make_multiresolution_mask_plan(
                    batch["patch_starts"].to(device), batch["patch_ends"].to(device),
                    resolution_ids, C, GYRO_IDX, channel_mask=channel_mask,
                    valid_patches=patch_pad)
            else:
                plan = make_mask_plan(B, P, C, GYRO_IDX, device=device,
                                      valid_patches=patch_pad, channel_mask=channel_mask)

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            # The filterbank DSP (rDFT + constant-Q einsum) runs in fp32 — fp16 has too little headroom
            # for the band-energy magnitudes. Sensor tokens retain gradients.
            with torch.amp.autocast(device.type, enabled=False):
                # Split analysis from projection so the EMA teacher can reuse the DSP. On the
                # fixed arm `analyze` is parameter-free, so the teacher's analysis would be
                # bit-identical anyway and running the rDFT + constant-Q einsum twice per step
                # was pure waste. The learnable arm's analysis reads EMA-diverging parameters,
                # so it keeps its own pass (shared_analysis stays None).
                _src_rate = batch.get("source_rates", rates).to(device)
                if cfg.frontend == "fixed":
                    shared_analysis = model.encoder.analyze(
                        patches.float(), rates, patch_len, source_rate_hz=_src_rate)
                    sensor_tokens = model.encoder.project_tokens(shared_analysis)
                else:
                    shared_analysis = None
                    sensor_tokens = model.encoder.tokenize(
                        patches.float(), rates, patch_len, source_rate_hz=_src_rate)
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
            clean = encode_fn(sensor_tokens, text_embs, text_masks, positions,
                                         patch_durations=patch_durations,
                                         resolution_ids=resolution_ids,
                                         channel_mask=enc_channel_mask,
                                         patch_padding_mask=patch_pad,
                                         sensor_text_embs=sensor_text_embs,
                                         sensor_text_masks=sensor_text_masks,
                                         sensor_id=enc_sensor_id)
            z = model.relation_projector(clean["pooled"])
            if cfg.jepa_weight > 0:
                masked = encode_fn(sensor_tokens, text_embs, text_masks, positions,
                                              patch_durations=patch_durations,
                                              resolution_ids=resolution_ids,
                                              cross_resolution_attention=not cfg.multiresolution,
                                              token_mask=plan.token_mask,
                                              channel_mask=enc_channel_mask,
                                              patch_padding_mask=patch_pad,
                                              sensor_text_embs=sensor_text_embs,
                                              sensor_text_masks=sensor_text_masks,
                                              sensor_id=enc_sensor_id)
                jepa_mask = plan.token_mask & enc_channel_mask.unsqueeze(1) & patch_pad.unsqueeze(2)
            else:
                masked = clean
                jepa_mask = torch.zeros(clean["tokens"].shape[:3], dtype=torch.bool, device=device)

            jepa_loss = clean["pooled"].new_zeros(())
            jepa_parts: dict[str, float] = {}
            if jepa_teacher is not None:
                # The teacher sees the clean view and never receives gradients. Reuse frozen text-LM
                # outputs, but run the teacher's own frontend/fusion/transformer weights.
                with torch.no_grad():
                    with torch.amp.autocast(device.type, enabled=False):
                        if shared_analysis is not None:
                            # Fixed arm: reuse the student's parameter-free analysis, then apply
                            # the TEACHER's own (EMA-lagged) projection. Detached — the teacher
                            # never receives gradient.
                            teacher_sensor_tokens = jepa_teacher.project_tokens(
                                shared_analysis.detach())
                        else:
                            teacher_sensor_tokens = jepa_teacher.tokenize(
                                patches.float(), rates, patch_len,
                                source_rate_hz=batch.get("source_rates", rates).to(device),
                            )
                    teacher_clean = jepa_teacher.encode(
                        teacher_sensor_tokens, text_embs, text_masks, positions,
                        patch_durations=patch_durations,
                        resolution_ids=resolution_ids,
                        channel_mask=enc_channel_mask,
                        patch_padding_mask=patch_pad,
                        sensor_text_embs=sensor_text_embs,
                        sensor_text_masks=sensor_text_masks,
                        sensor_id=enc_sensor_id,
                    )
                jepa_prediction = model.jepa_predictor(masked["tokens"])
                jepa_loss = masked_ema_latent_loss(
                    jepa_prediction, teacher_clean["tokens"], jepa_mask,
                    token_groups=resolution_ids,
                    token_durations=patch_durations,
                )
                # JEPA has no negatives, so its loss alone cannot distinguish "learned to predict
                # the teacher" from "the teacher collapsed and anything predicts it". Log the
                # margin over a random masked-position pairing (see losses_repr.pair_contrast).
                if bool(jepa_mask.any()):
                    jepa_diag = pair_contrast(jepa_prediction[jepa_mask].flatten(1),
                                              teacher_clean["tokens"][jepa_mask].flatten(1))
                    jepa_parts = {f"jepa/{k}": v for k, v in jepa_diag.items()}

            z_b = model.relation_projector(encode_clean_view_b(batch)["pooled"])
            pair_left, pair_right = verified_event_pairs(
                batch["event_ids"], batch["event_verified"], batch["streams"], device
            )
            relation = relation_loss(
                z, z_b, pair_left, pair_right,
                cross_placement_weight=cfg.cross_placement_weight,
                invariance_weight=cfg.vicreg_invariance_weight,
                variance_weight=cfg.vicreg_variance_weight,
                covariance_weight=cfg.vicreg_covariance_weight,
                target_std=cfg.vicreg_target_std,
            )

            out = phase_a_loss(
                jepa_loss, relation.total,
                jepa_weight=cfg.jepa_weight,
                relation_weight=cfg.relation_weight,
            )
            frontend_reg = model.encoder.filterbank.adaptation_regularization()
            out.total = out.total + cfg.frontend_reg_weight * frontend_reg
            parts = {
                "jepa": float(jepa_loss.detach()),
                "relation": float(relation.total.detach()),
                "relation/augmentation": float(relation.augmentation.total.detach()),
                "relation/augmentation_invariance": float(
                    relation.augmentation.invariance.detach()),
                "relation/variance": float(relation.augmentation.variance.detach()),
                "relation/covariance": float(relation.augmentation.covariance.detach()),
                "relation/min_std": float(relation.augmentation.min_std.detach()),
                "relation/cross_placement_invariance": float(
                    relation.cross_placement.detach()),
                "relation/cross_placement_weighted": float(
                    relation.cross_placement_weighted.detach()),
                "relation/placement_pairs": relation.placement_pairs,
                "frontend_reg": float(frontend_reg.detach()),
                "frontend_reg_weighted": float(
                    (cfg.frontend_reg_weight * frontend_reg).detach()),
            }
            parts.update(jepa_parts)
            parts.update({f"relation/augmentation_{key}": value
                          for key, value in pair_contrast(z, z_b).items()})
            if relation.placement_pairs >= 2:
                parts.update({f"relation/placement_{key}": value
                              for key, value in pair_contrast(
                                  z[pair_left], z[pair_right]).items()})

        do_log = step % 50 == 0 or step == 1
        do_objective_grad_log = step == 1 or step % 500 == 0
        objective_grad_norms = {}
        if do_objective_grad_log:
            objective_grad_norms = {
                f"grad_objective/{name}": objective_encoder_grad_norm(term, model.encoder)
                for name, term in out.terms.items()
            }
        if not bool(torch.isfinite(out.total)):
            raise FloatingPointError(f"non-finite Phase-A loss at step {step}: {parts}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(out.total).backward()
        scaler.unscale_(opt)
        gnorms = module_grad_norms(model) if do_log else {}     # pre-clip per-module grad norms
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        clip_coefficient = min(
            1.0, float(cfg.grad_clip) / (float(total_grad_norm.detach()) + 1e-6)
        )
        old_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        # GradScaler lowers its scale when it skips a non-finite optimizer step. The EMA target and
        # LR schedule must advance only with real student updates, or the teacher drifts toward
        # unchanged weights while the optimizer trajectory silently loses a scheduler step.
        optimizer_stepped = scaler.get_scale() >= old_scale
        if jepa_teacher is not None and optimizer_stepped:
            update_ema_encoder(model.encoder, jepa_teacher, cfg.jepa_ema_decay)
        if optimizer_stepped:
            sched.step()

        if do_log:
            lrs = sched.get_last_lr()
            rec = {"step": step, "lr": lrs[0],
                   "elapsed_s": round(time.time() - t0, 1),
                   "patch_seconds": batch["patch_seconds"],
                   "total": round(float(out.total.detach()), 4), **parts, **gnorms}
            rec["grad/total_preclip"] = float(total_grad_norm.detach())
            rec["grad/clip_coefficient"] = clip_coefficient
            rec["grad/clipped"] = float(clip_coefficient < 1.0)
            rec.update(objective_grad_norms)
            rec.update(representation_health(z))
            if cfg.jepa_weight > 0:
                # Windows that get NO masked token at all, per source. The mask planner correctly
                # refuses to mask when every real token overlaps the interval (there is no honest
                # visible/masked split), but that is silent, and it is systematic on the
                # short-window sources: sp_sw_har is 1.00 s windows and loses ~30-50% of its JEPA
                # supervision, uci_har (2.56 s) ~10%, unimib_shar (3.02 s) ~1%.
                # This depends only on the mask and remains comparable across tokenizer arms.
                with torch.no_grad():
                    unmasked = (jepa_mask.flatten(1).sum(dim=1) == 0).float()
                rec["jepa_unmasked_frac_by_source"] = per_source_mean(
                    unmasked, batch["sources"])
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
            val_z, val_y, val_src = embed_stratified(model, val_loader, device, cfg.val_per_label,
                                                     label_totals=val_label_totals)
            train_eval_gen.manual_seed(cfg.data_seed)   # same support bank at every val + across arms (#4)
            train_z, train_y, _ = embed_stratified(model, train_eval_loader, device,
                                                   cfg.val_per_label, target_labels=set(val_y.tolist()),
                                                   label_totals=train_label_totals)
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
