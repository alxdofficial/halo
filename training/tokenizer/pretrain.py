"""Pipeline A Phase-1 pretraining — the elite-3 objectives at corpus scale.

Everything the gates taught is baked in:
  * TWO forwards per step: masked -> A1 (feature targets from the CALIBRATED frozen
    filterbank); clean -> A2 SupCon + A3 grounding.        (M2 lesson 1)
  * Config conditioning is channel TEXT; the text-dropout/paraphrase augs supply the
    "unseen description" robustness.                        (M2 lesson 2, upgraded by M3)
  * Gravity alignment is disabled by default; signed DC preserves posture while SO(3)
    augmentation supplies orientation robustness.           (2026-07-19 decision)
  * The encoder's inner filterbank norm is CALIBRATED before training (copied from
    the target tokenizer's fitted stats).                    (M3 lesson)
  * A3 stays a rail (weight 0.1), targets validity-masked, computed on augmented views.

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
)
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
    a3_weight: float = 0.1
    calib_batches: int = 50               # filterbank norm calibration pass
    val_every: int = 1_000
    val_per_label: int = 40               # kNN val: windows PER LABEL (stratified, all classes scored)
    knn_k: int = 5
    num_workers: int = 12                 # profiled: nw=4 was data-bound (~35% GPU idle); 12 of 24
                                          # cores makes the step compute-bound (2026-07-19)
    seed: int = 20260718
    max_per_stream: int | None = None     # None = use ALL windows; the sampler is source-balanced
    device: str = "cpu"


class PipelineAModel(nn.Module):
    def __init__(self, cfg: PretrainConfig, a1_target_dim: int):
        super().__init__()
        # Fail loud rather than silently substitute: the mamba frontend is not yet integrated into
        # the training LIFECYCLE (A1 calibration, regularization/logging hooks, checkpoint config,
        # eval reconstruction — see docs/design/TOKENIZER_ABLATION.md "Audit blockers" #2/#3). Passing
        # frontend=cfg.frontend below WOULD build it, but the loop would then mishandle it; worse, the
        # old code never passed frontend at all, so cfg.frontend="mamba" silently built the FIXED
        # filterbank and stamped the checkpoint "mamba" — a falsely-labelled ablation. Refuse it here.
        if cfg.frontend not in ("fixed", "learnable"):
            raise NotImplementedError(
                f"frontend={cfg.frontend!r} is not wired into the training loop yet (calibration / "
                f"regularization / checkpoint / eval hooks assume the filterbank). See "
                f"docs/design/TOKENIZER_ABLATION.md. Prototype is reachable via "
                f"SetTokenizerEncoder(frontend=...) for standalone use only.")
        self.encoder = SetTokenizerEncoder(
            d_model=cfg.d_model, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout, dft_size=DFT_SIZE,
            frontend=cfg.frontend,          # route the ARM's frontend (was: silently ignored -> fixed)
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
    label vocab), stored in the checkpoint so it records WHICH corpus produced the weights (F5 —
    grid meta.json carries no raw fingerprint; this captures the corpus identity a checkpoint needs)."""
    import hashlib
    sig = [f"{r.dataset}/{r.stream}:{r.rate_hz}:{tuple(r.shape)}:{len(set(r.labels))}"
           for r in sorted(index.refs, key=lambda r: (r.dataset, r.stream))]
    sig.append("labels=" + ",".join(sorted(index.label_ids)))
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
            out = model.encoder(
                batch["patches"].to(device), batch["rates"].to(device),
                batch["patch_len"].to(device), batch["texts"],
                batch["positions"].to(device),
                patch_durations=(batch["patch_durations"].to(device)
                                 if "patch_durations" in batch else None),
                resolution_ids=(batch["resolution_ids"].to(device)
                                if "resolution_ids" in batch else None),
                channel_mask=batch["channel_mask"].to(device),
                patch_padding_mask=batch["patch_padding_mask"].to(device),
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
    mods = (("encoder", model.encoder), ("a1", model.a1_head), ("a2", model.a2_proj),
            ("a3_cad", model.a3_cadence), ("a3_eig", model.a3_eigen))
    out = {}
    for name, mod in mods:
        sq = sum(float(p.grad.detach().pow(2).sum()) for p in mod.parameters() if p.grad is not None)
        out[f"grad/{name}"] = sq ** 0.5
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
    parser.add_argument("--frontend", choices=("fixed", "learnable", "mamba"), default=None,
                        help="override the arm's frontend for an attribution diagnostic. 'mamba' is a "
                             "standalone prototype and raises NotImplementedError here until its "
                             "training lifecycle is integrated (docs/design/TOKENIZER_ABLATION.md).")
    parser.add_argument("--multiresolution", action=argparse.BooleanOptionalAction, default=None,
                        help="override multiresolution (default ON); --no-multiresolution is the "
                             "single-resolution ablation")
    args = parser.parse_args()

    cfg = PretrainConfig(
        device=args.device,
        arm=args.arm,
        frontend="learnable" if args.arm == "learnable" else "fixed",
        multiresolution=True,          # new Phase-A default: multiresolution ON (diagnostic-confirmed
                                       # winner, 0.835 held-out transfer); --no-multiresolution to ablate
    )
    if args.frontend is not None:
        cfg.frontend = args.frontend
    if args.multiresolution is not None:
        cfg.multiresolution = args.multiresolution
    if args.steps:
        cfg.steps = args.steps
    if args.smoke:
        cfg = PretrainConfig(
            d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
            classes_per_batch=8, samples_per_class=4, steps=args.steps or 10,
            warmup_steps=2, calib_batches=3, val_every=max(args.steps or 10, 5),
            val_per_label=10, num_workers=0, max_per_stream=200,
            device=args.device, arm=cfg.arm, frontend=cfg.frontend,
            multiresolution=cfg.multiresolution,
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
    (args.out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"arm={cfg.arm} frontend={cfg.frontend} multiresolution={cfg.multiresolution}", flush=True)

    # ------------------------------------------------------------------ data
    index = CorpusIndex(max_per_stream=cfg.max_per_stream, seed=cfg.seed)
    print(f"corpus: {index.summary()}", flush=True)
    train_ds = PretrainDataset(index, index.train, augment=True)
    val_ds = PretrainDataset(index, index.val, augment=False)
    train_collate = (MultiResolutionCollate(
        short_choices=cfg.short_patch_choices, long_choices=cfg.long_patch_choices,
        min_resolution_ratio=cfg.min_resolution_ratio, seed=cfg.seed,
    ) if cfg.multiresolution
                     else MultiScaleCollate(seed=cfg.seed))
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler(index.train, cfg.classes_per_batch,
                                           cfg.samples_per_class, cfg.steps, cfg.seed,
                                           stream_datasets=index.stream_datasets),
        collate_fn=train_collate,
        num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda",
    )
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
    train_eval_loader = DataLoader(
        PretrainDataset(index, index.train, augment=False), batch_size=256,
        shuffle=True, collate_fn=val_collate,
        num_workers=val_workers, persistent_workers=val_workers > 0,
        pin_memory=device.type == "cuda", worker_init_fn=_seed_worker,
    )

    # ---------------------------------------------------- A1 target tokenizer
    target_tok = PhysicalFilterbankTokenizer(d_model=1, dft_size=DFT_SIZE)
    target_tok.proj = nn.Identity()
    print(f"calibrating filterbank norm on {cfg.calib_batches} batches ...", flush=True)
    target_tok.reset_norm_accumulator()
    def _cycle(loader):
        while True:
            yield from loader
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

    # ------------------------------------------------------------------ model
    # A1 predicts only the SIGNAL-content dims of the filterbank feature (band energies +
    # amplitude + dc); the rate-metadata masks are dropped (they were ~81% of the target
    # norm and turned A1 into 'echo the rate' — second-agent audit 2026-07-18).
    signal_idx = torch.tensor(target_tok.signal_feature_indices(), device=device)
    model = PipelineAModel(cfg, a1_target_dim=len(signal_idx)).to(device)
    # M3 lesson: copy the calibrated norm into the encoder's inner filterbank.
    model.encoder.filterbank.norm_mu.copy_(target_tok.norm_mu.to(device))
    model.encoder.filterbank.norm_sd.copy_(target_tok.norm_sd.to(device))
    model.encoder.filterbank.dc_mu.copy_(target_tok.dc_mu.to(device))
    model.encoder.filterbank.dc_sd.copy_(target_tok.dc_sd.to(device))
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
        (adaptive_params if name in adaptive_names else base_params).append(parameter)
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
    weights = EliteLossWeights(a3_grounding=cfg.a3_weight)
    log_path = args.out / "log.jsonl"
    best_ba = -1.0
    t0 = time.time()

    def checkpoint(name: str, step: int, val_ba: float):
        import random as _stdrandom
        torch.save({
            "encoder": model.encoder.state_dict(),
            "heads": {k: v.state_dict() for k, v in
                      (("a1", model.a1_head), ("a2", model.a2_proj),
                       ("a3_cadence", model.a3_cadence), ("a3_eigen", model.a3_eigen))},
            "config": asdict(cfg),
            "label_ids": index.label_ids,
            "step": step, "val_ba": val_ba,
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
        for key in ("frontend", "multiresolution"):
            if saved_cfg.get(key, getattr(cfg, key)) != getattr(cfg, key):
                raise ValueError(
                    f"resume configuration mismatch for {key}: checkpoint="
                    f"{saved_cfg.get(key)!r}, requested={getattr(cfg, key)!r}"
                )
        model.encoder.load_state_dict(rk["encoder"])
        for k, head in (("a1", model.a1_head), ("a2", model.a2_proj),
                        ("a3_cadence", model.a3_cadence), ("a3_eigen", model.a3_eigen)):
            head.load_state_dict(rk["heads"][k])
        opt.load_state_dict(rk["optimizer"])
        sched.load_state_dict(rk["scheduler"])
        scaler.load_state_dict(rk["scaler"])
        if "rng" in rk:
            import random as _sr
            torch.set_rng_state(rk["rng"]["torch"])
            np.random.set_state(rk["rng"]["numpy"])
            _sr.setstate(rk["rng"]["python"])
        start_step = int(rk["step"])
        best_ba = float(rk["val_ba"])
        print(f"resumed from {args.resume} at step {start_step} (best_ba {best_ba:.3f})", flush=True)

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

        # validity-aware: temporal block lands on real patches, drops hit real channels,
        # so A1 supervision is non-empty for every window with >=2 real patches
        if cfg.multiresolution:
            plan = make_multiresolution_mask_plan(
                batch["patch_starts"].to(device), batch["patch_ends"].to(device),
                resolution_ids, C, GYRO_IDX, channel_mask=channel_mask,
                valid_patches=patch_pad,
            )
        else:
            plan = make_mask_plan(B, P, C, GYRO_IDX, device=device,
                                  valid_patches=patch_pad, channel_mask=channel_mask)
        targets = GroundingTargets(
            cadence_log2hz=batch["cadence_target"].to(device),
            cadence_valid=batch["cadence_valid"].to(device),
            eigen_ratios=batch["eigen_target"].to(device),
            eigen_valid=batch["eigen_valid"].to(device),
        )

        # A1 loss only counts tokens that are masked AND a real channel AND a real patch —
        # otherwise accel-only windows + rate-shortened phantom patches waste loss budget
        # "predicting" the zero-padding signature.
        a1_loss_mask = plan.token_mask & channel_mask.unsqueeze(1) & patch_pad.unsqueeze(2)

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            # The filterbank DSP (rDFT + constant-Q einsum) runs in fp32 — fp16 has too
            # little headroom for the band-energy magnitudes (sweep finding 15). The
            # transformer/heads stay in autocast fp16. sensor_tokens keeps grad (trainable
            # proj); only the A1 TARGET is under no_grad.
            with torch.amp.autocast(device.type, enabled=False):
                with torch.no_grad():
                    a1_target = target_tok(patches.float(), rates, patch_len)[..., signal_idx]
                    # observable-band validity over the SIGNAL dims: band dims use the
                    # Nyquist mask o (B,K); amplitude/dc dims are always valid.
                    o, _ = target_tok.masks(rates, patch_len)            # (B, K)
                    extra = o.new_ones(B, len(signal_idx) - o.shape[1])
                    a1_feature_valid = torch.cat([o, extra], dim=1).view(B, 1, 1, -1)
                sensor_tokens = model.encoder.tokenize(patches.float(), rates, patch_len)
            text_embs, text_masks = model.encoder.encode_texts(batch["texts"], device)
            masked = model.encoder.encode(sensor_tokens, text_embs, text_masks, positions,
                                          patch_durations=patch_durations,
                                          resolution_ids=resolution_ids,
                                          cross_resolution_attention=not cfg.multiresolution,
                                          token_mask=plan.token_mask,
                                          channel_mask=channel_mask,
                                          patch_padding_mask=patch_pad)
            clean = model.encoder.encode(sensor_tokens, text_embs, text_masks, positions,
                                         patch_durations=patch_durations,
                                         resolution_ids=resolution_ids,
                                         channel_mask=channel_mask,
                                         patch_padding_mask=patch_pad)
            z = model.a2_proj(clean["pooled"])
            a1_pred = model.a1_head(masked["tokens"])
            out = elite3_loss(
                a1_pred=a1_pred, a1_target=a1_target,
                a1_mask=a1_loss_mask,
                a2_embeddings=z, a2_labels=labels,
                a3_cadence_pred=model.a3_cadence(clean["pooled"]).squeeze(1),
                a3_eigen_pred=model.a3_eigen(clean["pooled"]).view(B, 4, 3),
                a3_targets=targets, weights=weights,
                a1_feature_valid=a1_feature_valid,
                a1_token_groups=resolution_ids,
            )
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
            with torch.no_grad():                               # per-source A1 (diagnostic, off-graph)
                a1_pw = masked_latent_per_window(a1_pred.float(), a1_target, a1_loss_mask,
                                                 feature_valid=a1_feature_valid)
            lrs = sched.get_last_lr()
            rec = {"step": step, "lr": lrs[0],
                   "elapsed_s": round(time.time() - t0, 1),
                   "patch_seconds": batch["patch_seconds"],
                   "total": round(float(out.total.detach()), 4), **out.parts, **gnorms,
                   "a1_by_source": per_source_mean(a1_pw, batch["sources"])}
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
