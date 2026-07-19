"""Pipeline A Phase-1 pretraining — the elite-3 objectives at corpus scale.

Everything the gates taught is baked in:
  * TWO forwards per step: masked -> A1 (feature targets from the CALIBRATED frozen
    filterbank); clean -> A2 SupCon + A3 grounding.        (M2 lesson 1)
  * Config conditioning is channel TEXT; the text-dropout/paraphrase augs supply the
    "unseen description" robustness.                        (M2 lesson 2, upgraded by M3)
  * Gravity-align is part of the front end, ALWAYS.         (M2 lesson 3)
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
    masked_latent_per_window,
)
from training.tokenizer.pretrain_data import (
    CHANNELS,
    DFT_SIZE,
    BalancedBatchSampler,
    CorpusIndex,
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
    classes_per_batch: int = 32
    samples_per_class: int = 8            # batch = 256
    steps: int = 20_000
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    a3_weight: float = 0.1
    calib_batches: int = 50               # filterbank norm calibration pass
    val_every: int = 1_000
    val_per_label: int = 40               # kNN val: windows PER LABEL (stratified, all classes scored)
    knn_k: int = 5
    num_workers: int = 4
    seed: int = 20260718
    max_per_stream: int | None = None     # None = use ALL windows; the sampler is source-balanced
    device: str = "cpu"


class PipelineAModel(nn.Module):
    def __init__(self, cfg: PretrainConfig, a1_target_dim: int):
        super().__init__()
        self.encoder = SetTokenizerEncoder(
            d_model=cfg.d_model, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout, dft_size=DFT_SIZE,
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
    # support cap happened to include.
    labels = sorted(set(test_y.tolist()))
    per_class = []
    for label in labels:
        idx = (test_y == label).nonzero().squeeze(1)
        if not len(idx):
            continue
        hits = 0
        for i in idx.tolist():
            d = (train_z - test_z[i]).norm(dim=1)
            nn_lab = train_y[d.argsort()[:k]]
            hits += int(nn_lab.mode().values) == label
        per_class.append(hits / len(idx))
    return float(np.mean(per_class)) if per_class else float("nan")


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
    args = parser.parse_args()

    cfg = PretrainConfig(device=args.device)
    if args.steps:
        cfg.steps = args.steps
    if args.smoke:
        cfg = PretrainConfig(
            d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
            classes_per_batch=8, samples_per_class=4, steps=args.steps or 10,
            warmup_steps=2, calib_batches=3, val_every=max(args.steps or 10, 5),
            val_per_label=10, num_workers=0, max_per_stream=200,
            device=args.device,
        )
    device = torch.device(cfg.device)
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

    # ------------------------------------------------------------------ data
    index = CorpusIndex(max_per_stream=cfg.max_per_stream, seed=cfg.seed)
    print(f"corpus: {index.summary()}", flush=True)
    train_ds = PretrainDataset(index, index.train, augment=True)
    val_ds = PretrainDataset(index, index.val, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler(index.train, cfg.classes_per_batch,
                                           cfg.samples_per_class, cfg.steps, cfg.seed,
                                           stream_datasets=index.stream_datasets),
        collate_fn=MultiScaleCollate(seed=cfg.seed),
        num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda",
    )
    # val: no aug, fixed 1.0 s patches, plain order
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=MultiScaleCollate(fixed_patch_seconds=1.0),
                            num_workers=0)
    train_eval_loader = DataLoader(
        PretrainDataset(index, index.train, augment=False), batch_size=256,
        shuffle=True, collate_fn=MultiScaleCollate(fixed_patch_seconds=1.0),
        num_workers=0,
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
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
        patches = batch["patches"].to(device, non_blocking=True)   # gravity-aligned in collate
        rates = batch["rates"].to(device)
        patch_len = batch["patch_len"].to(device)
        positions = batch["positions"].to(device)
        channel_mask = batch["channel_mask"].to(device)
        patch_pad = batch["patch_padding_mask"].to(device)
        labels = batch["labels"].to(device)
        B, P, _, C = patches.shape

        # validity-aware: temporal block lands on real patches, drops hit real channels,
        # so A1 supervision is non-empty for every window with >=2 real patches
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
                                          token_mask=plan.token_mask,
                                          channel_mask=channel_mask,
                                          patch_padding_mask=patch_pad)
            clean = model.encoder.encode(sensor_tokens, text_embs, text_masks, positions,
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
            )

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
            rec = {"step": step, "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1),
                   "patch_seconds": batch["patch_seconds"],
                   "total": round(float(out.total.detach()), 4), **out.parts, **gnorms,
                   "a1_by_source": per_source_mean(a1_pw, batch["sources"])}
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
            # per-source val kNN-BA — which datasets the representation clusters well
            vs = np.asarray(val_src)
            ba_by_src = {}
            for s in sorted(set(val_src)):
                mt = torch.from_numpy(vs == s)
                if int(mt.sum()) >= cfg.knn_k:
                    ba_by_src[s] = round(knn_balanced_acc(train_z, train_y, val_z[mt], val_y[mt],
                                                          cfg.knn_k), 4)
            rec = {"step": step, "val_knn_ba": ba, "val_ba_by_source": ba_by_src}
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
