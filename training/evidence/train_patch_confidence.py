"""Calibrate Phase-B reject confidence with a frozen patch evidence predictor.

The predictor is never updated here. The confidence target is one only when the ground truth is an
allowed candidate and the frozen predictor selects it; truth-absent and incorrect episodes target
zero. This learns selective prediction without introducing an ``UNKNOWN`` candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import get_sbert_encoder
from model.evidence.confidence import EvidenceConfidenceHead, aurc, binary_auroc
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.bank_guard import (
    assert_artifact_matches_bank,
    assert_bank_current,
    assert_patch_bank,
    bank_fingerprint,
)
from training.evidence.labeltext import ensemble_text
from training.evidence.patch_episodes import PatchTable
from training.evidence.train_patch_decoder import (
    OTHER_SUPPORT_PROBS,
    SUPPORT_CHOICES,
    TRUE_SUPPORT_PROBS,
    choose_candidates,
    choose_memory_labels,
    encode_bank_config_text,
    load_activity_families,
    physical_label_centroids,
    run_patch_episode,
    sample_queries,
    sample_support,
    synthetic_smoke_bank,
)

_DIR = Path(__file__).resolve().parent / "outputs"
_DEFAULT_BANK = _DIR / "memory_bank.pt"
_DEFAULT_PREDICTOR = _DIR / "patch_evidence_predictor.pt"
_DEFAULT_OUT = _DIR / "patch_evidence_confidence.pt"
SEED = 20260807


def expected_calibration_error(score: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not len(score):
        return float("nan")
    total = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        member = (score >= lower) & (score < upper if upper < 1 else score <= upper)
        if member.any():
            total += member.mean() * abs(score[member].mean() - target[member].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--predictor", type=Path, default=_DEFAULT_PREDICTOR)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--truth-absent-prob", type=float, default=0.25)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-episodes", type=int, default=8)
    ap.add_argument("--val-queries", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.device = "cpu"; args.steps = 2; args.batch = 4
        args.val_every = 1; args.val_episodes = 2; args.val_queries = 4
        args.predictor = Path("/tmp/halo_phase_b_predictor_smoke.pt")
        args.out = Path("/tmp/halo_phase_b_confidence_smoke.pt") \
            if args.out == _DEFAULT_OUT else args.out
    if args.steps < 1 or args.batch < 1 or args.val_every < 1:
        ap.error("steps, batch, and val-every must be positive")
    if not 0.0 < args.truth_absent_prob < 1.0:
        ap.error("truth-absent-prob must be in (0,1)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bank = synthetic_smoke_bank() if args.smoke else torch.load(
        args.bank, map_location="cpu", weights_only=True
    )
    if not args.smoke:
        assert_bank_current(bank, context="train_patch_confidence")
    assert_patch_bank(bank, context="train_patch_confidence")
    predictor = torch.load(args.predictor, map_location="cpu", weights_only=True)
    if not args.smoke:
        assert_artifact_matches_bank(
            predictor, bank, context="train_patch_confidence",
            artifact_name="patch evidence predictor",
        )
    if predictor.get("objective") != "candidate_cross_entropy":
        raise SystemExit("confidence calibration requires the consolidated CE predictor artifact")

    cfg = predictor["cfg"]
    decoder = EvidenceDecoder(DecoderConfig(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        candidate_tokens=True, structural_metadata=True,
    )).to(device)
    decoder.load_state_dict(predictor["decoder"])
    retriever = PatchSubspaceRetriever(
        cfg["d_model"], cfg["n_subspaces"], cfg["subspace_dim"], cfg["subspace_ema"]
    ).to(device)
    retriever.load_state_dict(predictor["retriever"])
    decoder.eval(); retriever.eval()
    for parameter in list(decoder.parameters()) + list(retriever.parameters()):
        parameter.requires_grad_(False)

    table = PatchTable(bank)
    Z = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y = bank["y"].long().to(device)
    subj = bank["subj"].long().to(device)
    config = bank["cfg"].long().to(device)
    vocab = list(bank["vocab"])
    n_vocab = len(vocab)
    heldout = torch.tensor(predictor["heldout_labels"], device=device, dtype=torch.long)
    fold = predictor["fold"]
    val_cfg = torch.tensor(fold["validation_config_ids"], device=device)
    val_subject = torch.tensor(fold["validation_subject_ids"], device=device)
    is_val_cfg = torch.isin(config, val_cfg)
    is_val_subject = torch.isin(subj, val_subject)
    base_train = torch.nonzero(~is_val_cfg & ~is_val_subject, as_tuple=True)[0]
    train_pool = base_train[~torch.isin(y[base_train], heldout)]
    val_pool = torch.nonzero(
        is_val_cfg & is_val_subject & torch.isin(y, heldout), as_tuple=True
    )[0]
    if not len(train_pool) or not len(val_pool):
        raise SystemExit("predictor fold does not provide confidence train/validation queries")
    physical = physical_label_centroids(Z[base_train], y[base_train], n_vocab)

    train_memory_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    train_memory_mask[train_pool] = True
    val_memory_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    val_memory_mask[base_train] = True
    retrieval_cfg = predictor["retrieval_cfg"]
    train_index_rows = table.sample_index_rows(
        train_memory_mask, retrieval_cfg["index_per_label"],
        np.random.default_rng(args.seed + 1),
    )
    val_index_rows = table.sample_index_rows(
        val_memory_mask, retrieval_cfg["index_per_label"],
        np.random.default_rng(args.seed + 2),
    )

    def build_index(rows):
        memory = F.normalize(bank["patch"]["Z"][rows].float().to(device), dim=-1)
        return retriever.build_index(memory)

    train_memory_index = build_index(train_index_rows)
    val_memory_index = build_index(val_index_rows)
    sbert = get_sbert_encoder()
    text = ensemble_text(vocab, sbert, 8, train_only=True).to(device)
    family_ids, _, _ = load_activity_families(vocab)
    family_ids = family_ids.to(device)
    config_text = encode_bank_config_text(bank, sbert, device) \
        if cfg.get("explicit_config_text", False) else None
    train_vocab = torch.arange(n_vocab, device=device)
    train_vocab = train_vocab[~torch.isin(train_vocab, heldout)]
    episode_cfg = predictor["episode_cfg"]

    @torch.no_grad()
    def draw_features(
        pool, memory_mask, index_rows, memory_index, allowed_vocab, local_rng,
        *, truth_present: bool, count: int,
    ):
        last_error = None
        for _ in range(20):
            present = torch.unique(y[pool])
            seed_count = min(4, len(present))
            seed_labels = torch.tensor(
                local_rng.choice(present.cpu().numpy(), size=seed_count, replace=False),
                device=device, dtype=torch.long,
            )
            qi = sample_queries(
                pool, seed_labels, y, count, local_rng,
                config_ids=config, subject_ids=subj, label_alpha=0.0,
                subject_alpha=float(episode_cfg["query_subject_alpha"]),
            )
            fraction = float(local_rng.choice(episode_cfg["candidate_fractions"]))
            candidate_count = max(2, min(n_vocab, int(round(fraction * n_vocab))))
            mode = str(local_rng.choice(episode_cfg["distractor_modes"]))
            candidates = choose_candidates(
                y[qi], candidate_count, n_vocab, text, physical,
                truth_present=truth_present, mode=mode, rng=local_rng,
                allowed_vocab=allowed_vocab, family_ids=family_ids,
            )
            true_support = sample_support(local_rng, SUPPORT_CHOICES, TRUE_SUPPORT_PROBS)
            if not truth_present:
                true_support = 0
            other_support = sample_support(local_rng, SUPPORT_CHOICES, OTHER_SUPPORT_PROBS)
            memory_fraction = float(local_rng.choice(episode_cfg["memory_fractions"]))
            memory_labels = choose_memory_labels(
                allowed_vocab, y[qi], memory_fraction, local_rng,
                require_truth=truth_present and true_support != 0,
            )
            config_mode = str(local_rng.choice(episode_cfg["config_modes"]))
            try:
                logits, aux = run_patch_episode(
                    decoder, retriever, table, bank, index_rows, memory_index,
                    qi, candidates, memory_labels, text, text,
                    truth_present=truth_present, true_support=true_support,
                    other_support=other_support, config_mode=config_mode, rng=local_rng,
                    topk_per_head=retrieval_cfg["topk_per_head"],
                    max_evidence=retrieval_cfg["max_evidence"],
                    max_per_window=retrieval_cfg["max_per_window"],
                    max_per_label=retrieval_cfg["max_per_label"],
                    tau=retrieval_cfg["tau_retr"], query_window_mask=memory_mask,
                    config_text=config_text,
                )
            except ValueError as error:
                last_error = error
                continue
            prediction = candidates[logits.argmax(1)]
            target = prediction.eq(y[qi]) if truth_present else torch.zeros_like(
                prediction, dtype=torch.bool
            )
            return aux["confidence_features"].detach(), target.float(), {
                "truth_present": truth_present,
                "correct": float(target.float().mean()),
                "candidate_count": len(candidates),
                "true_support": true_support,
            }
        raise RuntimeError("could not draw a feasible confidence episode") from last_error

    val_features, val_targets = [], []
    val_rng = np.random.default_rng(args.seed + 3)
    for index in range(args.val_episodes):
        features, targets, _ = draw_features(
            val_pool, val_memory_mask, val_index_rows, val_memory_index,
            torch.arange(n_vocab, device=device), val_rng,
            truth_present=index % 2 == 0, count=args.val_queries,
        )
        val_features.append(features); val_targets.append(targets)
    val_features = torch.cat(val_features)
    val_targets = torch.cat(val_targets)
    if not bool(val_targets.bool().any()) or bool(val_targets.bool().all()):
        raise SystemExit(
            "confidence validation has only one target class; the predictor/calibration fold "
            "cannot support a meaningful reject-confidence fit"
        )

    head = EvidenceConfidenceHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    @torch.no_grad()
    def evaluate():
        head.eval()
        logit = head(val_features)
        loss = head.loss(logit, val_targets)
        score = torch.sigmoid(logit).cpu().numpy()
        target = val_targets.cpu().numpy().astype(bool)
        return {
            "bce": float(loss),
            "auroc": binary_auroc(score, target),
            "aurc": aurc(1.0 - score, target),
            "ece": expected_calibration_error(score, target),
            "positive_rate": float(target.mean()),
        }

    best = {"bce": float("inf")}; best_state = None; best_step = 0
    started = time.time()
    for step in range(1, args.steps + 1):
        truth_present = bool(rng.random() >= args.truth_absent_prob)
        features, target, episode = draw_features(
            train_pool, train_memory_mask, train_index_rows, train_memory_index,
            train_vocab, rng, truth_present=truth_present, count=args.batch,
        )
        head.train()
        logit = head(features)
        loss = head.loss(logit, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite confidence loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if step == 1 or step % args.val_every == 0:
            metrics = evaluate()
            if metrics["bce"] < best["bce"]:
                best = dict(metrics); best_step = step
                best_state = {
                    key: value.detach().cpu().clone() for key, value in head.state_dict().items()
                }
            print(json.dumps({
                "step": step, "loss": round(float(loss.detach()), 5),
                "grad_norm": round(float(grad_norm), 4), **episode,
                **{key: round(value, 4) for key, value in metrics.items()},
                "elapsed_s": round(time.time() - started, 1),
            }), flush=True)

    if best_state is None:
        raise RuntimeError("confidence calibration completed without a valid checkpoint")
    predictor_fp = hashlib.sha256(args.predictor.read_bytes()).hexdigest()
    payload = {
        "confidence": best_state,
        "objective": "correct_and_answerable_bce",
        "predictor": str(args.predictor), "predictor_fp": predictor_fp,
        "bank_fp": bank.get("bank_fp") or bank_fingerprint(bank),
        "memory_schema": int(bank["schema_version"]), "vocab": vocab,
        "truth_absent_probability": args.truth_absent_prob,
        "best_step": best_step, "best_metrics": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"[confidence] best step {best_step}: {best} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
