"""T2.2 + T2.3 — episodic class-holdout trainer for the evidence decoder.

The M4a diagnostic proved the *loss* was the bug: closed-vocab CE over the fixed 59-way vocab
overfits the seen-label geometry and the untrained mechanism wins (40.9 vs 47.5). This trainer
replaces that with a **transfer-aligned episodic loss** and trains the §2.2 decoder as a residual
on the untrained mechanism (docs/design/EVIDENCE_ENGINE_TIER2.md §3):

  * **Class-disjoint episodes.** Each step samples a held-out label set H. Queries have labels in
    H; the retrievable memory EXCLUDES all of H (+ always subject-disjoint); candidates = H. So a
    neighbour (label ∉ H) is never itself a valid answer — the decoder must retrieve semantically
    related evidence and TEXT-TRANSFER its (refined) label text to the correct held-out candidate.
    This is genuine zero-shot per episode (Matching/Prototypical-net style), the faithful analog
    of the ZS-XD eval, and it removes M4a's "retrieve the same label" crutch (M4a was label-present).
  * **Reg-to-identity.** The refinement Δ and the pooling residual (KL to the retrieval prior) are
    penalized. NOTE: this is a soft prior, NOT a guarantee. Zero-init makes the decoder start AT the
    untrained mechanism, but nothing bounds test-time degradation, and it has in fact degraded it
    (46.7 -> 44.2 on the 93-label bank). Do not describe this as 'can only improve'.
  * **Checkpoint selection = held-out-CONFIG × held-out-SUBJECT × class-disjoint transfer** on
    FIXED val episodes. Validation queries retrieve from the training fold only (never validation or
    boundary-dropped rows).

Everything runs on the cached bank — the encoder never runs here. Retrieval is raw cosine over the
frozen memory (top-k); the decoder does all the learning. Smoke-testable core of the pivotal
experiment; the ZS-XD gate adapter is a separate step.

Run (smoke):
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    $PY -m training.evidence.train_decoder --device cuda --steps 40 --val-every 10
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import get_sbert_encoder
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.edl import DensityGate, acc_at_coverage, aurc, edl_loss
from training.evidence.bank_guard import assert_bank_current
from training.evidence.labeltext import build_label_variants, ensemble_text

_DIR = Path(__file__).resolve().parent / "outputs"
_DEFAULT_BANK = _DIR / "memory_bank.pt"
_DEFAULT_OUT = _DIR / "evidence_decoder.pt"
SEED = 20260720


def balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    accs = [float((pred[true == c] == c).mean()) for c in np.unique(true)]
    return float(np.mean(accs)) if accs else float("nan")


@torch.no_grad()
def retrieve(zq, Z, allowed_mask, k, tau):
    """Top-k subject/label-disjoint retrieval.

    Returns (idx (B,k), w_retr (B,k) normalized, vals (B,k) RAW cosine sims pre-softmax). ``vals``
    is the retrieval-density signal the density-gated evidential loss reads (mean of the top-m).
    """
    if allowed_mask.shape != (len(zq), len(Z)):
        raise ValueError(f"allowed_mask must have shape {(len(zq), len(Z))}, "
                         f"got {tuple(allowed_mask.shape)}")
    min_allowed = int(allowed_mask.sum(1).min())
    if min_allowed == 0:
        raise ValueError("at least one query has no eligible memory rows")
    k = min(int(k), min_allowed)
    sim = zq @ Z.t()                                    # (B, N) cosine (Z, zq pre-normalized)
    sim = sim.masked_fill(~allowed_mask, float("-inf"))
    vals, idx = sim.topk(k, dim=1)
    w = torch.softmax(vals / tau, dim=1)                # renormalized over the k neighbours
    return idx, w, vals


def label_index(H: torch.Tensor, n_vocab: int, device) -> torch.Tensor:
    """Map global label id -> its column in candidate set H (or -1)."""
    pos = torch.full((n_vocab,), -1, device=device, dtype=torch.long)
    pos[H] = torch.arange(len(H), device=device)
    return pos


def sample_label_set(
    present: torch.Tensor,
    min_labels: int,
    max_labels: int,
    rng: np.random.Generator,
    *,
    reserve_labels: int = 0,
    device=None,
) -> torch.Tensor:
    """Uniformly sample an episode label budget, clamped to the labels actually present.

    ``reserve_labels`` leaves that many labels outside the sampled set. Training uses one so an
    episode can never hold out the entire retrievable vocabulary.
    """
    n_present = int(present.numel())
    upper = min(int(max_labels), n_present - int(reserve_labels))
    lower = min(int(min_labels), upper)
    if lower < 2:
        raise ValueError("an episode needs at least two candidate labels after budget clamping")
    size = int(rng.integers(lower, upper + 1))
    pick = rng.choice(present.detach().cpu().numpy(), size=size, replace=False)
    return torch.tensor(sorted(pick.tolist()), device=device or present.device, dtype=torch.long)


@torch.no_grad()
def estimate_density_threshold(
    Z: torch.Tensor,
    y: torch.Tensor,
    subj: torch.Tensor,
    train_q: torch.Tensor,
    train_present: torch.Tensor,
    memory_mask: torch.Tensor,
    min_labels: int,
    max_labels: int,
    k: int,
    *,
    seed: int,
    n_episodes: int = 2,
    queries_per_episode: int = 64,
) -> float:
    """Median training-fold retrieval density under the real class-holdout protocol."""
    rng = np.random.default_rng(seed)
    densities = []
    for _ in range(n_episodes):
        H = sample_label_set(
            train_present, min_labels, max_labels, rng,
            reserve_labels=1, device=Z.device,
        )
        candidates = train_q[torch.isin(y[train_q], H)]
        if not len(candidates):
            continue
        n = min(int(queries_per_episode), len(candidates))
        selected = rng.choice(candidates.detach().cpu().numpy(), size=n, replace=False)
        qi = torch.tensor(selected, device=Z.device, dtype=torch.long)
        allowed = (
            (~torch.isin(y, H)).unsqueeze(0)
            & (subj.unsqueeze(0) != subj[qi].unsqueeze(1))
            & memory_mask.unsqueeze(0)
        )
        _idx, _weights, vals = retrieve(Z[qi], Z, allowed, k, tau=1.0)
        densities.append(vals[:, :min(8, vals.shape[1])].mean(1))
    if not densities:
        raise ValueError("could not form a density-calibration episode from the training fold")
    return float(torch.cat(densities).median().item())


def run_episode(dec, Z, y, subj, qi, H, t_ev, t_cand, k, tau, return_aux=False,
                memory_mask=None):
    """One class-disjoint episode forward. Memory excludes H (+ subject-disjoint); candidates = H.

    ``t_ev`` / ``t_cand`` are (L, 384) label-text tables for the EVIDENCE side and the CANDIDATE side.
    Passing two *independently sampled* paraphrase tables is what forces semantic matching (see
    ``sample_text_tables``); passing the same ensembled table twice reproduces the old behaviour.
    """
    zq = Z[qi]
    not_in_H = ~torch.isin(y, H)                                  # (N,) labels allowed in memory
    allowed = not_in_H.unsqueeze(0) & (subj.unsqueeze(0) != subj[qi].unsqueeze(1))
    if memory_mask is not None:
        if memory_mask.shape != y.shape:
            raise ValueError(f"memory_mask must have shape {tuple(y.shape)}, "
                             f"got {tuple(memory_mask.shape)}")
        allowed &= memory_mask.unsqueeze(0)
    idx, w, vals = retrieve(zq, Z, allowed, k, tau)
    out = dec(zq=zq, zev=Z[idx], ev_label_text=t_ev[y[idx]], w_retr=w,
              cand_text=t_cand[H], return_aux=return_aux)
    return (out, w, vals) if return_aux else out


def sample_text_tables(variants, gen):
    """Two (L, 384) tables: an INDEPENDENT paraphrase drawn per label for evidence vs candidates.

    FINDINGS §2 showed the decoder's gains were confined to labels it had seen (r=-0.973 vs
    unseen-label fraction): with a single fixed anchor per label it can memorize 53 points instead
    of using text semantics. Re-drawing the surface form every episode removes that shortcut, and
    drawing the evidence side and the candidate side *independently* means a correct match can never
    be made on identical surface strings — only on meaning. Variant 0 is the canonical name, so the
    canonical phrasing stays in the mix.

    Precisely: the two draws are independent, so they COLLIDE with probability 1/K per label
    (1/16 at the default), i.e. about 1 label per ~18-label episode still gets an identical
    evidence/candidate string. The shortcut is suppressed, not eliminated — do not write "never".
    """
    L, K, _ = variants.shape
    ev = variants[torch.arange(L, device=variants.device),
                  torch.randint(0, K, (L,), generator=gen, device=variants.device)]
    cd = variants[torch.arange(L, device=variants.device),
                  torch.randint(0, K, (L,), generator=gen, device=variants.device)]
    return ev, cd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topk", type=int, default=48, help="evidence set size per query")
    ap.add_argument("--tau-retr", type=float, default=0.05, help="retrieval softmax temperature")
    ap.add_argument("--episode-labels", type=int, nargs=2, default=(12, 24),
                    help="[min,max] held-out label-set size H per episode")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lambda-delta", type=float, default=0.1, help="reg-to-identity on Δ")
    ap.add_argument("--lambda-pool", type=float, default=0.1, help="reg-to-identity on pooling (KL to prior)")
    ap.add_argument("--n-subspaces", type=int, default=0,
                    help="[D2] number of learned similarity subspaces for evidence re-weighting "
                         "(0 = off; decoder byte-identical to today). Composes with --loss ce and edl.")
    ap.add_argument("--subspace-dim", type=int, default=64, help="[D2] per-subspace projection dim")
    ap.add_argument("--lambda-ms", type=float, default=0.0,
                    help="[D2] reg-to-identity weight on the multi-subspace gate |gamma_ms|")
    ap.add_argument("--loss", choices=("ce", "edl"), default="ce",
                    help="ce = closed-vocab cross-entropy (default, unchanged); edl = density-gated "
                         "evidential (Dirichlet) loss with selective uncertainty u=C/S "
                         "(calibration must be established on held-out data)")
    ap.add_argument("--kl-max", type=float, default=0.1,
                    help="[edl] peak weight of the KL-to-uniform-Dirichlet regularizer")
    ap.add_argument("--kl-anneal-steps", type=int, default=None,
                    help="[edl] linearly anneal lambda_kl 0->kl_max over this many steps "
                         "(default max(1, steps//2))")
    ap.add_argument("--evidence-scale-init", type=float, default=10.0,
                    help="[edl] init for the Dirichlet evidence scale beta=exp(log_beta) "
                         "(log_beta=log(this)); matches the decoder out_scale magnitude")
    ap.add_argument("--density-threshold-init", type=float, default=None,
                    help="[edl] initial cosine-density threshold delta; default calibrates the "
                         "median density on deterministic training-fold holdout episodes")
    ap.add_argument("--density-slope-init", type=float, default=10.0,
                    help="[edl] initial positive slope gamma=exp(log_gamma)")
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--label-variants", type=int, default=16,
                    help="paraphrase variants per label sampled per episode (0 = fixed ensemble, "
                         "the old behaviour that overfit to seen label strings)")
    ap.add_argument("--val-frac-cfg", type=float, default=0.2, help="fraction of CONFIGS held out")
    ap.add_argument("--val-episodes", type=int, default=6)
    ap.add_argument("--val-queries", type=int, default=800)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("--steps must be >= 1")
    if args.kl_anneal_steps is None:
        args.kl_anneal_steps = max(1, args.steps // 2)
    lo, hi = args.episode_labels
    if lo < 2 or hi < lo:
        ap.error("--episode-labels requires 2 <= min <= max")
    if args.topk < 1:
        ap.error("--topk must be >= 1")
    if args.batch < 1 or args.val_queries < 1 or args.val_episodes < 1:
        ap.error("--batch, --val-queries, and --val-episodes must be >= 1")
    if args.val_every < 1 or args.warmup < 1:
        ap.error("--val-every and --warmup must be >= 1")
    if not 0.0 < args.val_frac_cfg < 1.0:
        ap.error("--val-frac-cfg must be in (0, 1)")
    if args.tau_retr <= 0:
        ap.error("--tau-retr must be positive")
    if args.kl_anneal_steps < 1:
        ap.error("--kl-anneal-steps must be >= 1")
    if args.evidence_scale_init <= 0 or args.density_slope_init <= 0:
        ap.error("--evidence-scale-init and --density-slope-init must be positive")
    if args.density_threshold_init is not None and not np.isfinite(args.density_threshold_init):
        ap.error("--density-threshold-init must be finite")
    if args.n_subspaces < 0 or (args.n_subspaces > 0 and args.subspace_dim < 1):
        ap.error("--n-subspaces must be >= 0 and --subspace-dim must be >= 1 when enabled")
    if min(args.lambda_delta, args.lambda_pool, args.lambda_ms, args.kl_max) < 0:
        ap.error("regularization weights and --kl-max must be non-negative")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    assert_bank_current(bank, context="train_decoder")
    Z = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y = bank["y"].to(device)
    subj = bank["subj"].to(device)
    cfg = bank["cfg"].to(device)
    vocab = list(bank["vocab"])
    n_vocab, d = len(vocab), Z.shape[1]
    sbert = get_sbert_encoder()
    # train_only=True: paraphrases come ONLY from training-dataset tables (FINDINGS §6 F1).
    t_ens = ensemble_text(vocab, sbert, args.ensemble, train_only=True).to(device)
    variants = None
    if args.label_variants > 0:
        variants = build_label_variants(vocab, sbert, args.label_variants,
                                        train_only=True).to(device)      # (L, K, 384)
        print(f"[dec] label augmentation: {args.label_variants} variants/label, resampled per "
              f"episode, evidence vs candidate sides drawn INDEPENDENTLY", flush=True)
    txt_gen = torch.Generator(device=device).manual_seed(SEED)
    print(f"[dec] bank {Z.shape[0]} windows d={d} · {n_vocab} vocab · {int(cfg.max()) + 1} configs "
          f"· backbone {bank['backbone']['git']} (val_ba {bank['backbone']['val_ba']:.3f})", flush=True)

    # Held-out-CONFIG **and** held-out-SUBJECT split for selection.
    #
    # This used to split on `cfg` ALONE, which is not a transfer measurement: with only the config
    # held out, 30 of 59 val subjects also appeared among the train queries. For pamap2 that meant
    # the watch_wrist stream was "held out" while the chest and hand streams of the SAME subjects
    # in the SAME sessions were trained on -- so the selection metric partly rewarded memorising
    # those people. A subject is now on exactly one side, whatever config it appears under.
    cfg_ids = np.arange(int(cfg.max()) + 1); rng.shuffle(cfg_ids)
    n_val = max(1, int(len(cfg_ids) * args.val_frac_cfg))
    is_val_cfg = torch.isin(cfg, torch.tensor(cfg_ids[:n_val], device=device))

    subj_ids = torch.unique(subj).cpu().numpy()
    perm = rng.permutation(subj_ids)
    n_val_subj = max(1, int(len(perm) * args.val_frac_cfg))
    val_subj = torch.tensor(perm[:n_val_subj], device=device)
    is_val_subj = torch.isin(subj, val_subj)

    # val = held-out config AND held-out subject; train = neither. Windows that are one but not the
    # other are DROPPED -- they would leak a val subject into training or vice versa.
    val_q = torch.nonzero(is_val_cfg & is_val_subj, as_tuple=True)[0]
    train_q = torch.nonzero(~is_val_cfg & ~is_val_subj, as_tuple=True)[0]
    n_dropped = len(Z) - len(val_q) - len(train_q)
    if len(val_q) == 0:
        raise SystemExit(
            "[dec] the config x subject holdout is empty -- no window has both a held-out config "
            "and a held-out subject. Raise --val-frac-cfg or check the bank's cfg/subj fields.")

    train_present = torch.unique(y[train_q])
    val_present = torch.unique(y[val_q])
    if len(train_present) < 3:
        raise SystemExit("[dec] training fold needs at least three labels for class-holdout episodes")
    if len(val_present) < 2:
        raise SystemExit("[dec] validation fold needs at least two labels for transfer episodes")
    overlap = len(set(val_present.tolist()) & set(train_present.tolist()))
    assert not (set(subj[train_q].tolist()) & set(subj[val_q].tolist())), \
        "subject appears in both folds -- the holdout is not subject-disjoint"
    print(f"[dec] queries: {len(train_q)} train / {len(val_q)} val ({n_dropped} dropped at the "
          f"config x subject boundary) · {n_val}/{len(cfg_ids)} configs and "
          f"{n_val_subj}/{len(perm)} subjects held out", flush=True)
    print(f"[dec] labels present: {len(train_present)} train / {len(val_present)} val · "
          f"{overlap}/{len(val_present)} val labels ALSO seen in train "
          f"({100 * overlap / max(1, len(val_present)):.0f}%) -- the selection metric is only as "
          f"open-vocabulary as this number is low (REMEDIATION_PLAN 5.2)", flush=True)

    def sample_H(present, *, reserve_labels=0):
        return sample_label_set(
            present, lo, hi, rng, reserve_labels=reserve_labels, device=device
        )

    def sample_queries(pool, H, n):
        """n query indices from `pool` whose label ∈ H, sqrt-balanced across H."""
        in_H = torch.isin(y[pool], H)
        cand = pool[in_H]
        counts = torch.bincount(y[cand], minlength=n_vocab).float().clamp(min=1)
        wts = (1.0 / counts.sqrt())[y[cand]]
        return cand[torch.multinomial(wts, min(n, len(cand)), replacement=len(cand) < n)]

    # FIXED val episodes (same H + queries every eval → a stable selection metric)
    val_eps = []
    for _ in range(args.val_episodes):
        H = sample_H(val_present)
        qi = sample_queries(val_q, H, args.val_queries)
        val_eps.append((H, qi))

    # Both training and held-out evaluation retrieve ONLY from the training fold. Previously
    # run_episode searched all bank rows, so the held-out-config selection metric retrieved mostly
    # from the held-out config and the dropped split boundary. H still removes the episode's target
    # labels from this pool, preserving the class-disjoint objective.
    train_memory_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    train_memory_mask[train_q] = True
    density_threshold_init = args.density_threshold_init
    if args.loss == "edl" and density_threshold_init is None:
        density_threshold_init = estimate_density_threshold(
            Z, y, subj, train_q, train_present, train_memory_mask,
            lo, hi, args.topk, seed=SEED + 1,
        )
        print(f"[dec] density threshold calibrated on training memory: "
              f"delta={density_threshold_init:.4f}", flush=True)

    @torch.no_grad()
    def evaluate():
        """Selection metric = transfer bAcc (argmax; identical for ce/edl). Under --loss edl also
        returns calibration metrics on the per-query uncertainty u = C/S. Returns (bAcc, metrics)."""
        dec.eval()
        if gate is not None:
            gate.eval()
        accs, us, corrects = [], [], []
        for H, qi in val_eps:
            # val always uses the fixed canonical/ensembled text -> a stable selection metric
            true = y[qi].cpu().numpy()
            if args.loss == "edl":
                (logits, aux), _w, vals = run_episode(dec, Z, y, subj, qi, H, t_ens, t_ens,
                                                       args.topk, args.tau_retr, return_aux=True,
                                                       memory_mask=train_memory_mask)
                alpha, _g = gate.alpha(aux["evidence"], vals)
                us.append((alpha.shape[1] / alpha.sum(1)).cpu().numpy())   # u = C / S per query
            else:
                logits = run_episode(dec, Z, y, subj, qi, H, t_ens, t_ens,
                                     args.topk, args.tau_retr,
                                     memory_mask=train_memory_mask)
            pred = H[logits.argmax(1)].cpu().numpy()
            corrects.append(pred == true)
            accs.append(balanced_accuracy(pred, true))
        ba = float(np.mean(accs))
        if args.loss != "edl":
            return ba, {}
        u_all = np.concatenate(us)
        c_all = np.concatenate(corrects).astype(bool)
        metrics = {
            # sanity: correct predictions should carry LOWER uncertainty than incorrect ones
            "mean_u_correct": float(u_all[c_all].mean()) if c_all.any() else float("nan"),
            "mean_u_incorrect": float(u_all[~c_all].mean()) if (~c_all).any() else float("nan"),
            "aurc": aurc(u_all, c_all),           # area under risk-coverage (u = abstention score); lower better
            "acc@0.8cov": acc_at_coverage(u_all, c_all, 0.8),
        }
        return ba, metrics

    dec = EvidenceDecoder(DecoderConfig(d_model=d, n_layers=args.layers, n_heads=args.heads,
                                        n_subspaces=args.n_subspaces,
                                        subspace_dim=args.subspace_dim)).to(device)
    if args.n_subspaces > 0:
        print(f"[dec] D2 multi-subspace re-weighting: K={args.n_subspaces} subspaces of dim "
              f"{args.subspace_dim}, gated by gamma_ms (0 @ init -> identity), lambda_ms={args.lambda_ms}",
              flush=True)
    # Density-gated evidential head (opt-in). Created AFTER dec so dec's RNG-consuming init is
    # byte-for-byte identical to the ce path; the gate's scalars are constants (no RNG). Its params
    # (log_gamma, delta, log_beta) join the optimizer so they train alongside the decoder.
    gate = None
    param_groups = dec.param_groups(weight_decay=0.01)
    if args.loss == "edl":
        gate = DensityGate(gate_delta=density_threshold_init,
                           log_gamma=math.log(args.density_slope_init),
                           log_beta=math.log(args.evidence_scale_init)).to(device)
        param_groups = param_groups + [{"params": list(gate.parameters()), "weight_decay": 0.0}]
        print(f"[dec] loss=edl · density-gated evidential (Dirichlet) · kl_max={args.kl_max} "
              f"annealed over {args.kl_anneal_steps} steps · beta_init={args.evidence_scale_init} "
              f"· density init delta={density_threshold_init}, "
              f"gamma={args.density_slope_init}", flush=True)
    opt = torch.optim.AdamW(param_groups, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + np.cos(np.pi * max(0, s - args.warmup) / max(1, args.steps - args.warmup)))))

    init_val, init_metrics = evaluate()
    print(f"[dec] init val transfer bAcc (decoder≡untrained) = {init_val:.4f} "
          f"(chance≈{1/len(val_eps[0][0]):.3f})", flush=True)
    if init_metrics:
        print(f"[dec] init calibration (edl): " + json.dumps(
            {k_: round(v, 4) for k_, v in init_metrics.items()}), flush=True)

    best_val = init_val
    best_calibration = dict(init_metrics)
    best_step = 0
    best_sd = {k_: v.detach().cpu().clone() for k_, v in dec.state_dict().items()}
    best_gate_sd = ({k_: v.detach().cpu().clone() for k_, v in gate.state_dict().items()}
                    if gate is not None else None)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        dec.train()
        if gate is not None:
            gate.train()
        H = sample_H(train_present, reserve_labels=1)
        qi = sample_queries(train_q, H, args.batch)
        pos = label_index(H, n_vocab, device)
        target = pos[y[qi]]
        t_ev, t_cand = (t_ens, t_ens) if variants is None else sample_text_tables(variants, txt_gen)
        (logits, aux), w, vals = run_episode(dec, Z, y, subj, qi, H, t_ev, t_cand,
                                             args.topk, args.tau_retr, return_aux=True,
                                             memory_mask=train_memory_mask)
        a = aux["pool_weights"]
        kl_pool = (w * (torch.log(w.clamp_min(1e-12)) - torch.log(a.clamp_min(1e-12)))).sum(1).mean()
        # reg-to-identity: UNCHANGED between ce and edl (KEEP the existing soft prior).
        reg = args.lambda_delta * aux["delta"].norm(dim=-1).mean() + args.lambda_pool * kl_pool
        # D2: pull the multi-subspace gate back toward identity (only present when the head is built).
        if "gamma_ms" in aux:
            reg = reg + args.lambda_ms * aux["gamma_ms"]
        if args.loss == "edl":
            lambda_kl = args.kl_max * min(1.0, step / args.kl_anneal_steps)
            alpha, _g = gate.alpha(aux["evidence"], vals)      # argmax_c alpha == argmax_c e (do-no-harm)
            L_edl, L_data_m, L_kl_m = edl_loss(alpha, target, lambda_kl)
            task_loss = L_edl
            task_tel = {"edl": round(float(L_edl.detach()), 4),
                        "edl_data": round(float(L_data_m.detach()), 4),
                        "edl_kl": round(float(L_kl_m.detach()), 4), "lambda_kl": round(lambda_kl, 4)}
        else:
            ce = F.cross_entropy(logits, target)
            task_loss = ce
            task_tel = {"ce": round(float(ce.detach()), 4)}
        loss = task_loss + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        do_log = step % args.val_every == 0 or step == 1
        gtel = {}
        if do_log:
            gn = lambda ps: float(sum(float(p.grad.pow(2).sum()) for p in ps if p.grad is not None) ** 0.5)
            gtel = {"grad/refiner": round(gn(dec.refiner.parameters()), 4),
                    "grad/pool": round(gn(dec.pool_phi.parameters()), 4),
                    "grad/blocks": round(gn(dec.blocks.parameters()), 4)}
            if gate is not None:
                gtel["grad/gate"] = round(gn(gate.parameters()), 4)
            if dec.ms_head is not None:
                gtel["grad/ms"] = round(gn(list(dec.ms_head.parameters()) + [dec.gamma_ms]), 4)
        clipped_params = list(dec.parameters()) + (list(gate.parameters()) if gate is not None else [])
        torch.nn.utils.clip_grad_norm_(clipped_params, 1.0)
        opt.step(); sched.step()

        if do_log:
            va, va_metrics = evaluate()
            # Transfer bAcc remains primary. For EDL, AURC breaks exact bAcc ties so the saved gate
            # is selected for uncertainty quality rather than being an arbitrary final-step state.
            better = va > best_val
            if (not better and gate is not None and abs(va - best_val) <= 1e-12
                    and va_metrics.get("aurc", float("inf"))
                    < best_calibration.get("aurc", float("inf"))):
                better = True
            if better:
                best_val = va
                best_step = step
                best_calibration = dict(va_metrics)
                best_sd = {k_: v.detach().cpu().clone() for k_, v in dec.state_dict().items()}
                if gate is not None:
                    best_gate_sd = {
                        k_: v.detach().cpu().clone() for k_, v in gate.state_dict().items()
                    }
            eff_k = float((1.0 / a.detach().pow(2).sum(1).clamp(min=1e-12)).mean())
            cal = {k_: round(v, 4) for k_, v in va_metrics.items()}
            n_memory_labels = int(torch.unique(y[train_q][~torch.isin(y[train_q], H)]).numel())
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 4), **task_tel,
                              "reg": round(float(reg.detach()), 5),
                              "kl_pool": round(float(kl_pool.detach()), 5),
                              "val_transfer_ba": round(va, 4), "best": round(best_val, 4),
                              "n_heldout_labels": len(H), "n_memory_labels": n_memory_labels,
                              "delta_norm": round(aux["delta_norm"], 4),
                              "eff_k": round(eff_k, 1), **cal, **gtel,
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    dec.load_state_dict(best_sd)
    if gate is not None and best_gate_sd is not None:
        gate.load_state_dict(best_gate_sd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"decoder": {k_: v.cpu() for k_, v in dec.state_dict().items()},
               "cfg": {"d_model": d, "n_layers": args.layers, "n_heads": args.heads,
                       "n_subspaces": args.n_subspaces, "subspace_dim": args.subspace_dim},
               "topk": args.topk, "tau_retr": args.tau_retr, "ensemble": args.ensemble,
               "vocab": vocab, "best_val_transfer_ba": best_val, "init_val_transfer_ba": init_val,
               "best_step": best_step, "best_calibration": best_calibration,
               "loss": args.loss, "bank": str(args.bank), "backbone": bank["backbone"],
               "episode_labels": [lo, hi]}
    if gate is not None:
        payload["gate"] = {k_: v.cpu() for k_, v in gate.state_dict().items()}
        payload["gate_cfg"] = {
            "density_threshold_init": density_threshold_init,
            "density_slope_init": args.density_slope_init,
            "evidence_scale_init": args.evidence_scale_init,
            "kl_max": args.kl_max,
            "kl_anneal_steps": args.kl_anneal_steps,
        }
        payload["init_calibration"] = init_metrics
    torch.save(payload, str(args.out))
    print(f"[dec] done: init {init_val:.4f} -> best held-out transfer bAcc {best_val:.4f} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
