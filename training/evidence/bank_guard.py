"""Guard against MIXED-PROTOCOL results (REMEDIATION_PLAN Phase 0.2/0.3).

The memory bank is built against whatever global label vocabulary existed at build time and stores
its own copy. Every evidence-engine consumer then reads `bank["vocab"]`. If the global vocabulary
is regenerated (e.g. the 59 → 93 fix) but the bank is NOT rebuilt, the repo silently enters a state
where:

  * every ConSE head auto-refits at the NEW vocabulary (they compare their cached `labels`), while
  * the evidence engine keeps using the OLD, truncated bank vocabulary,

and a results table produced in that state blends two protocols with nothing to warn you. That is
exactly the failure this module prevents: consumers call :func:`assert_bank_current` at startup and
fail loudly with an actionable message instead of producing a quietly-wrong number.

Cheap by construction — a hash comparison, no data touched.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence


def vocab_fingerprint(labels: Sequence[str]) -> str:
    """Order-sensitive hash of a label vocabulary (order defines the label indices)."""
    return hashlib.sha256(json.dumps(list(labels)).encode()).hexdigest()[:16]


def _assert_build_params_recorded(bank: dict, context: str) -> None:
    """A matching vocabulary is NOT sufficient to call a bank 'current'.

    The guard originally compared vocabularies only, while its own name and docstring promised
    protection against "mixed protocols". Two banks can share a vocabulary and still be
    incomparable — different backbone checkpoint, different ``--max-per-label`` cap, different
    frontend. Those change every retrieval number the bank produces.

    We cannot retro-validate a bank that never recorded its build parameters, so the minimum
    enforceable guarantee is that they ARE recorded; consumers that know the expected backbone
    (eval_decoder, halo_evidence) additionally compare the checkpoint hash themselves.
    """
    missing = [k for k in ("vocab_fp", "backbone", "corpus") if k not in bank]
    if "backbone" in bank and not bank["backbone"].get("fingerprint"):
        missing.append("backbone.fingerprint")
    if "corpus" in bank and not bank["corpus"].get("streams"):
        missing.append("corpus.streams")               # which streams the bank was encoded over (F3)
    if not missing:
        return
    where = f" ({context})" if context else ""
    raise SystemExit(
        f"\n[bank_guard] BANK LACKS PROVENANCE{where}: missing {missing}.\n"
        f"  Its vocabulary matches, but without a backbone fingerprint there is no way to tell\n"
        f"  which encoder produced these vectors, so a retrieval number from it is unattributable.\n"
        f"  Rebuild with the current build_memory (which records them):\n\n"
        f"      HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \\\n"
        f"        python -m training.evidence.build_memory --device cuda\n")


def assert_bank_current(bank: dict, *, context: str = "") -> None:
    """Raise if the bank's vocabulary differs from the CURRENT global vocabulary.

    `bank` is the loaded memory-bank dict. Raises SystemExit with a rebuild instruction.
    """
    from eval.data import load_global_labels

    current = list(load_global_labels())
    stored = list(bank.get("vocab", []))
    if stored == current:
        _assert_build_params_recorded(bank, context)
        return

    cur_fp, old_fp = vocab_fingerprint(current), vocab_fingerprint(stored)
    missing = sorted(set(current) - set(stored))
    extra = sorted(set(stored) - set(current))
    where = f" ({context})" if context else ""
    raise SystemExit(
        f"\n[bank_guard] STALE MEMORY BANK{where} — refusing to produce a mixed-protocol result.\n"
        f"  bank vocabulary   : {len(stored):>4} labels (fp {old_fp})\n"
        f"  global vocabulary : {len(current):>4} labels (fp {cur_fp})\n"
        f"  in global but NOT in bank ({len(missing)}): {missing[:12]}{' ...' if len(missing) > 12 else ''}\n"
        f"  in bank but NOT in global ({len(extra)}): {extra[:12]}{' ...' if len(extra) > 12 else ''}\n"
        f"\n  The bank was built under a different vocabulary, so its windows were filtered by a\n"
        f"  different label set than the ConSE heads now use. Rebuild it:\n\n"
        f"      HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \\\n"
        f"        python -m training.evidence.build_memory --device cuda\n\n"
        f"  See docs/design/REMEDIATION_PLAN.md Phase 0.2 / Phase 2.\n")


#: Max relative L2 drift tolerated between the stored and live embedding probe. Well above CPU/GPU
#: float noise (~1e-6) and far below the smallest real behavioural change we have measured
#: (the F1 pooling fix moved real windows by 3e-3 .. 2e-2).
EMBED_PROBE_TOL = 1e-4


def assert_embedding_path_current(bank: dict, enc, device, *, context: str = "") -> None:
    """Raise if the live encode CODE no longer reproduces the bank's stored embedding probe.

    Vocabulary, backbone-weight and corpus fingerprints are all blind to a change in the *function*
    that maps weights+data to a vector. The F1 duration-weighted pooling fix is exactly that: it
    left every other fingerprint identical while moving real embeddings by 0.3-2%. A bank built
    before it, used after it, is a silent mixed-protocol result. Re-running the fixed probe catches
    any such change without anyone remembering to bump a version.
    """
    stored = bank.get("embed_probe")
    where = f" ({context})" if context else ""
    if stored is None:
        raise SystemExit(
            f"\n[bank_guard] BANK PREDATES THE EMBEDDING-PATH PROBE{where}.\n"
            f"  It cannot be shown to have been built by the CURRENT encode path (pooling / tail\n"
            f"  selection / text construction), and those have changed. Rebuild it:\n\n"
            f"      python -m training.evidence.build_memory --checkpoint <ckpt> --device cuda\n")
    import torch
    from training.tokenizer.eval_transfer import embedding_fingerprint
    live = embedding_fingerprint(enc, device)
    stored = torch.as_tensor(stored).float().cpu()
    if live.shape != stored.shape:
        raise SystemExit(
            f"\n[bank_guard] EMBEDDING-PATH MISMATCH{where}: probe shape {tuple(live.shape)} != "
            f"stored {tuple(stored.shape)} — the encoder/embedding path changed. Rebuild the bank.\n")
    drift = float((live - stored).norm() / stored.norm().clamp(min=1e-12))
    if drift > EMBED_PROBE_TOL:
        raise SystemExit(
            f"\n[bank_guard] EMBEDDING-PATH CHANGED{where} — refusing a mixed-protocol result.\n"
            f"  relative drift of the fixed probe: {drift:.2e}  (tolerance {EMBED_PROBE_TOL:.0e})\n"
            f"  The weights and corpus match, but the CODE that turns them into an embedding does\n"
            f"  not reproduce the vectors stored in this bank (e.g. pooling or tail handling\n"
            f"  changed since it was built). Rebuild the bank against the current code:\n\n"
            f"      python -m training.evidence.build_memory --checkpoint <ckpt> --device cuda\n")


def assert_bank_matches_backbone(bank: dict, ckpt: dict, *, context: str = "") -> None:
    """Verify the bank was built over the SAME Phase-A corpus as the encoder checkpoint (F3).

    A matching vocabulary + backbone fingerprint still is not enough: a bank encoded over a
    different dataset roster / cap than the encoder was trained on produces retrieval vectors that
    do not correspond to what the encoder learned. We compare the bank's recorded
    ``corpus.phase_a_corpus_fp`` against the checkpoint's ``corpus_fingerprint``. Skips silently only
    when BOTH predate corpus fingerprints (both None); raises if present and different.
    """
    where = f" ({context})" if context else ""
    corpus = bank.get("corpus") or {}
    bank_fp = corpus.get("phase_a_corpus_fp")
    ckpt_fp = ckpt.get("corpus_fingerprint")
    # 1) Phase-A corpus fingerprint (skip only when BOTH predate corpus fingerprints).
    if not (bank_fp is None and ckpt_fp is None) and bank_fp != ckpt_fp:
        raise SystemExit(
            f"\n[bank_guard] BANK/ENCODER CORPUS MISMATCH{where} — refusing an unattributable result.\n"
            f"  bank built over Phase-A corpus fp : {bank_fp!r}\n"
            f"  encoder checkpoint corpus fp      : {ckpt_fp!r}\n"
            f"  The retrieval vectors and the encoder disagree on which corpus they represent.\n"
            f"  Rebuild the bank against this checkpoint:\n\n"
            f"      python -m training.evidence.build_memory --checkpoint <this-ckpt> --device cuda\n")
    # 2) A COPIED fingerprint is not proof — compare the ACTUAL dataset rosters (F4). The bank's
    #    encoded datasets must equal the roster the encoder was trained on (build_memory now builds
    #    from the checkpoint roster, so a legit bank always matches).
    bank_ds = set(corpus.get("datasets") or [])
    roster = ckpt.get("config", {}).get("train_datasets")
    if roster is None:
        from training.tokenizer.pretrain_data import TRAIN_DATASETS
        roster = TRAIN_DATASETS
    ckpt_ds = set(roster)
    if bank_ds and ckpt_ds and bank_ds != ckpt_ds:
        raise SystemExit(
            f"\n[bank_guard] BANK/ENCODER ROSTER MISMATCH{where} — refusing an unattributable result.\n"
            f"  bank encoded datasets : {sorted(bank_ds)}\n"
            f"  encoder train roster  : {sorted(ckpt_ds)}\n"
            f"  in bank not encoder: {sorted(bank_ds - ckpt_ds)}; in encoder not bank: {sorted(ckpt_ds - bank_ds)}\n"
            f"  Rebuild the bank against this checkpoint (build_memory builds from its roster).\n")
