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


def assert_bank_matches_backbone(bank: dict, ckpt: dict, *, context: str = "") -> None:
    """Verify the bank was built over the SAME Phase-A corpus as the encoder checkpoint (F3).

    A matching vocabulary + backbone fingerprint still is not enough: a bank encoded over a
    different dataset roster / cap than the encoder was trained on produces retrieval vectors that
    do not correspond to what the encoder learned. We compare the bank's recorded
    ``corpus.phase_a_corpus_fp`` against the checkpoint's ``corpus_fingerprint``. Skips silently only
    when BOTH predate corpus fingerprints (both None); raises if present and different.
    """
    bank_fp = (bank.get("corpus") or {}).get("phase_a_corpus_fp")
    ckpt_fp = ckpt.get("corpus_fingerprint")
    if bank_fp is None and ckpt_fp is None:
        return
    if bank_fp != ckpt_fp:
        where = f" ({context})" if context else ""
        raise SystemExit(
            f"\n[bank_guard] BANK/ENCODER CORPUS MISMATCH{where} — refusing an unattributable result.\n"
            f"  bank built over Phase-A corpus fp : {bank_fp!r}\n"
            f"  encoder checkpoint corpus fp      : {ckpt_fp!r}\n"
            f"  The retrieval vectors and the encoder disagree on which corpus they represent.\n"
            f"  Rebuild the bank against this checkpoint:\n\n"
            f"      python -m training.evidence.build_memory --checkpoint <this-ckpt> --device cuda\n")
