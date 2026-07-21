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


def assert_bank_current(bank: dict, *, context: str = "") -> None:
    """Raise if the bank's vocabulary differs from the CURRENT global vocabulary.

    `bank` is the loaded memory-bank dict. Raises SystemExit with a rebuild instruction.
    """
    from eval.data import load_global_labels

    current = list(load_global_labels())
    stored = list(bank.get("vocab", []))
    if stored == current:
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
