"""Protocol fingerprint — makes a mixed-protocol table impossible to assemble by accident.

**The failure this prevents.** Result JSONs used to record only
``_baseline/_dataset/_stream/_alignment/_status/metrics``. Nothing identified *which protocol*
produced them — no vocabulary hash, no split-manifest hash, no code revision. So the 59-label
harnet results sat in ``eval/results/`` looking exactly like post-fix ones, and
``assemble_table`` reproduced the old 47.3 mean without a word of warning. Add one post-fix row
next to them and the published table silently mixes two protocols.

Two things changed underneath every cached head and every score in 2026-07:

  * the global label vocabulary (59 → 93 labels, 11.48% of training windows recovered);
  * the subject split manifest (per-dataset stratification; then ``hapt`` aliasing, 452 → 482).

Either one invalidates a result. The fingerprint below covers both, plus a manual
``PROTOCOL_VERSION`` for changes that are neither (a preprocessing fix, a metric change).

Usage: ``run_baselines`` stamps ``_protocol`` into every cell it writes; ``assemble_table``
refuses cells whose stamp is absent or differs from the current one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

# Bump MANUALLY when something changes that the vocabulary and split hashes do not capture:
# a preprocessing contract, a metric definition, the ConSE bridge, the probe architecture.
PROTOCOL_VERSION = 3        # 3 = 93-label vocab + aliased-cohort split + seeded 2-layer probe


def protocol_fingerprint() -> dict:
    """The identity of the *scoring protocol*, as a small JSON-able dict.

    Cheap enough to call per cell. Deliberately NOT a single opaque hash: when a mismatch is
    reported you want to see which component moved.
    """
    from eval.data import load_global_labels
    from eval.splits import manifest_fingerprint

    vocab = list(load_global_labels())
    return {
        "version": PROTOCOL_VERSION,
        "n_labels": len(vocab),
        "vocab_fp": hashlib.sha256(json.dumps(vocab).encode()).hexdigest()[:16],
        "split_fp": manifest_fingerprint(),
    }


def protocol_mismatch(stamped: Optional[dict], current: Optional[dict] = None) -> Optional[str]:
    """Return a human-readable reason ``stamped`` is not the current protocol, else ``None``."""
    cur = current or protocol_fingerprint()
    if not stamped:
        return (f"no _protocol stamp (pre-{PROTOCOL_VERSION} result — it predates protocol "
                f"stamping, so it cannot be shown to match the current "
                f"{cur['n_labels']}-label protocol)")
    diffs = [f"{k}: {stamped.get(k)!r} != {cur[k]!r}" for k in cur if stamped.get(k) != cur[k]]
    return "protocol mismatch — " + "; ".join(diffs) if diffs else None
