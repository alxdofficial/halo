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


def bank_fingerprint(bank: dict) -> str:
    """Identity of the bank an evidence artifact was trained against.

    Hash metadata that determines the vector population plus the fixed embedding-path probe. The
    full ``Z`` tensor is intentionally not re-hashed on every consumer startup; the corpus,
    checkpoint, caps and behavioral probe are the reproducible identity of a deterministic build.
    """
    probe = bank.get("embed_probe")
    if probe is None:
        probe_fp = None
    else:
        import torch
        tensor = torch.as_tensor(probe).detach().cpu().contiguous()
        probe_fp = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    corpus = bank.get("corpus") or {}
    payload = {
        "vocab_fp": bank.get("vocab_fp") or vocab_fingerprint(bank.get("vocab", [])),
        "backbone_fp": (bank.get("backbone") or {}).get("fingerprint"),
        "phase_a_corpus_fp": corpus.get("phase_a_corpus_fp"),
        "datasets": sorted(corpus.get("datasets") or []),
        "streams": sorted((corpus.get("streams") or {}).items()),
        "n_encoded_windows": corpus.get("n_encoded_windows"),
        "max_per_stream": bank.get("max_per_stream"),
        "max_per_label": bank.get("max_per_label"),
        "d_model": bank.get("d_model"),
        "embed_probe_fp": probe_fp,
    }
    if "balance_policy" in bank:
        payload["balance_policy"] = bank["balance_policy"]
    # Schema-v2 extends (rather than replaces) the historical pooled bank with a patch table. Keep
    # the exact v1 fingerprint payload for existing banks/artifacts; otherwise merely upgrading
    # this guard would invalidate a valid old pooled control.
    if int(bank.get("schema_version", 1)) >= 2:
        patch = bank.get("patch") or {}
        patch_probe = bank.get("patch_embed_probe")
        if patch_probe is None:
            patch_probe_fp = None
        else:
            import torch
            tensor = torch.as_tensor(patch_probe).detach().cpu().contiguous()
            patch_probe_fp = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        payload.update({
            "schema_version": int(bank["schema_version"]),
            "n_patches": len(patch.get("Z", [])),
            "patch_fields": sorted(patch),
            "n_events": len(bank.get("event_names", {})),
            "patch_embed_probe_fp": patch_probe_fp,
            "cfg_rate_hz": sorted((bank.get("cfg_rate_hz") or {}).items()),
        })
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def assert_patch_bank(bank: dict, *, context: str = "") -> None:
    """Validate the schema-v2 patch evidence table and its foreign-key invariants."""
    import torch

    where = f" ({context})" if context else ""
    if int(bank.get("schema_version", 1)) < 2 or "patch" not in bank:
        raise SystemExit(
            f"\n[bank_guard] PATCH BANK REQUIRED{where}: this bank only contains the legacy "
            "pooled evidence table. Rebuild it with the current build_memory.\n"
        )
    if "patch_embed_probe" not in bank:
        raise SystemExit(
            f"\n[bank_guard] PATCH BANK LACKS EMBEDDING-PATH PROVENANCE{where}. "
            "Rebuild it with the current build_memory.\n"
        )
    if "cfg_rate_hz" not in bank:
        raise SystemExit(
            f"\n[bank_guard] PATCH BANK LACKS CONFIG RATE METADATA{where}. Rebuild it.\n"
        )
    patch = bank["patch"]
    required = {
        "Z", "y", "subj", "cfg", "sensor", "window", "event", "event_verified",
        "time", "duration", "resolution",
    }
    missing = sorted(required - set(patch))
    if missing:
        raise SystemExit(f"\n[bank_guard] MALFORMED PATCH BANK{where}: missing {missing}.\n")
    n = len(patch["Z"])
    bad_lengths = {key: len(patch[key]) for key in required if len(patch[key]) != n}
    if bad_lengths:
        raise SystemExit(
            f"\n[bank_guard] MALFORMED PATCH BANK{where}: expected {n} rows, "
            f"mismatched fields {bad_lengths}.\n"
        )
    if n == 0:
        raise SystemExit(f"\n[bank_guard] EMPTY PATCH BANK{where}.\n")
    window = torch.as_tensor(patch["window"])
    if window.dtype not in (torch.int32, torch.int64) or int(window.min()) < 0 \
            or int(window.max()) >= len(bank["Z"]):
        raise SystemExit(
            f"\n[bank_guard] MALFORMED PATCH BANK{where}: patch.window contains an invalid "
            "pooled-window foreign key.\n"
        )
    for key in ("y", "subj", "cfg", "event", "event_verified"):
        expected = torch.as_tensor(bank[key])[window]
        actual = torch.as_tensor(patch[key])
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise SystemExit(
                f"\n[bank_guard] MALFORMED PATCH BANK{where}: patch.{key} disagrees with its "
                "parent pooled window.\n"
            )
    sensor = torch.as_tensor(patch["sensor"])
    if sensor.dtype not in (torch.int32, torch.int64) or int(sensor.min()) < 0:
        raise SystemExit(
            f"\n[bank_guard] MALFORMED PATCH BANK{where}: patch.sensor must contain "
            "nonnegative integer structural sensor ids.\n"
        )
    duration = torch.as_tensor(patch["duration"])
    time = torch.as_tensor(patch["time"])
    if not bool(torch.isfinite(duration).all() and torch.isfinite(time).all()) \
            or not bool((duration > 0).all()) or not bool((time >= 0).all()):
        raise SystemExit(
            f"\n[bank_guard] MALFORMED PATCH BANK{where}: non-finite/non-physical time metadata.\n"
        )


def assert_artifact_matches_bank(artifact: dict, bank: dict, *, context: str,
                                 artifact_name: str) -> None:
    """Fail if a learned evidence artifact was trained against a different bank."""
    expected = bank.get("bank_fp") or bank_fingerprint(bank)
    recorded = artifact.get("bank_fp")
    if recorded != expected:
        raise SystemExit(
            f"\n[bank_guard] {artifact_name.upper()} / BANK MISMATCH ({context}).\n"
            f"  artifact bank fp : {recorded!r}\n"
            f"  current bank fp  : {expected!r}\n"
            f"  The artifact was not trained against this exact vector population. Re-train "
            f"{artifact_name} against the current bank before evaluating it.\n"
        )
    if list(artifact.get("vocab", [])) != list(bank.get("vocab", [])):
        raise SystemExit(
            f"\n[bank_guard] {artifact_name.upper()} VOCABULARY MISMATCH ({context}): "
            f"{len(artifact.get('vocab', []))} vs {len(bank.get('vocab', []))} labels.\n"
        )


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
    missing = [
        k for k in ("vocab_fp", "backbone", "corpus", "embed_probe", "bank_fp")
        if k not in bank
    ]
    if "backbone" in bank and not bank["backbone"].get("fingerprint"):
        missing.append("backbone.fingerprint")
    if "corpus" in bank and not bank["corpus"].get("streams"):
        missing.append("corpus.streams")               # which streams the bank was encoded over (F3)
    if not missing:
        calculated = bank_fingerprint(bank)
        if bank["bank_fp"] != calculated:
            raise SystemExit(
                f"\n[bank_guard] BANK FINGERPRINT MISMATCH ({context}): stored "
                f"{bank['bank_fp']!r} != calculated {calculated!r}. The bank metadata was modified "
                "after it was built; rebuild it before evaluation.\n"
            )
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


def assert_patch_embedding_path_current(bank: dict, enc, device, *, context: str = "") -> None:
    """Raise if live per-patch export no longer reproduces the schema-v2 bank."""
    import torch
    from training.tokenizer.eval_transfer import patch_embedding_fingerprint

    where = f" ({context})" if context else ""
    stored = bank.get("patch_embed_probe")
    if stored is None:
        raise SystemExit(
            f"\n[bank_guard] PATCH BANK PREDATES THE PATCH-PATH PROBE{where}. Rebuild it.\n"
        )
    live = patch_embedding_fingerprint(enc, device)
    stored = torch.as_tensor(stored).float().cpu()
    if live.shape != stored.shape:
        raise SystemExit(
            f"\n[bank_guard] PATCH EMBEDDING-PATH MISMATCH{where}: probe shape "
            f"{tuple(live.shape)} != stored {tuple(stored.shape)}. Rebuild the bank.\n"
        )
    drift = float((live - stored).norm() / stored.norm().clamp(min=1e-12))
    if drift > EMBED_PROBE_TOL:
        raise SystemExit(
            f"\n[bank_guard] PATCH EMBEDDING-PATH CHANGED{where}: relative drift "
            f"{drift:.2e} exceeds {EMBED_PROBE_TOL:.0e}. Rebuild the bank.\n"
        )


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
