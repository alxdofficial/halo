"""Shared label-text tooling for the evidence engine — the single source of truth for
paraphrase ensembling and (optional) fine-grained descriptions.

Used by the T2.0 retrieval adapter (`baselines/halo_evidence`), the tier-1 sweep, and the
decoder trainer, so every code path anchors labels in the *same* SBERT text the way the
47.5 untrained mechanism did (raw canonical name + template/synonym paraphrases averaged).

- ``global_label_paraphrases`` merges every per-dataset synonym/template table into one
  dataset-agnostic pool. This is load-bearing: ``augment_label(label, dataset_name="")``
  falls into a generic fallback that ignores the synonym tables, so the global vocab labels
  (owned by no single dataset) would otherwise collapse to near-duplicate template wraps.
- ``ensemble_text`` returns the (L, 384) L2-normalized mean SBERT anchor per label. Variant
  0 is always the canonical name (so E=1 reproduces the plain-ConSE text); higher E adds
  paraphrase variants; a fine-grained **description** (T2.1), when present, is appended as an
  extra anchor. The description file is optional — absent ⇒ this is a no-op and the mechanism
  reproduces the exact 47.5 tier-1 config.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
_DESCRIPTIONS = _REPO / "data/labels/label_descriptions.json"
SBERT_DIM = 384


@lru_cache(maxsize=1)
def global_label_paraphrases():
    """(synonyms, templates): merged pool over every dataset's augmentation config.

    ``synonyms``: ``{label -> [surface forms]}``; ``templates``: ``["{}", "a person {}", ...]``.
    """
    from data.scripts.labels.label_augmentation import DATASET_CONFIGS
    synonyms: dict[str, set] = {}
    templates: set = set()
    for cfg in DATASET_CONFIGS.values():
        for lab, forms in cfg.get("synonyms", {}).items():
            synonyms.setdefault(lab, set()).update(forms)
        templates.update(cfg.get("templates", []))
    return ({k: sorted(v) for k, v in synonyms.items()}, sorted(templates) or ["{}"])


@lru_cache(maxsize=1)
def load_descriptions() -> dict:
    """Optional ``{label -> fine-grained description}`` (T2.1). ``{}`` if the file is absent."""
    if _DESCRIPTIONS.exists():
        return dict(json.loads(_DESCRIPTIONS.read_text()))
    return {}


def label_variant_rows(labels, E: int, seed: int = 0, use_descriptions: bool = True):
    """List of E (+1 if descriptions) rows, each a list of surface strings per label.

    Row 0 = canonical names (``label.replace("_", " ")``). Rows 1..E-1 = ``template.format(synonym)``
    sampled deterministically. A trailing description row is appended for any label that has one.
    """
    syn, templates = global_label_paraphrases()
    descs = load_descriptions() if use_descriptions else {}
    rng = random.Random(seed)
    rows = []
    for e in range(max(1, E)):
        if e == 0:
            rows.append([l.replace("_", " ") for l in labels])
        else:
            rows.append([
                templates[rng.randrange(len(templates))].format(
                    rng.choice(syn.get(l, [l.replace("_", " ")])))
                for l in labels
            ])
    if descs:
        rows.append([descs.get(l, l.replace("_", " ")) for l in labels])
    return rows


def ensemble_text(labels, sbert, E: int = 8, seed: int = 0,
                  use_descriptions: bool = True) -> torch.Tensor:
    """(L, 384) L2-normalized mean SBERT over the paraphrase (+ description) variants."""
    rows = label_variant_rows(labels, E, seed, use_descriptions)
    acc = np.zeros((len(labels), SBERT_DIM), dtype=np.float32)
    for r in rows:
        acc += sbert(r).astype(np.float32)
    v = torch.from_numpy(acc / len(rows))
    return F.normalize(v, dim=-1)
