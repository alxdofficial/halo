"""Fetch NFI-FARED (IMU subset) from Hugging Face.

Source: huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED_IMU
Reproducibly downloads the per-subject `*final*.csv` protocol runs (the labelled files the
converter reads) into downloads/. Pinned to a revision for reproducibility; bump REVISION to
re-pull. Idempotent (snapshot_download reuses the local cache).
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "NetherlandsForensicInstitute/NFI_FARED_IMU"
REVISION = "main"          # pin to a commit SHA for a fully frozen snapshot
HERE = Path(__file__).resolve().parent
OUT = HERE / "downloads"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                      local_dir=str(OUT), allow_patterns=["*final*.csv", "**/*final*.csv"])
    n = len(list(OUT.rglob("*final*.csv")))
    print(f"nfi_fared: fetched {n} labelled protocol CSVs into {OUT}")


if __name__ == "__main__":
    main()
