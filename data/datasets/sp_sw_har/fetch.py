"""Fetch SP-SW-HAR from its GitHub/Zenodo repository.

Source: github.com/GeoTecINIT/sp-sw-har-dataset (CC-BY-4.0; Zenodo-archived).
Clones the repo into downloads/ so convert.py can read downloads/DATA/s*/*.csv. Pinned to a tag
for reproducibility. Idempotent: skips if downloads/DATA already exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = "https://github.com/GeoTecINIT/sp-sw-har-dataset"
REF = "main"               # pin to a release tag / commit SHA for a frozen snapshot
HERE = Path(__file__).resolve().parent
OUT = HERE / "downloads"


def main() -> None:
    if (OUT / "DATA").is_dir():
        print(f"sp_sw_har: downloads/DATA already present at {OUT}"); return
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, REPO, str(OUT)], check=True)
    n = len(list((OUT / "DATA").rglob("s*_s[pw].csv")))
    print(f"sp_sw_har: cloned {n} device CSVs into {OUT}")


if __name__ == "__main__":
    main()
