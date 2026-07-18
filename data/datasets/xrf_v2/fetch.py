"""Fetch the XRF V2 "Plus" (16-volunteer, 853-sequence) IMU + AirPods streams from Kaggle.

Source: kaggle.com/datasets/airslab2020/xrfv2-multimodal-tal-caption-qa-no-rgb (video-aligned
release of XRF V2; arXiv 2501.19034). We pull ONLY the small IMU + AirPods h5 and the TAL
label/manifest files (~340 MB); the wifi CSI (4.9 GB), densepose (40 GB), pose/mesh/video
modalities are not used by HALO. Idempotent: existing files are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
from kaggle.api.kaggle_api_extended import KaggleApi

REF = "airslab2020/xrfv2-multimodal-tal-caption-qa-no-rgb"
HERE = Path(__file__).resolve().parent
OUT = HERE / "downloads"
_TAL = "annotations/annotations/tal_annotations_video_aligned"
WANT = [
    "imu_50hz_853_video_aligned.h5",
    "airpods_25hz_853_video_aligned.h5",
    f"{_TAL}/activitynet_tal_853.json",
    f"{_TAL}/label_map.json",
    f"{_TAL}/tal_sample_manifest.csv",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate()
    for fn in WANT:
        dest = OUT / Path(fn).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {dest.name} (exists)"); continue
        api.dataset_download_file(REF, fn, path=str(OUT), force=True)
        z = OUT / (Path(fn).name + ".zip")
        if z.exists():
            import zipfile
            with zipfile.ZipFile(z) as zf:
                zf.extractall(OUT)
            z.unlink()
        print(f"got {dest.name}")
    print("fetch complete")


if __name__ == "__main__":
    main()
