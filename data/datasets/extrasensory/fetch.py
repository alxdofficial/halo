"""Fetch the public ExtraSensory raw accelerometer archives.

The upstream server exposes stable, byte-addressable zip files.  Downloads are
large, so this script validates Content-Length, writes through a temporary file,
and supports importing the already-downloaded archives from another local tree
without duplicating bytes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path


DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"

FILES = {
    "raw_acc.zip": (
        "http://extrasensory.ucsd.edu/data/raw_measurements/"
        "ExtraSensory.raw_measurements.raw_acc.zip",
        6_475_399_720,
    ),
    "watch_acc.zip": (
        "http://extrasensory.ucsd.edu/data/raw_measurements/"
        "ExtraSensory.raw_measurements.watch_acc.zip",
        837_898_335,
    ),
    "labels.zip": (
        "http://extrasensory.ucsd.edu/data/primary_data_files/"
        "ExtraSensory.per_uuid_features_labels.zip",
        225_374_973,
    ),
    "cv5Folds.zip": (
        "http://extrasensory.ucsd.edu/data/cv5Folds.zip",
        10_803,
    ),
}


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _download(url: str, destination: Path, expected_size: int) -> None:
    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"[extrasensory] present: {destination.name} ({expected_size:,} bytes)")
        return
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "HALO-dataset-fetch/1.0"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as output:
        remote_size = int(response.headers.get("Content-Length", expected_size))
        if remote_size != expected_size:
            raise RuntimeError(
                f"{url}: upstream size changed: expected {expected_size}, got {remote_size}"
            )
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"{url}: incomplete download: expected {expected_size}, got {tmp.stat().st_size}"
        )
    os.replace(tmp, destination)
    print(f"[extrasensory] downloaded: {destination.name} ({expected_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--import-from",
        type=Path,
        help="directory containing raw_acc.zip, watch_acc.zip, and labels.zip; link them",
    )
    args = parser.parse_args()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    if args.import_from:
        for name, (_, expected_size) in FILES.items():
            source = args.import_from / name
            if source.exists() and source.stat().st_size == expected_size:
                _link_or_copy(source, DOWNLOADS / name)
                print(f"[extrasensory] imported: {name} -> {source.resolve()}")
                continue
            if name == "cv5Folds.zip":
                url, _ = FILES[name]
                _download(url, DOWNLOADS / name, expected_size)
                continue
            if not source.exists() or source.stat().st_size != expected_size:
                raise FileNotFoundError(
                    f"{source}: missing or wrong size (expected {expected_size:,} bytes)"
                )
        return

    for name, (url, expected_size) in FILES.items():
        _download(url, DOWNLOADS / name, expected_size)


if __name__ == "__main__":
    main()
