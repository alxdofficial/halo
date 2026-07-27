"""Download and verify the official H-MOG dataset archive.

H-MOG is licensed for non-commercial educational and research use. The
official terms prohibit redistribution and re-identification. Pass
``--accept-license`` before this script performs a new download; verification
of an archive already present on disk does not require the flag.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
ARCHIVE = DOWNLOADS / "hmog_dataset.zip"
OFFICIAL_PAGE = "https://hmog-dataset.github.io/hmog/"
DOWNLOAD_URL = (
    "https://wm1693.app.box.com/index.php"
    "?rm=box_download_shared_file"
    "&shared_name=jkjodaw2scunua7b7qaxt3ker5w9lwt7"
    "&file_id=f_770692248712"
)
EXPECTED_SIZE = 6_132_356_276
EXPECTED_MD5 = "4fd46756abec13815f426c66a58e626f"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path = ARCHIVE) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_SIZE:,} bytes, found {size:,}"
        )
    actual_md5 = _md5(path)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(
            f"{path}: MD5 mismatch, expected {EXPECTED_MD5}, found {actual_md5}"
        )
    print(f"[hmog] verified {path} ({size:,} bytes, MD5 {actual_md5})")


def download(path: Path = ARCHIVE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify(path)
        return

    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "HALO-dataset-fetch/1.0"},
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        remote_size = int(response.headers.get("Content-Length", EXPECTED_SIZE))
        if remote_size != EXPECTED_SIZE:
            raise RuntimeError(
                f"H-MOG upstream size changed: expected {EXPECTED_SIZE:,}, "
                f"found {remote_size:,}"
            )
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    os.replace(temporary, path)
    verify(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help=f"confirm acceptance of the non-commercial terms at {OFFICIAL_PAGE}",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the local archive without downloading",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify()
        return
    if not ARCHIVE.exists() and not args.accept_license:
        parser.error(
            f"read {OFFICIAL_PAGE} and pass --accept-license to download H-MOG"
        )
    download()


if __name__ == "__main__":
    main()
