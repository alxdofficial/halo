"""Fetch ONLY the HARMES right-wrist IMU recordings + event logs from the RAW zip.

HARMES-RAW.zip on Zenodo is 19.9 GB, but 18.9 GB of that is audio `.h5`. We need
only the WearOS right-wrist recordings (`recording_*.csv`, ~1.1 GB total) and the
tiny start/end event logs (`<epoch>.csv`). We pull just those members out of the
remote zip via HTTP range requests (Zenodo honours `Range`), skipping all audio and
the left-wrist Puck.js `*_merged.csv` (see convert.py for why the left wrist is
excluded). Idempotent: existing files are skipped.
"""

from __future__ import annotations

from pathlib import Path

from remotezip import RemoteZip

URL = "https://zenodo.org/records/19425719/files/HARMES-RAW.zip?download=1"
HERE = Path(__file__).resolve().parent
OUT = HERE / "downloads"


def is_wanted(name: str) -> bool:
    parts = name.split("/")
    if len(parts) != 4 or not name.endswith(".csv"):
        return False
    fn = parts[3]
    if fn.endswith("_merged.csv") or fn.endswith("_TASKLIST.csv"):
        return False  # left-wrist Puck.js / planned-protocol list — not used
    # right-wrist recordings + the small numeric event-log csvs
    return fn.startswith("recording_") or fn.split(".")[0].isdigit()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        members = [i for i in z.infolist() if is_wanted(i.filename)]
        tot = sum(i.file_size for i in members)
        print(f"fetching {len(members)} members ({tot/1e9:.2f} GB) of right-wrist + event logs")
        for k, i in enumerate(members, 1):
            dest = OUT / i.filename
            if dest.exists() and dest.stat().st_size == i.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            z.extract(i, OUT)
            if k % 20 == 0 or k == len(members):
                print(f"  {k}/{len(members)} done", flush=True)
    print("fetch complete")


if __name__ == "__main__":
    main()
