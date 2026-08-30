#!/usr/bin/env python3
"""Acquire the selected application datasets without touching the HALO corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
INVENTORY_PATH = HERE / "SOURCE_INVENTORY.json"
CHECKSUMS_PATH = HERE / "PAYLOAD_CHECKSUMS.json"
SOURCES_ROOT = HERE / "sources"
BOX_SHARE = "4zwus5h4khsxpullnm45o59xlfpba6a0"
CROSSFIT_FOLDER = (
    "https://drive.google.com/drive/folders/"
    "1s_-2MI0eoNBUo0P-zSNij16uhyUpK0QK?usp=sharing"
)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "HALO-dataset-audit/1.0"})


def fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(_request(url), timeout=120) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url))


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_digest: str | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    if expected_digest is None:
        raise ValueError(
            "downloads require a frozen SHA-256 digest; update PAYLOAD_CHECKSUMS.json "
            "explicitly when adopting a new source release"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        digest = sha256_file(destination)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"SHA-256 mismatch for cached {destination}")
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": digest,
            "cached": True,
        }

    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            hasher = hashlib.sha256()
            size = 0
            with (
                urllib.request.urlopen(_request(url), timeout=120) as response,
                partial.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
            if expected_size is not None and size != expected_size:
                raise ValueError(
                    f"size mismatch for {destination}: {size} != {expected_size}"
                )
            digest = hasher.hexdigest()
            if expected_digest is not None and digest != expected_digest:
                raise ValueError(f"SHA-256 mismatch for {destination}")
            partial.replace(destination)
            return {
                "path": str(destination),
                "bytes": size,
                "sha256": digest,
                "cached": False,
            }
        except Exception as error:  # network retries are intentionally bounded
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def download_many(jobs: Iterable[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    jobs = list(jobs)
    if workers <= 0:
        raise ValueError("workers must be positive")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(
                download,
                job["url"],
                job["destination"],
                expected_size=job.get("expected_size"),
                expected_digest=job.get("expected_digest"),
            ): job
            for job in jobs
        }
        for index, future in enumerate(as_completed(pending), 1):
            result = future.result()
            results.append(result)
            if index % 50 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)} files")
    return results


def write_manifest(dataset: str, results: list[dict[str, Any]]) -> None:
    path = SOURCES_ROOT / dataset / "manifests" / "download.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for result in results:
        row = dict(result)
        row["path"] = str(
            Path(row["path"]).resolve().relative_to(SOURCES_ROOT.resolve())
        )
        normalized.append(row)
    payload = {
        "dataset": dataset,
        "checksum_contract": "PAYLOAD_CHECKSUMS.json",
        "files": sorted(normalized, key=lambda item: item["path"]),
        "file_count": len(normalized),
        "total_bytes": sum(item["bytes"] for item in normalized),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


@lru_cache(maxsize=1)
def _frozen_checksum_payload() -> dict[str, Any]:
    payload = json.loads(CHECKSUMS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported payload-checksum schema in {CHECKSUMS_PATH}")
    return payload


def _bind_frozen_checksums(
    dataset: str, jobs: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind a discovered release listing to the tracked immutable payload contract."""

    jobs = [dict(job) for job in jobs]
    try:
        frozen_rows = _frozen_checksum_payload()["datasets"][dataset]["files"]
    except KeyError as error:
        raise KeyError(f"no frozen payload checksums for {dataset}") from error
    frozen = {row["path"]: row for row in frozen_rows}
    discovered: dict[str, dict[str, Any]] = {}
    for job in jobs:
        destination = Path(job["destination"]).resolve()
        try:
            relative = destination.relative_to(SOURCES_ROOT.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(
                f"download destination escapes source root: {destination}"
            ) from error
        if relative in discovered:
            raise ValueError(f"duplicate download destination: {relative}")
        discovered[relative] = job

    missing = sorted(set(frozen) - set(discovered))
    unexpected = sorted(set(discovered) - set(frozen))
    if missing or unexpected:
        raise RuntimeError(
            f"{dataset} release listing differs from the frozen payload contract; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for relative, job in discovered.items():
        row = frozen[relative]
        job["expected_size"] = int(row["bytes"])
        job["expected_digest"] = str(row["sha256"])
    return jobs


def acquire_http_archive(dataset: str, url: str, filename: str) -> None:
    jobs = _bind_frozen_checksums(
        dataset,
        [
            {
                "url": url,
                "destination": SOURCES_ROOT / dataset / "downloads" / filename,
            }
        ],
    )
    write_manifest(dataset, download_many(jobs, workers=1))


def acquire_aidlab_har() -> None:
    acquire_http_archive(
        "aidlab_har",
        "https://aidlab-production-datasets.s3.eu-central-1.amazonaws.com/"
        "AIDLAB-HAR-DATASET_v3.zip",
        "AIDLAB-HAR-DATASET_v3.zip",
    )


def acquire_oca() -> None:
    acquire_http_archive(
        "oca",
        "https://fordatis.fraunhofer.de/bitstream/fordatis/195/1/OCA.zip",
        "OCA.zip",
    )


def acquire_recofit() -> None:
    acquire_http_archive(
        "recofit",
        "https://media.githubusercontent.com/media/microsoft/"
        "Exercise-Recognition-from-Wearable-Sensors/main/"
        "exercise_data.50.0000_multionly.mat",
        "exercise_data.50.0000_multionly.mat",
    )


def acquire_openpack() -> None:
    record = fetch_json("https://zenodo.org/api/records/8145223")
    jobs = []
    for item in record["files"]:
        if not re.fullmatch(r"U\d+\.zip", item["key"]):
            continue
        jobs.append(
            {
                "url": item["links"]["self"],
                "destination": SOURCES_ROOT / "openpack" / "downloads" / item["key"],
                "expected_size": item["size"],
            }
        )
    if len(jobs) != 21:
        raise RuntimeError(f"expected 21 OpenPack subject archives, found {len(jobs)}")
    write_manifest(
        "openpack",
        download_many(_bind_frozen_checksums("openpack", jobs), workers=4),
    )


def _index_links(index_url: str, suffix: str) -> list[str]:
    html = fetch_bytes(index_url).decode("utf-8")
    links = re.findall(r'href="([^"?#]+)"', html)
    return sorted(
        urllib.parse.urljoin(index_url, link)
        for link in links
        if link.lower().endswith(suffix)
    )


def acquire_wear() -> None:
    jobs = []
    for url in _index_links(
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/",
        ".csv",
    ):
        jobs.append(
            {
                "url": url,
                "destination": SOURCES_ROOT
                / "wear"
                / "raw"
                / "inertial_50hz"
                / Path(url).name,
            }
        )
    for url in _index_links(
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/annotations/60fps/",
        ".json",
    ):
        jobs.append(
            {
                "url": url,
                "destination": SOURCES_ROOT
                / "wear"
                / "raw"
                / "annotations_60fps"
                / Path(url).name,
            }
        )
    if len([job for job in jobs if job["destination"].suffix == ".csv"]) != 24:
        raise RuntimeError(
            "WEAR release no longer exposes the expected 24 inertial files"
        )
    write_manifest(
        "wear", download_many(_bind_frozen_checksums("wear", jobs), workers=8)
    )


def _box_items(folder_id: int) -> list[dict[str, Any]]:
    url = f"https://utdallas.app.box.com/s/{BOX_SHARE}/folder/{folder_id}"
    html = fetch_bytes(url).decode("utf-8")
    match = re.search(r"Box\.postStreamData = (\{.*?\});</script>", html)
    if match is None:
        raise RuntimeError(f"could not parse Box folder {folder_id}")
    payload = json.loads(match.group(1))
    return payload["/app-api/enduserapp/shared-folder"]["items"]


def _box_download_url(file_id: int) -> str:
    query = urllib.parse.urlencode(
        {
            "rm": "box_download_shared_file",
            "shared_name": BOX_SHARE,
            "file_id": f"f_{file_id}",
        }
    )
    return f"https://utdallas.app.box.com/index.php?{query}"


def acquire_c_mhad() -> None:
    application_folders = {
        "TVGestureApplication": 96_786_265_756,
        "TransitionMovementsApplication": 96_783_661_866,
    }
    jobs = []
    for application, application_id in application_folders.items():
        subjects = [
            item for item in _box_items(application_id) if item["type"] == "folder"
        ]
        if len(subjects) != 12:
            raise RuntimeError(f"expected 12 C-MHAD subjects in {application}")
        for subject in subjects:
            children = _box_items(subject["id"])
            inertial = next(
                item
                for item in children
                if item["type"] == "folder" and item["name"] == "InertialData"
            )
            labels = [
                item
                for item in children
                if item["type"] == "file" and item["extension"] == "xlsx"
            ]
            if len(labels) != 1:
                raise RuntimeError(
                    f"unexpected annotation files for {application}/{subject['name']}"
                )
            label = labels[0]
            jobs.append(
                {
                    "url": _box_download_url(label["id"]),
                    "destination": SOURCES_ROOT
                    / "c_mhad"
                    / "raw"
                    / application
                    / subject["name"]
                    / label["name"],
                    "expected_size": label["itemSize"],
                }
            )
            csv_items = [
                item
                for item in _box_items(inertial["id"])
                if item["type"] == "file" and item["extension"] == "csv"
            ]
            if len(csv_items) != 10:
                raise RuntimeError(
                    f"expected 10 C-MHAD CSVs for {application}/{subject['name']}"
                )
            for item in csv_items:
                jobs.append(
                    {
                        "url": _box_download_url(item["id"]),
                        "destination": SOURCES_ROOT
                        / "c_mhad"
                        / "raw"
                        / application
                        / subject["name"]
                        / item["name"],
                        "expected_size": item["itemSize"],
                    }
                )
    write_manifest(
        "c_mhad", download_many(_bind_frozen_checksums("c_mhad", jobs), workers=8)
    )


def _crossfit_listing() -> list[dict[str, str]]:
    result = subprocess.run(
        ["uvx", "gdown", "--folder", "--json", CROSSFIT_FOLDER],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def acquire_crossfit() -> None:
    jobs = []
    for item in _crossfit_listing():
        path = Path(item["path"])
        if (
            path.name == ".DS_Store"
            or path.parts[0] == "HAR_Crossfit_Sensors_PretrainedModels"
        ):
            continue
        match = re.search(r"[?&]id=([^&]+)", item["url"])
        if match is None:
            raise RuntimeError(f"missing Google Drive id for {path}")
        url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
            {"id": match.group(1), "export": "download", "confirm": "t"}
        )
        jobs.append(
            {
                "url": url,
                "destination": SOURCES_ROOT / "crossfit" / "raw" / path,
            }
        )
    write_manifest(
        "crossfit", download_many(_bind_frozen_checksums("crossfit", jobs), workers=12)
    )


ACQUIRERS = {
    "aidlab_har": acquire_aidlab_har,
    "c_mhad": acquire_c_mhad,
    "crossfit": acquire_crossfit,
    "oca": acquire_oca,
    "openpack": acquire_openpack,
    "recofit": acquire_recofit,
    "wear": acquire_wear,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=sorted(ACQUIRERS))
    args = parser.parse_args()
    selected = (
        args.datasets or json.loads(INVENTORY_PATH.read_text())["selected_new_sources"]
    )
    for dataset in selected:
        print(f"Acquiring {dataset}")
        ACQUIRERS[dataset]()


if __name__ == "__main__":
    main()
