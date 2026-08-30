from __future__ import annotations

from pathlib import Path

import pytest

from applications.motion_monitoring.data import acquire


def test_download_requires_frozen_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen SHA-256"):
        acquire.download("https://example.invalid/file", tmp_path / "file")


def test_bind_frozen_checksums_supplies_exact_size_and_digest() -> None:
    destination = (
        acquire.SOURCES_ROOT
        / "aidlab_har"
        / "downloads"
        / "AIDLAB-HAR-DATASET_v3.zip"
    )
    jobs = acquire._bind_frozen_checksums(
        "aidlab_har", [{"url": "https://example.invalid/file", "destination": destination}]
    )

    assert jobs[0]["expected_size"] == 3_292_987
    assert jobs[0]["expected_digest"] == (
        "f5274a59bdec3fef7b6fb45e35279b13f1589572bfb906f4f45da4cf3f285089"
    )


def test_bind_frozen_checksums_rejects_release_drift() -> None:
    with pytest.raises(RuntimeError, match="differs from the frozen payload"):
        acquire._bind_frozen_checksums("aidlab_har", [])


def test_bind_frozen_checksums_rejects_destination_outside_source_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="escapes source root"):
        acquire._bind_frozen_checksums(
            "aidlab_har",
            [{"url": "https://example.invalid/file", "destination": tmp_path / "file"}],
        )
