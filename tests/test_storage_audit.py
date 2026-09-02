from pathlib import Path

from data.scripts.storage_audit import _exact_duplicates, apparent_bytes, build_inventory


def test_apparent_bytes_ignores_partial_and_python_cache(tmp_path: Path) -> None:
    (tmp_path / "kept.bin").write_bytes(b"kept")
    (tmp_path / "download.part").write_bytes(b"partial")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cache")

    assert apparent_bytes(tmp_path) == 4


def test_exact_duplicates_require_matching_content(tmp_path: Path) -> None:
    content = b"same payload"
    (tmp_path / "first.bin").write_bytes(content)
    (tmp_path / "second.bin").write_bytes(content)
    (tmp_path / "different.bin").write_bytes(b"other content")

    groups = _exact_duplicates((tmp_path,), minimum_bytes=1)

    assert len(groups) == 1
    assert groups[0]["copies"] == 2
    assert groups[0]["reclaimable_bytes_if_one_copy_retained"] == len(content)


def test_inventory_summary_matches_detail() -> None:
    inventory = build_inventory(hash_minimum_bytes=0)

    assert inventory["summary"]["core_dataset_bytes"] == sum(
        entry["total_bytes"] for entry in inventory["core_datasets"].values()
    )
    assert inventory["summary"]["application_source_bytes"] == sum(
        entry["total_bytes"] for entry in inventory["application_sources"].values()
    )
    assert inventory["summary"]["tracked_output_bytes"] == sum(
        inventory["output_roots"].values()
    )
