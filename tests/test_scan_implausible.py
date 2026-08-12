"""Regressions for fail-closed physical-plausibility cache loading."""

import json

import pytest


def test_implausible_cache_is_required_and_grid_fingerprinted(tmp_path, monkeypatch):
    from data.scripts import scan_implausible as si
    from data.scripts.eda import grid_io

    cache = tmp_path / "implausible_windows.json"
    monkeypatch.setattr(si, "OUT", cache)
    with pytest.raises(FileNotFoundError):
        si.load(require=True)

    cache.write_text(json.dumps({
        "alignment": "native", "grid_fingerprint": "old", "windows": {},
    }))
    monkeypatch.setattr(grid_io, "grid_corpus_fingerprint", lambda alignment: "current")
    assert si.load(require=False) == {}
    with pytest.raises(ValueError, match="does not match"):
        si.load(require=True)


def test_implausible_cache_paths_do_not_overwrite_alignments(tmp_path, monkeypatch):
    from data.scripts import scan_implausible as si
    monkeypatch.setattr(si, "OUT", tmp_path / "implausible_windows.json")
    assert si.cache_path("native").name == "implausible_windows.json"
    assert si.cache_path("non_harmonised").name == "implausible_windows_non_harmonised.json"


def test_implausible_cache_allows_absent_optional_cached_streams(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from data.scripts import scan_implausible as si
    from data.scripts.eda import grid_io
    cache = tmp_path / "implausible_windows.json"
    cache.write_text(json.dumps({
        "alignment": "native",
        "stream_fingerprints": {"required/stream": "ok", "optional/stream": "extra"},
        "windows": {},
    }))
    monkeypatch.setattr(si, "OUT", cache)
    monkeypatch.setattr(grid_io, "discover_grids",
                        lambda alignment: [SimpleNamespace(key="required/stream")])
    monkeypatch.setattr(grid_io, "grid_corpus_fingerprint",
                        lambda alignment, refs=None: "ok")
    assert si.load("native", require=True) == {}
