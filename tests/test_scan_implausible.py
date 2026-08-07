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
