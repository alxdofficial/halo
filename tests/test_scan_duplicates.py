"""Regressions for the byte-identical-window screen (data/scripts/scan_duplicates)."""

import json

import numpy as np
import pytest

from data.scripts.scan_duplicates import OUT, scan_stream


def _grid(windows):
    return np.asarray(windows, dtype=np.float32)


def test_unique_windows_are_never_dropped():
    rng = np.random.default_rng(0)
    data = _grid(rng.normal(size=(8, 30, 6)))
    drop, groups, conflict = scan_stream(data, ["walking"] * 8)
    assert drop == [] and groups == 0 and conflict == 0


def test_same_label_group_keeps_exactly_one_genuine_observation():
    rng = np.random.default_rng(1)
    stale = rng.normal(size=(30, 6)).astype(np.float32)
    data = _grid([stale, rng.normal(size=(30, 6)), stale, stale])
    drop, groups, conflict = scan_stream(data, ["sitting"] * 4)
    assert drop == [2, 3]          # index 0 survives, its two repeats do not
    assert (groups, conflict) == (1, 0)


def test_label_conflicted_group_is_dropped_whole():
    """The ExtraSensory case: the Pebble buffer went stale across a label change, so the
    window is real but its label is unknowable — keeping any member injects a wrong label."""
    rng = np.random.default_rng(2)
    stale = rng.normal(size=(30, 6)).astype(np.float32)
    data = _grid([stale, stale, rng.normal(size=(30, 6))])
    drop, groups, conflict = scan_stream(data, ["lying", "sitting", "walking"])
    assert drop == [0, 1]          # NOT [1] — the survivor would be a coin flip
    assert (groups, conflict) == (1, 1)


def test_scan_spans_hash_block_boundaries():
    """Duplicates must be found even when the pair straddles the 4096-window read block."""
    from data.scripts import scan_duplicates

    rng = np.random.default_rng(3)
    data = _grid(rng.normal(size=(20, 4, 6)))
    data[19] = data[0]
    monkey = scan_duplicates.BLOCK
    try:
        scan_duplicates.BLOCK = 8          # force three blocks
        drop, groups, _ = scan_duplicates.scan_stream(data, ["walking"] * 20)
    finally:
        scan_duplicates.BLOCK = monkey
    assert drop == [19] and groups == 1


def test_cached_scan_covers_the_stale_pebble_stream():
    """The committed cache must actually carry the ExtraSensory finding, not an empty dict."""
    if not OUT.exists():
        pytest.skip("duplicate scan has not been run in this checkout")
    blob = json.loads(OUT.read_text())
    assert blob["alignment"] == "native"
    windows = blob["windows"]
    if "extrasensory/watch_wrist" not in windows:
        pytest.skip("extrasensory grids are not built in this checkout")
    assert len(windows["extrasensory/watch_wrist"]) > 10_000
    assert len(set(windows["extrasensory/watch_wrist"])) == \
        len(windows["extrasensory/watch_wrist"])       # indices are unique


def test_load_require_refuses_to_silently_return_an_empty_screen(tmp_path, monkeypatch):
    """A missing cache must not look like 'this corpus has no duplicates'."""
    from data.scripts import scan_duplicates as sd

    monkeypatch.setattr(sd, "OUT", tmp_path / "absent.json")
    assert sd.load(require=False) == {}
    with pytest.raises(FileNotFoundError):
        sd.load(require=True)

    (tmp_path / "absent.json").write_text(json.dumps({"alignment": "harmonised", "windows": {}}))
    assert sd.load("native", require=False) == {}
    with pytest.raises(ValueError):
        sd.load("native", require=True)
