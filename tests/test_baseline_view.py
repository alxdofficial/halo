"""Tests for the harmonised / non-harmonised baseline input views."""

import numpy as np
import pandas as pd
import pytest

from data.scripts.baseline_view import (
    HARMONISED_CHANNELS,
    to_view,
)
from data.scripts.deployment_policy import (
    curate_frame,
    get_stream_spec,
)

ACC = ("acc_x", "acc_y", "acc_z")
ACC_GYRO = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def _stream(T, channels):
    """(T, len(channels)) with column j filled with value (j+1) so provenance is checkable."""
    return np.stack([np.full(T, j + 1.0, dtype=np.float32) for j in range(len(channels))], axis=1)


def test_harmonised_six_channel_is_identity_layout():
    data = _stream(5, ACC_GYRO)
    out, chans, mask = to_view(data, ACC_GYRO, "harmonised")
    assert chans == HARMONISED_CHANNELS and out.shape == (5, 6)
    assert mask.all()
    assert np.array_equal(out, data)


def test_harmonised_acc_only_zero_pads_gyro_with_mask():
    data = _stream(4, ACC)
    out, chans, mask = to_view(data, ACC, "harmonised")
    assert chans == HARMONISED_CHANNELS and out.shape == (4, 6)
    assert list(mask) == [True, True, True, False, False, False]
    assert np.array_equal(out[:, :3], data)           # real accel preserved
    assert np.count_nonzero(out[:, 3:]) == 0          # gyro slots EXACTLY zero — never fabricated


def test_non_harmonised_keeps_native_width():
    out, chans, mask = to_view(_stream(4, ACC), ACC, "non_harmonised")
    assert chans == ACC and out.shape == (4, 3) and mask.all()

    out6, chans6, mask6 = to_view(_stream(4, ACC_GYRO), ACC_GYRO, "non_harmonised")
    assert chans6 == ACC_GYRO and out6.shape == (4, 6) and mask6.all()


def test_non_harmonised_padded_to_six_collapses_onto_harmonised():
    """The documented finding: for a fixed-input model both views are byte-identical."""
    data = _stream(6, ACC)
    h_out, h_chans, h_mask = to_view(data, ACC, "harmonised")
    n_out, n_chans, n_mask = to_view(data, ACC, "non_harmonised", pad_to=6)
    assert n_chans == h_chans
    assert np.array_equal(n_out, h_out)
    assert np.array_equal(n_mask, h_mask)


def test_non_canonical_input_order_is_reordered_canonically():
    scrambled = ("gyro_x", "acc_z", "acc_x")
    data = _stream(3, scrambled)  # cols: gyro_x=1, acc_z=2, acc_x=3
    out, chans, mask = to_view(data, scrambled, "non_harmonised")
    assert chans == ("acc_x", "acc_z", "gyro_x")       # canonical order restored
    assert np.array_equal(out[:, 0], data[:, 2])       # acc_x <- col 2
    assert np.array_equal(out[:, 1], data[:, 1])       # acc_z <- col 1
    assert np.array_equal(out[:, 2], data[:, 0])       # gyro_x <- col 0


def test_rejects_bad_shape_unknown_channels_and_alignment():
    with pytest.raises(ValueError):
        to_view(_stream(4, ACC), ACC_GYRO, "harmonised")           # shape/channel mismatch
    with pytest.raises(ValueError):
        to_view(_stream(4, ("mag_x", "mag_y", "mag_z")), ("mag_x", "mag_y", "mag_z"), "harmonised")
    with pytest.raises(ValueError):
        to_view(_stream(4, ACC), ACC, "bogus")
    with pytest.raises(ValueError):
        to_view(_stream(4, ACC), ACC, "non_harmonised", pad_to=3)  # only None or 6 allowed


def _full_source_frame(dataset, spec, n=8):
    cols = {"timestamp_sec": np.arange(n, dtype=float) / 50.0}
    from data.scripts.deployment_policy import all_source_channels
    for source in all_source_channels(dataset, role=spec.role):
        cols[source] = np.ones(n, dtype=float)
    return pd.DataFrame(cols)


def test_end_to_end_from_deployment_policy_six_and_three_channel():
    # 6-ch dataset (uci_har) -> harmonised full, mask all True.
    spec6 = get_stream_spec("uci_har", "phone_waist")
    curated6, meta6 = curate_frame(_full_source_frame("uci_har", spec6), spec6)
    data6 = curated6[list(meta6.channels)].to_numpy(np.float32)
    out6, chans6, mask6 = to_view(data6, meta6.channels, "harmonised")
    assert chans6 == HARMONISED_CHANNELS and out6.shape[1] == 6 and mask6.all()

    # 3-ch dataset (capture24, acc-only wrist) -> harmonised pads gyro, non_harmonised stays 3.
    spec3 = get_stream_spec("capture24", "watch_wrist")
    curated3, meta3 = curate_frame(_full_source_frame("capture24", spec3), spec3)
    data3 = curated3[list(meta3.channels)].to_numpy(np.float32)
    assert meta3.channels == ACC
    h3, _, hmask3 = to_view(data3, meta3.channels, "harmonised")
    n3, nchans3, nmask3 = to_view(data3, meta3.channels, "non_harmonised")
    assert h3.shape[1] == 6 and list(hmask3) == [True, True, True, False, False, False]
    assert n3.shape[1] == 3 and nchans3 == ACC and nmask3.all()
