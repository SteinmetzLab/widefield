"""Session discovery and the component-prefix reader.

The reader is the performance-critical piece (it turns a 2 GB transfer into 500 MB), and it is
only correct because the files are Fortran-ordered — so these tests check both the values and
that assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from widefield.io import (
    LoadCancelled,
    default_channel,
    discover_channels,
    load_uvt,
    npy_header,
    read_u_from_npy,
    read_v_from_npy,
)

# ------------------------------------------------------------------------- discovery


def test_discovers_all_three_channels(synthetic_session):
    ch = discover_channels(synthetic_session)
    assert set(ch) == {"blue", "violet", "corr"}


def test_corr_borrows_blue_spatial_basis(synthetic_session):
    ch = discover_channels(synthetic_session)
    assert ch["corr"].spatial == ch["blue"].spatial
    assert ch["corr"].mean_image == ch["blue"].mean_image


def test_default_channel_prefers_corr(synthetic_session):
    assert default_channel(discover_channels(synthetic_session)) == "corr"


def test_default_channel_falls_back_to_blue():
    assert default_channel({"blue": None, "violet": None}) == "blue"
    assert default_channel({"violet": None}) == "violet"
    assert default_channel({}) is None


def test_discover_on_missing_directory_is_empty(tmp_path):
    assert discover_channels(tmp_path / "nope") == {}


def test_no_corr_without_blue(tmp_path):
    """corr is meaningless on its own — it has no spatial components."""
    (tmp_path / "corr").mkdir()
    np.save(tmp_path / "corr" / "svdTemporalComponents_corr.npy", np.zeros((5, 2)))
    assert discover_channels(tmp_path) == {}


def test_channel_metadata_without_reading_pixels(synthetic_session):
    ch = discover_channels(synthetic_session)["blue"]
    assert ch.image_shape == (8, 6)
    assert ch.n_components == 10
    assert ch.n_frames == 240


def test_corr_reports_its_own_shorter_shape(synthetic_session):
    ch = discover_channels(synthetic_session)["corr"]
    assert ch.n_components == 4  # fewer than blue's 10
    assert ch.n_frames == 239  # one frame short


# ------------------------------------------------------------------------- npy reading


def test_fixture_files_are_fortran_ordered(synthetic_session):
    """Guard the premise of the fast path — matches what npy-matlab writes on the server."""
    ch = discover_channels(synthetic_session)["blue"]
    assert npy_header(ch.spatial).fortran_order
    assert npy_header(ch.temporal).fortran_order


def test_read_u_full_matches_plain_load(synthetic_session):
    ch = discover_channels(synthetic_session)["blue"]
    np.testing.assert_array_equal(read_u_from_npy(ch.spatial), np.load(ch.spatial))


def test_read_u_prefix_is_the_leading_components(synthetic_session):
    """The whole optimization: reading n components must equal loading all and slicing."""
    ch = discover_channels(synthetic_session)["blue"]
    full = np.load(ch.spatial)
    got = read_u_from_npy(ch.spatial, 3)
    assert got.shape == (8, 6, 3)
    np.testing.assert_array_equal(got, full[:, :, :3])


def test_read_v_transposes_to_component_major(synthetic_session):
    ch = discover_channels(synthetic_session)["blue"]
    full = np.load(ch.temporal)  # (nFrames, nSV) on disk
    got = read_v_from_npy(ch.temporal)
    assert got.shape == (full.shape[1], full.shape[0])
    np.testing.assert_array_equal(got, full.T)


def test_read_v_prefix_is_the_leading_components(synthetic_session):
    ch = discover_channels(synthetic_session)["blue"]
    full = np.load(ch.temporal)
    np.testing.assert_array_equal(read_v_from_npy(ch.temporal, 3), full.T[:3])


def test_asking_for_more_components_than_exist_returns_all(synthetic_session):
    ch = discover_channels(synthetic_session)["blue"]
    assert read_u_from_npy(ch.spatial, 999).shape[-1] == 10


def test_read_rejects_zero_components(synthetic_session):
    ch = discover_channels(synthetic_session)["blue"]
    with pytest.raises(ValueError, match="nsv must be"):
        read_u_from_npy(ch.spatial, 0)


def test_c_ordered_file_still_reads_correctly(tmp_path):
    """A C-ordered file can't use the prefix trick; it must fall back, not return garbage."""
    path = tmp_path / "c_order.npy"
    arr = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)  # C-ordered
    np.save(path, arr)
    assert not npy_header(path).fortran_order
    np.testing.assert_array_equal(read_u_from_npy(path, 2), arr[:, :, :2])


def test_read_is_cancellable(synthetic_session):
    ch = discover_channels(synthetic_session)["blue"]
    with pytest.raises(LoadCancelled):
        read_u_from_npy(ch.spatial, 10, should_cancel=lambda: True)


def test_chunked_read_matches_single_chunk(synthetic_session):
    """Force many small chunks; the assembled array must be identical."""
    from widefield.io import _read_component_prefix

    ch = discover_channels(synthetic_session)["blue"]
    flat_small, _ = _read_component_prefix(ch.spatial, 5, chunk_bytes=64)
    flat_big, _ = _read_component_prefix(ch.spatial, 5, chunk_bytes=1 << 30)
    np.testing.assert_array_equal(flat_small, flat_big)


# ------------------------------------------------------------------------- load_uvt


def test_load_uvt_default_channel(synthetic_session):
    d = load_uvt(synthetic_session, use_cache=False)
    assert d.channel == "corr"
    assert d.u.shape[:2] == (8, 6)


def test_load_uvt_truncates_u_to_corr_component_count(synthetic_session):
    """corr has 4 components; loading it must not drag in blue's other 6."""
    d = load_uvt(synthetic_session, channel="corr", use_cache=False)
    assert d.u.shape[-1] == 4
    assert d.v.shape[0] == 4


def test_load_uvt_respects_nsv(synthetic_session):
    d = load_uvt(synthetic_session, nsv=2, channel="blue", use_cache=False)
    assert d.u.shape[-1] == 2 and d.v.shape[0] == 2


def test_load_uvt_trims_the_extra_trailing_timestamp(synthetic_session):
    """Real sessions have one more blue timestamp than corrected frame; it must be dropped."""
    d = load_uvt(synthetic_session, channel="corr", use_cache=False)
    assert d.t is not None
    assert d.t.size == d.v.shape[1]


def test_load_uvt_reports_frame_rate(synthetic_session):
    d = load_uvt(synthetic_session, channel="blue", use_cache=False)
    assert d.fs == pytest.approx(35.0)


def test_load_uvt_binning_shrinks_spatially(synthetic_session):
    d = load_uvt(synthetic_session, channel="blue", binning=2, use_cache=False)
    assert d.u.shape[:2] == (4, 3)
    assert d.mean_image.shape == (4, 3)


def test_load_uvt_binning_leaves_time_alone(synthetic_session):
    full = load_uvt(synthetic_session, channel="blue", use_cache=False)
    binned = load_uvt(synthetic_session, channel="blue", binning=2, use_cache=False)
    np.testing.assert_array_equal(full.v, binned.v)


def test_load_uvt_unknown_channel_raises(synthetic_session):
    with pytest.raises(KeyError, match="not found"):
        load_uvt(synthetic_session, channel="green", use_cache=False)


def test_load_uvt_on_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no widefield SVD data"):
        load_uvt(tmp_path, use_cache=False)


def test_load_uvt_detrend_changes_v(synthetic_session):
    plain = load_uvt(synthetic_session, channel="blue", use_cache=False)
    filt = load_uvt(synthetic_session, channel="blue", use_cache=False, detrend=True)
    assert not np.allclose(plain.v, filt.v)


def test_cache_roundtrip(synthetic_session, tmp_path, monkeypatch):
    """Second load must come from the local cache and be byte-identical."""
    monkeypatch.setenv("WIDEFIELD_CACHE", str(tmp_path / "cache"))
    first = load_uvt(synthetic_session, channel="blue", nsv=3, use_cache=True)
    cached_files = list((tmp_path / "cache").glob("*.npy"))
    assert cached_files, "nothing was cached"
    second = load_uvt(synthetic_session, channel="blue", nsv=3, use_cache=True)
    np.testing.assert_array_equal(first.u, second.u)
    np.testing.assert_array_equal(first.v, second.v)


def test_cache_never_written_under_the_session(synthetic_session, tmp_path, monkeypatch):
    """The server is read-only; nothing may appear next to the data."""
    monkeypatch.setenv("WIDEFIELD_CACHE", str(tmp_path / "cache"))
    before = set(synthetic_session.rglob("*"))
    load_uvt(synthetic_session, channel="blue", use_cache=True)
    assert set(synthetic_session.rglob("*")) == before


def test_cache_key_separates_binning(synthetic_session, tmp_path, monkeypatch):
    """A cached full-res U must not be served for a binning=2 request."""
    monkeypatch.setenv("WIDEFIELD_CACHE", str(tmp_path / "cache"))
    full = load_uvt(synthetic_session, channel="blue", use_cache=True)
    binned = load_uvt(synthetic_session, channel="blue", binning=2, use_cache=True)
    assert full.u.shape != binned.u.shape


def test_cache_key_separates_nsv(synthetic_session, tmp_path, monkeypatch):
    monkeypatch.setenv("WIDEFIELD_CACHE", str(tmp_path / "cache"))
    a = load_uvt(synthetic_session, channel="blue", nsv=2, use_cache=True)
    b = load_uvt(synthetic_session, channel="blue", nsv=5, use_cache=True)
    assert a.u.shape[-1] == 2 and b.u.shape[-1] == 5


def test_corrupt_cache_is_rebuilt(synthetic_session, tmp_path, monkeypatch):
    monkeypatch.setenv("WIDEFIELD_CACHE", str(tmp_path / "cache"))
    load_uvt(synthetic_session, channel="blue", nsv=3, use_cache=True)
    for p in (tmp_path / "cache").glob("*.npy"):
        p.write_bytes(b"garbage")
    d = load_uvt(synthetic_session, channel="blue", nsv=3, use_cache=True)  # must not raise
    np.testing.assert_array_equal(
        d.u, read_u_from_npy(discover_channels(synthetic_session)["blue"].spatial, 3)
    )
