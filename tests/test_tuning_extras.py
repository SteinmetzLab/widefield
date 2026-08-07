"""Tuning viewer additions: upsampling, s.e.m., band-pass; and the movie's fast GPU view."""

from __future__ import annotations

import numpy as np
import pytest

from widefield.events import event_locked_avg_svd, peri_event_series, peri_event_window

pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui  # noqa: E402

from widefield.gui.movie import _get_class as _movie_class  # noqa: E402
from widefield.gui.pixel_tuning_curve import _SEM_ALL, _SEM_NONE, _SEM_SELECTED  # noqa: E402
from widefield.gui.pixel_tuning_curve import _get_class as _tuning_class  # noqa: E402


def press(widget, key, mods=QtCore.Qt.NoModifier):
    widget.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, mods))


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    ypix, xpix, nsv, nframes = 14, 11, 5, 2000
    fs = 35.0
    q, _ = np.linalg.qr(rng.standard_normal((ypix * xpix, nsv)))
    u = q.reshape(ypix, xpix, nsv)
    t = np.arange(nframes) / fs
    v = rng.standard_normal((nsv, nframes)).astype(np.float32)
    # Deliberately jittered relative to the frame grid, which is what upsampling exploits.
    times = np.sort(rng.uniform(2.0, t[-1] - 2.0, 120))
    labels = np.tile([0.0, 0.5, 1.0], 40)
    return u, v, t, times, labels, fs


@pytest.fixture
def viewer(qtbot, data):
    u, v, t, times, labels, _ = data
    w = _tuning_class()(u, v, t, times, labels, (-0.3, 0.8), upsample=4)
    qtbot.addWidget(w)
    return w


# ============================================================ upsampling (numerics)


def test_upsample_makes_the_window_denser(data):
    _, _, t, _, _, fs = data
    w1, _ = peri_event_window(t, (-0.3, 0.8), upsample=1)
    w4, _ = peri_event_window(t, (-0.3, 0.8), upsample=4)
    assert w4.size > 3.5 * w1.size
    np.testing.assert_allclose(np.diff(w4)[0], np.diff(w1)[0] / 4, rtol=1e-9)


def test_upsample_one_reproduces_the_matlab_grid(ref):
    """upsample=1 must still match the reference grid exactly."""
    win, _ = peri_event_window(ref["t"], ref["ela_calcWin"], upsample=1)
    np.testing.assert_allclose(win, ref["ela_winSamps"], atol=1e-12)


def test_upsample_rejects_zero(data):
    _, _, t, _, _, _ = data
    with pytest.raises(ValueError, match="upsample must be"):
        peri_event_window(t, (-0.3, 0.8), upsample=0)


def test_upsample_recovers_subframe_detail(data):
    """The point of upsampling: a response faster than the frame rate is resolved.

    A brief transient locked to jittered event times is smeared by a frame-rate grid but
    recovered by a dense one, because each event samples it at a different sub-frame phase.
    """
    _, _, t, _, _, fs = data
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(2.0, t[-1] - 2.0, 400))
    # A narrow Gaussian bump after each event, much narrower than one frame interval.
    series = np.zeros_like(t)
    for e in times:
        series += np.exp(-(((t - e - 0.05) / 0.006) ** 2))

    peri1, win1 = peri_event_series(series, t, times, (-0.05, 0.15), upsample=1)
    peri8, win8 = peri_event_series(series, t, times, (-0.05, 0.15), upsample=8)
    m1, m8 = np.nanmean(peri1, axis=0), np.nanmean(peri8, axis=0)

    # The dense grid should localize the peak far closer to the true 50 ms latency.
    assert abs(win8[np.argmax(m8)] - 0.05) <= abs(win1[np.argmax(m1)] - 0.05) + 1e-9
    assert m8.max() > m1.max()  # and not flatten it as much


def test_keep_peri_false_gives_the_same_average(data):
    _, v, t, times, labels, _ = data
    a = event_locked_avg_svd(v, t, times, labels, (-0.3, 0.8), upsample=2, keep_peri=True)
    b = event_locked_avg_svd(v, t, times, labels, (-0.3, 0.8), upsample=2, keep_peri=False)
    np.testing.assert_allclose(a.avg_v, b.avg_v, rtol=1e-10, atol=1e-10)
    assert a.peri_v.shape[0] == times.size
    assert b.peri_v.shape[0] == 0  # not materialized


# ============================================================ s.e.m.


def test_sem_defaults_to_selected_only(viewer):
    assert viewer._sem_mode == _SEM_SELECTED
    assert "selected" in viewer._status.text()


def test_e_cycles_sem_modes(viewer):
    press(viewer, QtCore.Qt.Key_E)
    assert viewer._sem_mode == _SEM_ALL
    press(viewer, QtCore.Qt.Key_E)
    assert viewer._sem_mode == _SEM_NONE
    press(viewer, QtCore.Qt.Key_E)
    assert viewer._sem_mode == _SEM_SELECTED


def test_only_the_selected_band_is_visible_by_default(viewer):
    visible = [band.isVisible() for band, _lo, _hi in viewer._sem_bands]
    assert visible.count(True) == 1
    assert visible[viewer.condition_index]


def test_all_bands_visible_in_all_mode(viewer):
    press(viewer, QtCore.Qt.Key_E)  # -> all
    assert all(band.isVisible() for band, _lo, _hi in viewer._sem_bands)


def test_no_bands_visible_in_none_mode(viewer):
    press(viewer, QtCore.Qt.Key_E)
    press(viewer, QtCore.Qt.Key_E)  # -> none
    assert not any(band.isVisible() for band, _lo, _hi in viewer._sem_bands)
    assert not viewer._tc_errors.isVisible()


def test_sem_matches_a_direct_computation(viewer, data):
    """s.e.m. across events, per condition and time point."""
    u, v, t, times, labels, _ = data
    viewer.set_pixel(6, 5)
    trace = u[6, 5, :] @ v
    peri, _ = peri_event_series(trace, t, times, (-0.3, 0.8), upsample=viewer._upsample)
    order = np.argsort(times, kind="stable")
    sorted_labels = np.asarray(labels)[order]
    for c, label in enumerate(viewer._conditions):
        block = peri[sorted_labels == label]
        n = np.sum(np.isfinite(block), axis=0)
        expected = np.nanstd(block, axis=0, ddof=1) / np.sqrt(n)
        np.testing.assert_allclose(viewer._sem[c], expected, rtol=1e-4, atol=1e-6)


def test_sem_is_positive_and_finite(viewer):
    assert np.isfinite(viewer._sem).all()
    assert (viewer._sem >= 0).all()


def test_sem_band_brackets_the_mean(viewer):
    c = viewer.condition_index
    _band, lo, hi = viewer._sem_bands[c]
    _, ylo = lo.getData()
    _, yhi = hi.getData()
    np.testing.assert_allclose(ylo, viewer.traces[c] - viewer._sem[c], atol=1e-9)
    np.testing.assert_allclose(yhi, viewer.traces[c] + viewer._sem[c], atol=1e-9)


def test_tuning_curve_error_bars_use_the_sem_at_the_selected_time(viewer):
    viewer._time_idx = 7
    viewer._refresh_all()
    opts = viewer._tc_errors.opts
    np.testing.assert_allclose(opts["top"], viewer._sem[:, 7], atol=1e-9)
    np.testing.assert_allclose(opts["bottom"], viewer._sem[:, 7], atol=1e-9)


def test_sem_follows_the_pixel(viewer):
    viewer.set_pixel(2, 2)
    a = viewer._sem.copy()
    viewer.set_pixel(11, 8)
    assert not np.allclose(a, viewer._sem)


# ============================================================ band-pass


def test_tuning_bandpass_starts_unfiltered(viewer):
    assert viewer.bandpass.description == "unfiltered"


def test_tuning_bandpass_changes_the_traces(viewer):
    viewer.set_pixel(6, 5)
    before = viewer.traces.copy()
    viewer.bandpass.hp_edit.setText("1.0")
    viewer.bandpass.apply()
    assert not np.allclose(before, viewer.traces)


def test_tuning_bandpass_changes_the_brain_image(viewer):
    before = viewer.brain_image.copy()
    viewer.bandpass.hp_edit.setText("1.0")
    viewer.bandpass.apply()
    assert not np.allclose(before, viewer.brain_image)


def test_tuning_bandpass_reset_restores(viewer):
    viewer.set_pixel(6, 5)
    original = viewer.traces.copy()
    viewer.bandpass.hp_edit.setText("1.0")
    viewer.bandpass.apply()
    viewer.bandpass.reset()
    np.testing.assert_allclose(viewer.traces, original, atol=1e-5)


def test_tuning_bandpass_shown_in_the_status(viewer):
    viewer.bandpass.hp_edit.setText("2")
    viewer.bandpass.apply()
    assert "high-pass 2 Hz" in viewer._status.text()


def test_tuning_bandpass_invalid_leaves_traces_alone(viewer):
    before = viewer.traces.copy()
    viewer.bandpass.hp_edit.setText("nope")
    viewer.bandpass.apply()
    assert "invalid" in viewer.bandpass.label.text()
    np.testing.assert_allclose(viewer.traces, before, atol=1e-9)


# ============================================================ movie fast (GPU) view


@pytest.fixture
def movie(qtbot, data):
    u, v, t, _, _, _ = data
    w = _movie_class()(u, v, t=t)
    qtbot.addWidget(w)
    return w


def test_fast_mode_off_by_default(movie):
    assert movie.fast_mode is False
    assert movie._img_stack.currentIndex() == 0


def test_g_toggles_fast_mode(movie):
    if movie._fast_view is None:
        pytest.skip("no OpenGL image widget on this machine")
    press(movie, QtCore.Qt.Key_G)
    assert movie.fast_mode is True
    assert movie._img_stack.currentIndex() == 1
    press(movie, QtCore.Qt.Key_G)
    assert movie.fast_mode is False


def test_fast_rgba_has_the_right_shape_and_type(movie):
    if movie._fast_view is None:
        pytest.skip("no OpenGL image widget on this machine")
    oriented = movie._orient.apply(movie.frame_image)
    rgba = movie._fast_rgba(oriented)
    assert rgba.shape == (*oriented.shape, 4)
    assert rgba.dtype == np.uint8


def test_fast_rgba_maps_levels_to_the_colormap_ends(movie):
    if movie._fast_view is None:
        pytest.skip("no OpenGL image widget on this machine")
    lo, hi = movie._cax
    probe = np.array([[lo - 10.0, hi + 10.0]], dtype=np.float32)
    rgba = movie._fast_rgba(probe)
    np.testing.assert_array_equal(rgba[0, 0], movie._fast_lut[0])
    np.testing.assert_array_equal(rgba[0, 1], movie._fast_lut[255])


def test_fast_rgba_stamps_the_pixel_markers(movie):
    """Markers are drawn into the pixels, since a bare GL view cannot overlay items."""
    if movie._fast_view is None:
        pytest.skip("no OpenGL image widget on this machine")
    movie.set_pixel(7, 5)
    oriented = movie._orient.apply(movie.frame_image)
    plain = movie._fast_lut[
        np.clip(
            (oriented - movie._cax[0]) * (255.0 / (movie._cax[1] - movie._cax[0])), 0, 255
        ).astype(np.uint8)
    ]
    stamped = movie._fast_rgba(oriented)
    assert not np.array_equal(plain[7, 5], stamped[7, 5])


def test_fast_mode_still_updates_frames(movie):
    if movie._fast_view is None:
        pytest.skip("no OpenGL image widget on this machine")
    press(movie, QtCore.Qt.Key_G)
    movie.set_frame(100)
    a = movie.frame_image.copy()
    movie.set_frame(200)
    assert not np.allclose(a, movie.frame_image)
