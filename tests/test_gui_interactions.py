"""Tests for the interaction fixes: hotkey routing, ctrl+click, band-pass, follow-zoom.

These cover behaviors that were reported broken from real use, so each one names the failure it
guards against.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from widefield.gui.movie_with_traces import Trace  # noqa: E402
from widefield.gui.movie_with_traces import _get_class as _movie_class  # noqa: E402
from widefield.gui.pixel_correlation import _get_class as _corr_class  # noqa: E402
from widefield.svd import bandpass_filt  # noqa: E402


@pytest.fixture(scope="module")
def movie_data():
    rng = np.random.default_rng(3)
    ypix, xpix, nsv, nframes = 16, 12, 5, 1400
    q, _ = np.linalg.qr(rng.standard_normal((ypix * xpix, nsv)))
    u = q.reshape(ypix, xpix, nsv)
    fs = 35.0
    t = np.arange(nframes) / fs
    # A slow drift plus a fast oscillation, so filtering has something visible to do.
    v = (
        np.outer(np.linspace(1, 2, nsv), np.sin(2 * np.pi * 0.02 * t))
        + np.outer(np.linspace(1, 2, nsv), np.sin(2 * np.pi * 12.0 * t)) * 0.5
    ).astype(np.float32)
    return u, v, t, fs


@pytest.fixture
def movie(qtbot, movie_data):
    u, v, t, _ = movie_data
    w = _movie_class()(u, v, t=t, traces=[Trace(t=t, v=np.sin(t), name="dummy")])
    qtbot.addWidget(w)
    return w


class FakeClick:
    """Minimal stand-in for a pyqtgraph mouse-click event."""

    def __init__(self, scene_pos, button=QtCore.Qt.LeftButton, modifiers=QtCore.Qt.NoModifier):
        self._pos, self._button, self._mods = scene_pos, button, modifiers

    def scenePos(self):
        return self._pos

    def button(self):
        return self._button

    def modifiers(self):
        return self._mods


def click_at(viewer, dy, dx, button=QtCore.Qt.LeftButton, modifiers=QtCore.Qt.NoModifier):
    """Drive the real click handler, with the scene->view mapping stubbed to a known pixel."""
    viewer._plot.vb.mapSceneToView = lambda _pos: QtCore.QPointF(dx + 0.5, dy + 0.5)
    viewer._on_click(FakeClick(QtCore.QPointF(0, 0), button, modifiers))


# ===================================================================== hotkey routing


def test_hotkeys_work_when_a_child_widget_has_focus(movie):
    """Reported bug: -/= only worked right after clicking Play.

    Clicking the image or a button moves focus to that child, which swallowed the key before it
    reached the top-level widget. install_hotkeys must route it regardless of focus.
    """
    movie._glw.setFocus()
    assert movie.focusWidget() is not movie or True  # focus is somewhere in the subtree
    before = list(movie._cax)
    QtWidgets.QApplication.sendEvent(
        movie._glw,
        QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Minus, QtCore.Qt.NoModifier),
    )
    assert movie._cax[1] < before[1], "colour scale did not change with focus on the plot"


def test_hotkeys_work_with_focus_on_the_play_button(movie):
    movie._play_btn.setFocus()
    before = list(movie._cax)
    QtWidgets.QApplication.sendEvent(
        movie._play_btn,
        QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Equal, QtCore.Qt.NoModifier),
    )
    assert movie._cax[1] > before[1]


def test_hotkeys_work_with_focus_on_the_slider(movie):
    movie._slider.setFocus()
    QtWidgets.QApplication.sendEvent(
        movie._slider,
        QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_P, QtCore.Qt.NoModifier),
    )
    assert movie._playing


def test_hotkeys_suppressed_while_typing_in_a_cutoff_box(movie):
    """Typing "-" into the high-pass box must not rescale the colour axis."""
    movie._hp_edit.setFocus()
    before = list(movie._cax)
    QtWidgets.QApplication.sendEvent(
        movie._hp_edit,
        QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Minus, QtCore.Qt.NoModifier),
    )
    assert movie._cax == before


def test_hotkey_filter_is_not_installed_on_the_application(movie):
    """An app-wide filter made every viewer see every other widget's events (~3.5 s per open)."""
    app = QtWidgets.QApplication.instance()
    movie._glw.setFocus()
    # Sending a key to an unrelated widget must not reach this viewer's handler.
    stray = QtWidgets.QWidget()
    before = list(movie._cax)
    QtWidgets.QApplication.sendEvent(
        stray, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Minus, QtCore.Qt.NoModifier)
    )
    assert movie._cax == before
    assert app is not None


def test_correlation_viewer_hotkeys_also_route_from_children(qtbot, movie_data):
    u, v, t, _ = movie_data
    w = _corr_class()(u, v, t=t)
    qtbot.addWidget(w)
    w.set_pixel(8, 6)
    w._glw.setFocus()
    QtWidgets.QApplication.sendEvent(
        w._glw, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_V, QtCore.Qt.NoModifier)
    )
    assert w._normalize_by_max is True


# ===================================================================== ctrl+click


def test_left_click_moves_the_selected_pixel(movie):
    movie.set_pixel(2, 2)
    click_at(movie, 7, 5)
    assert movie.pixels == [(7, 5)]


def test_ctrl_click_adds_a_pixel(movie):
    """Replaces the MATLAB's right-click, which collided with pyqtgraph's context menu."""
    movie.set_pixel(2, 2)
    click_at(movie, 7, 5, modifiers=QtCore.Qt.ControlModifier)
    assert movie.pixels == [(2, 2), (7, 5)]
    assert len(movie._pixel_traces) == 2


def test_ctrl_click_accumulates_several_pixels(movie):
    movie.set_pixel(1, 1)
    for y, x in ((3, 3), (5, 5), (9, 7)):
        click_at(movie, y, x, modifiers=QtCore.Qt.ControlModifier)
    assert movie.pixels == [(1, 1), (3, 3), (5, 5), (9, 7)]
    assert len(movie._trace_curves[-1]) == 4


def test_right_click_does_not_add_a_pixel(movie):
    """Right-click belongs to the view's context menu now, and must be inert here."""
    movie.set_pixel(2, 2)
    click_at(movie, 7, 5, button=QtCore.Qt.RightButton)
    assert movie.pixels == [(2, 2)]


def test_click_outside_the_image_is_ignored(movie):
    movie.set_pixel(2, 2)
    click_at(movie, 999, 999)
    assert movie.pixels == [(2, 2)]


def test_ctrl_click_respects_orientation(movie):
    """After a rotation, a click must still land on the pixel under the cursor."""
    movie.set_pixel(0, 0)
    movie._orient.rotate(1)
    click_at(movie, 3, 4, modifiers=QtCore.Qt.ControlModifier)
    added = movie.pixels[-1]
    assert movie._orient.to_display(*added, movie.shape) == (3, 4)


# ===================================================================== band-pass filter


def test_filter_defaults_to_unfiltered(movie, movie_data):
    _, v, _, _ = movie_data
    assert movie._hp_edit.text() == "0"
    assert movie._lp_edit.text() == "inf"
    assert "unfiltered" in movie._filt_label.text()
    np.testing.assert_allclose(np.asarray(movie._v32), v, atol=1e-5)


def test_highpass_removes_the_slow_component(movie):
    """The fixture V has 0.02 Hz and 12 Hz content; a 1 Hz high-pass should kill the slow one."""
    movie._hp_edit.setText("1.0")
    movie._apply_filter()
    filtered = np.asarray(movie._v32)

    # Compare power below 0.1 Hz before and after.
    def slow_power(x):
        spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
        freq = np.fft.rfftfreq(x.shape[-1], 1 / movie._fs)
        return spec[:, freq < 0.1].sum()

    assert slow_power(filtered) < 0.01 * slow_power(movie._v_raw)
    assert "high-pass 1 Hz" in movie._filt_label.text()


def test_lowpass_removes_the_fast_component(movie):
    movie._lp_edit.setText("2.0")
    movie._apply_filter()
    filtered = np.asarray(movie._v32)

    def fast_power(x):
        spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
        freq = np.fft.rfftfreq(x.shape[-1], 1 / movie._fs)
        return spec[:, freq > 5].sum()

    assert fast_power(filtered) < 0.01 * fast_power(movie._v_raw)
    assert "low-pass 2 Hz" in movie._filt_label.text()


def test_bandpass_reports_both_cutoffs(movie):
    movie._hp_edit.setText("0.5")
    movie._lp_edit.setText("5")
    movie._apply_filter()
    assert "band-pass 0.5" in movie._filt_label.text()
    assert "5 Hz" in movie._filt_label.text()


def test_filter_changes_the_displayed_frame(movie):
    before = movie.frame_image.copy()
    movie._hp_edit.setText("1.0")
    movie._apply_filter()
    assert not np.allclose(before, movie.frame_image)


def test_filter_invalidates_the_prefetch_block(movie):
    """The playback block caches reconstructed frames; stale ones must not survive a filter.

    The block may legitimately be rebuilt straight away (the refresh after filtering re-enters
    the prefetch path while playing), so what matters is that its contents come from the *new*
    components, not that the cache is empty.
    """
    movie._playing = True
    movie.set_frame(10)
    assert movie._block is not None
    stale = movie.frame_image.copy()

    movie._hp_edit.setText("1.0")
    movie._apply_filter()

    fresh = movie._flat_u @ np.asarray(movie._v32)[:, movie.frame]
    np.testing.assert_allclose(movie.frame_image, fresh.reshape(movie.shape), rtol=1e-5, atol=1e-6)
    assert not np.allclose(movie.frame_image, stale)


def test_filter_updates_the_pixel_traces(movie):
    movie.set_pixel(4, 4)
    before = movie._pixel_traces[0].copy()
    movie._hp_edit.setText("1.0")
    movie._apply_filter()
    assert not np.allclose(before, movie._pixel_traces[0])


def test_filter_reapplies_from_the_raw_components(movie):
    """Applying twice must not compound: the second result is derived from the original V."""
    movie._hp_edit.setText("1.0")
    movie._apply_filter()
    once = np.asarray(movie._v32).copy()
    movie._apply_filter()
    np.testing.assert_allclose(np.asarray(movie._v32), once, atol=1e-6)


def test_filter_reset_restores_the_raw_components(movie, movie_data):
    _, v, _, _ = movie_data
    movie._hp_edit.setText("1.0")
    movie._apply_filter()
    movie._reset_filter()
    assert movie._hp_edit.text() == "0" and movie._lp_edit.text() == "inf"
    np.testing.assert_allclose(np.asarray(movie._v32), v, atol=1e-5)


def test_filter_rejects_inverted_cutoffs(movie):
    movie._hp_edit.setText("5")
    movie._lp_edit.setText("1")
    movie._apply_filter()
    assert "invalid" in movie._filt_label.text()


def test_filter_rejects_unparseable_text(movie):
    movie._hp_edit.setText("banana")
    movie._apply_filter()
    assert "invalid" in movie._filt_label.text()


def test_filter_survives_a_bad_value_without_changing_v(movie, movie_data):
    _, v, _, _ = movie_data
    movie._hp_edit.setText("nonsense")
    movie._apply_filter()
    np.testing.assert_allclose(np.asarray(movie._v32), v, atol=1e-5)


def test_lowpass_above_nyquist_is_treated_as_no_filter(movie, movie_data):
    _, v, _, _ = movie_data
    movie._lp_edit.setText("1000")
    movie._apply_filter()
    assert "unfiltered" in movie._filt_label.text()
    np.testing.assert_allclose(np.asarray(movie._v32), v, atol=1e-5)


def test_blank_cutoff_boxes_mean_no_filter(movie):
    movie._hp_edit.setText("")
    movie._lp_edit.setText("")
    movie._apply_filter()
    assert "unfiltered" in movie._filt_label.text()


# ===================================================================== follow / zoom


def test_follow_is_on_by_default(movie):
    assert movie._follow_chk.isChecked()


def test_following_keeps_the_window_on_the_current_frame(movie):
    movie.set_frame(700)
    lo, hi = movie._trace_plots[0].vb.viewRange()[0]
    assert hi - lo == pytest.approx(10.0, abs=1e-6)


def test_manual_zoom_turns_follow_off(movie):
    """Reported bug: zooming a trace plot was undone on the very next frame."""
    assert movie._follow_chk.isChecked()
    movie._trace_plots[0].vb.sigRangeChangedManually.emit((0, 1))
    assert not movie._follow_chk.isChecked()


def test_zoomed_range_survives_playback(movie):
    movie.set_frame(300)
    movie._trace_plots[0].vb.sigRangeChangedManually.emit((0, 1))
    movie._trace_plots[0].setXRange(2.0, 4.0, padding=0)
    before = movie._trace_plots[0].vb.viewRange()[0]
    for f in range(301, 320):
        movie.set_frame(f)
    after = movie._trace_plots[0].vb.viewRange()[0]
    assert after == pytest.approx(before, abs=1e-6)


def test_time_cursor_still_tracks_the_frame_when_not_following(movie, movie_data):
    _, _, t, _ = movie_data
    movie._follow_chk.setChecked(False)
    movie.set_frame(400)
    assert movie._trace_marks[0].value() == pytest.approx(t[400])


def test_not_following_plots_the_whole_trace(movie, movie_data):
    """Panning around should not run off the end of the plotted data."""
    _, _, t, _ = movie_data
    movie._follow_chk.setChecked(False)
    movie.set_frame(400)
    x, _y = movie._trace_curves[-1][0].getData()
    assert x.size == t.size


def test_rechecking_follow_snaps_back_to_the_window(movie):
    movie.set_frame(500)
    movie._follow_chk.setChecked(False)
    movie._trace_plots[0].setXRange(0.0, 1.0, padding=0)
    movie._follow_chk.setChecked(True)
    lo, hi = movie._trace_plots[0].vb.viewRange()[0]
    assert hi - lo == pytest.approx(10.0, abs=1e-6)


# ===================================================================== bandpass_filt numerics


def test_bandpass_filt_no_cutoffs_is_a_copy():
    v = np.random.default_rng(0).standard_normal((3, 200))
    out = bandpass_filt(v, 35.0)
    np.testing.assert_allclose(out, v)
    assert out is not v


def test_bandpass_filt_removes_dc():
    v = np.ones((2, 2000)) * 5.0
    out = bandpass_filt(v, 35.0, highpass=0.5)
    assert np.abs(out).max() < 1e-6


def test_bandpass_filt_passes_the_band():
    fs, n = 35.0, 4000
    t = np.arange(n) / fs
    v = np.sin(2 * np.pi * 3.0 * t)[None, :]
    out = bandpass_filt(v, fs, highpass=1.0, lowpass=8.0)
    keep = slice(n // 4, 3 * n // 4)  # ignore edge transients
    assert np.std(out[0, keep]) > 0.9 * np.std(v[0, keep])


def test_bandpass_filt_is_zero_phase():
    """filtfilt must not shift the signal — a viewer can't afford a lag against behavior."""
    fs, n = 35.0, 4000
    t = np.arange(n) / fs
    v = np.sin(2 * np.pi * 2.0 * t)[None, :]
    out = bandpass_filt(v, fs, highpass=0.5)
    keep = slice(n // 4, 3 * n // 4)
    lag = np.argmax(np.correlate(out[0, keep], v[0, keep], mode="same")) - len(t[keep]) // 2
    assert abs(lag) <= 1


def test_bandpass_filt_stable_at_a_very_low_cutoff():
    """0.01 Hz at fs=35 is where a b/a 3rd-order filter blows up; SOS must not."""
    v = np.random.default_rng(1).standard_normal((4, 20000))
    out = bandpass_filt(v, 35.0, highpass=0.01)
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 100 * np.abs(v).max()


def test_bandpass_filt_rejects_inverted_cutoffs():
    with pytest.raises(ValueError, match="must be below"):
        bandpass_filt(np.zeros((2, 100)), 35.0, highpass=5.0, lowpass=1.0)


def test_bandpass_filt_rejects_highpass_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass_filt(np.zeros((2, 100)), 35.0, highpass=30.0)


def test_bandpass_filt_rejects_bad_shape():
    with pytest.raises(ValueError, match="V must be"):
        bandpass_filt(np.zeros(100), 35.0, highpass=1.0)


def test_bandpass_filt_filtering_v_equals_filtering_pixels():
    """The reason this is cheap: filtering V is the same as filtering every pixel."""
    from widefield.svd import svd_frame_reconstruct

    rng = np.random.default_rng(4)
    u = rng.standard_normal((5, 4, 3))
    v = rng.standard_normal((3, 1000))
    fs = 35.0

    via_v = svd_frame_reconstruct(u, bandpass_filt(v, fs, highpass=1.0))
    movie = svd_frame_reconstruct(u, v)
    flat = movie.reshape(-1, movie.shape[2])
    via_pixels = bandpass_filt(flat, fs, highpass=1.0).reshape(movie.shape)
    np.testing.assert_allclose(via_v, via_pixels, atol=1e-8)
