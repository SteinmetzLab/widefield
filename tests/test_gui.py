"""Behavioral tests for the three viewers.

Run headless (``QT_QPA_PLATFORM=offscreen``, set in conftest). These check *behavior* — that a
key or click changes the state it is supposed to, and that what is displayed matches what the
numerics say — rather than pixels. Where a viewer computes something the pure-numpy layer also
computes, the two are cross-checked, so a GUI regression can't quietly diverge from the math.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui  # noqa: E402

from widefield.correlation import SeedCorrelation  # noqa: E402
from widefield.events import event_locked_avg_svd  # noqa: E402
from widefield.gui._common import polygon_mask  # noqa: E402
from widefield.gui.movie import Trace  # noqa: E402
from widefield.gui.movie import _get_class as _movie_class  # noqa: E402
from widefield.gui.pixel_correlation import _get_class as _corr_class  # noqa: E402
from widefield.gui.pixel_tuning_curve import _get_class as _tuning_class  # noqa: E402
from widefield.svd import pixel_timecourse, svd_frame_reconstruct  # noqa: E402

# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def movie_data():
    rng = np.random.default_rng(7)
    ypix, xpix, nsv, nframes = 18, 12, 6, 1200
    q, _ = np.linalg.qr(rng.standard_normal((ypix * xpix, nsv)))
    u = q.reshape(ypix, xpix, nsv)
    v = rng.standard_normal((nsv, nframes)).astype(np.float32)
    t = np.arange(nframes) / 35.0
    return u, v, t


@pytest.fixture(scope="module")
def event_data(movie_data):
    """Invented stimulus times with four 'contrast' conditions — enough to drive the viewer."""
    _, _, t = movie_data
    rng = np.random.default_rng(11)
    times = np.sort(rng.uniform(1.0, t[-1] - 2.0, 60))
    labels = np.tile([0.0, 0.25, 0.5, 1.0], 15)
    return times, labels


def press(widget, key, mods=QtCore.Qt.NoModifier):
    widget.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, mods))


# =========================================================================== correlation


@pytest.fixture
def corr_viewer(qtbot, movie_data):
    u, v, t = movie_data
    w = _corr_class()(u, v, t=t)
    qtbot.addWidget(w)
    return w


def test_corr_opens_with_seed_in_the_middle(corr_viewer, movie_data):
    u, _, _ = movie_data
    assert corr_viewer.pixel == (u.shape[0] // 2, u.shape[1] // 2)


def test_corr_map_matches_the_numpy_layer(corr_viewer, movie_data):
    """The GUI must display exactly what SeedCorrelation computes — no drift."""
    u, v, _ = movie_data
    corr_viewer.set_pixel(4, 3)
    expected = SeedCorrelation(u, v, dtype=np.float32).map((4, 3))
    np.testing.assert_allclose(corr_viewer.correlation_map, expected, atol=1e-6)


def test_corr_seed_is_perfectly_correlated_with_itself(corr_viewer):
    corr_viewer.set_pixel(6, 5)
    assert corr_viewer.correlation_map[6, 5] == pytest.approx(1.0, abs=1e-5)


def test_corr_arrow_keys_step_five_pixels(corr_viewer):
    corr_viewer.set_pixel(8, 5)
    press(corr_viewer, QtCore.Qt.Key_Right)
    assert corr_viewer.pixel == (8, 10)
    press(corr_viewer, QtCore.Qt.Key_Down)
    assert corr_viewer.pixel == (13, 10)


def test_corr_ctrl_arrow_steps_one_pixel(corr_viewer):
    corr_viewer.set_pixel(8, 5)
    press(corr_viewer, QtCore.Qt.Key_Right, QtCore.Qt.ControlModifier)
    assert corr_viewer.pixel == (8, 6)


def test_corr_arrows_clamp_at_the_border(corr_viewer):
    corr_viewer.set_pixel(0, 0)
    for _ in range(10):
        press(corr_viewer, QtCore.Qt.Key_Up)
        press(corr_viewer, QtCore.Qt.Key_Left)
    assert corr_viewer.pixel == (0, 0)


def test_corr_v_key_toggles_variance_normalization(corr_viewer, movie_data):
    u, v, _ = movie_data
    corr_viewer.set_pixel(5, 5)
    true_corr = corr_viewer.correlation_map.copy()
    press(corr_viewer, QtCore.Qt.Key_V)
    normalized = corr_viewer.correlation_map
    assert not np.allclose(true_corr, normalized)
    expected = SeedCorrelation(u, v, dtype=np.float32).map((5, 5), normalize_by_max=True)
    np.testing.assert_allclose(normalized, expected, atol=1e-6)
    press(corr_viewer, QtCore.Qt.Key_V)
    np.testing.assert_allclose(corr_viewer.correlation_map, true_corr, atol=1e-9)


def test_corr_hover_is_on_by_default(corr_viewer):
    """Unlike the MATLAB: a map costs ~9 ms, so sweeping the mouse is the fastest way to read."""
    assert corr_viewer._hover is True
    assert "hover on" in corr_viewer._status.text()


def test_corr_h_key_toggles_hover(corr_viewer):
    assert corr_viewer._hover is True
    press(corr_viewer, QtCore.Qt.Key_H)
    assert corr_viewer._hover is False
    assert "hover off" in corr_viewer._status.text()
    press(corr_viewer, QtCore.Qt.Key_H)
    assert corr_viewer._hover is True


def test_corr_alt_arrows_rotate_and_transpose_the_display(corr_viewer):
    before = corr_viewer._image.image.shape
    press(corr_viewer, QtCore.Qt.Key_Right, QtCore.Qt.AltModifier)
    assert corr_viewer._image.image.shape == before[::-1]


def test_corr_alt_updown_flips(corr_viewer):
    press(corr_viewer, QtCore.Qt.Key_Up, QtCore.Qt.AltModifier)
    assert corr_viewer._orient.flip is True


def test_corr_r_resets_orientation(corr_viewer):
    press(corr_viewer, QtCore.Qt.Key_Right, QtCore.Qt.AltModifier)
    press(corr_viewer, QtCore.Qt.Key_Up, QtCore.Qt.AltModifier)
    press(corr_viewer, QtCore.Qt.Key_R)
    assert (corr_viewer._orient.rot, corr_viewer._orient.flip) == (0, False)


def test_corr_arrows_stay_screen_relative_after_rotation(corr_viewer):
    """The MATLAB remaps arrow keys when rotated; we must match that behavior."""
    corr_viewer.set_pixel(8, 5)
    press(corr_viewer, QtCore.Qt.Key_Right, QtCore.Qt.AltModifier)
    before = corr_viewer._orient.to_display(*corr_viewer.pixel, corr_viewer.shape)
    press(corr_viewer, QtCore.Qt.Key_Right)
    after = corr_viewer._orient.to_display(*corr_viewer.pixel, corr_viewer.shape)
    assert (after[0] - before[0], after[1] - before[1]) == (0, 5)


def test_corr_displayed_array_follows_the_orientation(corr_viewer):
    corr_viewer.set_pixel(4, 3)
    press(corr_viewer, QtCore.Qt.Key_Right, QtCore.Qt.AltModifier)
    expected = corr_viewer._orient.apply(corr_viewer.correlation_map)
    np.testing.assert_allclose(corr_viewer._image.image, expected, atol=1e-9)


def test_corr_marker_tracks_the_seed_in_display_space(corr_viewer):
    corr_viewer.set_pixel(3, 7)
    press(corr_viewer, QtCore.Qt.Key_Left, QtCore.Qt.AltModifier)
    dy, dx = corr_viewer._orient.to_display(3, 7, corr_viewer.shape)
    spot = corr_viewer._marker.getData()
    assert spot[0][0] == pytest.approx(dx + 0.5)
    assert spot[1][0] == pytest.approx(dy + 0.5)


def test_corr_trace_matches_pixel_timecourse(corr_viewer, movie_data):
    u, v, _ = movie_data
    corr_viewer.set_pixel(2, 9)
    _, y = corr_viewer._trace_curve.getData()
    np.testing.assert_allclose(y, pixel_timecourse(u, v, (2, 9)), rtol=1e-5, atol=1e-4)


def test_corr_levels_fixed_for_true_correlation(corr_viewer):
    assert corr_viewer._levels() == [-1.0, 1.0]


def test_corr_set_pixel_rejects_out_of_bounds(corr_viewer):
    with pytest.raises(IndexError, match="outside image"):
        corr_viewer.set_pixel(999, 0)


def test_corr_max_components_is_respected(qtbot, movie_data):
    u, v, t = movie_data
    w = _corr_class()(u, v, t=t, max_components=3)
    qtbot.addWidget(w)
    assert w._corr.n_components == 3


# =========================================================================== tuning


@pytest.fixture
def tuning_viewer(qtbot, movie_data, event_data):
    u, v, t = movie_data
    times, labels = event_data
    w = _tuning_class()(u, v, t, times, labels, (-0.3, 0.8))
    qtbot.addWidget(w)
    return w


def test_tuning_finds_four_conditions(tuning_viewer):
    np.testing.assert_allclose(tuning_viewer._conditions, [0.0, 0.25, 0.5, 1.0])


def test_tuning_traces_match_the_numpy_layer(tuning_viewer, movie_data, event_data):
    """Cross-check the middle panel against event_locked_avg_svd at the viewer's own upsampling.

    The viewer builds its traces from the *pixel's* peri-event matrix (so it can also report a
    per-event s.e.m.), while avg_v averages in component space. Projection and averaging are both
    linear, so the two must agree exactly — this is the test that keeps them honest.
    """
    u, v, t = movie_data
    times, labels = event_data
    tuning_viewer.set_pixel(5, 4)
    avg = event_locked_avg_svd(v, t, times, labels, (-0.3, 0.8), upsample=tuning_viewer._upsample)
    expected = np.stack([u[5, 4, :] @ avg.avg_v[c] for c in range(avg.conditions.size)])
    np.testing.assert_allclose(tuning_viewer.traces, expected, rtol=1e-4, atol=1e-4)


def test_tuning_curve_is_the_traces_column_at_the_selected_time(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_Right)
    press(tuning_viewer, QtCore.Qt.Key_Right)
    np.testing.assert_allclose(
        tuning_viewer.tuning_curve, tuning_viewer.traces[:, tuning_viewer.time_index]
    )


def test_tuning_brain_image_matches_reconstruction(tuning_viewer, movie_data):
    u, _, _ = movie_data
    c, k = tuning_viewer.condition_index, tuning_viewer.time_index
    expected = svd_frame_reconstruct(u, tuning_viewer._avg.avg_v[c, :, k])
    np.testing.assert_allclose(tuning_viewer.brain_image, expected, rtol=1e-3, atol=1e-3)


def test_tuning_time_steps_and_clamps(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_Right)
    assert tuning_viewer.time_index == 1
    for _ in range(5):
        press(tuning_viewer, QtCore.Qt.Key_Left)
    assert tuning_viewer.time_index == 0
    for _ in range(tuning_viewer._n_time + 5):
        press(tuning_viewer, QtCore.Qt.Key_Right)
    assert tuning_viewer.time_index == tuning_viewer._n_time - 1


def test_tuning_condition_wraps_both_ways(tuning_viewer):
    """MATLAB wraps condition selection rather than clamping it."""
    assert tuning_viewer.condition_index == 0
    press(tuning_viewer, QtCore.Qt.Key_Down)
    assert tuning_viewer.condition_index == tuning_viewer._n_cond - 1
    press(tuning_viewer, QtCore.Qt.Key_Up)
    assert tuning_viewer.condition_index == 0


def test_tuning_ijkl_move_the_pixel(tuning_viewer):
    tuning_viewer.set_pixel(9, 6)
    press(tuning_viewer, QtCore.Qt.Key_L)
    assert tuning_viewer.pixel == (9, 11)
    press(tuning_viewer, QtCore.Qt.Key_J)
    assert tuning_viewer.pixel == (9, 6)
    press(tuning_viewer, QtCore.Qt.Key_I)
    assert tuning_viewer.pixel == (14, 6)
    press(tuning_viewer, QtCore.Qt.Key_K)
    assert tuning_viewer.pixel == (9, 6)


def test_tuning_ijkl_stay_screen_relative_after_rotation(tuning_viewer):
    tuning_viewer.set_pixel(9, 6)
    press(tuning_viewer, QtCore.Qt.Key_Right, QtCore.Qt.AltModifier)
    before = tuning_viewer._orient.to_display(*tuning_viewer.pixel, tuning_viewer.shape)
    press(tuning_viewer, QtCore.Qt.Key_L)
    after = tuning_viewer._orient.to_display(*tuning_viewer.pixel, tuning_viewer.shape)
    assert (after[0] - before[0], after[1] - before[1]) == (0, 5)


def test_tuning_caxis_keys_scale_symmetrically(tuning_viewer):
    start = list(tuning_viewer._cax)
    press(tuning_viewer, QtCore.Qt.Key_Minus)
    np.testing.assert_allclose(tuning_viewer._cax, [c * 0.75 for c in start])
    press(tuning_viewer, QtCore.Qt.Key_Equal)
    np.testing.assert_allclose(tuning_viewer._cax, [c * 0.9375 for c in start])


def test_tuning_caxis_stays_centerd_on_zero(tuning_viewer):
    for _ in range(4):
        press(tuning_viewer, QtCore.Qt.Key_Minus)
    assert tuning_viewer._cax[0] == pytest.approx(-tuning_viewer._cax[1])


def test_tuning_playback_advances_and_wraps(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_P)
    assert tuning_viewer._playing
    tuning_viewer._time_idx = tuning_viewer._n_time - 1
    tuning_viewer._on_tick()
    assert tuning_viewer.time_index == 0
    press(tuning_viewer, QtCore.Qt.Key_P)
    assert not tuning_viewer._playing


def test_tuning_rate_doubles_and_floors_at_one(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_F)
    assert tuning_viewer._rate == 2
    press(tuning_viewer, QtCore.Qt.Key_F)
    assert tuning_viewer._rate == 4
    for _ in range(6):
        press(tuning_viewer, QtCore.Qt.Key_S)
    assert tuning_viewer._rate == 1


def test_tuning_playback_honours_the_rate(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_F)  # rate 2
    tuning_viewer._time_idx = 0
    tuning_viewer._on_tick()
    assert tuning_viewer.time_index == 2


def _wheel(mods=QtCore.Qt.NoModifier):
    return QtGui.QWheelEvent(
        QtCore.QPointF(0, 0),
        QtCore.QPointF(0, 0),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton,
        mods,
        QtCore.Qt.NoScrollPhase,
        False,
    )


def test_tuning_ctrl_wheel_steps_time(tuning_viewer):
    tuning_viewer._time_idx = 5
    tuning_viewer.wheelEvent(_wheel(QtCore.Qt.ControlModifier))
    assert tuning_viewer.time_index != 5


def test_tuning_plain_wheel_is_left_to_pyqtgraph(tuning_viewer):
    """A bare scroll should zoom the plots (pyqtgraph's own behavior), not step time."""
    tuning_viewer._time_idx = 5
    tuning_viewer.wheelEvent(_wheel())
    assert tuning_viewer.time_index == 5


def test_tuning_roi_gives_the_mean_over_its_pixels(tuning_viewer, movie_data):
    u, _, _ = movie_data
    press(tuning_viewer, QtCore.Qt.Key_R)
    assert tuning_viewer.roi is not None
    mask = tuning_viewer.roi["mask"]
    assert mask.any() and mask.shape == tuning_viewer.shape

    # ROI traces must equal the mean of the per-pixel weights, applied to the averaged V.
    weights = tuning_viewer._flat_u[mask.reshape(-1)].mean(axis=0)
    expected = np.einsum("s,csw->cw", weights, tuning_viewer._avg_v)
    np.testing.assert_allclose(tuning_viewer.traces, expected, rtol=1e-5, atol=1e-5)


def test_tuning_roi_result_carries_shape_metadata(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_R)
    roi = tuning_viewer.roi
    assert roi["traces"].shape == (tuning_viewer._n_cond, tuning_viewer._n_time)
    assert roi["n_pixels"] == int(roi["mask"].sum())
    np.testing.assert_allclose(roi["win_samps"], tuning_viewer._win_samps)


def test_tuning_roi_toggles_off_and_restores_pixel_traces(tuning_viewer):
    tuning_viewer.set_pixel(6, 4)
    pixel_traces = tuning_viewer.traces.copy()
    press(tuning_viewer, QtCore.Qt.Key_R)
    assert not np.allclose(tuning_viewer.traces, pixel_traces)
    press(tuning_viewer, QtCore.Qt.Key_R)
    assert tuning_viewer.roi is None
    np.testing.assert_allclose(tuning_viewer.traces, pixel_traces)


def test_tuning_marker_hidden_while_roi_active(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_R)
    assert not tuning_viewer._marker.isVisible()
    press(tuning_viewer, QtCore.Qt.Key_R)
    assert tuning_viewer._marker.isVisible()


def test_tuning_string_labels_get_evenly_spaced_ticks(qtbot, movie_data):
    u, v, t = movie_data
    times = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array(["left", "right", "left", "right"])
    w = _tuning_class()(u, v, t, times, labels, (-0.2, 0.4))
    qtbot.addWidget(w)
    assert not w._numeric_labels
    np.testing.assert_allclose(w._cond_x, [0.0, 1.0])
    assert "left" in w._status.text()


def test_tuning_numeric_labels_are_their_own_x_axis(tuning_viewer):
    assert tuning_viewer._numeric_labels
    np.testing.assert_allclose(tuning_viewer._cond_x, [0.0, 0.25, 0.5, 1.0])


def test_tuning_empty_window_raises(qtbot, movie_data, event_data):
    u, v, t = movie_data
    times, labels = event_data
    with pytest.raises(ValueError, match="no samples"):
        _tuning_class()(u, v, t, times, labels, (0.5, 0.0))


def test_tuning_selected_condition_line_is_thicker(tuning_viewer):
    press(tuning_viewer, QtCore.Qt.Key_Up)  # condition 1
    widths = [c.opts["pen"].width() for c in tuning_viewer._trace_curves]
    assert widths[tuning_viewer.condition_index] > widths[0]


# =========================================================================== movie


@pytest.fixture
def movie_viewer(qtbot, movie_data):
    u, v, t = movie_data
    traces = [
        Trace(t=t, v=np.sin(2 * np.pi * 0.3 * t), name="wheel"),
        Trace(t=t, v=np.cos(2 * np.pi * 0.1 * t), name="pupil", lims=(-1.5, 1.5)),
    ]
    w = _movie_class()(u, v, t=t, traces=traces)
    qtbot.addWidget(w)
    return w


def test_movie_starts_paused_on_frame_zero(movie_viewer):
    assert movie_viewer.frame == 0
    assert not movie_viewer._playing


def test_movie_frame_image_matches_reconstruction(movie_viewer, movie_data):
    u, v, _ = movie_data
    movie_viewer.set_frame(17)
    expected = svd_frame_reconstruct(u, v[:, 17])
    np.testing.assert_allclose(movie_viewer.frame_image, expected, rtol=1e-4, atol=1e-4)


def test_movie_play_toggle(movie_viewer):
    press(movie_viewer, QtCore.Qt.Key_P)
    assert movie_viewer._playing and movie_viewer._play_btn.text() == "pause"
    press(movie_viewer, QtCore.Qt.Key_P)
    assert not movie_viewer._playing


def test_movie_speed_doubles_and_halves(movie_viewer):
    assert movie_viewer.speed == 1.0
    press(movie_viewer, QtCore.Qt.Key_Up)
    assert movie_viewer.speed == 2.0
    press(movie_viewer, QtCore.Qt.Key_Up)
    assert movie_viewer.speed == 4.0
    press(movie_viewer, QtCore.Qt.Key_Down)
    assert movie_viewer.speed == 2.0


def test_movie_speed_allows_slow_motion(movie_viewer):
    """Unlike the MATLAB's integer frame-step, speed can go below 1 for slow motion."""
    for _ in range(3):
        press(movie_viewer, QtCore.Qt.Key_Down)
    assert movie_viewer.speed == pytest.approx(0.125)


def test_movie_speed_is_clamped(movie_viewer):
    for _ in range(20):
        press(movie_viewer, QtCore.Qt.Key_Up)
    assert movie_viewer.speed <= 64
    for _ in range(40):
        press(movie_viewer, QtCore.Qt.Key_Down)
    assert movie_viewer.speed >= 1 / 32


def test_movie_playback_follows_the_wall_clock(movie_viewer, monkeypatch):
    """Frames come from elapsed time, not one-per-tick, so playback holds its speed.

    This is the fix for the reported "plays at ~2 fps": rendering a big window is slower than the
    frame rate, and advancing one frame per tick made the recording crawl instead of dropping
    frames.
    """
    import widefield.gui.movie as mwt

    clock = [1000.0]
    monkeypatch.setattr(mwt.time, "perf_counter", lambda: clock[0])

    movie_viewer._playing = True
    movie_viewer.set_frame(0)  # anchors the clock at t=1000
    fs = movie_viewer._fs

    clock[0] += 1.0  # one second of wall time
    movie_viewer._on_tick()
    assert movie_viewer.frame == pytest.approx(int(fs), abs=1)  # ~fs frames later

    clock[0] += 2.0
    movie_viewer._on_tick()
    assert movie_viewer.frame == pytest.approx(int(3 * fs), abs=1)


def test_movie_playback_drops_frames_rather_than_slowing(movie_viewer, monkeypatch):
    """A single slow tick must jump, not fall behind."""
    import widefield.gui.movie as mwt

    clock = [500.0]
    monkeypatch.setattr(mwt.time, "perf_counter", lambda: clock[0])
    movie_viewer._playing = True
    movie_viewer.set_frame(0)

    clock[0] += 0.5  # a single 500 ms stall
    movie_viewer._on_tick()
    assert movie_viewer.frame > 10  # jumped ahead, not one frame


def test_movie_speed_scales_the_advance(movie_viewer, monkeypatch):
    import widefield.gui.movie as mwt

    clock = [0.0]
    monkeypatch.setattr(mwt.time, "perf_counter", lambda: clock[0])
    movie_viewer._playing = True
    movie_viewer.set_speed(4.0)
    movie_viewer.set_frame(0)
    clock[0] += 1.0
    movie_viewer._on_tick()
    assert movie_viewer.frame == pytest.approx(int(4 * movie_viewer._fs), abs=2)


def test_movie_playback_restarts_at_the_end(movie_viewer, monkeypatch):
    """MATLAB restarts at frame 1 rather than wrapping modulo; keep that."""
    import widefield.gui.movie as mwt

    clock = [0.0]
    monkeypatch.setattr(mwt.time, "perf_counter", lambda: clock[0])
    movie_viewer._playing = True
    movie_viewer.set_frame(movie_viewer._n_frames - 2)
    clock[0] += 10.0
    movie_viewer._on_tick()
    assert movie_viewer.frame == 0


def test_movie_b_jumps_back_half_a_second(movie_viewer):
    fs = movie_viewer._fs
    movie_viewer.set_frame(500)
    press(movie_viewer, QtCore.Qt.Key_B)
    assert movie_viewer.frame == 500 - int(round(0.5 * fs))
    movie_viewer.set_frame(500)
    press(movie_viewer, QtCore.Qt.Key_Up)  # speed 2
    press(movie_viewer, QtCore.Qt.Key_B)
    assert movie_viewer.frame == 500 - int(round(0.5 * fs * 2))


def test_movie_b_clamps_at_zero(movie_viewer):
    movie_viewer.set_frame(5)
    press(movie_viewer, QtCore.Qt.Key_B)
    assert movie_viewer.frame == 0


def test_movie_home_returns_to_start(movie_viewer):
    movie_viewer.set_frame(300)
    press(movie_viewer, QtCore.Qt.Key_Home)
    assert movie_viewer.frame == 0


def test_movie_slider_and_frame_stay_in_sync(movie_viewer):
    movie_viewer.set_frame(123)
    assert movie_viewer._slider.value() == 123
    movie_viewer._slider.setValue(45)
    assert movie_viewer.frame == 45


def test_movie_pixel_traces_match_pixel_timecourse(movie_viewer, movie_data):
    u, v, _ = movie_data
    movie_viewer.set_pixel(4, 6)
    np.testing.assert_allclose(
        movie_viewer._pixel_traces[-1], pixel_timecourse(u, v, (4, 6)), rtol=1e-4, atol=1e-3
    )


def test_movie_add_pixel_keeps_the_previous_one(movie_viewer):
    movie_viewer.set_pixel(3, 3)
    movie_viewer.add_pixel(9, 8)
    assert movie_viewer.pixels == [(3, 3), (9, 8)]
    assert len(movie_viewer._pixel_traces) == 2


def test_movie_c_clears_all_but_the_last_pixel(movie_viewer):
    movie_viewer.add_pixel(2, 2)
    movie_viewer.add_pixel(7, 7)
    press(movie_viewer, QtCore.Qt.Key_C)
    assert movie_viewer.pixels == [(7, 7)]
    assert len(movie_viewer._trace_curves[-1]) == 1


def test_movie_each_pixel_gets_a_curve(movie_viewer):
    movie_viewer.add_pixel(2, 2)
    movie_viewer.add_pixel(5, 5)
    assert len(movie_viewer._trace_curves[-1]) == len(movie_viewer.pixels) == 3


def test_movie_pixel_colors_cycle_after_five(movie_viewer):
    """MATLAB cycles through 5 colors even though 7 are defined."""
    from widefield.gui.movie import _N_CYCLE, _PIXEL_COLORS

    assert _N_CYCLE == 5
    np.testing.assert_allclose(_PIXEL_COLORS[_N_CYCLE % _N_CYCLE], _PIXEL_COLORS[0])


def test_movie_caxis_scales_symmetrically(movie_viewer):
    assert movie_viewer._cax == [-0.4, 0.4]
    press(movie_viewer, QtCore.Qt.Key_Minus)
    np.testing.assert_allclose(movie_viewer._cax, [-0.3, 0.3])
    press(movie_viewer, QtCore.Qt.Key_Equal)
    np.testing.assert_allclose(movie_viewer._cax, [-0.375, 0.375])


def test_movie_alt_arrows_rotate_the_display(movie_viewer):
    before = movie_viewer._image.image.shape
    press(movie_viewer, QtCore.Qt.Key_Right, QtCore.Qt.AltModifier)
    assert movie_viewer._image.image.shape == before[::-1]


def test_movie_trace_window_follows_the_current_time(movie_viewer, movie_data):
    _, _, t = movie_data
    movie_viewer.set_frame(600)
    now = t[600]
    x, _ = movie_viewer._trace_curves[0][0].getData()
    assert x.min() >= now - 5.001 and x.max() <= now + 5.001


def test_movie_trace_window_is_ten_seconds_wide(movie_viewer):
    movie_viewer.set_frame(600)
    lo, hi = movie_viewer._trace_plots[0].vb.viewRange()[0]
    assert hi - lo == pytest.approx(10.0, abs=1e-6)


def test_movie_has_a_panel_per_trace_plus_pixels(movie_viewer):
    assert len(movie_viewer._trace_plots) == 3  # wheel, pupil, selected pixels


def test_movie_respects_explicit_trace_limits(movie_viewer):
    lo, hi = movie_viewer._trace_plots[1].vb.viewRange()[1]
    assert (lo, hi) == pytest.approx((-1.5, 1.5))


def test_movie_accepts_dict_traces(qtbot, movie_data):
    """Parity with the MATLAB struct array — dicts must work as well as Trace objects."""
    u, v, t = movie_data
    w = _movie_class()(u, v, t=t, traces=[{"t": t, "v": t * 0.0, "name": "flat"}])
    qtbot.addWidget(w)
    assert len(w._trace_plots) == 2


def test_movie_rejects_bad_trace_type(qtbot, movie_data):
    u, v, t = movie_data
    with pytest.raises(TypeError, match="Trace or dict"):
        _movie_class()(u, v, t=t, traces=[object()])


def test_movie_rejects_mismatched_trace_lengths(movie_data):
    _, _, t = movie_data
    with pytest.raises(ValueError, match="samples"):
        Trace(t=t, v=t[:-1], name="bad")


def test_movie_rejects_t_of_wrong_length(qtbot, movie_data):
    u, v, t = movie_data
    with pytest.raises(ValueError, match="frames"):
        _movie_class()(u, v, t=t[:-5])


def test_movie_defaults_t_to_frame_indices(qtbot, movie_data):
    u, v, _ = movie_data
    w = _movie_class()(u, v)
    qtbot.addWidget(w)
    np.testing.assert_allclose(w._t, np.arange(v.shape[1]))


def test_movie_nsv_display_caps_components(qtbot, movie_data):
    u, v, t = movie_data
    w = _movie_class()(u, v, t=t, nsv_display=2)
    qtbot.addWidget(w)
    assert w._flat_u.shape[1] == 2
    expected = svd_frame_reconstruct(u[..., :2], v[:2, 0])
    np.testing.assert_allclose(w.frame_image, expected, rtol=1e-4, atol=1e-4)


def test_movie_aux_video_renderer_is_called(qtbot, movie_data):
    from widefield.gui.movie import AuxVideo

    u, v, t = movie_data
    calls = []

    def render(item, time, data):
        calls.append(time)
        item.setImage(np.zeros((4, 4)))

    w = _movie_class()(u, v, t=t, aux_videos=[AuxVideo(render=render, name="eye")])
    qtbot.addWidget(w)
    assert calls, "aux renderer was never called"
    w.set_frame(50)
    assert calls[-1] == pytest.approx(t[50])


def test_movie_broken_aux_video_does_not_crash_playback(qtbot, movie_data):
    from widefield.gui.movie import AuxVideo

    u, v, t = movie_data

    def boom(item, time, data):
        raise RuntimeError("camera unplugged")

    w = _movie_class()(u, v, t=t, aux_videos=[AuxVideo(render=boom, name="eye")])
    qtbot.addWidget(w)
    w.set_frame(10)  # must not raise
    assert "camera unplugged" in w._readout.text()


def test_movie_no_record_button_without_a_path(movie_viewer):
    assert not hasattr(movie_viewer, "_rec_btn")


def test_movie_set_pixel_rejects_out_of_bounds(movie_viewer):
    with pytest.raises(IndexError, match="outside image"):
        movie_viewer.set_pixel(-1, 0)


# =========================================================================== polygon mask


def test_polygon_mask_fills_a_square():
    mask = polygon_mask(np.array([[2, 2], [6, 2], [6, 6], [2, 6]]), (10, 10))
    assert mask[3, 3] and mask[5, 5]
    assert not mask[0, 0] and not mask[9, 9]
    assert mask.sum() == 16  # pixel centers 2.5..5.5 in both axes


def test_polygon_mask_handles_a_triangle():
    mask = polygon_mask(np.array([[0, 0], [10, 0], [0, 10]]), (10, 10))
    assert mask[0, 0]
    assert not mask[9, 9]
    assert 0 < mask.sum() < 100


def test_polygon_mask_degenerate_polygon_is_empty():
    assert not polygon_mask(np.array([[1, 1], [2, 2]]), (5, 5)).any()


def test_polygon_mask_outside_image_is_empty():
    assert not polygon_mask(np.array([[-9, -9], [-5, -9], [-5, -5]]), (5, 5)).any()


def test_polygon_mask_concave_shape():
    """Even-odd rule must handle a non-convex outline."""
    verts = np.array([[0, 0], [10, 0], [10, 10], [5, 3], [0, 10]])
    mask = polygon_mask(verts, (10, 10))
    assert mask[1, 5]  # near the top, inside
    assert not mask[8, 5]  # in the notch


def test_polygon_mask_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        polygon_mask(np.array([1, 2, 3]), (5, 5))
