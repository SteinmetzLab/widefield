"""Frame stepping, fixed-width readout, session titles, and the s.e.m. band actually rendering."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui  # noqa: E402

from widefield.gui._common import session_label, window_title  # noqa: E402
from widefield.gui.movie import _get_class as _movie_class  # noqa: E402
from widefield.gui.pixel_correlation import _get_class as _corr_class  # noqa: E402
from widefield.gui.pixel_tuning_curve import _get_class as _tuning_class  # noqa: E402
from widefield.gui.svd_viewer import _get_class as _svd_class  # noqa: E402


def press(widget, key, mods=QtCore.Qt.NoModifier):
    widget.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, mods))


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    ypix, xpix, nsv, nframes = 12, 10, 4, 1500
    q, _ = np.linalg.qr(rng.standard_normal((ypix * xpix, nsv)))
    u = q.reshape(ypix, xpix, nsv)
    t = np.arange(nframes) / 35.0
    v = rng.standard_normal((nsv, nframes)).astype(np.float32)
    times = np.sort(rng.uniform(2.0, t[-1] - 2.0, 90))
    labels = np.tile([0.0, 0.5, 1.0], 30)
    return u, v, t, times, labels


@pytest.fixture
def movie(qtbot, data):
    u, v, t, _, _ = data
    w = _movie_class()(u, v, t=t)
    qtbot.addWidget(w)
    return w


# ===================================================================== session labels


def test_session_label_from_a_server_path():
    assert session_label(r"Y:\Subjects\AB_0032\2024-07-24\1") == "AB_0032 / 2024-07-24 / 1"


def test_session_label_from_a_posix_path():
    assert session_label("/mnt/data/Subjects/ZYE_0057/2022-01-10/1") == "ZYE_0057 / 2022-01-10 / 1"


def test_session_label_passes_through_a_plain_string():
    assert session_label("my favourite session") == "my favourite session"


def test_session_label_from_an_object_with_attributes():
    class Sess:
        subject = "AB_0032"
        date = "2024-07-24"
        number = 1

    assert session_label(Sess()) == "AB_0032 / 2024-07-24 / 1"


def test_session_label_of_none_is_empty():
    assert session_label(None) == ""


def test_window_title_without_a_session_is_just_the_name():
    assert window_title("SVD viewer") == "SVD viewer"


def test_window_title_includes_the_session():
    title = window_title("SVD viewer", r"Y:\Subjects\AB_0032\2024-07-24\1")
    assert "SVD viewer" in title and "AB_0032 / 2024-07-24 / 1" in title


@pytest.mark.parametrize("viewer", ["movie", "corr", "tuning", "svd"])
def test_every_viewer_shows_the_session_in_its_title(qtbot, data, viewer):
    u, v, t, times, labels = data
    sess = r"Y:\Subjects\AB_0032\2024-07-24\1"
    if viewer == "movie":
        w = _movie_class()(u, v, t=t, session=sess)
    elif viewer == "corr":
        w = _corr_class()(u, v, t=t, session=sess)
    elif viewer == "tuning":
        w = _tuning_class()(u, v, t, times, labels, (-0.2, 0.5), session=sess)
    else:
        w = _svd_class()(u, np.arange(4, 0, -1.0), v, fs=35.0, session=sess)
    qtbot.addWidget(w)
    assert "AB_0032 / 2024-07-24 / 1" in w.windowTitle()


# ===================================================================== frame stepping


def test_right_arrow_steps_one_frame(movie):
    movie.set_frame(500)
    press(movie, QtCore.Qt.Key_Right)
    assert movie.frame == 501


def test_left_arrow_steps_back_one_frame(movie):
    movie.set_frame(500)
    press(movie, QtCore.Qt.Key_Left)
    assert movie.frame == 499


def test_shift_arrow_steps_ten_frames(movie):
    movie.set_frame(500)
    press(movie, QtCore.Qt.Key_Right, QtCore.Qt.ShiftModifier)
    assert movie.frame == 510
    press(movie, QtCore.Qt.Key_Left, QtCore.Qt.ShiftModifier)
    assert movie.frame == 500


def test_stepping_pauses_playback(movie):
    """Stepping means you want to look at something, so it should not keep running away."""
    press(movie, QtCore.Qt.Key_P)
    assert movie._playing
    press(movie, QtCore.Qt.Key_Right)
    assert not movie._playing


def test_stepping_clamps_at_the_ends(movie):
    movie.set_frame(0)
    press(movie, QtCore.Qt.Key_Left)
    assert movie.frame == 0
    movie.set_frame(movie._n_frames - 1)
    press(movie, QtCore.Qt.Key_Right)
    assert movie.frame == movie._n_frames - 1


def test_arrows_no_longer_change_speed(movie):
    """up/down own the speed; left/right own the frame. They must not overlap."""
    before = movie.speed
    press(movie, QtCore.Qt.Key_Right)
    press(movie, QtCore.Qt.Key_Left)
    assert movie.speed == before


# ===================================================================== fixed-width readout


def test_readout_uses_a_monospace_font(movie):
    assert movie._readout.font().styleHint() == QtGui.QFont.Monospace


def test_readout_width_is_stable_across_frames(movie):
    """The row must not reflow as the frame number grows a digit."""
    lengths = set()
    for f in (0, 9, 99, 999, 1400):
        movie.set_frame(f)
        lengths.add(len(movie._readout.text()))
    assert len(lengths) == 1, f"readout changed length: {sorted(lengths)}"


def test_readout_keeps_the_fps_slot_when_unmeasured(movie):
    """Previously the fps field appeared and vanished, shifting everything beside it."""
    movie.set_frame(10)
    assert "fps drawn" in movie._readout.text()
    assert "--" in movie._readout.text()  # placeholder, not a missing field


def test_readout_width_is_stable_when_fps_appears(movie, monkeypatch):
    movie.set_frame(100)
    without = len(movie._readout.text())
    monkeypatch.setattr(type(movie), "_achieved_fps", lambda self: 17.0)
    movie.set_frame(101)
    assert len(movie._readout.text()) == without


def test_readout_width_is_stable_across_speeds(movie):
    lengths = set()
    for speed in (0.25, 1.0, 8.0):
        movie.set_speed(speed)
        lengths.add(len(movie._readout.text()))
    assert len(lengths) == 1, f"readout changed length with speed: {sorted(lengths)}"


def test_readout_width_is_stable_across_color_scales(movie):
    lengths = set()
    for _ in range(4):
        press(movie, QtCore.Qt.Key_Minus)
        lengths.add(len(movie._readout.text()))
    assert len(lengths) == 1, f"readout changed length with scale: {sorted(lengths)}"


# ===================================================================== s.e.m. band renders


@pytest.fixture
def tuning(qtbot, data):
    u, v, t, times, labels = data
    w = _tuning_class()(u, v, t, times, labels, (-0.2, 0.5))
    qtbot.addWidget(w)
    return w


def test_sem_band_has_a_non_empty_shape(tuning):
    """A FillBetweenItem with no path draws nothing — this is the 'I can't see it' guard."""
    band, _lo, _hi = tuning._sem_bands[tuning.condition_index]
    assert band.isVisible()
    assert not band.shape().isEmpty()


def test_sem_band_spans_a_real_area(tuning):
    band, _lo, _hi = tuning._sem_bands[tuning.condition_index]
    rect = band.boundingRect()
    assert rect.width() > 0 and rect.height() > 0


def test_sem_band_is_half_transparent_and_matches_the_trace(tuning):
    c = tuning.condition_index
    band, _lo, _hi = tuning._sem_bands[c]
    color = band.brush().color()
    expected = (tuning._colors[c] * 255).astype(int)
    assert (color.red(), color.green(), color.blue()) == tuple(expected)
    assert color.alpha() == 128  # ~50%


def test_sem_band_sits_behind_the_mean_line(tuning):
    band, _lo, _hi = tuning._sem_bands[0]
    assert band.zValue() < tuning._trace_curves[0].zValue()


# ===================================================================== tuning-curve points


def test_tuning_points_are_colored_per_condition(tuning):
    spots = tuning._tc_points.points()
    assert len(spots) == tuning._n_cond
    for c, spot in enumerate(spots):
        expected = (tuning._colors[c] * 255).astype(int)
        color = spot.brush().color()
        assert (color.red(), color.green(), color.blue()) == tuple(expected)


def test_selected_tuning_point_is_a_star(tuning):
    tuning._cond_idx = 1
    tuning._refresh_all()
    symbols = [spot.symbol() for spot in tuning._tc_points.points()]
    assert symbols[1] == "star"
    assert all(sym == "o" for i, sym in enumerate(symbols) if i != 1)


def test_selected_tuning_point_is_larger(tuning):
    tuning._cond_idx = 2
    tuning._refresh_all()
    sizes = [spot.size() for spot in tuning._tc_points.points()]
    assert sizes[2] > max(s for i, s in enumerate(sizes) if i != 2)


def test_star_follows_the_selection(tuning):
    for idx in range(tuning._n_cond):
        tuning._cond_idx = idx
        tuning._refresh_all()
        symbols = [spot.symbol() for spot in tuning._tc_points.points()]
        assert symbols.index("star") == idx


def test_no_condition_color_is_black(tuning):
    """The old copper-reversed scheme started at pure black, invisible on this background."""
    lum = tuning._colors @ [0.2126, 0.7152, 0.0722]
    assert lum.min() > 0.15
