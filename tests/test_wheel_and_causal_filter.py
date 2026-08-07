"""Two fixes that the earlier tests were structurally unable to catch.

**Ctrl+wheel.** ``test_tuning_ctrl_wheel_steps_time`` called ``viewer.wheelEvent(...)`` directly
and passed, while ctrl+wheel did nothing in the running app: a real wheel event goes to the child
widget under the cursor, and pyqtgraph's view accepts it for its own zoom, so the container's
override was never reached. These tests deliver the event the way Qt does, to a child.

**Causal high-pass.** Zero-phase filtering runs the filter backwards as well as forwards, so a
response *after* an event shifts the trace *before* it — on an opto session the pre-stimulus
baselines fan out in proportion to laser power, which reads as anticipation. The event-locked
viewer therefore high-passes forwards only.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from widefield.gui._common import install_wheel  # noqa: E402
from widefield.gui.movie import _get_class as _movie_class  # noqa: E402
from widefield.gui.pixel_tuning_curve import _get_class as _tuning_class  # noqa: E402
from widefield.svd import bandpass_filt  # noqa: E402

FS = 35.0


def wheel(notches=-1.0, mods=QtCore.Qt.NoModifier):
    """A wheel event of ``notches`` clicks; negative is scrolling towards the user."""
    return QtGui.QWheelEvent(
        QtCore.QPointF(0, 0),
        QtCore.QPointF(0, 0),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, int(round(120 * notches))),
        QtCore.Qt.NoButton,
        mods,
        QtCore.Qt.NoScrollPhase,
        False,
    )


def send(widget, event):
    QtWidgets.QApplication.sendEvent(widget, event)


# ============================================================== install_wheel, in isolation


class Spinner(QtWidgets.QLabel):
    """A child that handles the wheel itself — pyqtgraph's view, in miniature."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.wheels = 0

    def wheelEvent(self, event):
        self.wheels += 1
        event.accept()


@pytest.fixture
def nest(qtbot):
    parent = QtWidgets.QWidget()
    child = Spinner(parent)
    qtbot.addWidget(parent)
    return parent, child


def test_wheel_on_a_child_reaches_the_handler(nest):
    """The regression in one line: the container never sees this event by itself."""
    parent, child = nest
    seen = []
    install_wheel(parent, lambda steps, _mods: seen.append(steps) or True)
    send(child, wheel(notches=1))
    assert seen == [1.0]


def test_wheel_handler_gets_the_modifiers(nest):
    parent, child = nest
    seen = []
    install_wheel(parent, lambda _steps, mods: seen.append(mods) or True)
    send(child, wheel(mods=QtCore.Qt.ControlModifier))
    assert seen[0] & QtCore.Qt.ControlModifier


def test_claiming_the_wheel_stops_the_child_handling_it(nest):
    """Otherwise ctrl+wheel would step time *and* zoom the plot."""
    parent, child = nest
    install_wheel(parent, lambda _steps, _mods: True)
    send(child, wheel())
    assert child.wheels == 0


def test_declining_the_wheel_leaves_the_child_alone(nest):
    """This is what keeps pyqtgraph's plain-scroll zoom working."""
    parent, child = nest
    install_wheel(parent, lambda _steps, _mods: False)
    send(child, wheel())
    assert child.wheels == 1


def test_wheel_notches_are_signed_and_fractional(nest):
    parent, child = nest
    seen = []
    install_wheel(parent, lambda steps, _mods: seen.append(steps) or True)
    for n in (1, -1, 3, -0.5):
        send(child, wheel(notches=n))
    assert seen == pytest.approx([1.0, -1.0, 3.0, -0.5])


def test_wheel_filter_ignores_other_event_types(nest):
    parent, child = nest
    seen = []
    install_wheel(parent, lambda steps, _mods: seen.append(steps) or True)
    send(child, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_A, QtCore.Qt.NoModifier))
    assert seen == []


# ============================================================== ctrl+wheel in the tuning viewer


@pytest.fixture(scope="module")
def tuning_data():
    rng = np.random.default_rng(11)
    ypix, xpix, nsv, nframes = 10, 8, 4, 2000
    q, _ = np.linalg.qr(rng.standard_normal((ypix * xpix, nsv)))
    u = q.reshape(ypix, xpix, nsv)
    t = np.arange(nframes) / FS
    v = rng.standard_normal((nsv, nframes)).astype(np.float32)
    times = np.sort(rng.uniform(2.0, t[-1] - 3.0, 60))
    labels = np.tile([0.0, 0.5, 1.0], 20)
    return u, v, t, times, labels


@pytest.fixture
def tuning(qtbot, tuning_data):
    u, v, t, times, labels = tuning_data
    w = _tuning_class()(u, v, t, times, labels, (-0.3, 0.8))
    qtbot.addWidget(w)
    return w


def plot_area(viewer):
    """The widget a real wheel event is delivered to when the cursor is over the plots."""
    return viewer._glw.viewport()


def test_ctrl_wheel_over_the_plots_steps_time(tuning):
    """The actual bug: this is where the cursor is, and nothing happened."""
    tuning._time_idx = 5
    send(plot_area(tuning), wheel(notches=-1, mods=QtCore.Qt.ControlModifier))
    assert tuning.time_index == 6


def test_ctrl_wheel_up_steps_back_in_time(tuning):
    tuning._time_idx = 5
    send(plot_area(tuning), wheel(notches=1, mods=QtCore.Qt.ControlModifier))
    assert tuning.time_index == 4


def test_ctrl_wheel_steps_once_per_notch(tuning):
    tuning._time_idx = 2
    send(plot_area(tuning), wheel(notches=-3, mods=QtCore.Qt.ControlModifier))
    assert tuning.time_index == 5


def test_a_fractional_notch_still_moves_a_frame(tuning):
    """Trackpads send fractions; rounding to zero would make scrolling feel dead."""
    tuning._time_idx = 5
    send(plot_area(tuning), wheel(notches=-0.25, mods=QtCore.Qt.ControlModifier))
    assert tuning.time_index == 6


def test_plain_wheel_over_the_plots_does_not_step_time(tuning):
    """A bare scroll belongs to pyqtgraph's zoom."""
    tuning._time_idx = 5
    send(plot_area(tuning), wheel(notches=-1))
    assert tuning.time_index == 5


def test_ctrl_wheel_clamps_at_the_ends(tuning):
    """Started one step inside each end, so a dead wheel cannot pass this by standing still."""
    tuning._time_idx = 1
    send(plot_area(tuning), wheel(notches=40, mods=QtCore.Qt.ControlModifier))
    assert tuning.time_index == 0
    tuning._time_idx = tuning._n_time - 2
    send(plot_area(tuning), wheel(notches=-40, mods=QtCore.Qt.ControlModifier))
    assert tuning.time_index == tuning._n_time - 1


def test_ctrl_wheel_works_over_any_child_widget(tuning):
    """Hotkeys work whatever has focus; the wheel should work wherever the cursor is."""
    for child in (tuning, plot_area(tuning), tuning._status, tuning.bandpass):
        tuning._time_idx = 5
        send(child, wheel(notches=-1, mods=QtCore.Qt.ControlModifier))
        assert tuning.time_index == 6, f"ctrl+wheel dead over {type(child).__name__}"


def test_ctrl_wheel_redraws_the_brain_image(tuning):
    """Stepping time must move all three panels, not just the index."""
    tuning._time_idx = 5
    before = tuning.brain_image.copy()
    send(plot_area(tuning), wheel(notches=-1, mods=QtCore.Qt.ControlModifier))
    assert not np.allclose(before, tuning.brain_image)


# ============================================================== causal high-pass numerics


def isolated_event(n=20000, onset_s=200.0, width_s=0.5, amplitude=1.0):
    """One boxcar in an otherwise silent recording, plus its onset index."""
    t = np.arange(n) / FS
    v = np.zeros((1, n))
    v[0, (t >= onset_s) & (t < onset_s + width_s)] = amplitude
    return v, int(np.searchsorted(t, onset_s))


def test_causal_highpass_cannot_see_the_future():
    """The defining property: data after sample T cannot change the output before T."""
    quiet = np.zeros((1, 8000))
    loud = quiet.copy()
    loud[0, 4000:] = 1.0
    a = bandpass_filt(quiet, FS, highpass=0.5, causal_highpass=True)
    b = bandpass_filt(loud, FS, highpass=0.5, causal_highpass=True)
    assert np.array_equal(a[:, :4000], b[:, :4000])


def test_zero_phase_highpass_does_see_the_future():
    """The contrast case — this is the artifact the causal option exists to remove."""
    quiet = np.zeros((1, 8000))
    loud = quiet.copy()
    loud[0, 4000:] = 1.0
    a = bandpass_filt(quiet, FS, highpass=0.5)
    b = bandpass_filt(loud, FS, highpass=0.5)
    assert np.abs(a - b)[:, :4000].max() > 0.1


def test_causal_highpass_leaves_the_pre_event_baseline_untouched():
    v, i0 = isolated_event()
    out = bandpass_filt(v, FS, highpass=1.0, causal_highpass=True)
    assert np.abs(out[0, i0 - 200 : i0]).max() == 0.0


def test_zero_phase_highpass_puts_a_deflection_before_the_event():
    """Roughly a quarter of the response amplitude, an eighth of a second early."""
    v, i0 = isolated_event()
    out = bandpass_filt(v, FS, highpass=1.0)
    assert np.abs(out[0, i0 - 200 : i0]).max() > 0.2


def test_causal_pre_event_deflection_scales_with_nothing():
    """Doubling the response must not double a baseline that is not there."""
    small, i0 = isolated_event(amplitude=1.0)
    big, _ = isolated_event(amplitude=10.0)
    a = bandpass_filt(small, FS, highpass=1.0, causal_highpass=True)
    b = bandpass_filt(big, FS, highpass=1.0, causal_highpass=True)
    assert np.array_equal(a[0, :i0], b[0, :i0])


def test_zero_phase_pre_event_deflection_scales_with_the_response():
    """Why it reads as tuning: the fake baseline is proportional to the real response."""
    small, i0 = isolated_event(amplitude=1.0)
    big, _ = isolated_event(amplitude=10.0)
    a = bandpass_filt(small, FS, highpass=1.0)[0, i0 - 200 : i0]
    b = bandpass_filt(big, FS, highpass=1.0)[0, i0 - 200 : i0]
    assert np.allclose(b, 10.0 * a, rtol=1e-6, atol=1e-9)


def test_causal_highpass_still_removes_dc():
    rng = np.random.default_rng(4)
    v = 7.0 + rng.standard_normal((3, 8000))
    out = bandpass_filt(v, FS, highpass=0.5, causal_highpass=True)
    assert np.abs(out[:, 2000:].mean(axis=1)).max() < 0.05


def test_causal_highpass_starts_settled_not_ringing():
    """Zero initial conditions would ring for ~1/cutoff seconds; sosfilt_zi does not."""
    v = np.full((2, 8000), 5.0)
    out = bandpass_filt(v, FS, highpass=0.5, causal_highpass=True)
    assert np.abs(out).max() < 1e-9


def test_causal_flag_is_a_no_op_without_a_highpass():
    rng = np.random.default_rng(5)
    v = rng.standard_normal((3, 4000))
    assert np.allclose(
        bandpass_filt(v, FS, lowpass=5.0),
        bandpass_filt(v, FS, lowpass=5.0, causal_highpass=True),
    )


def test_causal_flag_changes_the_result_when_there_is_a_highpass():
    rng = np.random.default_rng(6)
    v = rng.standard_normal((3, 4000))
    assert not np.allclose(
        bandpass_filt(v, FS, highpass=0.5),
        bandpass_filt(v, FS, highpass=0.5, causal_highpass=True),
    )


def test_causal_band_pass_still_low_passes():
    t = np.arange(8000) / FS
    v = (np.sin(2 * np.pi * 0.9 * t) + np.sin(2 * np.pi * 14.0 * t))[None, :]
    out = bandpass_filt(v, FS, highpass=0.2, lowpass=4.0, causal_highpass=True)
    spectrum = np.abs(np.fft.rfft(out[0, 1000:-1000]))
    freqs = np.fft.rfftfreq(out[0, 1000:-1000].size, 1 / FS)
    assert spectrum[np.argmin(np.abs(freqs - 14.0))] < 0.02 * spectrum.max()


def test_causal_band_pass_keeps_the_low_pass_zero_phase():
    """Only the high-pass is one-directional; a causal low-pass would delay every peak."""
    t = np.arange(8000) / FS
    bump = np.exp(-(((t - t[4000]) / 0.15) ** 2))[None, :]
    out = bandpass_filt(bump, FS, highpass=0.05, lowpass=5.0, causal_highpass=True)
    assert int(np.argmax(out)) == int(np.argmax(bump))


def test_causal_highpass_survives_a_very_low_cutoff():
    """0.01 Hz at 35 Hz is where a b/a design diverges; SOS plus zi must not."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal((2, 20000)) + 100.0
    out = bandpass_filt(v, FS, highpass=0.01, causal_highpass=True)
    assert np.isfinite(out).all() and np.abs(out).max() < 20.0


# ============================================================== the viewer, end to end


@pytest.fixture(scope="module")
def opto_data():
    """A synthetic opto session: sparse events, response amplitude set by condition.

    Deliberately shaped like the real thing — sustained responses, three power levels in a
    shuffled order so that any leakage from *previous* events is uncorrelated with the current
    condition, which is what makes the pre-event spread interpretable at all.
    """
    rng = np.random.default_rng(12)
    n = int(900 * FS)
    t = np.arange(n) / FS
    events = np.arange(10.0, t[-1] - 6.0, 8.0)
    amps = np.array([0.2, 0.6, 1.0])
    labels = amps[rng.permutation(np.arange(events.size) % 3)]

    signal = np.zeros(n)
    for e, amp in zip(events, labels, strict=True):
        signal[(t >= e) & (t < e + 0.6)] += amp

    # Rank-2 so the viewer has something to truncate; component 0 carries the whole signal and
    # every pixel sees it, so a pixel's trace is the signal itself.
    ypix, xpix = 6, 5
    u = np.zeros((ypix, xpix, 2), dtype=np.float32)
    u[..., 0] = 1.0
    v = np.zeros((2, n), dtype=np.float32)
    v[0] = signal
    v[1] = rng.standard_normal(n) * 0.01
    return u, v, t, events, labels


@pytest.fixture
def opto(qtbot, opto_data):
    u, v, t, events, labels = opto_data
    w = _tuning_class()(u, v, t, events, labels, (-2.0, 2.0), upsample=1)
    qtbot.addWidget(w)
    return w


def baseline_spread(viewer):
    """Range across conditions of the mean trace just before the event."""
    pre = viewer._win_samps < -0.05
    return float(np.ptp(viewer.traces[:, pre].mean(axis=1)))


def response_spread(viewer):
    post = (viewer._win_samps > 0.05) & (viewer._win_samps < 0.6)
    return float(np.ptp(viewer.traces[:, post].mean(axis=1)))


def test_the_unfiltered_baseline_is_already_flat(opto):
    """Sanity: the artifact under test is created by filtering, not by the fixture."""
    assert baseline_spread(opto) < 1e-6
    assert response_spread(opto) > 0.5


def test_the_viewer_high_passes_causally(opto):
    opto.bandpass.hp_edit.setText("0.5")
    opto.bandpass.apply()
    assert baseline_spread(opto) < 0.02 * response_spread(opto)


def test_a_zero_phase_high_pass_would_have_split_the_baselines(qtbot, opto_data):
    """The same session filtered the other way — the effect the viewer now avoids.

    Fed in pre-filtered rather than monkeypatched, because the spread has to be measured through
    the same peri-event averaging the viewer does; that is where a modest per-event deflection
    turns into a clean condition-ordered fan.
    """
    u, v, t, events, labels = opto_data
    v_zero_phase = bandpass_filt(v, FS, highpass=0.5).astype(np.float32)
    w = _tuning_class()(u, v_zero_phase, t, events, labels, (-2.0, 2.0), upsample=1)
    qtbot.addWidget(w)
    assert baseline_spread(w) > 0.1 * response_spread(w)


def test_the_causal_label_names_both_halves(opto):
    opto.bandpass.hp_edit.setText("0.5")
    opto.bandpass.lp_edit.setText("5")
    opto.bandpass.apply()
    text = opto.bandpass.label.text()
    assert "causal high-pass" in text and "zero-phase low-pass" in text


def test_a_low_pass_alone_is_still_called_zero_phase(opto):
    opto.bandpass.lp_edit.setText("5")
    opto.bandpass.apply()
    assert "zero-phase" in opto.bandpass.label.text()
    assert "causal" not in opto.bandpass.label.text()


def test_the_high_pass_box_explains_itself(opto):
    assert "causal" in opto.bandpass.hp_edit.toolTip()


def test_the_continuous_viewers_stay_zero_phase(qtbot, tuning_data):
    """A movie must not lag the behavioral traces beside it; nothing there is event-locked."""
    u, v, t, _times, _labels = tuning_data
    w = _movie_class()(u, v, t=t)
    qtbot.addWidget(w)
    w.bandpass.hp_edit.setText("0.5")
    w.bandpass.apply()
    assert "zero-phase" in w.bandpass.label.text()
    assert "causal" not in w.bandpass.label.text()
