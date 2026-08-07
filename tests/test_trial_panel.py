"""Mean/median toggle, the single-trial panel, and picking one trial out of a condition.

The motivating case is real: on an opto session one condition's average looked completely unlike
its neighbors, because a single trial out of ~90 had gone somewhere strange. A mean cannot show
you that; a median plus the individual trials can. The fixtures here reproduce it deliberately —
one trial with a huge response — so the tests check the thing that actually matters, which is
whether that trial can be found and inspected, not just whether the widgets exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from widefield.events import peri_event_components, peri_event_series, peri_event_window
from widefield.stats import MEAN, MEDIAN, median_ci_rank, trial_summary

FS = 35.0


# ===================================================================== stats: the CI itself


def test_median_ci_rank_at_ninety_three_trials():
    """Textbook value: with n = 93 the 95% interval runs from the 37th to the 57th smallest."""
    k = median_ci_rank(93)
    assert k == 37
    assert 93 + 1 - k == 57


def test_median_ci_rank_is_the_tightest_that_still_covers():
    """One rank further in and the interval would drop below 95%."""
    from scipy.stats import binom

    for n in (10, 25, 60, 93, 200):
        k = median_ci_rank(n)
        assert 1 - 2 * binom.cdf(k - 1, n, 0.5) >= 0.95
        assert 1 - 2 * binom.cdf(k, n, 0.5) < 0.95


def test_no_distribution_free_ci_below_six_trials():
    """With 5 trials both extremes together still leave 6.25% outside — so we say so."""
    assert median_ci_rank(5) is None
    assert median_ci_rank(6) == 1


def test_median_ci_actually_covers_the_median():
    """Monte Carlo the claim rather than trusting the algebra."""
    rng = np.random.default_rng(0)
    n, reps = 31, 4000
    samples = rng.standard_normal((reps, n))
    k = median_ci_rank(n)
    ordered = np.sort(samples, axis=1)
    lo, hi = ordered[:, k - 1], ordered[:, n - k]
    coverage = np.mean((lo <= 0.0) & (hi >= 0.0))
    assert coverage > 0.95  # conservative by construction, so strictly above


def test_median_ci_covers_a_skewed_distribution_too():
    """Distribution-free is the point: no normality anywhere in the derivation."""
    rng = np.random.default_rng(1)
    n, reps = 31, 4000
    samples = rng.exponential(1.0, (reps, n))
    truth = np.log(2.0)  # median of Exp(1)
    k = median_ci_rank(n)
    ordered = np.sort(samples, axis=1)
    coverage = np.mean((ordered[:, k - 1] <= truth) & (ordered[:, n - k] >= truth))
    assert coverage > 0.95


# ===================================================================== stats: trial_summary


@pytest.fixture
def block():
    """40 trials of a bump, one of which went ten times too far."""
    rng = np.random.default_rng(2)
    t = np.linspace(-0.5, 1.5, 60)
    bump = np.exp(-((t - 0.3) ** 2) / 0.02)
    out = bump[None, :] + rng.standard_normal((40, 60)) * 0.05
    out[17] = 10.0 * bump + rng.standard_normal(60) * 0.05
    return out


def test_mean_summary_is_mean_and_sem(block):
    center, lo, hi = trial_summary(block, MEAN)
    np.testing.assert_allclose(center, block.mean(axis=0))
    sem = block.std(axis=0, ddof=1) / np.sqrt(block.shape[0])
    np.testing.assert_allclose(hi - center, sem)
    np.testing.assert_allclose(center - lo, sem)


def test_median_summary_is_the_median(block):
    center, _lo, _hi = trial_summary(block, MEDIAN)
    np.testing.assert_allclose(center, np.median(block, axis=0))


def test_the_median_ignores_the_rogue_trial_and_the_mean_does_not(block):
    """The whole reason the toggle exists."""
    peak = int(np.argmax(block.mean(axis=0)))
    clean = np.delete(block, 17, axis=0)
    mean_shift = abs(block.mean(axis=0)[peak] - clean.mean(axis=0)[peak])
    median_shift = abs(np.median(block, axis=0)[peak] - np.median(clean, axis=0)[peak])
    assert mean_shift > 0.15
    assert median_shift < 0.02


def test_median_band_edges_are_observed_values(block):
    """An order-statistic interval cannot point anywhere no trial actually went."""
    _center, lo, hi = trial_summary(block, MEDIAN)
    for j in (0, 25, 59):
        assert lo[j] in block[:, j]
        assert hi[j] in block[:, j]


def test_median_band_brackets_the_median(block):
    center, lo, hi = trial_summary(block, MEDIAN)
    assert np.all(lo <= center) and np.all(center <= hi)


def test_median_band_is_allowed_to_be_asymmetric(block):
    """Unlike a s.e.m., which is why the API returns edges rather than a half-width."""
    center, lo, hi = trial_summary(block, MEDIAN)
    assert not np.allclose(center - lo, hi - center)


def test_median_band_is_nan_when_there_are_too_few_trials():
    center, lo, hi = trial_summary(np.random.default_rng(3).standard_normal((4, 10)), MEDIAN)
    assert np.isfinite(center).all()
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_summary_excludes_nan_column_by_column():
    """A window running off the end of the recording still contributes where it has data."""
    b = np.ones((10, 4))
    b[0, 0] = np.nan
    b[:3, 3] = np.nan
    for stat in (MEAN, MEDIAN):
        center, _lo, _hi = trial_summary(b, stat)
        np.testing.assert_allclose(center, 1.0)


def test_median_band_narrows_where_fewer_trials_survive():
    """Different columns can have different n; each gets its own rank."""
    rng = np.random.default_rng(4)
    b = rng.standard_normal((40, 3))
    b[:34, 2] = np.nan  # only 6 trials left in the last column
    _center, lo, hi = trial_summary(b, MEDIAN)
    assert np.isfinite(lo).all()
    assert (hi[2] - lo[2]) > (hi[0] - lo[0])  # 6 trials is a much looser interval than 40


def test_summary_of_no_trials_is_all_nan():
    center, lo, hi = trial_summary(np.zeros((0, 5)), MEDIAN)
    assert np.isnan(center).all() and np.isnan(lo).all() and np.isnan(hi).all()


def test_summary_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="statistic"):
        trial_summary(np.zeros((3, 4)), "mode")


def test_summary_rejects_a_non_2d_block():
    with pytest.raises(ValueError, match="nTrials"):
        trial_summary(np.zeros(4), MEAN)


# ===================================================================== peri_event_components


def test_components_match_the_per_event_series_of_a_projection():
    """Reconstructing one trial must agree with windowing that trial's pixel trace."""
    rng = np.random.default_rng(5)
    n = 3000
    t = np.arange(n) / FS
    v = rng.standard_normal((6, n))
    weights = rng.standard_normal(6)
    event = 40.0
    peri_v, win = peri_event_components(v, t, event, (-0.4, 0.9), upsample=3)
    direct, win2 = peri_event_series(weights @ v, t, [event], (-0.4, 0.9), upsample=3)
    np.testing.assert_allclose(win, win2)
    np.testing.assert_allclose(weights @ peri_v, direct[0], rtol=1e-9, atol=1e-9)


def test_components_use_the_shared_window_grid():
    t = np.arange(2000) / FS
    v = np.zeros((2, 2000))
    _peri, win = peri_event_components(v, t, 10.0, (-0.3, 0.8), upsample=4)
    expected, _ = peri_event_window(t, (-0.3, 0.8), upsample=4)
    np.testing.assert_allclose(win, expected)


def test_components_are_nan_off_the_end_of_the_recording():
    t = np.arange(500) / FS
    v = np.ones((2, 500))
    peri, win = peri_event_components(v, t, float(t[-1]), (-0.2, 0.5))
    assert np.isnan(peri[:, win > 0.02]).all()
    assert np.isfinite(peri[:, win < 0]).all()


def test_components_reject_a_1d_v():
    with pytest.raises(ValueError, match="nSV"):
        peri_event_components(np.zeros(100), np.arange(100) / FS, 1.0, (-0.1, 0.1))


# ===================================================================== the viewer

pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui  # noqa: E402

from widefield.gui.pixel_tuning_curve import _get_class as _tuning_class  # noqa: E402

ROGUE_CONDITION = 1
ROGUE_TRIAL = 4


def press(widget, key, mods=QtCore.Qt.NoModifier):
    widget.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, mods))


@pytest.fixture(scope="module")
def rogue_data():
    """Three conditions, ~20 trials each, and one trial that ran away.

    The bad trial is in condition 1 and is 25x the normal response, which is enough to make that
    condition's *mean* the largest of the three while its *median* stays the middle one — the
    exact situation the toggle is for.
    """
    rng = np.random.default_rng(7)
    n = int(600 * FS)
    t = np.arange(n) / FS
    events = np.arange(5.0, t[-1] - 3.0, 9.0)
    amps = np.array([0.3, 0.6, 0.9])
    labels = amps[np.arange(events.size) % 3]

    signal = rng.standard_normal(n) * 0.01
    seen = dict.fromkeys(amps, 0)
    for e, amp in zip(events, labels, strict=True):
        k = seen[amp]
        seen[amp] += 1
        scale = 25.0 if (amp == amps[ROGUE_CONDITION] and k == ROGUE_TRIAL) else 1.0
        signal[(t >= e) & (t < e + 0.5)] += amp * scale

    ypix, xpix = 6, 5
    u = np.zeros((ypix, xpix, 2), dtype=np.float32)
    u[..., 0] = 1.0  # every pixel sees component 0, so a pixel trace is the signal
    u[..., 1] = np.linspace(0, 1, ypix * xpix).reshape(ypix, xpix)
    v = np.zeros((2, n), dtype=np.float32)
    v[0] = signal
    v[1] = rng.standard_normal(n) * 0.01
    return u, v, t, events, labels


@pytest.fixture
def viewer(qtbot, rogue_data):
    u, v, t, events, labels = rogue_data
    w = _tuning_class()(u, v, t, events, labels, (-0.5, 1.0), upsample=1)
    qtbot.addWidget(w)
    return w


def peak_response(viewer, cond):
    win = (viewer._win_samps > 0.05) & (viewer._win_samps < 0.45)
    return float(np.nanmean(viewer.traces[cond][win]))


# ---------------------------------------------------------------- the 2x2 layout


def test_there_are_four_panels(viewer):
    assert {viewer._plot, viewer._trace_plot, viewer._tc_plot, viewer._trial_plot} <= set(
        viewer._glw.ci.items
    )


def test_the_panels_are_in_a_two_by_two_grid(viewer):
    at = {tuple(viewer._glw.ci.items[p][0]): p for p in viewer._glw.ci.items}
    assert at[(0, 0)] is viewer._plot
    assert at[(0, 1)] is viewer._trace_plot
    assert at[(1, 0)] is viewer._tc_plot
    assert at[(1, 1)] is viewer._trial_plot


def test_the_two_time_panels_share_an_x_axis(viewer):
    """Stacked in one column so zooming the averages zooms the trials with them."""
    viewer._trace_plot.setXRange(-0.2, 0.4, padding=0)
    lo, hi = viewer._trial_plot.vb.viewRange()[0]
    assert lo == pytest.approx(-0.2, abs=1e-6) and hi == pytest.approx(0.4, abs=1e-6)


def test_every_panel_axis_is_labeled(viewer):
    for plot in (viewer._trace_plot, viewer._tc_plot, viewer._trial_plot):
        for side in ("bottom", "left"):
            assert plot.getAxis(side).labelText, f"{plot.titleLabel.text} has no {side} label"


# ---------------------------------------------------------------- the color scale


def test_the_color_scale_starts_from_the_data(viewer):
    """A fixed +/-1 default would pin every trace off the panel on real, non-dF/F components."""
    assert viewer._cax[0] == -viewer._cax[1]
    assert 0 < viewer._cax[1] < 10.0


def test_the_traces_fit_inside_the_starting_scale(viewer):
    """The point of auto-scaling: you can see the averages without touching a key."""
    inside = np.mean(np.abs(viewer.traces) <= viewer._cax[1])
    assert inside > 0.9


def test_the_scale_covers_the_strongest_condition_not_just_the_first(viewer):
    """Up/down cycles conditions without rescaling, so the weakest one must not set the scale."""
    strongest = int(np.argmax(np.nanmax(np.abs(viewer.traces), axis=1)))
    assert viewer._cax[1] > 0.5 * np.nanmax(np.abs(viewer.traces[strongest]))


def test_the_scale_survives_a_hundredfold_change_of_units(qtbot, rogue_data):
    u, v, t, events, labels = rogue_data
    small = _tuning_class()(u, v, t, events, labels, (-0.5, 1.0), upsample=1)
    big = _tuning_class()(u, v * 100.0, t, events, labels, (-0.5, 1.0), upsample=1)
    qtbot.addWidget(small)
    qtbot.addWidget(big)
    assert big._cax[1] == pytest.approx(100.0 * small._cax[1], rel=1e-3)


def test_an_explicit_scale_is_honored(qtbot, rogue_data):
    u, v, t, events, labels = rogue_data
    w = _tuning_class()(u, v, t, events, labels, (-0.5, 1.0), upsample=1, cax=0.4)
    qtbot.addWidget(w)
    assert w._cax == [-0.4, 0.4]


# ---------------------------------------------------------------- mean vs median


def test_the_viewer_starts_on_the_mean(viewer):
    assert viewer.statistic == MEAN


def test_m_toggles_the_statistic(viewer):
    press(viewer, QtCore.Qt.Key_M)
    assert viewer.statistic == MEDIAN
    press(viewer, QtCore.Qt.Key_M)
    assert viewer.statistic == MEAN


def test_the_rogue_trial_dominates_the_mean(viewer):
    """Condition 1 is the middle amplitude, but its mean is the biggest. That is the bug."""
    means = [peak_response(viewer, c) for c in range(viewer._n_cond)]
    assert np.argmax(means) == ROGUE_CONDITION


def test_the_median_puts_the_conditions_back_in_order(viewer):
    press(viewer, QtCore.Qt.Key_M)
    medians = [peak_response(viewer, c) for c in range(viewer._n_cond)]
    assert list(np.argsort(medians)) == [0, 1, 2]


def test_the_traces_really_are_medians(viewer):
    press(viewer, QtCore.Qt.Key_M)
    for c in range(viewer._n_cond):
        block = viewer._peri[viewer._cond_rows[c]]
        np.testing.assert_allclose(viewer.traces[c], np.nanmedian(block, axis=0), atol=1e-9)


def test_the_band_becomes_the_ci_in_median_mode(viewer):
    press(viewer, QtCore.Qt.Key_M)
    lo, hi = viewer.band
    c = ROGUE_CONDITION
    assert not np.allclose(viewer.traces[c] - lo[c], hi[c] - viewer.traces[c])


def test_the_half_width_still_reads_as_a_sem_in_mean_mode(viewer):
    c = viewer.condition_index
    block = viewer._peri[viewer._cond_rows[c]]
    expected = np.nanstd(block, axis=0, ddof=1) / np.sqrt(block.shape[0])
    np.testing.assert_allclose(viewer._sem[c], expected, rtol=1e-6)


def test_the_statistic_is_named_in_the_status_line(viewer):
    assert "mean" in viewer._status.text()
    press(viewer, QtCore.Qt.Key_M)
    assert "median" in viewer._status.text() and "95% CI" in viewer._status.text()


def test_the_tuning_curve_follows_the_statistic(viewer):
    before = viewer.tuning_curve.copy()
    press(viewer, QtCore.Qt.Key_M)
    assert not np.allclose(before, viewer.tuning_curve)


# ---------------------------------------------------------------- the single-trial panel


def test_the_trial_panel_holds_every_trial_of_the_condition(viewer):
    """One polyline, trials separated by a NaN, so the item count does not grow with trials."""
    _xs, ys = viewer._trial_curves.getData()
    n_trials, n_win = viewer.trial_traces.shape
    assert ys.size == n_trials * (n_win + 1)
    assert np.count_nonzero(np.isnan(ys)) == n_trials


def test_the_trial_panel_x_values_stay_finite(viewer):
    """A NaN x drops the point instead of breaking the line, which loses a sample per trial."""
    xs, _ys = viewer._trial_curves.getData()
    assert np.isfinite(xs).all()


def test_the_trial_panel_follows_the_selected_condition(viewer):
    before = viewer.trial_traces.shape
    press(viewer, QtCore.Qt.Key_Up)
    _xs, ys = viewer._trial_curves.getData()
    assert ys.size == viewer.trial_traces.shape[0] * (viewer.trial_traces.shape[1] + 1)
    assert viewer.trial_traces.shape[0] == before[0]  # equal counts here, but recomputed


def test_the_condition_average_is_overlaid_in_its_own_color(viewer):
    _x, y = viewer._trial_mean.getData()
    np.testing.assert_allclose(y, viewer.traces[viewer.condition_index])
    pen = viewer._trial_mean.opts["pen"].color()
    expected = (viewer._colors[viewer.condition_index] * 255).astype(int)
    assert (pen.red(), pen.green(), pen.blue()) == tuple(expected)


def test_the_trial_panel_scales_to_its_own_data(viewer):
    """Individual trials are far noisier than their average; sharing the color scale hides them."""
    viewer._cond_idx = ROGUE_CONDITION
    viewer._refresh_all()
    lo, hi = viewer._trial_plot.vb.viewRange()[1]
    block = viewer.trial_traces
    assert lo <= np.nanpercentile(block, 1) and hi >= np.nanpercentile(block, 99)


def test_trials_are_grouped_by_the_right_condition(viewer):
    """Rows come from a time-sorted peri matrix; the labels they are grouped by are too."""
    for c in range(viewer._n_cond):
        rows = viewer._cond_rows[c]
        np.testing.assert_array_equal(viewer._avg.sorted_labels[rows], viewer._conditions[c])


def test_trial_rows_survive_unsorted_event_times(qtbot, rogue_data):
    """The labels are sorted by time inside the averager, so the traces must be too."""
    u, v, t, events, labels = rogue_data
    shuffle = np.random.default_rng(8).permutation(events.size)
    a = _tuning_class()(u, v, t, events, labels, (-0.5, 1.0), upsample=1)
    b = _tuning_class()(u, v, t, events[shuffle], labels[shuffle], (-0.5, 1.0), upsample=1)
    qtbot.addWidget(a)
    qtbot.addWidget(b)
    np.testing.assert_allclose(a.traces, b.traces, atol=1e-9)


# ---------------------------------------------------------------- picking one trial out


def test_no_trial_is_selected_to_begin_with(viewer):
    assert viewer.trial_index is None


def test_selecting_a_trial_changes_the_brain_panel(viewer):
    viewer._cond_idx = ROGUE_CONDITION
    viewer._refresh_all()
    average = viewer.brain_image.copy()
    viewer.select_trial(ROGUE_TRIAL)
    assert not np.allclose(average, viewer.brain_image)


def test_the_brain_panel_shows_that_trial_exactly(viewer):
    """Not an approximation of it: the same reconstruction, from that event's own components."""
    viewer._cond_idx = ROGUE_CONDITION
    viewer.select_trial(2)
    viewer._time_idx = 20
    peri_v, _ = peri_event_components(
        viewer._v[: viewer._nsv],
        viewer._t,
        viewer.trial_time,
        viewer._calc_win,
        upsample=viewer._upsample,
    )
    expected = (viewer._flat_u @ peri_v.astype(np.float32))[:, 20].reshape(viewer.shape)
    np.testing.assert_allclose(viewer.brain_image, expected, rtol=1e-4, atol=1e-6)


def test_the_rogue_trial_is_visibly_huge_in_the_brain_panel(viewer):
    """Finding it is the point of the whole feature."""
    viewer._cond_idx = ROGUE_CONDITION
    viewer._time_idx = int(np.argmin(np.abs(viewer._win_samps - 0.25)))
    viewer.select_trial(ROGUE_TRIAL)
    rogue = np.abs(viewer.brain_image).max()
    viewer.select_trial(ROGUE_TRIAL + 1)
    ordinary = np.abs(viewer.brain_image).max()
    assert rogue > 5 * ordinary


def test_trial_time_and_event_point_back_at_the_original_arrays(viewer, rogue_data):
    _u, _v, _t, events, labels = rogue_data
    viewer._cond_idx = ROGUE_CONDITION
    viewer.select_trial(ROGUE_TRIAL)
    assert events[viewer.trial_event] == pytest.approx(viewer.trial_time)
    assert labels[viewer.trial_event] == viewer._conditions[ROGUE_CONDITION]


def test_asking_for_the_trial_time_without_a_selection_is_an_error(viewer):
    with pytest.raises(ValueError, match="no single trial"):
        _ = viewer.trial_time


def test_selecting_a_trial_out_of_range_is_an_error(viewer):
    with pytest.raises(IndexError, match="outside"):
        viewer.select_trial(viewer.n_trials)


def test_the_selected_trial_is_drawn_bold_and_the_others_are_not(viewer):
    assert not viewer._trial_highlight.isVisible()
    viewer.select_trial(3)
    assert viewer._trial_highlight.isVisible()
    _x, y = viewer._trial_highlight.getData()
    np.testing.assert_allclose(y, viewer.trial_traces[3])
    assert viewer._trial_highlight.opts["pen"].width() > viewer._trial_curves.opts["pen"].width()


def test_escape_goes_back_to_the_condition_average(viewer):
    viewer.select_trial(3)
    press(viewer, QtCore.Qt.Key_Escape)
    assert viewer.trial_index is None


def test_a_also_goes_back_to_the_average(viewer):
    viewer.select_trial(3)
    press(viewer, QtCore.Qt.Key_A)
    assert viewer.trial_index is None


def test_brackets_step_through_the_trials(viewer):
    press(viewer, QtCore.Qt.Key_BracketRight)
    assert viewer.trial_index == 0  # stepping in from the average lands on the first
    press(viewer, QtCore.Qt.Key_BracketRight)
    assert viewer.trial_index == 1
    press(viewer, QtCore.Qt.Key_BracketLeft)
    assert viewer.trial_index == 0


def test_stepping_trials_clamps_at_the_ends(viewer):
    viewer.select_trial(0)
    press(viewer, QtCore.Qt.Key_BracketLeft)
    assert viewer.trial_index == 0
    viewer.select_trial(viewer.n_trials - 1)
    press(viewer, QtCore.Qt.Key_BracketRight)
    assert viewer.trial_index == viewer.n_trials - 1


def test_changing_condition_releases_the_trial(viewer):
    """Trial 4 of one condition is a different trial in the next; keeping the index would lie."""
    viewer.select_trial(3)
    press(viewer, QtCore.Qt.Key_Up)
    assert viewer.trial_index is None


def test_choosing_a_condition_from_the_tuning_curve_releases_the_trial(viewer):
    viewer.select_trial(3)
    viewer._set_condition(viewer._n_cond - 1)
    assert viewer.trial_index is None
    assert viewer.condition_index == viewer._n_cond - 1


# ---------------------------------------------------------------- clicking


@pytest.fixture
def clickable(viewer):
    """Give the viewer the click helper the tests use, in view coordinates."""

    def click(plot, x, y):
        if plot is viewer._trace_plot:
            viewer._click_averages(QtCore.QPointF(x, y))
        elif plot is viewer._trial_plot:
            viewer._click_trials(QtCore.QPointF(x, y))
        else:
            viewer._cond_idx = int(np.argmin(np.abs(viewer._cond_x - x)))
            viewer._trial_idx = None
            viewer._refresh_all()

    viewer._on_scene_click_at = click
    return viewer


def test_clicking_a_trial_selects_it(clickable):
    v = clickable
    v._cond_idx = ROGUE_CONDITION
    v._trial_idx = None
    v._refresh_all()
    col = int(np.argmin(np.abs(v._win_samps - 0.25)))
    target = 3
    v._on_scene_click_at(v._trial_plot, float(v._win_samps[col]), v.trial_traces[target, col])
    assert v.trial_index == target


def test_clicking_the_selected_trial_again_releases_it(clickable):
    v = clickable
    v._cond_idx = ROGUE_CONDITION
    v._refresh_all()
    col = int(np.argmin(np.abs(v._win_samps - 0.25)))
    y = float(v.trial_traces[3, col])
    v._on_scene_click_at(v._trial_plot, float(v._win_samps[col]), y)
    assert v.trial_index == 3
    v._on_scene_click_at(v._trial_plot, float(v._win_samps[col]), y)
    assert v.trial_index is None


def test_clicking_far_from_every_trial_only_moves_the_time(clickable):
    """Otherwise any stray click in the panel would grab whichever trace was least far away."""
    v = clickable
    v._refresh_all()
    lo, hi = v._trial_plot.vb.viewRange()[1]
    v._on_scene_click_at(v._trial_plot, float(v._win_samps[5]), hi + 10 * (hi - lo))
    assert v.trial_index is None
    assert v.time_index == 5


def test_clicking_the_averages_panel_returns_to_the_average(clickable):
    v = clickable
    v.select_trial(3)
    v._on_scene_click_at(v._trace_plot, float(v._win_samps[4]), 0.0)
    assert v.trial_index is None


def test_clicking_a_condition_average_selects_that_condition(clickable):
    v = clickable
    v._cond_idx = 0
    v._time_idx = 0
    v._refresh_all()
    col = int(np.argmin(np.abs(v._win_samps - 0.25)))
    v._on_scene_click_at(v._trace_plot, float(v._win_samps[col]), v.traces[2, col])
    assert v.condition_index == 2


def test_clicking_between_the_averages_leaves_the_condition_alone(clickable):
    v = clickable
    v._cond_idx = 0
    v._refresh_all()
    lo, hi = v._trace_plot.vb.viewRange()[1]
    v._on_scene_click_at(v._trace_plot, float(v._win_samps[3]), hi + 10 * (hi - lo))
    assert v.condition_index == 0
    assert v.time_index == 3


def test_clicking_a_panel_still_sets_the_time(clickable):
    v = clickable
    v._on_scene_click_at(v._trace_plot, float(v._win_samps[9]), 0.0)
    assert v.time_index == 9
