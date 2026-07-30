"""Validate widefield.events against the MATLAB reference, and the tuning-curve helpers."""

from __future__ import annotations

import numpy as np
import pytest

from widefield.events import (
    event_locked_avg_svd,
    matlab_range,
    peri_event_series,
    peri_event_window,
    tuning_by_condition,
)
from widefield.svd import pixel_timecourse, svd_frame_reconstruct

# --------------------------------------------------------------------- MATLAB's colon


def test_matlab_range_matches_reference_window(ref):
    """1/35 is not representable, so the sample count is the thing that can drift."""
    win = matlab_range(ref["ela_calcWin"][0], 1.0 / ref["Fs"], ref["ela_calcWin"][1])
    np.testing.assert_allclose(win, ref["ela_winSamps"], atol=1e-12)


def test_matlab_range_includes_endpoint_when_it_lands_exactly():
    np.testing.assert_allclose(matlab_range(0, 0.5, 2.0), [0, 0.5, 1.0, 1.5, 2.0])


def test_matlab_range_excludes_endpoint_when_it_overshoots():
    np.testing.assert_allclose(matlab_range(0, 0.3, 1.0), [0, 0.3, 0.6, 0.9])


def test_matlab_range_rejects_zero_step():
    with pytest.raises(ValueError, match="non-zero"):
        matlab_range(0, 0, 1)


def test_peri_event_window_infers_fs_from_median_interval():
    t = np.arange(100) / 35.0
    _, fs = peri_event_window(t, (-0.1, 0.1))
    assert fs == pytest.approx(35.0)


def test_peri_event_window_fs_is_robust_to_dropped_frames():
    """A couple of long gaps must not drag the inferred rate down — hence median, not mean."""
    t = np.arange(200) / 35.0
    t[150:] += 2.0  # a 2 s dropout
    _, fs = peri_event_window(t, (-0.1, 0.1))
    assert fs == pytest.approx(35.0)


# --------------------------------------------------------------------- eventLockedAvgSVD


def test_event_locked_avg_matches_matlab(ref):
    r = event_locked_avg_svd(
        ref["V"], ref["t"], ref["ela_eventTimes"], ref["ela_eventLabels"], ref["ela_calcWin"]
    )
    np.testing.assert_allclose(r.win_samps, ref["ela_winSamps"], atol=1e-12)
    assert r.avg_v.shape == ref["ela_avg"].shape
    np.testing.assert_allclose(r.avg_v, ref["ela_avg"], rtol=1e-9, atol=1e-9)


def test_event_locked_peri_matches_matlab(ref):
    r = event_locked_avg_svd(
        ref["V"], ref["t"], ref["ela_eventTimes"], ref["ela_eventLabels"], ref["ela_calcWin"]
    )
    assert r.peri_v.shape == ref["ela_peri"].shape
    np.testing.assert_allclose(r.peri_v, ref["ela_peri"], rtol=1e-9, atol=1e-9)


def test_event_locked_sorted_labels_match_matlab(ref):
    r = event_locked_avg_svd(
        ref["V"], ref["t"], ref["ela_eventTimes"], ref["ela_eventLabels"], ref["ela_calcWin"]
    )
    np.testing.assert_allclose(r.sorted_labels, ref["ela_sortedLabels"], atol=1e-12)


def test_event_locked_conditions_are_sorted_unique(ref):
    r = event_locked_avg_svd(
        ref["V"], ref["t"], ref["ela_eventTimes"], ref["ela_eventLabels"], ref["ela_calcWin"]
    )
    np.testing.assert_allclose(r.conditions, [0, 0.25, 0.5, 1.0])


def test_event_locked_sorts_unsorted_events(ref):
    """Events arrive in whatever order; the result must not depend on that order."""
    order = np.random.default_rng(0).permutation(ref["ela_eventTimes"].size)
    r = event_locked_avg_svd(
        ref["V"],
        ref["t"],
        ref["ela_eventTimes"][order],
        ref["ela_eventLabels"][order],
        ref["ela_calcWin"],
    )
    np.testing.assert_allclose(r.avg_v, ref["ela_avg"], rtol=1e-9, atol=1e-9)


def test_event_locked_avg_is_mean_of_its_own_peri(ref):
    """Internal consistency: avg_v[c] must be the nanmean of peri_v over that condition."""
    r = event_locked_avg_svd(
        ref["V"], ref["t"], ref["ela_eventTimes"], ref["ela_eventLabels"], ref["ela_calcWin"]
    )
    for c, label in enumerate(r.conditions):
        mask = r.sorted_labels == label
        np.testing.assert_allclose(r.avg_v[c], np.nanmean(r.peri_v[mask], axis=0), atol=1e-12)


def test_event_locked_string_labels_work(small_uv):
    u, v, t = small_uv
    times = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array(["left", "right", "left", "right"])
    r = event_locked_avg_svd(v, t, times, labels, (-0.2, 0.4))
    assert list(r.conditions) == ["left", "right"]
    assert r.avg_v.shape[0] == 2


def test_event_locked_events_outside_recording_become_nan(small_uv):
    """A window past the end of the recording must yield NaN, not silently clamp."""
    u, v, t = small_uv
    r = event_locked_avg_svd(v, t, np.array([t[-1] + 10.0]), np.array([0]), (-0.1, 0.1))
    assert np.isnan(r.avg_v).all()


def test_event_locked_partial_window_keeps_the_in_range_part(small_uv):
    """An event near the end contributes its valid samples and NaNs only the rest."""
    u, v, t = small_uv
    r = event_locked_avg_svd(v, t, np.array([t[-1] - 0.1]), np.array([0]), (-0.2, 0.2))
    assert np.isfinite(r.avg_v).any() and np.isnan(r.avg_v).any()


def test_event_locked_nanmean_ignores_a_bad_event(small_uv):
    """One out-of-range event must not poison the condition average (the viewer_nans fix)."""
    u, v, t = small_uv
    good = np.array([1.0, 2.0])
    r_good = event_locked_avg_svd(v, t, good, np.array([0, 0]), (-0.1, 0.1))
    with_bad = np.array([1.0, 2.0, t[-1] + 50.0])
    r_bad = event_locked_avg_svd(v, t, with_bad, np.array([0, 0, 0]), (-0.1, 0.1))
    np.testing.assert_allclose(r_good.avg_v, r_bad.avg_v, atol=1e-12)


def test_event_locked_rejects_mismatched_labels(small_uv):
    u, v, t = small_uv
    with pytest.raises(ValueError, match="same length"):
        event_locked_avg_svd(v, t, np.array([1.0, 2.0]), np.array([0]), (-0.1, 0.1))


def test_event_locked_reconstructs_to_a_movie(ref):
    """The averaged V must reconstruct into an image stack of the expected shape."""
    r = event_locked_avg_svd(
        ref["V"], ref["t"], ref["ela_eventTimes"], ref["ela_eventLabels"], ref["ela_calcWin"]
    )
    stack = svd_frame_reconstruct(ref["U"], r.avg_v[0])
    assert stack.shape == (int(ref["Ypix"]), int(ref["Xpix"]), r.win_samps.size)


def test_tuning_viewer_traces_match_matlab(ref):
    """The middle panel of pixelTuningCurveViewerSVD: per-condition trace at one pixel."""
    r = event_locked_avg_svd(
        ref["V"], ref["t"], ref["ela_eventTimes"], ref["ela_eventLabels"], ref["ela_calcWin"]
    )
    pixel = tuple(int(p) - 1 for p in ref["corr_pixel"])
    pix_u = np.asarray(ref["U"])[pixel[0], pixel[1], :]
    traces = np.stack([pix_u @ r.avg_v[c] for c in range(r.conditions.size)])
    np.testing.assert_allclose(traces, ref["tuning_traces"], rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------- 1-D helpers


def test_peri_event_series_agrees_with_svd_path(small_uv):
    """The raw-pixel route and the SVD route must give the same peri-event matrix."""
    u, v, t = small_uv
    times = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 1, 0])
    win = (-0.2, 0.3)

    trace = pixel_timecourse(u, v, (4, 3))
    peri_raw, win_samps = peri_event_series(trace, t, times, win)

    r = event_locked_avg_svd(v, t, times, labels, win)
    pix_u = u[4, 3, :]
    peri_svd = np.einsum("s,esw->ew", pix_u, r.peri_v)

    # r.peri_v is time-sorted; times were already sorted here.
    np.testing.assert_allclose(peri_raw, peri_svd, atol=1e-10)
    np.testing.assert_allclose(win_samps, r.win_samps, atol=1e-12)


def test_tuning_by_condition_averages_per_condition():
    peri = np.array([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [20.0, 20.0]])
    labels = np.array([0, 0, 1, 1])
    win = np.array([0.0, 0.1])
    conds, resp, sem = tuning_by_condition(peri, labels, win)
    np.testing.assert_allclose(conds, [0, 1])
    np.testing.assert_allclose(resp, [2.0, 15.0])
    np.testing.assert_allclose(sem, [1.0, 5.0])


def test_tuning_by_condition_response_window_selects_samples():
    peri = np.array([[0.0, 10.0], [0.0, 10.0]])
    labels = np.array([0, 0])
    win = np.array([-0.1, 0.1])
    _, all_win, _ = tuning_by_condition(peri, labels, win)
    _, late, _ = tuning_by_condition(peri, labels, win, response_win=(0.0, 0.2))
    assert all_win[0] == pytest.approx(5.0)
    assert late[0] == pytest.approx(10.0)


def test_tuning_by_condition_single_event_has_zero_sem():
    conds, resp, sem = tuning_by_condition(np.array([[4.0]]), np.array([7]), np.array([0.0]))
    np.testing.assert_allclose(conds, [7])
    np.testing.assert_allclose(resp, [4.0])
    np.testing.assert_allclose(sem, [0.0])


def test_tuning_by_condition_all_nan_condition_is_nan():
    peri = np.array([[np.nan, np.nan], [1.0, 3.0]])
    conds, resp, _ = tuning_by_condition(peri, np.array([0, 1]), np.array([0.0, 0.1]))
    assert np.isnan(resp[0]) and resp[1] == pytest.approx(2.0)


def test_tuning_by_condition_rejects_empty_response_window():
    with pytest.raises(ValueError, match="selects no samples"):
        tuning_by_condition(
            np.zeros((2, 2)), np.array([0, 1]), np.array([0.0, 0.1]), response_win=(5.0, 6.0)
        )


def test_tuning_by_condition_rejects_label_mismatch():
    with pytest.raises(ValueError, match="events but labels"):
        tuning_by_condition(np.zeros((3, 2)), np.array([0, 1]), np.array([0.0, 0.1]))
