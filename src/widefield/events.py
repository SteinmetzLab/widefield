"""Event-locked (peri-stimulus) averaging in SVD space.

Averaging in ``V`` space and reconstructing afterwards is the whole point: an
``(nConditions, nSV, nWindow)`` average is a few MB, whereas the equivalent pixel-space
stack would be tens of GB. Port of ``eventLockedAvgSVD.m``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "matlab_range",
    "peri_event_window",
    "EventLockedAvg",
    "event_locked_avg_svd",
    "peri_event_series",
    "tuning_by_condition",
]


def matlab_range(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB's ``start:step:stop``.

    ``np.arange`` is not a drop-in replacement: it computes its length in a way that can
    add or drop a final sample relative to MATLAB for non-representable steps like
    ``1/35``. MATLAB derives the count first and then multiplies out, which is both
    stabler and what the reference data was generated with — so the window lengths match.
    """
    if step == 0:
        raise ValueError("step must be non-zero")
    n = int(np.floor((stop - start) / step + 1e-10)) + 1
    if n <= 0:
        return np.array([], dtype=float)
    return start + np.arange(n, dtype=float) * step


def peri_event_window(
    t: np.ndarray,
    calc_win: tuple[float, float],
    fs: float | None = None,
    upsample: int = 1,
):
    """Time offsets sampled inside ``calc_win``. Returns ``(win_samps, fs)``.

    ``fs`` is derived from the *median* inter-frame interval (not the mean), so a session
    with a few dropped frames still reports its true rate.

    ``upsample`` makes the grid that many times denser than the frame rate, and it is worth
    doing *before* averaging. Events almost never land at the same offset within a frame, so
    each event samples the underlying response at a different sub-frame phase; interpolating
    every event onto a dense grid and only then averaging recovers detail finer than one
    frame. The MATLAB interpolates too, but only ever onto a 1/fs grid, so it gets that
    benefit at the grid points and no finer.
    """
    t = np.asarray(t, dtype=float).ravel()
    if fs is None:
        if t.size < 2:
            raise ValueError("need at least 2 timestamps to infer the frame rate")
        fs = 1.0 / float(np.median(np.diff(t)))
    upsample = int(upsample)
    if upsample < 1:
        raise ValueError(f"upsample must be >= 1, got {upsample}")
    return matlab_range(calc_win[0], 1.0 / (fs * upsample), calc_win[1]), fs


class EventLockedAvg(NamedTuple):
    """Result of :func:`event_locked_avg_svd`."""

    avg_v: np.ndarray  # (nConditions, nSV, nWindow) — mean V per condition
    win_samps: np.ndarray  # (nWindow,) time offsets relative to the event
    peri_v: np.ndarray  # (nEvents, nSV, nWindow) — every event, time-sorted
    sorted_labels: np.ndarray  # (nEvents,) labels reordered to match peri_v
    conditions: np.ndarray  # (nConditions,) sorted unique labels


def event_locked_avg_svd(
    v: np.ndarray,
    t: np.ndarray,
    event_times: np.ndarray,
    event_labels: np.ndarray,
    calc_win: tuple[float, float],
    fs: float | None = None,
    upsample: int = 1,
    keep_peri: bool = True,
) -> EventLockedAvg:
    """Average temporal components around each event, grouped by condition.

    Parameters
    ----------
    v : (nSV, nTimePoints)
    t : (nTimePoints,) frame times, same clock as ``event_times``
    event_times : (nEvents,)
    event_labels : (nEvents,) condition of each event. Numeric labels are used directly as
        tuning-curve x-values; string labels are treated as categorical.
    calc_win : (start, stop) seconds relative to the event.
    upsample : sample the window this many times more finely than the frame rate, exploiting
        the sub-frame jitter of event times (see :func:`peri_event_window`).
    keep_peri : hold every event's windowed components in ``peri_v``. Set False to average
        condition by condition instead: ``peri_v`` comes back empty, but the peak allocation
        drops from nSV x nEvents x nWindow to one condition's worth. On a 2220-event opto
        session at upsample=4 that is ~550 MB versus ~25 MB.

    Notes
    -----
    The MATLAB signature also takes ``U``, but never reads it — the whole computation is in
    ``V`` space. It is dropped here so callers need not hold a multi-GB spatial array to
    compute an average that does not depend on it.

    Windows extending past the ends of the recording contribute NaN and are excluded
    per-sample by the nanmean, rather than dropping the whole event.
    """
    v = np.asarray(v)
    t = np.asarray(t, dtype=float).ravel()
    event_times = np.asarray(event_times, dtype=float).ravel()
    event_labels = np.asarray(event_labels).ravel()
    if event_labels.size != event_times.size:
        raise ValueError(
            f"event_times ({event_times.size}) and event_labels ({event_labels.size}) "
            "must be the same length"
        )

    order = np.argsort(event_times, kind="stable")
    event_times = event_times[order]
    sorted_labels = event_labels[order]

    win_samps, _ = peri_event_window(t, calc_win, fs, upsample)
    n_sv, n_ev, n_win = v.shape[0], event_times.size, win_samps.size
    conditions = np.unique(sorted_labels)

    def windowed(component: int, times: np.ndarray) -> np.ndarray:
        """One component interpolated onto every given event window -> (nEvents, nWindow)."""
        flat = (times[:, None] + win_samps[None, :]).ravel()
        return np.interp(
            flat, t, np.asarray(v[component], dtype=float), left=np.nan, right=np.nan
        ).reshape(times.size, n_win)

    avg_v = np.empty((conditions.size, n_sv, n_win), dtype=float)

    # An event window wholly outside the recording makes a column all-NaN; that is a
    # legitimate "no data here", so the nanmean helper silences the empty-slice warning
    # rather than letting it surface as a scary numerical error.
    if keep_peri:
        peri_v = np.empty((n_sv, n_ev, n_win), dtype=float)
        for comp in range(n_sv):
            peri_v[comp] = windowed(comp, event_times)
        for c, label in enumerate(conditions):
            avg_v[c] = _nanmean_quiet(peri_v[:, sorted_labels == label, :], axis=1)
        peri_out = np.transpose(peri_v, (1, 0, 2))
    else:
        for c, label in enumerate(conditions):
            times_c = event_times[sorted_labels == label]
            for comp in range(n_sv):
                avg_v[c, comp] = _nanmean_quiet(windowed(comp, times_c), axis=0)
        peri_out = np.empty((0, n_sv, n_win), dtype=float)

    return EventLockedAvg(
        avg_v=avg_v,
        win_samps=win_samps,
        peri_v=peri_out,
        sorted_labels=sorted_labels,
        conditions=conditions,
    )


def _nanmean_quiet(a: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` without the all-NaN-slice RuntimeWarning (result is still NaN)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def peri_event_series(
    series: np.ndarray,
    t: np.ndarray,
    event_times: np.ndarray,
    calc_win: tuple[float, float],
    fs: float | None = None,
    upsample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Peri-event matrix of a single 1-D trace → ``(peri, win_samps)``, ``peri`` (nEvents, nWindow).

    For the "raw" viewers, and for pixel traces obtained from
    :func:`widefield.svd.pixel_timecourse`.
    """
    t = np.asarray(t, dtype=float).ravel()
    series = np.asarray(series, dtype=float).ravel()
    event_times = np.asarray(event_times, dtype=float).ravel()
    win_samps, _ = peri_event_window(t, calc_win, fs, upsample)
    peri_times = (event_times[:, None] + win_samps[None, :]).ravel()
    peri = np.interp(peri_times, t, series, left=np.nan, right=np.nan)
    return peri.reshape(event_times.size, win_samps.size), win_samps


def tuning_by_condition(
    peri: np.ndarray,
    labels: np.ndarray,
    win_samps: np.ndarray,
    response_win: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a peri-event matrix into a tuning curve → ``(conditions, response, sem)``.

    ``response_win`` restricts which part of the window is averaged into the per-event
    scalar response; ``None`` uses the whole window. The s.e.m. is across events within a
    condition, so single-event conditions get 0.
    """
    peri = np.asarray(peri, dtype=float)
    labels = np.asarray(labels).ravel()
    win_samps = np.asarray(win_samps, dtype=float).ravel()
    if peri.shape[0] != labels.size:
        raise ValueError(f"peri has {peri.shape[0]} events but labels has {labels.size}")

    if response_win is None:
        sel = slice(None)
    else:
        mask = (win_samps >= response_win[0]) & (win_samps <= response_win[1])
        if not mask.any():
            raise ValueError(f"response_win {response_win} selects no samples")
        sel = mask

    per_event = _nanmean_quiet(peri[:, sel], axis=1)

    conditions = np.unique(labels)
    response = np.full(conditions.size, np.nan)
    sem = np.zeros(conditions.size)
    for i, c in enumerate(conditions):
        vals = per_event[labels == c]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            response[i] = float(np.mean(vals))
        if vals.size > 1:
            sem[i] = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    return conditions, response, sem
