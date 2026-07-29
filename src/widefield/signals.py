"""Threshold-crossing detection for sync/strobe traces.

Widefield acquisition needs to know when each LED was on and when each camera exposure
happened, and those signals arrive as noisy analogue traces on Timeline. A plain threshold
chatters on noise near the crossing; a Schmitt trigger with separate low/high thresholds does
not. Port of ``schmitt.m`` (Mike Brookes' VOICEBOX, GPL) and ``schmittTimes.m``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["schmitt", "schmitt_times"]


def schmitt(
    x: np.ndarray,
    thresh: float | tuple[float, float] = 0.5,
    min_width: int = 0,
) -> np.ndarray:
    """Pass ``x`` through a Schmitt trigger → array of -1 / +1 (0 before the first crossing).

    Parameters
    ----------
    x : 1-D signal.
    thresh : ``(low, high)`` explicit thresholds, or a scalar *hysteresis* fraction in [0, 1].
        As a scalar it means thresholds at ``min + delta`` and ``max - delta`` where
        ``delta = (max - min) * (1 - thresh) / 2``.
    min_width : discard pulses narrower than this many samples (0 keeps everything). Useful
        for rejecting single-sample glitches on a noisy photodiode.

    Returns
    -------
    ``y`` with the same length as ``x``: +1 where ``x`` most recently exceeded ``high``, -1
    where it most recently fell below ``low``, and 0 for any leading stretch where neither has
    happened yet.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(0)

    thresh_arr = np.atleast_1d(np.asarray(thresh, dtype=float))
    if thresh_arr.size < 2:
        hysteresis = float(thresh_arr[0])
        if not 0.0 <= hysteresis <= 1.0:
            raise ValueError("scalar hysteresis must be in [0, 1]")
        xmax, xmin = float(np.max(x)), float(np.min(x))
        delta = (xmax - xmin) * (1.0 - hysteresis) / 2.0
        low, high = xmin + delta, xmax - delta
    else:
        low, high = float(thresh_arr[0]), float(thresh_arr[1])
    if low > high:
        raise ValueError(f"low threshold {low} exceeds high threshold {high}")

    # +1 above high, -1 below low, 0 in the hysteresis band; then keep only the samples where
    # the state actually changes (this is what makes it a trigger and not a comparator).
    c = (x > high).astype(int) - (x < low).astype(int)
    c[1:] = c[1:] * (c[1:] != c[:-1])
    t = np.flatnonzero(c)
    t = _drop_repeats(c, t)

    if min_width >= 1 and t.size > 1:
        t = np.delete(t, np.flatnonzero(np.diff(t) < min_width))
        t = _drop_repeats(c, t)

    y = np.zeros_like(c, dtype=float)
    if t.size:
        y[t] = 2 * c[t]
        y[t[0]] = c[t[0]]
        y = np.cumsum(y)
    return y


def _drop_repeats(c: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove crossings that repeat the previous crossing's direction.

    Mirrors the MATLAB's ``t(1+find(c(t(2:end))==c(t(1:end-1)))) = []``: after masking to
    state *changes* a run can still contain two same-direction entries (e.g. when the signal
    dips into the hysteresis band and back out the same side), and those are not real flips.
    """
    if t.size < 2:
        return t
    same = c[t[1:]] == c[t[:-1]]
    return np.delete(t, 1 + np.flatnonzero(same))


def schmitt_times(
    t: np.ndarray,
    sig: np.ndarray,
    thresh: float | tuple[float, float],
    min_width: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times at which ``sig`` flips state. Port of ``schmittTimes.m``.

    Returns ``(flip_times, flips_up, flips_down)`` — all sorted, in the units of ``t``.

    A flip is timed at the sample *before* the transition completes, matching the MATLAB (it
    indexes ``t`` where consecutive trigger states differ). For a 1 kHz Timeline trace that is
    a sub-millisecond bias, but it is a bias, so don't mix these with times derived any other
    way without checking.
    """
    t = np.asarray(t, dtype=float).ravel()
    sig = np.asarray(sig, dtype=float).ravel()
    if t.size != sig.size:
        raise ValueError(f"t has {t.size} samples but sig has {sig.size}")

    s = schmitt(sig, thresh, min_width)
    # The masks are one shorter than t (they compare consecutive samples). MATLAB tolerates a
    # short logical index and silently applies it to the leading elements; numpy raises, so
    # slice t explicitly. Indexing t[:-1] is also what makes the flip land on the sample
    # *before* the transition, matching the MATLAB.
    t_lead = t[:-1]
    down = t_lead[(s[:-1] == 1) & (s[1:] == -1)]
    up = t_lead[(s[:-1] == -1) & (s[1:] == 1)]
    return np.sort(np.concatenate([up, down])), up, down
