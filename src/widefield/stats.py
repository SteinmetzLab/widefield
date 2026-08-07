"""Center-and-spread estimators for a stack of repeated trials.

One block of trials, ``(nTrials, nWindow)``, in; a center estimate and a band, out. The two
supported statistics answer different questions, and the choice matters more than it looks:

* **mean +/- s.e.m.** — the classical answer. One trial that went somewhere strange moves the
  mean by ``1/n`` of however strange it went, which on 90 trials is enough to make a whole
  condition look different from its neighbors.
* **median + 95% CI** — the median moves by at most one rank no matter how extreme that trial is.

The CI here is the **distribution-free order-statistic interval**, not a bootstrap. For ``n``
finite trials it is the pair of order statistics ``[x_(k), x_(n+1-k)]`` where ``k`` is the largest
rank whose binomial tail is under ``alpha/2`` — the interval you get by inverting the sign test.
Three reasons it is the right default here over a bootstrap:

1. It is *exact*, not asymptotic and not sampled: the stated coverage is the real coverage under
   nothing more than "the trials are independent".
2. It costs one sort, which we are doing anyway to take the median. A percentile bootstrap over
   the same data needs ~1000 resampled medians per condition; measured on an opto session
   (11 conditions, ~90 trials, ~150 samples) that is seconds, and this runs again on every pixel
   the user clicks. Interactivity is not a nice-to-have for a viewer.
3. Being built from actual observed values, it cannot suggest a limit no trial ever reached.

The price is granularity: the interval can only land on values that occur in the data, so it
steps rather than glides, and below ``n = 6`` no interval reaches 95% coverage at all (with 5
trials, both extremes together still leave 6.25% outside). Those columns come back NaN rather
than silently widened.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import binom

__all__ = ["MEAN", "MEDIAN", "STATISTICS", "trial_summary", "median_ci_rank"]

MEAN = "mean"
MEDIAN = "median"
STATISTICS = (MEAN, MEDIAN)

# Band labels, for status lines and legends.
BAND_LABEL = {MEAN: "s.e.m.", MEDIAN: "95% CI"}


def median_ci_rank(n: int, coverage: float = 0.95) -> int | None:
    """Rank ``k`` for which ``[x_(k), x_(n+1-k)]`` covers the median with at least ``coverage``.

    1-based, as order statistics conventionally are. ``None`` when ``n`` is too small for any
    pair of order statistics to reach the coverage — at 95% that is ``n < 6``.

    The interval covers the true median unless more than ``n - k`` or fewer than ``k`` of the
    trials fall below it, so its miss probability is twice the binomial tail
    ``P(Bin(n, 1/2) < k)``. Take the largest ``k`` keeping that tail under ``alpha/2``, which
    makes the interval the shortest one meeting the requirement.
    """
    n = int(n)
    if n < 1:
        return None
    alpha = 1.0 - float(coverage)
    tail = binom.cdf(np.arange(n), n, 0.5)  # tail[j] = P(Bin <= j), for k = j + 1
    ok = np.nonzero(tail <= alpha / 2.0)[0]
    return int(ok[-1]) + 1 if ok.size else None


def _quiet(func, a, axis):
    """All-NaN slices are a legitimate "no data in this column", not a numerical problem."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(a, axis=axis)


def trial_summary(
    block: np.ndarray, statistic: str = MEAN, coverage: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize ``(nTrials, nWindow)`` trials into ``(center, lo, hi)``, each ``(nWindow,)``.

    ``lo``/``hi`` are absolute values, not offsets — the band for ``mean`` is symmetric but the
    one for ``median`` is not, and giving both edges keeps callers from assuming otherwise.

    NaN entries (a window running off the end of the recording) are excluded column by column,
    so an event near the edge still contributes wherever it has data. Columns with too few finite
    trials for a band get NaN edges; the center is still returned.
    """
    block = np.asarray(block, dtype=float)
    if block.ndim != 2:
        raise ValueError(f"block must be (nTrials, nWindow); got shape {block.shape}")
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}; got {statistic!r}")

    n_win = block.shape[1]
    if block.shape[0] == 0:
        empty = np.full(n_win, np.nan)
        return empty, empty.copy(), empty.copy()

    counts = np.count_nonzero(np.isfinite(block), axis=0)

    if statistic == MEAN:
        center = _quiet(np.nanmean, block, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = _quiet(lambda a, axis: np.nanstd(a, axis=axis, ddof=1), block, 0)
            sem = np.where(counts > 1, sd / np.sqrt(np.maximum(counts, 1)), np.nan)
        return center, center - sem, center + sem

    center = _quiet(np.nanmedian, block, 0)
    # np.sort puts NaN last, so the first `counts[j]` entries of column j are its finite values
    # in ascending order — exactly the order statistics the interval is defined on.
    ordered = np.sort(block, axis=0)
    lo = np.full(n_win, np.nan)
    hi = np.full(n_win, np.nan)
    for n in np.unique(counts):  # usually one value; edge columns differ
        k = median_ci_rank(int(n), coverage)
        if k is None:
            continue
        cols = np.nonzero(counts == n)[0]
        lo[cols] = ordered[k - 1, cols]
        hi[cols] = ordered[int(n) - k, cols]
    return center, lo, hi
