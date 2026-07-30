"""Small helpers ported from ``matlab/generalUtils``."""

from __future__ import annotations

import numpy as np

__all__ = ["find_nearest_point"]


def find_nearest_point(of_these: np.ndarray, in_these: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each value in ``of_these``, the closest value in ``in_these`` and its index.

    Port of ``findNearestPoint.m``. Used to line up event times against frame times — e.g. which
    imaging frame each stimulus onset falls nearest to.

    Both inputs must already be sorted ascending, as in the MATLAB. Values before the start or
    after the end of ``in_these`` clamp to the first/last element rather than erroring.

    Returns
    -------
    ``(nearest, indices)``, both shaped like ``of_these``.

    Notes
    -----
    Implemented with ``searchsorted`` rather than the MATLAB's double-argsort trick: same result,
    O(n log m) instead of O((n+m) log(n+m)), and it does not allocate a combined array. Ties
    resolve to the *earlier* element, matching the MATLAB's ``nextDiff >= prevDiff`` test.
    """
    of_these = np.asarray(of_these, dtype=float).ravel()
    in_these = np.asarray(in_these, dtype=float).ravel()
    if in_these.size == 0:
        raise ValueError("in_these is empty; there is no nearest point")

    # Index of the first element >= each query, then compare against the one before it.
    right = np.searchsorted(in_these, of_these, side="left")
    left = np.clip(right - 1, 0, in_these.size - 1)
    right = np.clip(right, 0, in_these.size - 1)

    dist_left = np.abs(of_these - in_these[left])
    dist_right = np.abs(of_these - in_these[right])
    # Strict < keeps ties on the earlier element.
    take_right = dist_right < dist_left
    indices = np.where(take_right, right, left)
    return in_these[indices], indices
