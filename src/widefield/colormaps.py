"""The colormaps the MATLAB viewers use, reproduced exactly.

Widefield activity is signed (dF/F above and below baseline), so the viewers use diverging
maps with a hard zero anchor — ``blueblackred`` in particular puts black at zero, which reads
as "no change" far better than a mid-grey does. Keeping the exact tables means a Python figure
is directly comparable to a MATLAB one from the same data.

Arrays are ``(N, 3)`` float64 in [0, 1]. The matplotlib and pyqtgraph converters import their
backends lazily so this module stays usable with neither installed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "blueblackred",
    "redblackblue",
    "blue_white_red",
    "red_white_blue",
    "copper",
    "to_matplotlib",
    "to_pyqtgraph",
    "to_lookup_table",
]


def _blackred_table() -> np.ndarray:
    """The 101-entry cyan→blue→black→red→yellow ramp, in ``colormap_redblackblue`` order.

    Built in the same four 0.04-spaced segments as the MATLAB literal rather than copied as
    a 101-row constant, so it is checkable by eye — and verified against the MATLAB table in
    the test suite.
    """
    step = 0.04
    yellow_to_red = [(1.0, 1.0 - step * k, 0.0) for k in range(25)]  # 1 1 0 → 1 0.04 0
    red_to_black = [(1.0 - step * k, 0.0, 0.0) for k in range(26)]  # 1 0 0 → 0 0 0
    black_to_blue = [(0.0, 0.0, step * k) for k in range(1, 26)]  # 0 0 0.04 → 0 0 1
    blue_to_cyan = [(0.0, step * k, 1.0) for k in range(1, 26)]  # 0 0.04 1 → 0 1 1
    return np.array(yellow_to_red + red_to_black + black_to_blue + blue_to_cyan, dtype=float)


def redblackblue() -> np.ndarray:
    """Yellow→red→black→blue→cyan (101 entries). Port of ``colormap_redblackblue.m``."""
    return _blackred_table()


def blueblackred() -> np.ndarray:
    """Cyan→blue→black→red→yellow (101 entries). Port of ``colormap_blueblackred.m``.

    The default for every activity image in the viewers.
    """
    return _blackred_table()[::-1].copy()


def _white_table(n: int, gamma: float) -> np.ndarray:
    """Red→white→blue ramp of ``2n + 1`` entries, gamma-corrected."""
    r = np.concatenate([np.full(n, n, dtype=float), np.arange(n, -1, -1, dtype=float)])
    g = np.concatenate([np.arange(0, n + 1, dtype=float), np.arange(n - 1, -1, -1, dtype=float)])
    b = np.concatenate([np.arange(0, n + 1, dtype=float), np.full(n, n, dtype=float)])
    return (np.stack([r, g, b], axis=1) / n) ** gamma


def red_white_blue(n: int = 100, gamma: float = 0.6) -> np.ndarray:
    """Red→white→blue (``2n + 1`` entries). Port of ``colormap_RedWhiteBlue.m``."""
    return _white_table(n, gamma)


def blue_white_red(n: int = 100, gamma: float = 0.6) -> np.ndarray:
    """Blue→white→red (``2n + 1`` entries). Port of ``colormap_BlueWhiteRed.m``."""
    return _white_table(n, gamma)[::-1].copy()


def copper(m: int = 64) -> np.ndarray:
    """MATLAB's ``copper(m)``, used for the tuning viewer's per-condition line colours.

    Reimplemented rather than taken from matplotlib: matplotlib's ``copper`` uses different
    channel scalings, so the condition colours would not match a MATLAB figure.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    gray = np.linspace(0.0, 1.0, m) if m > 1 else np.zeros(1)
    return np.minimum(1.0, gray[:, None] * np.array([1.25, 0.7812, 0.4975]))


def condition_colors(n_conditions: int) -> np.ndarray:
    """Line colours for ``n_conditions`` tuning-curve traces.

    ``pixelTuningCurveViewerSVD`` uses ``copper(n)`` with its channels reversed
    (``colors(:, [3 2 1])``), turning the copper ramp into a black→pale-blue one.
    """
    return copper(n_conditions)[:, ::-1].copy()


# ------------------------------------------------------------------ backend converters


def to_lookup_table(cmap: np.ndarray, n: int = 256) -> np.ndarray:
    """Resample a ``(N, 3)`` float table to an ``(n, 3)`` uint8 LUT (for image display)."""
    cmap = np.asarray(cmap, dtype=float)
    src = np.linspace(0.0, 1.0, cmap.shape[0])
    dst = np.linspace(0.0, 1.0, n)
    out = np.stack([np.interp(dst, src, cmap[:, c]) for c in range(3)], axis=1)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def to_matplotlib(cmap: np.ndarray, name: str = "widefield"):
    """Wrap a table as a matplotlib ``ListedColormap``."""
    from matplotlib.colors import ListedColormap

    return ListedColormap(np.asarray(cmap, dtype=float), name=name)


def to_pyqtgraph(cmap: np.ndarray):
    """Wrap a table as a ``pyqtgraph.ColorMap``."""
    import pyqtgraph as pg

    cmap = np.asarray(cmap, dtype=float)
    pos = np.linspace(0.0, 1.0, cmap.shape[0])
    rgba = np.concatenate([cmap, np.ones((cmap.shape[0], 1))], axis=1) * 255.0
    return pg.ColorMap(pos, rgba.astype(np.uint8))
