"""Seed-pixel correlation maps computed directly from the SVD.

The trick (from ``pixelCorrelationViewerSVD.m``) is that the pixel x pixel covariance of the
movie is ``Ur @ cov(V') @ Ur.T``, so the correlation of *one* seed pixel against all others
is a single row of that product — obtainable without ever forming the ``nPix x nPix`` matrix,
which for a full-resolution session would be ~500 GB.

Precompute once (``cov(V')`` and the per-pixel variance), then each seed costs one
``(nSV,) @ (nSV, nSV) @ (nSV, nPix)`` chain: fast enough to run on mouse-hover.
"""

from __future__ import annotations

import numpy as np

from widefield.svd import flatten_u

__all__ = ["SeedCorrelation", "correlation_map_raw"]

# Rows of Ur processed per chunk when accumulating per-pixel variance. Ur @ cov_v is the same
# size as Ur (nPix x nSV), which at full resolution is several GB in float64 — so it is never
# materialized whole. 65536 rows x 2000 components x 4 bytes is ~500 MB worst case; measured on
# a real 512x512 session this is ~20% faster than 8192 (fewer, larger GEMMs) and 40% faster
# than doing it in one shot.
_VAR_CHUNK = 65536


class SeedCorrelation:
    """Precomputed state for seed-pixel correlation maps of one (U, V) pair.

    Parameters
    ----------
    u : (Ypix, Xpix, nSV)
    v : (nSV, nFrames)
    max_components : cap on components used. ``None`` uses all. Capping bounds both the
        precompute cost and ``Ur``'s footprint while retaining nearly all the variance,
        since SVD components are ordered by it.
    dtype : accumulation dtype. float32 halves memory and roughly doubles throughput on the
        per-seed matmul; correlation values are stable to ~1e-6, well below anything
        visible in a map scaled to [-1, 1].
    """

    def __init__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        max_components: int | None = None,
        dtype: np.dtype | str = np.float32,
    ):
        u = np.asarray(u)
        v = np.asarray(v)
        nsv = min(u.shape[-1], v.shape[0])
        if max_components is not None:
            nsv = min(nsv, int(max_components))
        if nsv < 1:
            raise ValueError("need at least one component")

        self.shape: tuple[int, int] = (int(u.shape[0]), int(u.shape[1]))
        self.n_components = nsv
        self.dtype = np.dtype(dtype)

        # copy=False: U off disk is already float32, so the common case is a free view.
        self.ur = flatten_u(u[..., :nsv]).astype(self.dtype, copy=False)
        # cov(V') in MATLAB — normalized by (nFrames - 1).
        self.cov_v = np.atleast_2d(np.cov(np.asarray(v[:nsv], dtype=np.float64))).astype(
            self.dtype, copy=False
        )
        self.var_p = self._per_pixel_variance()
        # Precompute the normalization once. Each map is otherwise dominated by streaming Ur
        # (210 MB for a 512x512 x 200 session), so per-call sqrt/where passes over nPix are pure
        # overhead on top of a memory-bandwidth-bound GEMV. Dead pixels (zero variance) get an
        # inverse std of 0, which makes their correlation 0 rather than NaN without a `where`.
        self._std_p = np.sqrt(self.var_p)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._inv_std_p = np.where(self._std_p > 0, 1.0 / self._std_p, 0.0).astype(
                self.dtype, copy=False
            )
        self._inv_std_max = (
            1.0 / np.sqrt(self.var_p.max()) if self.var_p.max() > 0 else np.array(0.0)
        )
        # Scratch buffer for the per-seed product, reused across calls.
        self._buf = np.empty(self.ur.shape[0], dtype=self.dtype)

    def _per_pixel_variance(self) -> np.ndarray:
        """diag(Ur @ cov_v @ Ur.T) — the variance of each pixel's timecourse.

        Flat pixel order is **row-major** (``index = y * Xpix + x``), because that is numpy's
        reshape convention. MATLAB's equivalent ``varP`` is column-major, so the two flat
        vectors are permutations of one another even though every derived *image* agrees. Use
        :attr:`variance_image` rather than comparing flat vectors across the two languages.
        """
        out = np.empty(self.ur.shape[0], dtype=self.dtype)
        for i in range(0, self.ur.shape[0], _VAR_CHUNK):
            block = self.ur[i : i + _VAR_CHUNK]
            out[i : i + _VAR_CHUNK] = np.einsum("ps,ps->p", block @ self.cov_v, block)
        return out

    @property
    def variance_image(self) -> np.ndarray:
        """Per-pixel variance as a ``(Ypix, Xpix)`` image — a useful "where is the signal" map."""
        return self.var_p.reshape(self.shape)

    def map(self, pixel: tuple[int, int], normalize_by_max: bool = False) -> np.ndarray:
        """Correlation of every pixel with the seed ``pixel``, as ``(Ypix, Xpix)``.

        ``pixel`` is ``(row, col)``, **0-based**.

        ``normalize_by_max=True`` reproduces the viewer's ``V`` key: divide by the *global*
        maximum pixel standard deviation instead of each pixel's own. The result is no longer
        a correlation (it is bounded well inside [-1, 1]) but it stops low-variance pixels
        from being amplified to full scale, which makes the strong-signal areas stand out.
        """
        ypix, xpix = self.shape
        y, x = int(pixel[0]), int(pixel[1])
        if not (0 <= y < ypix and 0 <= x < xpix):
            raise IndexError(f"pixel {(y, x)} outside image of shape {self.shape}")
        seed = y * xpix + x

        # The GEMV over Ur is the whole cost; write it into a reused buffer, then scale in place.
        np.dot(self.ur, self.cov_v @ self.ur[seed], out=self._buf)
        seed_std = self._std_p[seed]
        if seed_std <= 0:  # a dead seed correlates with nothing
            return np.zeros((ypix, xpix), dtype=self.dtype)

        scale = self._inv_std_max if normalize_by_max else self._inv_std_p
        out = self._buf * scale  # broadcasts for the scalar (max) case
        out /= seed_std
        return out.reshape(ypix, xpix)


def correlation_map_raw(movie: np.ndarray, pixel: tuple[int, int]) -> np.ndarray:
    """Seed correlation computed directly on a pixel-space movie ``(Ypix, Xpix, nFrames)``.

    Port of ``pixelCorrelationViewer.m`` (the non-SVD viewer). Mathematically the same as
    :meth:`SeedCorrelation.map` when the movie is the full-rank reconstruction, but it needs
    the movie in RAM — so reconstruct it binned and/or time-subsampled first.
    """
    movie = np.asarray(movie)
    if movie.ndim != 3:
        raise ValueError(f"movie must be (Ypix, Xpix, nFrames); got shape {movie.shape}")
    ypix, xpix, _ = movie.shape
    flat = movie.reshape(ypix * xpix, -1).astype(np.float64)
    centered = flat - flat.mean(axis=1, keepdims=True)
    norm = np.sqrt((centered * centered).sum(axis=1))
    seed = int(pixel[0]) * xpix + int(pixel[1])
    num = centered @ centered[seed]
    denom = norm * norm[seed]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, num / denom, 0.0)
    return corr.reshape(ypix, xpix)
