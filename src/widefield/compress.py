"""SVD-compress a raw movie that does not fit in RAM. Port of ``get_svdcomps.m``.

A widefield session is tens of GB of raw frames, so the SVD cannot be taken directly. The trick
is that the *spatial* components can be estimated from far fewer frames than the recording has:

1. **Pass 1** — stream the movie, averaging every ``nt0`` consecutive frames into one, to build a
   small stack of ``n_avg_frames`` temporally-binned frames.
2. Take the eigendecomposition of that stack's frame-by-frame covariance (a small
   ``n_avg x n_avg`` matrix), and project the stack onto its leading eigenvectors to get ``U``.
3. **Pass 2** — stream the movie again, projecting every real frame onto ``U`` to get ``V`` at
   full temporal resolution.

Averaging frames costs high-frequency spatial detail in the *estimate of the basis* only; ``V``
is still computed from every original frame, so no temporal resolution is lost.

Following the MATLAB (and the README), the singular values are returned separately as ``Sv`` and
are **eigenvalues of the covariance matrix** — i.e. the squared singular values of the data, per
dimension — not the singular values of the movie.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = ["SVDResult", "svd_compress", "iter_raw_frames"]

log = logging.getLogger(__name__)

# Frames per read. The MATLAB uses nt0 * floor(1000 / nt0) — about 1000 frames, rounded down to
# a whole number of averaging groups so a group never straddles two reads.
_TARGET_BATCH = 1000


class SVDResult(NamedTuple):
    u: np.ndarray  # (Ypix, Xpix, nSV) spatial components, unit-norm columns
    sv: np.ndarray  # (nSV,) eigenvalues of the covariance matrix
    v: np.ndarray  # (nSV, nFrames) temporal components
    total_var: float  # trace of the covariance — sum(sv) if all components are kept


def _frame_count(path: Path, ypix: int, xpix: int, dtype: np.dtype) -> int:
    size = path.stat().st_size
    per_frame = ypix * xpix * np.dtype(dtype).itemsize
    if size % per_frame:
        log.warning(
            "%s: size %d is not a whole number of %dx%d %s frames; ignoring the trailing bytes",
            path.name,
            size,
            ypix,
            xpix,
            np.dtype(dtype).name,
        )
    return size // per_frame


def iter_raw_frames(
    path: Path | str,
    ypix: int,
    xpix: int,
    dtype: np.dtype | str = np.uint16,
    batch_frames: int = _TARGET_BATCH,
    should_cancel: Callable[[], bool] | None = None,
):
    """Yield ``(start_frame, frames)`` batches from a flat binary movie.

    The file layout is the one the MATLAB pipeline writes: frames back to back, each stored
    column-major (``y`` fastest), i.e. exactly what ``fwrite`` of a ``Ly x Lx x n`` array
    produces. Batches come out as ``(Ypix, Xpix, n)`` float32.
    """
    path = Path(path)
    dtype = np.dtype(dtype)
    n_frames = _frame_count(path, ypix, xpix, dtype)
    per_frame = ypix * xpix

    with open(path, "rb") as f:
        start = 0
        while start < n_frames:
            if should_cancel is not None and should_cancel():
                return
            n = min(batch_frames, n_frames - start)
            buf = np.frombuffer(f.read(n * per_frame * dtype.itemsize), dtype=dtype)
            if buf.size == 0:
                return
            n = buf.size // per_frame
            if n == 0:
                return
            # (n, Xpix, Ypix) C-order is the same memory as n column-major (Ypix, Xpix) frames.
            frames = buf[: n * per_frame].reshape(n, xpix, ypix).transpose(2, 1, 0)
            yield start, np.ascontiguousarray(frames, dtype=np.float32)
            start += n


def svd_compress(
    reg_file: Path | str,
    mean_image: np.ndarray,
    n_svd: int = 500,
    n_avg_frames_svd: int = 5000,
    dtype: np.dtype | str = np.uint16,
    yrange: slice | np.ndarray | None = None,
    xrange: slice | np.ndarray | None = None,
    roi: np.ndarray | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SVDResult:
    """SVD-compress a flat binary movie. Port of ``get_svdcomps.m``.

    Parameters
    ----------
    reg_file : flat binary movie (frames back to back, each column-major). Usually the
        motion-registered output of the preprocessing pipeline.
    mean_image : ``(Ypix, Xpix)`` mean fluorescence, subtracted from every frame. Defines the
        full frame size, so it must match the file's geometry.
    n_svd : components to keep.
    n_avg_frames_svd : how many temporally-binned frames to estimate the basis from. Higher is
        more faithful and slower; it is capped at the number of frames available.
    dtype : on-disk sample type (``uint16`` for the lab's cameras).
    yrange, xrange : optional crop applied after mean subtraction, as the MATLAB's
        ``ops.yrange``/``ops.xrange`` do. ``U`` comes back at the cropped size.
    roi : optional ``(Ypix, Xpix)`` boolean brain mask (post-crop). Pixels outside are zeroed
        before the covariance, so out-of-brain signal does not shape the components.

    Returns
    -------
    :class:`SVDResult`
    """
    reg_file = Path(reg_file)
    mean_image = np.asarray(mean_image, dtype=np.float32)
    if mean_image.ndim != 2:
        raise ValueError(f"mean_image must be 2-D; got shape {mean_image.shape}")
    ypix, xpix = mean_image.shape
    n_frames = _frame_count(reg_file, ypix, xpix, dtype)
    if n_frames < 2:
        raise ValueError(f"{reg_file.name}: only {n_frames} frames — nothing to compress")

    ys = slice(None) if yrange is None else yrange
    xs = slice(None) if xrange is None else xrange

    # How many real frames go into each averaged frame, and how many averaged frames result.
    n_avg = min(int(n_avg_frames_svd), n_frames)
    nt0 = int(np.ceil(n_frames / n_avg))
    n_avg = n_frames // nt0
    if n_avg < 2:
        raise ValueError(
            f"n_avg_frames_svd={n_avg_frames_svd} gives only {n_avg} averaged frames; "
            "raise it or use a longer recording"
        )
    # Read in whole groups so an averaging group never straddles a batch boundary.
    batch_frames = nt0 * max(1, _TARGET_BATCH // nt0)

    log.info("pass 1: %d frames -> %d averaged frames (%d per average)", n_frames, n_avg, nt0)
    sample = mean_image[ys, xs]
    mov = np.zeros((sample.shape[0], sample.shape[1], n_avg), dtype=np.float32)

    filled = 0
    for _, frames in iter_raw_frames(reg_file, ypix, xpix, dtype, batch_frames, should_cancel):
        usable = (frames.shape[2] // nt0) * nt0
        if usable == 0:
            continue
        # Group *consecutive* frames. MATLAB writes reshape(data, Ly, Lx, nt0, []) because it is
        # column-major, so its third axis is the within-group index. The C-order equivalent puts
        # the within-group index last — reshaping to (..., nt0, -1) here instead would silently
        # average frames g, g+nAvg, g+2*nAvg (strided across the whole recording) and inflate the
        # variance by a few percent while still looking plausible.
        grouped = frames[:, :, :usable].reshape(ypix, xpix, -1, nt0)
        davg = grouped.mean(axis=3)
        davg = (davg - mean_image[:, :, None])[ys, xs, :]
        take = min(davg.shape[2], n_avg - filled)
        if take <= 0:
            break
        mov[:, :, filled : filled + take] = davg[:, :, :take]
        filled += take
    mov = mov[:, :, :filled]
    if filled < 2:
        raise ValueError("pass 1 produced fewer than 2 averaged frames")

    cropped_shape = mov.shape[:2]
    flat_mov = mov.reshape(-1, mov.shape[2])
    if roi is not None:
        roi = np.asarray(roi, dtype=bool)
        if roi.shape != cropped_shape:
            raise ValueError(f"roi shape {roi.shape} does not match cropped image {cropped_shape}")
        flat_mov[~roi.reshape(-1), :] = 0.0

    # Covariance *between averaged frames* — small (n_avg x n_avg), so this is the cheap step.
    log.info("computing the %dx%d frame covariance", filled, filled)
    cov = (flat_mov.T @ flat_mov) / flat_mov.shape[0]
    total_var = float(np.trace(cov))

    keep = int(min(n_svd, cov.shape[0] - 2))
    if keep < 1:
        raise ValueError(
            f"cannot keep {n_svd} components from {cov.shape[0]} averaged frames "
            "(the MATLAB reserves 2); use more averaged frames"
        )

    # Symmetric by construction, so eigh — faster and better conditioned than a general svd,
    # and it cannot return complex values the way a general eig can on a near-symmetric matrix.
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1][:keep]  # eigh returns ascending
    sv = eigvals[order]
    basis = eigvecs[:, order]

    # Spatial components: project the averaged stack onto the basis, then normalise each column
    # (MATLAB's normc). Zero-norm columns are left at zero rather than producing NaN.
    u_flat = flat_mov @ basis
    norms = np.linalg.norm(u_flat, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_flat = np.where(norms > 0, u_flat / norms, 0.0).astype(np.float32)

    log.info("pass 2: projecting all %d frames onto %d components", n_frames, keep)
    v = np.zeros((keep, n_frames), dtype=np.float32)
    written = 0
    for start, frames in iter_raw_frames(reg_file, ypix, xpix, dtype, batch_frames, should_cancel):
        n = frames.shape[2]
        data = (frames - mean_image[:, :, None])[ys, xs, :]
        v[:, start : start + n] = u_flat.T @ data.reshape(-1, n)
        written = start + n
    if written < n_frames:
        log.warning("pass 2 wrote %d of %d frames; truncating V", written, n_frames)
        v = v[:, :written]

    u = np.ascontiguousarray(u_flat.reshape(cropped_shape[0], cropped_shape[1], keep))
    return SVDResult(u=u, sv=sv.astype(np.float32), v=v, total_var=total_var)
