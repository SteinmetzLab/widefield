"""Hemodynamic correction for widefield imaging, in SVD space.

Blood volume and oxygenation change the amount of light coming back from cortex, so a
GCaMP-excitation (blue) movie carries a large non-neural component. The standard fix is to
interleave a hemodynamic-only wavelength (violet, where GCaMP barely absorbs), then subtract
the part of blue that the violet channel predicts.

Both estimators fit the correction *in the heartbeat band* (9-13 Hz), where essentially all the
signal is vascular, and then apply it at all frequencies:

* :func:`hemo_correct_local` — a separate gain per pixel (blood vessels differ across cortex),
  expressed as a single ``V``-space matrix so it can be applied without leaving the SVD.
* :func:`hemo_correct_nonlocal` — one global least-squares mixing matrix. Simpler, and what you
  want when the spatial structure of the correction is not trustworthy.

Ports ``HemoCorrectLocal.m`` and ``HemoCorrectNonlocal.m``.

API note
--------
The two MATLAB functions disagree with each other about input orientation: ``HemoCorrectLocal``
transposes internally (so it takes ``nSV x nTimes``) while ``HemoCorrectNonlocal`` does not (so
it takes ``nTimes x nSV``, and silently produces garbage if fed the other way). Both functions
here take the ``(nSV, nTimes)`` convention used everywhere else in this package.

Plotting is *not* done here — the MATLAB versions pop up figures mid-computation, which makes
them unusable in a batch pipeline. The scale-factor map and the transformation matrix are
returned instead, ready to hand to a viewer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import butter, lfilter

from widefield.svd import flatten_u

__all__ = [
    "HemoCorrection",
    "hemo_correct_local",
    "hemo_correct_nonlocal",
    "variance_explained",
]

DEFAULT_FREQ_RANGE = (9.0, 13.0)  # heartbeat band
DEFAULT_PIX_SPACE = 3


class HemoCorrection(NamedTuple):
    """Result of a hemodynamic correction."""

    v_corrected: np.ndarray  # (nSV, nTimes) corrected temporal components
    transform: np.ndarray  # (nSV, nSV) matrix predicting V from Vaux
    scale_factor_map: np.ndarray | None  # (nSubY, nSubX) per-pixel gain, local method only
    subgrid: tuple[np.ndarray, np.ndarray] | None  # (y, x) pixel coords of the subgrid
    heart_variance_explained: float  # % of heartbeat-band power removed
    slow_variance_explained: float  # % of >0.1 Hz power removed


def _zero_mean(v: np.ndarray) -> np.ndarray:
    """Remove each component's temporal mean.

    The filters below have no business seeing a large DC offset, and the mean is not something
    we are trying to predict.
    """
    return v - v.mean(axis=1, keepdims=True)


def _band_filter(v: np.ndarray, fs: float, freq_range) -> np.ndarray:
    """Causal 2nd-order Butterworth bandpass along time, matching MATLAB's ``filter``."""
    b, a = butter(2, [f / (fs / 2) for f in freq_range], btype="bandpass")
    return lfilter(b, a, v, axis=-1)


def variance_explained(signal: np.ndarray, residual: np.ndarray) -> float:
    """Percent of ``signal``'s total power removed in ``residual``.

    Reported over the whole array (not per component), as the MATLAB does — it answers "how much
    of the hemodynamic signal did we actually get rid of", which is the number worth sanity
    checking before trusting a correction. Typical good values: >90% in the heartbeat band.
    """
    power = float(np.sum(np.asarray(signal, dtype=float) ** 2))
    if power == 0:
        return 0.0
    residual_power = float(np.sum(np.asarray(residual, dtype=float) ** 2))
    return 100.0 * (power - residual_power) / power


def hemo_correct_local(
    u: np.ndarray,
    v: np.ndarray,
    v_aux: np.ndarray,
    fs: float,
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
    pix_space: int = DEFAULT_PIX_SPACE,
) -> HemoCorrection:
    """Per-pixel hemodynamic correction. Port of ``HemoCorrectLocal.m``.

    Parameters
    ----------
    u : (Ypix, Xpix, nSV) spatial components of the signal channel.
    v : (nSV, nTimes) temporal components to correct (blue).
    v_aux : (nSV, nTimes) hemodynamic channel (violet), **already expressed in the same ``u``
        basis** (use :func:`widefield.svd.change_u`) and **already time-aligned** to ``v``
        (use :func:`widefield.svd.subsample_shift` for alternating illumination).
    fs : frame rate of ``v`` in Hz.
    freq_range : band in which the gains are estimated.
    pix_space : stride of the pixel subgrid the gains are computed on. Larger is faster and
        coarser; the gains vary smoothly across cortex, so 3 is usually plenty.

    Returns
    -------
    :class:`HemoCorrection`. ``scale_factor_map`` is the per-pixel gain on the subgrid, oriented
    ``(y, x)`` like an image.

    Notes
    -----
    The gain is a per-pixel regression of the filtered signal onto the filtered auxiliary
    channel, then lifted back into ``V`` space as ``pinv(Usub) @ diag(gain) @ Usub`` so the whole
    correction is one small matrix.

    MATLAB's displayed scale-factor map is transposed: it reshapes to ``size(pixY)``, which is
    ``(nSubX, nSubY)``, then hands that to ``imagesc`` as if it were ``(nSubY, nSubX)``. That
    goes unnoticed on the square (512x512) images the lab records. The map returned here is
    correctly oriented, which means it will look transposed relative to the MATLAB figure on a
    non-square image — the Python one is right.
    """
    u = np.asarray(u)
    v = np.asarray(v, dtype=np.float64)
    v_aux = np.asarray(v_aux, dtype=np.float64)
    if v.shape != v_aux.shape:
        raise ValueError(f"v {v.shape} and v_aux {v_aux.shape} must have the same shape")
    if pix_space < 1:
        raise ValueError("pix_space must be >= 1")

    ypix, xpix = int(u.shape[0]), int(u.shape[1])
    nsv = min(u.shape[-1], v.shape[0])
    u = u[..., :nsv]
    v, v_aux = v[:nsv], v_aux[:nsv]

    zv = _zero_mean(v)
    zv_aux = _zero_mean(v_aux)
    fv = _band_filter(zv, fs, freq_range)
    fv_aux = _band_filter(zv_aux, fs, freq_range)

    # Pixel subgrid, in image order so the resulting map is (y, x).
    y_span = np.arange(0, ypix, pix_space)
    x_span = np.arange(0, xpix, pix_space)
    flat_u = flatten_u(u)
    sub_idx = (y_span[:, None] * xpix + x_span[None, :]).ravel()
    u_sub = flat_u[sub_idx]  # (nSub, nSV)

    # Per-subgrid-pixel traces, then the regression gain. Both are mean-zero already, so the
    # least-squares slope is just cov / var.
    pix_trace = u_sub @ fv  # (nSub, nTimes)
    pix_aux = u_sub @ fv_aux
    numer = np.einsum("pt,pt->p", pix_trace, pix_aux)
    denom = np.einsum("pt,pt->p", pix_aux, pix_aux)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(denom > 0, numer / denom, 0.0)
    # Zero out pixels with no auxiliary signal at all (outside the ROI, or artifacts).
    scale = np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    # Lift the per-pixel gains into a V-space transform.
    transform = np.linalg.pinv(u_sub) @ (scale[:, None] * u_sub)

    # MATLAB applies this as `V - zVaux*T'` on (nTimes, nSV) arrays. Transposed into our
    # (nSV, nTimes) convention that is `V - T @ zVaux` — *not* `T.T @ zVaux`. The distinction is
    # easy to get wrong because T is very nearly symmetric here (pinv(Usub) @ diag(g) @ Usub with
    # a near-orthonormal Usub), so the wrong version still looks almost right; it changed the
    # reported variance explained by only 0.006 percentage points on the reference data.
    v_out = v - transform @ zv_aux

    heart_pct = variance_explained(fv, fv - transform @ fv_aux)
    b1, a1 = butter(2, 0.1 / (fs / 2), btype="high")
    f1v = lfilter(b1, a1, zv, axis=-1)
    f1v_aux = lfilter(b1, a1, zv_aux, axis=-1)
    slow_pct = variance_explained(f1v, f1v - transform @ f1v_aux)

    return HemoCorrection(
        v_corrected=v_out,
        transform=transform,
        scale_factor_map=scale.reshape(y_span.size, x_span.size),
        subgrid=(y_span, x_span),
        heart_variance_explained=heart_pct,
        slow_variance_explained=slow_pct,
    )


def hemo_correct_nonlocal(
    v: np.ndarray,
    v_aux: np.ndarray,
    fs: float | None = None,
    freq_range: tuple[float, float] | None = None,
) -> HemoCorrection:
    """Global least-squares hemodynamic correction. Port of ``HemoCorrectNonlocal.m``.

    Predicts ``v`` from ``v_aux`` with one mixing matrix and subtracts the prediction. With
    ``fs`` and ``freq_range`` given, the fit is restricted to that band (but applied at all
    frequencies); without them, it is fit on the broadband signal.

    Parameters
    ----------
    v, v_aux : (nSV, nTimes). Note this is the transpose of what the MATLAB expects — see the
        module docstring.
    """
    v = np.asarray(v, dtype=np.float64)
    v_aux = np.asarray(v_aux, dtype=np.float64)
    if v.shape != v_aux.shape:
        raise ValueError(f"v {v.shape} and v_aux {v_aux.shape} must have the same shape")
    if (fs is None) != (freq_range is None):
        raise ValueError("pass both fs and freq_range, or neither")

    zv = _zero_mean(v)
    zv_aux = _zero_mean(v_aux)
    if fs is not None:
        fv = _band_filter(zv, fs, freq_range)
        fv_aux = _band_filter(zv_aux, fs, freq_range)
    else:
        fv, fv_aux = zv, zv_aux

    # MATLAB's fV2 \ fV1 on (nTimes, nSV) matrices; transposed into our convention this is a
    # least-squares solve for the (nSV, nSV) matrix mapping aux -> signal.
    weights, *_ = np.linalg.lstsq(fv_aux.T, fv.T, rcond=None)
    v_out = v - weights.T @ zv_aux

    heart_pct = variance_explained(fv, fv - weights.T @ fv_aux)
    if fs is not None:
        b1, a1 = butter(2, 0.1 / (fs / 2), btype="high")
        f1v = lfilter(b1, a1, zv, axis=-1)
        f1v_aux = lfilter(b1, a1, zv_aux, axis=-1)
        slow_pct = variance_explained(f1v, f1v - weights.T @ f1v_aux)
    else:
        slow_pct = heart_pct

    return HemoCorrection(
        v_corrected=v_out,
        transform=weights,
        scale_factor_map=None,
        subgrid=None,
        heart_variance_explained=heart_pct,
        slow_variance_explained=slow_pct,
    )
