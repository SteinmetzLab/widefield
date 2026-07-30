"""Reading SVD sessions off the lab server.

On-disk layout, as written by ``Pipelines/widefield/hemoCorrect.m`` and friends (one folder
per LED channel, plus the hemo-corrected temporal components)::

    <session>/blue/svdSpatialComponents.npy              U   (Ypix, Xpix, nSV)
    <session>/blue/svdTemporalComponents.npy             V   (nFrames, nSV)
    <session>/blue/svdTemporalComponents.timestamps.npy      (nFrames, 1)
    <session>/blue/meanImage.npy                             (Ypix, Xpix)
    <session>/blue/dataSummary.mat                           Sv, and acquisition metadata
    <session>/violet/...                                     same, self-contained
    <session>/corr/svdTemporalComponents_corr.npy        hemo-corrected V (nFrames, nSV')
    <session>/corr/svdTemporalComponents_corr.timestamps.npy

``corr`` has no spatial components of its own: it is the corrected temporal in *blue*'s ``U``
basis, and its component count is routinely smaller (500 vs 2000 in practice), so ``U`` gets
truncated to match.

This supersedes the older flat layout that ``matlab/generalUtils/quickLoadUVt.m`` reads
(``svdSpatialComponents_blue.npy`` at the date root, channel named ``purple``); real sessions
on the server use the layout above.

Performance note
----------------
These ``.npy`` files are written by npy-matlab in **Fortran order**, which puts the component
axis slowest-varying — so component ``s`` occupies one contiguous block. Truncating to the
first ``nsv`` components is therefore a *contiguous prefix read*: asking for 500 of 2000
components transfers 500 MB instead of 2 GB, sequentially. That is the single biggest lever on
load time over a network mount. Spatial binning, by contrast, saves no I/O at all here (it
strides within each component's block), only RAM and downstream compute — the opposite of what
you would expect from a C-ordered file.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "npy_header",
    "read_u_from_npy",
    "read_v_from_npy",
    "Channel",
    "discover_channels",
    "default_channel",
    "UVData",
    "load_uvt",
    "cache_dir",
    "LoadCancelled",
]

CHANNEL_PREFERENCE = ("corr", "blue", "violet")


class LoadCancelled(Exception):
    """Raised by a cancellable read when its ``should_cancel()`` callback returns True.

    Lets a GUI abandon a multi-GB read (window closing, user hit Cancel) instead of blocking
    until it finishes.
    """


# ----------------------------------------------------------------------------- npy access


class NpyHeader(NamedTuple):
    shape: tuple[int, ...]
    fortran_order: bool
    dtype: np.dtype
    data_offset: int


def npy_header(path: Path | str) -> NpyHeader:
    """Read an ``.npy`` header without touching the data."""
    path = Path(path)
    with open(path, "rb") as f:
        major, minor = np.lib.format.read_magic(f)
        if (major, minor) == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    return NpyHeader(tuple(int(s) for s in shape), bool(fortran), np.dtype(dtype), int(offset))


def _read_component_prefix(
    path: Path,
    nsv: int | None,
    should_cancel: Callable[[], bool] | None = None,
    chunk_bytes: int = 64 << 20,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Read the first ``nsv`` components of a component-major ``.npy`` as a flat array.

    Returns ``(flat_values, leading_shape)`` where ``leading_shape`` is the file's shape with
    the component axis dropped. Raises if the file is not Fortran-ordered, because then the
    prefix would not be the components (silently returning the wrong data is far worse than
    refusing).
    """
    hdr = npy_header(path)
    if len(hdr.shape) < 2:
        raise ValueError(f"{path.name}: expected >= 2 dimensions, got shape {hdr.shape}")

    *leading, n_total = hdr.shape
    if nsv is None or nsv >= n_total:
        nsv = n_total
    if nsv < 1:
        raise ValueError(f"nsv must be >= 1, got {nsv}")

    if not hdr.fortran_order:
        # C-ordered files interleave components across the whole file; fall back to a full
        # read + slice rather than reading the wrong bytes. Returned in the same F-style
        # layout the fast path produces, so callers need not care which route was taken.
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr[..., :nsv]).reshape(-1, order="F"), tuple(leading)

    per_component = int(np.prod(leading))
    count = per_component * nsv
    itemsize = hdr.dtype.itemsize

    out = np.empty(count, dtype=hdr.dtype)
    with open(path, "rb") as f:
        f.seek(hdr.data_offset)
        step = max(1, chunk_bytes // itemsize)
        pos = 0
        while pos < count:
            if should_cancel is not None and should_cancel():
                raise LoadCancelled
            n = min(step, count - pos)
            buf = f.read(n * itemsize)
            if len(buf) != n * itemsize:
                raise OSError(
                    f"{path.name}: truncated read at element {pos} "
                    f"({len(buf)} of {n * itemsize} bytes)"
                )
            out[pos : pos + n] = np.frombuffer(buf, dtype=hdr.dtype, count=n)
            pos += n
    return out, tuple(leading)


def read_u_from_npy(
    path: Path | str,
    nsv: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Spatial components as ``(Ypix, Xpix, nsv)``. Port of ``readUfromNPY.m``.

    ``nsv=None`` reads every component. Otherwise only the first ``nsv`` are transferred.
    """
    path = Path(path)
    flat, leading = _read_component_prefix(path, nsv, should_cancel)
    if len(leading) != 2:
        raise ValueError(f"{path.name}: expected U shape (Ypix, Xpix, nSV), got {leading + (-1,)}")
    ypix, xpix = leading
    n = flat.size // (ypix * xpix)

    # The file is component-major with (y, x) Fortran-ordered inside each component, i.e. the
    # buffer is a C-ordered (nSV, Xpix, Ypix). Transpose it to a C-contiguous (Ypix, Xpix, nSV)
    # here, once, so that `flatten_u` downstream is a free reshape.
    #
    # This costs one pass over the array on load (a few hundred ms for 210 MB) and removes the
    # same copy from every reconstruction path. MATLAB gets its equivalent reshape for free
    # because it is column-major throughout; paying it once at the boundary is how we match.
    return np.ascontiguousarray(flat.reshape(n, xpix, ypix).transpose(2, 1, 0))


def read_v_from_npy(
    path: Path | str,
    nsv: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Temporal components as ``(nsv, nFrames)``. Port of ``readVfromNPY.m``.

    Note the transpose: on disk ``V`` is ``(nFrames, nSV)`` (that is what makes a component
    prefix contiguous), but every consumer wants component-major.
    """
    path = Path(path)
    flat, leading = _read_component_prefix(path, nsv, should_cancel)
    if len(leading) != 1:
        raise ValueError(f"{path.name}: expected V shape (nFrames, nSV), got {leading + (-1,)}")
    (n_frames,) = leading
    n = flat.size // n_frames
    return np.ascontiguousarray(flat.reshape(n_frames, n, order="F").T)


# ----------------------------------------------------------------------------- discovery


@dataclass(frozen=True)
class Channel:
    """Resolved paths for one channel's SVD representation. Nothing is read on construction."""

    name: str  # "blue" | "violet" | "corr"
    spatial: Path
    temporal: Path
    timestamps: Path | None
    mean_image: Path | None
    data_summary: Path | None

    @property
    def image_shape(self) -> tuple[int, int]:
        """``(Ypix, Xpix)`` from the spatial header, without reading pixels."""
        ypix, xpix, _ = npy_header(self.spatial).shape
        return int(ypix), int(xpix)

    @property
    def n_components(self) -> int:
        """Components actually present in this channel's ``V``."""
        return int(npy_header(self.temporal).shape[-1])

    @property
    def n_frames(self) -> int:
        return int(npy_header(self.temporal).shape[0])


def discover_channels(session_path: Path | str) -> dict[str, Channel]:
    """Find the widefield channels present in a session folder. Empty dict if none."""
    session_path = Path(session_path)
    found: dict[str, Channel] = {}
    if not session_path.is_dir():
        return found

    def opt(p: Path) -> Path | None:
        return p if p.exists() else None

    for name in ("blue", "violet"):
        d = session_path / name
        spatial = d / "svdSpatialComponents.npy"
        temporal = d / "svdTemporalComponents.npy"
        if spatial.exists() and temporal.exists():
            found[name] = Channel(
                name=name,
                spatial=spatial,
                temporal=temporal,
                timestamps=opt(d / "svdTemporalComponents.timestamps.npy"),
                mean_image=opt(d / "meanImage.npy"),
                data_summary=opt(d / "dataSummary.mat"),
            )

    corr = session_path / "corr" / "svdTemporalComponents_corr.npy"
    if corr.exists() and "blue" in found:
        blue = found["blue"]
        found["corr"] = Channel(
            name="corr",
            spatial=blue.spatial,  # corr lives in blue's spatial basis
            temporal=corr,
            timestamps=opt(session_path / "corr" / "svdTemporalComponents_corr.timestamps.npy"),
            mean_image=blue.mean_image,
            data_summary=blue.data_summary,
        )
    return found


def default_channel(channels: dict[str, Channel]) -> str | None:
    """Channel to show first: hemo-corrected if present, else blue, else violet."""
    for name in CHANNEL_PREFERENCE:
        if name in channels:
            return name
    return next(iter(channels), None)


# ----------------------------------------------------------------------------- loading


class UVData(NamedTuple):
    """A loaded channel, ready for the viewers."""

    u: np.ndarray  # (Ypix, Xpix, nSV)
    v: np.ndarray  # (nSV, nFrames)
    t: np.ndarray | None  # (nFrames,) seconds, Timeline clock
    mean_image: np.ndarray | None  # (Ypix, Xpix)
    channel: str
    fs: float | None  # frame rate from the median inter-frame interval

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.u.shape[0]), int(self.u.shape[1])


def cache_dir() -> Path:
    """Local cache root. Never on the (read-only) server. Override with ``WIDEFIELD_CACHE``."""
    root = Path(os.environ.get("WIDEFIELD_CACHE", Path.home() / ".widefield" / "cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


# Bump when the *layout or content* of a cached array changes, so existing caches miss instead
# of being served in a format the current code no longer expects. v2: U became C-contiguous.
_CACHE_FORMAT = 2


def _cache_path(src: Path, tag: str) -> Path:
    """Cache filename keyed on the source's identity *and* size+mtime, so stale data misses."""
    st = src.stat()
    key = f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}|{tag}|v{_CACHE_FORMAT}"
    return cache_dir() / f"{src.stem}.{tag}.v{_CACHE_FORMAT}.{hashlib.sha1(key.encode()).hexdigest()[:16]}.npy"


def _cached(src: Path, tag: str, build: Callable[[], np.ndarray], use_cache: bool) -> np.ndarray:
    if not use_cache:
        return build()
    path = _cache_path(src, tag)
    if path.exists():
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError):  # truncated/corrupt cache — rebuild rather than die
            log.warning("Ignoring unreadable cache file %s", path)
    arr = build()
    try:
        np.save(path, arr)
    except OSError:  # caching is best-effort
        log.debug("Could not write cache %s", path, exc_info=True)
    return arr


def load_uvt(
    session_path: Path | str,
    nsv: int | None = None,
    channel: str | None = None,
    binning: int = 1,
    use_cache: bool = True,
    should_cancel: Callable[[], bool] | None = None,
    detrend: bool = False,
) -> UVData:
    """Load one channel of a session. Port of ``Pipelines/widefield/loadUVt.m``.

    Parameters
    ----------
    session_path : session folder containing ``blue/``, ``violet/``, ``corr/``.
    nsv : keep only the first ``nsv`` components (a cheap contiguous read — see the module
        docstring). ``None`` loads all, which for a real session means a 2 GB transfer.
    channel : ``"blue"``, ``"violet"``, ``"corr"``, or ``None`` for the best available.
    binning : spatially subsample ``U`` by this factor. Saves RAM and per-frame compute; does
        *not* reduce the network transfer.
    detrend : apply :func:`widefield.svd.detrend_and_filt` to ``V``. The MATLAB does this only
        for *uncorrected* components (``corr`` has already been filtered by the hemo step), and
        so does this — passing ``detrend=True`` on ``corr`` would filter twice.

    Returns
    -------
    :class:`UVData`
    """
    session_path = Path(session_path)
    channels = discover_channels(session_path)
    if not channels:
        raise FileNotFoundError(f"no widefield SVD data found in {session_path}")
    name = channel or default_channel(channels)
    if name not in channels:
        raise KeyError(f"channel {name!r} not found; available: {sorted(channels)}")
    ch = channels[name]

    # corr's V has fewer components than blue's U; never read more U than V can use.
    available = ch.n_components
    want = available if nsv is None else min(int(nsv), available)

    def build_u() -> np.ndarray:
        u = read_u_from_npy(ch.spatial, want, should_cancel)
        if binning > 1:
            u = np.ascontiguousarray(u[::binning, ::binning, :])
        return u

    u = _cached(ch.spatial, f"U.n{want}.b{binning}", build_u, use_cache)
    v = _cached(
        ch.temporal, f"V.n{want}", lambda: read_v_from_npy(ch.temporal, want, should_cancel), use_cache
    )

    t = None
    if ch.timestamps is not None:
        t = np.asarray(np.load(ch.timestamps, allow_pickle=False)).ravel()
        # The blue channel can carry one more frame than V (a trailing exposure that never
        # made it into the SVD); loadUVt.m has this same guard commented out, but real
        # sessions do exhibit it (23075 timestamps vs 23074 corrected frames).
        if t.size == v.shape[1] + 1:
            t = t[:-1]
        elif t.size != v.shape[1]:
            log.warning(
                "%s: %d timestamps for %d frames — leaving both as-is",
                ch.name,
                t.size,
                v.shape[1],
            )

    mean_image = None
    if ch.mean_image is not None:
        mean_image = np.asarray(np.load(ch.mean_image, allow_pickle=False))
        if binning > 1:
            mean_image = np.ascontiguousarray(mean_image[::binning, ::binning])

    fs = None
    if t is not None and t.size > 1:
        fs = float(1.0 / np.median(np.diff(t)))

    if detrend:
        if fs is None:
            raise ValueError("detrend=True needs timestamps to determine the frame rate")
        from widefield.svd import detrend_and_filt

        v = detrend_and_filt(v, fs)

    return UVData(u=u, v=v, t=t, mean_image=mean_image, channel=name, fs=fs)
