"""Widefield calcium imaging analysis and interactive viewers.

A movie is kept SVD-compressed as ``M = U @ V`` throughout (singular values folded into
``V``), because a full session is tens of GB in pixel space and a few hundred MB factored.
Everything here works on that representation and reconstructs frames only on demand.

Quick start::

    from widefield import load_uvt
    from widefield.gui import pixel_correlation_viewer

    d = load_uvt(r"Y:\\Subjects\\AB_0004\\2021-03-24\\1", nsv=500)
    pixel_correlation_viewer(d.u, d.v)

Or straight from arrays you already have, exactly like the MATLAB::

    pixel_correlation_viewer(U, V)

Submodules
----------
``svd``         reconstruction, basis changes, dF/F, temporal filters, spatial binning
``events``      event-locked (peri-stimulus) averaging in V space
``correlation`` seed-pixel correlation maps straight from the SVD
``hemo``        hemodynamic correction from an interleaved reflectance channel
``compress``    SVD-compressing a raw movie too large for RAM
``io``          reading sessions off the lab server
``signals``     Schmitt-trigger threshold crossings for sync traces
``utils``       small helpers (nearest-point matching)
``colormaps``   the MATLAB viewers' color tables
``gui``         the interactive viewers (needs the ``[gui]`` extra)

``widefield.gui`` is *not* imported here, so importing this package never pulls in Qt.
"""

from __future__ import annotations

from widefield.compress import SVDResult, iter_raw_frames, svd_compress
from widefield.correlation import SeedCorrelation, correlation_map_raw
from widefield.events import (
    EventLockedAvg,
    event_locked_avg_svd,
    peri_event_series,
    peri_event_window,
    tuning_by_condition,
)
from widefield.hemo import (
    HemoCorrection,
    hemo_correct_local,
    hemo_correct_nonlocal,
    variance_explained,
)
from widefield.io import UVData, discover_channels, load_uvt
from widefield.signals import schmitt, schmitt_times
from widefield.svd import (
    bin_image,
    change_u,
    detrend_and_filt,
    dff_from_svd,
    flatten_u,
    hp_filt,
    pixel_timecourse,
    subsample_shift,
    svd_frame_reconstruct,
)
from widefield.utils import find_nearest_point

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # svd
    "svd_frame_reconstruct",
    "pixel_timecourse",
    "flatten_u",
    "change_u",
    "dff_from_svd",
    "hp_filt",
    "detrend_and_filt",
    "subsample_shift",
    "bin_image",
    # events
    "event_locked_avg_svd",
    "EventLockedAvg",
    "peri_event_window",
    "peri_event_series",
    "tuning_by_condition",
    # correlation
    "SeedCorrelation",
    "correlation_map_raw",
    # hemodynamic correction
    "hemo_correct_local",
    "hemo_correct_nonlocal",
    "HemoCorrection",
    "variance_explained",
    # compression
    "svd_compress",
    "SVDResult",
    "iter_raw_frames",
    # io
    "load_uvt",
    "discover_channels",
    "UVData",
    # signals
    "schmitt",
    "schmitt_times",
    # utils
    "find_nearest_point",
]
