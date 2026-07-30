"""Interactive viewers, ported from ``matlab/miniGUIs``.

Each viewer is a plain function you call on arrays you already have, exactly like the MATLAB::

    from widefield.gui import pixel_correlation_viewer
    pixel_correlation_viewer(U, V)

and each also exposes the underlying ``QWidget`` class (``PixelCorrelationViewer``, ...) for
embedding in a larger application. Nothing here imports Qt until you actually call something, so
``import widefield.gui`` is safe on a headless machine.

Needs the GUI extra: ``pip install 'widefield[gui]'``.
"""

from __future__ import annotations

from widefield.gui.movie_with_traces import AuxVideo, Trace, movie_with_traces
from widefield.gui.pixel_correlation import pixel_correlation_viewer
from widefield.gui.pixel_tuning_curve import pixel_tuning_curve_viewer
from widefield.gui.svd_viewer import svd_viewer

__all__ = [
    "pixel_correlation_viewer",
    "pixel_tuning_curve_viewer",
    "movie_with_traces",
    "svd_viewer",
    "Trace",
    "AuxVideo",
]


def __getattr__(name):
    """Forward the widget classes, which are built lazily so importing needs no Qt."""
    if name == "PixelCorrelationViewer":
        from widefield.gui import pixel_correlation

        return pixel_correlation.PixelCorrelationViewer
    if name == "PixelTuningCurveViewer":
        from widefield.gui import pixel_tuning_curve

        return pixel_tuning_curve.PixelTuningCurveViewer
    if name == "MovieWithTracesViewer":
        from widefield.gui import movie_with_traces as mwt

        return mwt.MovieWithTracesViewer
    if name == "SVDViewer":
        from widefield.gui import svd_viewer as sv

        return sv.SVDViewer
    raise AttributeError(name)
