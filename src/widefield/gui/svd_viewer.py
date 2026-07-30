"""SVD component browser. Port of ``svdViewer.m``.

Steps through the decomposition one component at a time — spatial map, timecourse, and power
spectrum — against the variance-explained curve. This is the first thing to look at after
compressing a session: it tells you how many components carry real signal, and it makes
artefacts (a light leak, a stuck LED, motion) obvious as a component with a recognisable
spatial pattern.

Controls
--------
========================  ==============================================================
left / right               previous / next component
click variance plot        jump to that component
click spatial map, ``p``   toggle pixel mode: full vs cumulative reconstruction of a pixel
``-`` / ``=``              shrink / grow the spatial colour scale
alt + left/right           rotate 90 degrees
alt + up/down              flip vertically
``z``                      reset the zoom on the trace plots
========================  ==============================================================

Pixel mode answers "how many components do I need before this pixel's timecourse looks right",
by overlaying the full reconstruction with the partial sum up to the selected component.
"""

from __future__ import annotations

import numpy as np

from widefield.colormaps import blueblackred, to_pyqtgraph
from widefield.gui._common import Orientation, ensure_app, require_qt, run_app

__all__ = ["svd_viewer"]


def _build():
    pg, QtCore, _QtGui, QtWidgets = require_qt()

    class SVDViewer(QtWidgets.QWidget):
        def __init__(self, u, sv, v, fs=1.0, total_variance=None, parent=None):
            super().__init__(parent)
            self._u = np.asarray(u)
            self._sv = np.asarray(sv, dtype=float).ravel()
            self._v = np.asarray(v)
            self._fs = float(fs)
            self.shape = (int(self._u.shape[0]), int(self._u.shape[1]))

            self._n_comp = min(self._sv.size, self._u.shape[-1], self._v.shape[0])
            if self._n_comp < 1:
                raise ValueError("need at least one component")
            self._total_variance = (
                float(np.sum(self._sv)) if total_variance is None else float(total_variance)
            )

            self._index = 0
            self._pixel = (self.shape[0] // 2, self.shape[1] // 2)
            self._pixel_mode = False
            self._orient = Orientation()
            self._cax_scale = 1.0
            self._time = np.arange(self._v.shape[1]) / self._fs

            self.setWindowTitle("SVD viewer")
            self._build_ui(pg, QtWidgets)
            self._refresh(full=True)

        def _build_ui(self, pg, QtWidgets):
            layout = QtWidgets.QVBoxLayout(self)
            self._glw = pg.GraphicsLayoutWidget()
            layout.addWidget(self._glw, stretch=1)

            self._map_plot = self._glw.addPlot(row=0, col=0)
            self._map_plot.setAspectLocked(True)
            self._map_plot.invertY(True)
            self._map_plot.hideAxis("bottom")
            self._map_plot.hideAxis("left")
            self._image = pg.ImageItem()
            self._image.setLookupTable(to_pyqtgraph(blueblackred()).getLookupTable(nPts=256))
            self._map_plot.addItem(self._image)
            self._marker = pg.ScatterPlotItem(
                size=12, pen=pg.mkPen((0, 204, 0), width=2), brush=None, symbol="o"
            )
            self._marker.setZValue(10)
            self._map_plot.addItem(self._marker)

            # Variance explained: singular values are variances, so the cumulative fraction is
            # the natural "how many components do I need" curve.
            self._var_plot = self._glw.addPlot(row=0, col=1)
            self._var_plot.setLabel("bottom", "component")
            self._var_plot.setLabel("left", "cumulative variance explained", units="%")
            self._var_plot.showGrid(x=True, y=True, alpha=0.2)
            cumulative = 100.0 * np.cumsum(self._sv[: self._n_comp]) / self._total_variance
            self._var_plot.plot(np.arange(1, self._n_comp + 1), cumulative, pen=pg.mkPen("w"))
            self._var_marker = pg.InfiniteLine(pos=1, angle=90, pen=pg.mkPen("y", width=2))
            self._var_plot.addItem(self._var_marker)

            self._trace_plot = self._glw.addPlot(row=1, col=0, colspan=2)
            self._trace_plot.setLabel("bottom", "time", units="s")
            self._trace_plot.setLabel("left", "amplitude")
            self._trace_plot.showGrid(x=True, y=True, alpha=0.2)
            self._trace_curve = self._trace_plot.plot(pen=pg.mkPen("w"))
            self._trace_extra = self._trace_plot.plot(pen=pg.mkPen((255, 128, 0), width=1))

            self._spec_plot = self._glw.addPlot(row=2, col=0, colspan=2)
            self._spec_plot.setLabel("bottom", "frequency", units="Hz")
            self._spec_plot.setLabel("left", "power")
            self._spec_plot.setLogMode(x=True, y=True)
            self._spec_plot.showGrid(x=True, y=True, alpha=0.2)
            self._spec_curve = self._spec_plot.plot(pen=pg.mkPen("c"))

            self._glw.ci.layout.setRowStretchFactor(0, 3)
            self._glw.ci.layout.setRowStretchFactor(1, 2)
            self._glw.ci.layout.setRowStretchFactor(2, 2)

            self._status = QtWidgets.QLabel()
            layout.addWidget(self._status)
            hint = QtWidgets.QLabel(
                "left/right: component · click variance plot: jump · p or click map: pixel mode · "
                "-/=: colour scale · alt+arrows: rotate/flip · z: reset zoom"
            )
            hint.setStyleSheet("color: gray;")
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self._glw.scene().sigMouseClicked.connect(self._on_click)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # ---------------------------------------------------------------- interaction

        def _on_click(self, event):
            pos = event.scenePos()
            if self._map_plot.sceneBoundingRect().contains(pos):
                pt = self._map_plot.vb.mapSceneToView(pos)
                dx, dy = int(np.floor(pt.x())), int(np.floor(pt.y()))
                dh, dw = self._orient.display_shape(self.shape)
                if 0 <= dy < dh and 0 <= dx < dw:
                    self._pixel = self._orient.to_data(dy, dx, self.shape)
                    self._pixel_mode = True
                    self._refresh()
            elif self._var_plot.sceneBoundingRect().contains(pos):
                x = self._var_plot.vb.mapSceneToView(pos).x()
                self.set_component(int(round(x)) - 1)

        def keyPressEvent(self, event):
            key, mods = event.key(), event.modifiers()

            if mods & QtCore.Qt.AltModifier:
                if key == QtCore.Qt.Key_Right:
                    self._orient.rotate(-1)
                elif key == QtCore.Qt.Key_Left:
                    self._orient.rotate(1)
                elif key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                    self._orient.toggle_flip()
                else:
                    return super().keyPressEvent(event)
                self._refresh(full=True)
                return

            if key == QtCore.Qt.Key_Right:
                self.set_component(self._index + 1)
            elif key == QtCore.Qt.Key_Left:
                self.set_component(self._index - 1)
            elif key == QtCore.Qt.Key_P:
                self._pixel_mode = not self._pixel_mode
                self._refresh()
            elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
                self._cax_scale *= 0.75
                self._refresh()
            elif key in (QtCore.Qt.Key_Equal, QtCore.Qt.Key_Plus):
                self._cax_scale *= 1.25
                self._refresh()
            elif key == QtCore.Qt.Key_Z:
                self._trace_plot.enableAutoRange()
                self._spec_plot.enableAutoRange()
            else:
                super().keyPressEvent(event)

        # ---------------------------------------------------------------- rendering

        def _power_spectrum(self, trace):
            """One-sided periodogram. Log-log axes make the 1/f trend and heartbeat peak legible."""
            n = trace.size
            if n < 4:
                return np.array([1.0]), np.array([1.0])
            windowed = trace - trace.mean()
            spec = np.abs(np.fft.rfft(windowed)) ** 2 / n
            freq = np.fft.rfftfreq(n, 1.0 / self._fs)
            keep = freq > 0  # log axes cannot show DC
            return freq[keep], np.maximum(spec[keep], 1e-30)

        def _refresh(self, full: bool = False) -> None:
            k = self._index
            spatial = np.asarray(self._u[:, :, k], dtype=float)
            peak = float(np.abs(spatial).max()) or 1.0
            limit = peak * self._cax_scale
            self._image.setImage(
                self._orient.apply(spatial), autoLevels=False, levels=[-limit, limit]
            )
            dy, dx = self._orient.to_display(*self._pixel, self.shape)
            self._marker.setData([dx + 0.5], [dy + 0.5])
            self._marker.setVisible(self._pixel_mode)
            self._var_marker.setPos(k + 1)

            if self._pixel_mode:
                y, x = self._pixel
                weights = np.asarray(self._u[y, x, : self._n_comp], dtype=float)
                full_trace = weights @ np.asarray(self._v[: self._n_comp], dtype=float)
                partial = weights[: k + 1] @ np.asarray(self._v[: k + 1], dtype=float)
                self._trace_curve.setData(self._time, full_trace)
                self._trace_extra.setData(self._time, partial)
                self._trace_extra.setVisible(True)
                self._trace_plot.setLabel("left", f"pixel ({y}, {x})")
                freq, spec = self._power_spectrum(full_trace)
            else:
                trace = np.asarray(self._v[k], dtype=float)
                self._trace_curve.setData(self._time, trace)
                self._trace_extra.setVisible(False)
                self._trace_plot.setLabel("left", f"V[{k}]")
                freq, spec = self._power_spectrum(trace)
            self._spec_curve.setData(freq, spec)

            if full:
                self._map_plot.vb.autoRange()

            pct = 100.0 * self._sv[k] / self._total_variance
            cum = 100.0 * np.sum(self._sv[: k + 1]) / self._total_variance
            bits = [
                f"component {k + 1}/{self._n_comp}",
                f"this: {pct:.3g}%",
                f"cumulative: {cum:.3g}%",
            ]
            if self._pixel_mode:
                bits.append(f"PIXEL MODE (orange = first {k + 1} components)")
            self._status.setText("  |  ".join(bits))

        # ---------------------------------------------------------------- programmatic API

        def set_component(self, index: int) -> None:
            self._index = int(np.clip(index, 0, self._n_comp - 1))
            self._refresh()

        @property
        def component(self) -> int:
            return self._index

        @property
        def pixel_mode(self) -> bool:
            return self._pixel_mode

    return SVDViewer


_CLASS = None


def _get_class():
    global _CLASS
    if _CLASS is None:
        _CLASS = _build()
    return _CLASS


def __getattr__(name):
    if name == "SVDViewer":
        return _get_class()
    raise AttributeError(name)


def svd_viewer(
    u: np.ndarray,
    sv: np.ndarray,
    v: np.ndarray,
    fs: float = 1.0,
    total_variance: float | None = None,
    block: bool = True,
):
    """Open the component browser. Equivalent to ``svdViewer(U, Sv, V, Fs[, totalVariance])``.

    ``sv`` and ``total_variance`` are what :func:`widefield.compress.svd_compress` returns.
    Without ``total_variance`` the percentages are relative to ``sum(sv)``, so they only mean
    "of the variance retained" rather than "of the movie".
    """
    app = ensure_app()
    viewer = _get_class()(u, sv, v, fs=fs, total_variance=total_variance)
    viewer.resize(1200, 900)
    viewer.show()
    viewer.setFocus()
    run_app(app, block)
    return viewer
