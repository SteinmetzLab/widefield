"""Seed-pixel correlation viewer. Port of ``pixelCorrelationViewerSVD.m``.

Click (or hover) a pixel to see how every other pixel's timecourse correlates with it — the
standard way to pull functional areas out of a widefield movie without any stimulus.

MATLAB parity
-------------
============================  ==========================================================
click                          set the seed pixel
arrow keys                     move the seed 5 px (1 px with Ctrl)
``v``                          toggle variance normalisation (emphasise strong-signal areas)
``h``                          toggle hover mode (recompute continuously under the cursor)
alt + left/right               rotate the image 90 degrees
alt + up/down                  flip vertically
============================  ==========================================================

Additions over the MATLAB: a live value readout, the seed's own timecourse underneath (the
thing you usually check next), ``r`` to reset the orientation, and ``s`` to save the map.
"""

from __future__ import annotations

import numpy as np

from widefield.colormaps import blueblackred, to_pyqtgraph
from widefield.correlation import SeedCorrelation
from widefield.gui._common import Orientation, ensure_app, require_qt, run_app
from widefield.svd import pixel_timecourse

# PixelCorrelationViewer is served by the module __getattr__ below (it cannot be defined at
# module level without importing Qt), which static analysis cannot see.
__all__ = ["PixelCorrelationViewer", "pixel_correlation_viewer"]  # noqa: F822

_STEP = 5  # arrow-key step in pixels; Ctrl gives single pixels (as in the MATLAB)


def _build():
    pg, QtCore, _QtGui, QtWidgets = require_qt()

    class PixelCorrelationViewer(QtWidgets.QWidget):
        """Widget form, so this can be embedded (e.g. in DataBrowser) as well as run standalone."""

        def __init__(self, u, v, t=None, max_components=None, parent=None):
            super().__init__(parent)
            self._u = np.asarray(u)
            self._v = np.asarray(v)
            self._t = None if t is None else np.asarray(t, dtype=float).ravel()
            self.shape = (int(self._u.shape[0]), int(self._u.shape[1]))

            self._orient = Orientation()
            self._pixel = (self.shape[0] // 2, self.shape[1] // 2)
            self._normalize_by_max = False
            self._hover = False

            self.setWindowTitle("Pixel correlation (SVD)")
            self._build_ui(pg, QtWidgets)

            # Precompute is the slow step (cov of V, then per-pixel variance). Show a wait
            # cursor rather than looking hung; on a real session this is a few seconds.
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self._corr = SeedCorrelation(self._u, self._v, max_components=max_components)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

            self._refresh(full=True)

        # ---------------------------------------------------------------- construction

        def _build_ui(self, pg, QtWidgets):
            layout = QtWidgets.QVBoxLayout(self)

            self._glw = pg.GraphicsLayoutWidget()
            layout.addWidget(self._glw, stretch=4)

            self._plot = self._glw.addPlot(row=0, col=0)
            self._plot.setAspectLocked(True)
            self._plot.invertY(True)  # image convention: row 0 at the top
            self._plot.hideAxis("bottom")
            self._plot.hideAxis("left")

            self._image = pg.ImageItem()
            self._image.setLookupTable(to_pyqtgraph(blueblackred()).getLookupTable(nPts=256))
            self._image.setLevels([-1.0, 1.0])  # correlations: fixed scale, as in the MATLAB
            self._plot.addItem(self._image)

            # Green marker: deliberately outside the colormap so it stays visible everywhere.
            self._marker = pg.ScatterPlotItem(
                size=12, pen=pg.mkPen((0, 204, 0), width=2), brush=None, symbol="o"
            )
            self._marker.setZValue(10)
            self._plot.addItem(self._marker)

            bar = pg.ColorBarItem(values=(-1, 1), colorMap=to_pyqtgraph(blueblackred()))
            bar.setImageItem(self._image, insert_in=self._plot)

            # The seed's own timecourse — the natural follow-up question to "where correlates".
            self._trace_plot = self._glw.addPlot(row=1, col=0)
            self._trace_plot.setMaximumHeight(130)
            self._trace_plot.showGrid(x=True, y=True, alpha=0.2)
            self._trace_plot.setLabel("bottom", "time", units="s" if self._t is not None else None)
            self._trace_plot.setLabel("left", "activity")
            self._trace_curve = self._trace_plot.plot(pen=pg.mkPen((0, 204, 0), width=1))

            self._status = QtWidgets.QLabel()
            self._status.setTextFormat(QtCore.Qt.PlainText)
            layout.addWidget(self._status)

            hint = QtWidgets.QLabel(
                "click / arrows: move seed (Ctrl = 1 px) · v: variance norm · h: hover · "
                "alt+arrows: rotate/flip · r: reset view · s: save map"
            )
            hint.setStyleSheet("color: gray;")
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self._image.scene().sigMouseClicked.connect(self._on_click)
            self._plot.scene().sigMouseMoved.connect(self._on_move)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # ---------------------------------------------------------------- interaction

        def _scene_to_data(self, scene_pos):
            """Scene coords -> data pixel, or None if outside the image."""
            pt = self._plot.vb.mapSceneToView(scene_pos)
            dx, dy = int(np.floor(pt.x())), int(np.floor(pt.y()))
            dh, dw = self._orient.display_shape(self.shape)
            if not (0 <= dy < dh and 0 <= dx < dw):
                return None
            return self._orient.to_data(dy, dx, self.shape)

        def _on_click(self, event):
            pixel = self._scene_to_data(event.scenePos())
            if pixel is not None:
                self._pixel = pixel
                self._refresh()

        def _on_move(self, scene_pos):
            pixel = self._scene_to_data(scene_pos)
            if pixel is None:
                return
            if self._hover:
                self._pixel = pixel
                self._refresh()
            else:
                # Even without hover mode, report the value under the cursor.
                self._update_status(cursor=pixel)

        def keyPressEvent(self, event):
            key = event.key()
            mods = event.modifiers()
            alt = bool(mods & QtCore.Qt.AltModifier)
            step = 1 if mods & QtCore.Qt.ControlModifier else _STEP

            if alt:
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

            moves = {
                QtCore.Qt.Key_Up: (-step, 0),
                QtCore.Qt.Key_Down: (step, 0),
                QtCore.Qt.Key_Left: (0, -step),
                QtCore.Qt.Key_Right: (0, step),
            }
            if key in moves:
                d_row, d_col = moves[key]
                self._pixel = self._orient.step_on_screen(*self._pixel, self.shape, d_row, d_col)
                self._refresh()
            elif key == QtCore.Qt.Key_V:
                self._normalize_by_max = not self._normalize_by_max
                self._refresh()
            elif key == QtCore.Qt.Key_H:
                self._hover = not self._hover
                self._update_status()
            elif key == QtCore.Qt.Key_R:
                self._orient.reset()
                self._refresh(full=True)
            elif key == QtCore.Qt.Key_S:
                self._save_map()
            else:
                super().keyPressEvent(event)

        # ---------------------------------------------------------------- rendering

        def _refresh(self, full: bool = False) -> None:
            self._map = self._corr.map(self._pixel, normalize_by_max=self._normalize_by_max)
            self._image.setImage(
                self._orient.apply(self._map), autoLevels=False, levels=self._levels()
            )
            dy, dx = self._orient.to_display(*self._pixel, self.shape)
            self._marker.setData([dx + 0.5], [dy + 0.5])  # centre of the pixel
            self._update_trace()
            self._update_status()
            if full:
                self._plot.vb.autoRange()

        def _levels(self):
            """Correlations use a fixed [-1, 1]; the variance-normalised map is much flatter.

            Autoscaling the latter symmetrically about zero keeps it readable — otherwise it
            renders as a nearly uniform black field, which is what the MATLAB does show.
            """
            if not self._normalize_by_max:
                return [-1.0, 1.0]
            peak = float(np.nanmax(np.abs(self._map))) or 1.0
            return [-peak, peak]

        def _update_trace(self) -> None:
            trace = pixel_timecourse(self._u, self._v, self._pixel)
            x = self._t if self._t is not None and self._t.size == trace.size else None
            if x is None:
                self._trace_curve.setData(trace)
            else:
                self._trace_curve.setData(x, trace)

        def _update_status(self, cursor=None) -> None:
            y, x = self._pixel
            bits = [f"seed ({y}, {x})"]
            if cursor is not None:
                cy, cx = cursor
                bits.append(f"cursor ({cy}, {cx}) r = {self._map[cy, cx]:+.3f}")
            bits.append(
                "normalised by max variance" if self._normalize_by_max else "true correlation"
            )
            if self._hover:
                bits.append("HOVER ON")
            if self._orient.rot or self._orient.flip:
                bits.append(
                    f"rot {self._orient.rot * 90}deg{' flipped' if self._orient.flip else ''}"
                )
            bits.append(f"{self._corr.n_components} components")
            self._status.setText("  |  ".join(bits))

        def _save_map(self) -> None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save correlation map", "correlation_map.npy", "NumPy (*.npy)"
            )
            if path:
                np.save(path, self._map)
                self._status.setText(f"saved {path}")

        # ---------------------------------------------------------------- programmatic API

        def set_pixel(self, y: int, x: int) -> None:
            """Move the seed. Exposed so tests and embedding code don't synthesise clicks."""
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise IndexError(f"pixel {(y, x)} outside image of shape {self.shape}")
            self._pixel = (int(y), int(x))
            self._refresh()

        @property
        def pixel(self) -> tuple[int, int]:
            return self._pixel

        @property
        def correlation_map(self) -> np.ndarray:
            """The currently displayed map, in data orientation."""
            return self._map

    return PixelCorrelationViewer


_CLASS = None


def _get_class():
    """Build the widget class once, on first use.

    The class body needs Qt at definition time, so it lives inside ``_build()`` — that is what
    keeps ``import widefield`` free of any Qt dependency. Caching matters beyond speed: two
    calls to ``_build()`` would produce two unrelated classes, so ``isinstance`` checks and
    ``qtbot`` bookkeeping would quietly disagree.
    """
    global _CLASS
    if _CLASS is None:
        _CLASS = _build()
    return _CLASS


def __getattr__(name):
    """Expose ``PixelCorrelationViewer`` lazily so importing this module needs no Qt."""
    if name == "PixelCorrelationViewer":
        return _get_class()
    raise AttributeError(name)


def pixel_correlation_viewer(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray | None = None,
    max_components: int | None = None,
    block: bool = True,
):
    """Open the seed-pixel correlation viewer. Equivalent to ``pixelCorrelationViewerSVD(U, V)``.

    Parameters
    ----------
    u : (Ypix, Xpix, nSV) spatial components.
    v : (nSV, nFrames) temporal components.
    t : optional frame times, used to label the seed's timecourse.
    max_components : cap the components used, to bound the precompute on huge sessions.
    block : run the Qt event loop until the window closes. Set False when embedding, or when
        driving the widget from code.

    Returns
    -------
    The viewer widget, so callers can keep a reference or inspect it.
    """
    app = ensure_app()
    viewer = _get_class()(u, v, t=t, max_components=max_components)
    viewer.resize(760, 820)
    viewer.show()
    viewer.setFocus()
    run_app(app, block)
    return viewer
