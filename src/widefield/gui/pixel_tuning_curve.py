"""Peri-event tuning viewer. Port of ``pixelTuningCurveViewerSVD.m``.

Three linked panels answering three questions about the same event-locked average:

1. **brain image** — what the cortex is doing at the selected time, for the selected condition
2. **traces** — how the selected *pixel* responds over time, one line per condition
3. **tuning curve** — how that pixel's response at the selected *time* varies with condition

Clicking in any panel changes the corresponding variable, and all three update.

MATLAB parity
-------------
==========================  ============================================================
click brain / traces / TC    set pixel / time / condition
left / right                 step time
up / down                    step condition (wraps)
scroll wheel                 step time
``i`` / ``k``                move pixel down / up 5 px    (screen-relative here)
``j`` / ``l``                move pixel left / right 5 px
``p``                        play / pause the movie of the selected condition
``f`` / ``s``                faster / slower playback
``-`` / ``=``                shrink / grow the color scale (and the y-limits with it)
alt + left/right             rotate 90 degrees
alt + up/down                flip vertically
``r``                        draw an ROI; traces become the ROI mean
==========================  ============================================================

Differences, all deliberate: ``i``/``j``/``k``/``l`` and the arrow keys move relative to the
screen rather than the data, so they stay correct after a rotation (the MATLAB's ``i``/``k`` are
data-space and invert visually once rotated). The ROI is a live, draggable polygon whose result
stays available as :attr:`roi` instead of being dumped into the base workspace by ``assignin``.
"""

from __future__ import annotations

import numpy as np

from widefield.colormaps import blueblackred, condition_colors, to_pyqtgraph
from widefield.events import event_locked_avg_svd
from widefield.gui._common import (
    Orientation,
    ensure_app,
    install_hotkeys,
    polygon_mask,
    require_qt,
    run_app,
    text_entry_focused,
)
from widefield.svd import flatten_u

__all__ = ["pixel_tuning_curve_viewer"]

_PIXEL_STEP = 5  # matches the MATLAB's i/j/k/l step
_TIMER_MS = 100  # MATLAB timer Period 0.1 s


def _build():
    pg, QtCore, _QtGui, QtWidgets = require_qt()

    class PixelTuningCurveViewer(QtWidgets.QWidget):
        def __init__(self, u, v, t, event_times, event_labels, calc_win, parent=None):
            super().__init__(parent)
            self._u = np.asarray(u)
            self.shape = (int(self._u.shape[0]), int(self._u.shape[1]))

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self._avg = event_locked_avg_svd(v, t, event_times, event_labels, calc_win)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

            self._win_samps = self._avg.win_samps
            self._conditions = self._avg.conditions
            self._n_cond = int(self._conditions.size)
            self._n_time = int(self._win_samps.size)
            if self._n_time == 0:
                raise ValueError(f"calc_win {calc_win} contains no samples")

            # Numeric labels are their own x-axis; categorical labels get evenly spaced slots.
            self._numeric_labels = np.issubdtype(self._conditions.dtype, np.number)
            self._cond_x = (
                self._conditions.astype(float)
                if self._numeric_labels
                else np.arange(self._n_cond, dtype=float)
            )

            # avg_v is (nCond, nSV, nTime); truncate U once to whatever rank V provided.
            self._nsv = min(self._u.shape[-1], self._avg.avg_v.shape[1])
            self._flat_u = flatten_u(self._u[..., : self._nsv]).astype(np.float32, copy=False)
            self._avg_v = np.ascontiguousarray(self._avg.avg_v[:, : self._nsv, :], dtype=np.float32)
            # Reconstructed image stack for whichever condition is selected, built on demand.
            # The peri-event window is short (tens of samples), so the whole condition costs
            # nPix * nWindow * 4 bytes — ~41 MB at 512x512 x 39 samples — and makes playback and
            # time-scrubbing free instead of a GEMV per step. Only one condition is held at a
            # time; switching condition rebuilds it with a single GEMM.
            self._stack: np.ndarray | None = None
            self._stack_cond: int | None = None

            self._orient = Orientation()
            self._pixel = (self.shape[0] // 2, self.shape[1] // 2)
            self._time_idx = 0
            self._cond_idx = 0
            self._cax = [-1.0, 1.0]
            self._rate = 1
            self._playing = False
            self._roi = None
            self.roi: dict | None = None

            self.setWindowTitle("Pixel tuning curve (SVD)")
            self._colors = condition_colors(self._n_cond)
            self._build_ui(pg, QtWidgets)

            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(_TIMER_MS)
            self._timer.timeout.connect(self._on_tick)

            self._recompute_traces()
            self._refresh_all()
            self._plot.vb.autoRange()

        # ---------------------------------------------------------------- construction

        def _build_ui(self, pg, QtWidgets):
            layout = QtWidgets.QVBoxLayout(self)
            self._glw = pg.GraphicsLayoutWidget()
            layout.addWidget(self._glw, stretch=1)

            # Brain image spans two columns, as subtightplot(1,4,1:2) does in the MATLAB.
            self._plot = self._glw.addPlot(row=0, col=0, colspan=2)
            self._plot.setAspectLocked(True)
            self._plot.invertY(True)
            self._plot.hideAxis("bottom")
            self._plot.hideAxis("left")
            self._image = pg.ImageItem()
            self._lut = to_pyqtgraph(blueblackred()).getLookupTable(nPts=256)
            self._image.setLookupTable(self._lut)
            self._plot.addItem(self._image)
            self._marker = pg.ScatterPlotItem(
                size=12, pen=pg.mkPen((0, 204, 0), width=2), brush=None, symbol="o"
            )
            self._marker.setZValue(10)
            self._plot.addItem(self._marker)
            self._colorbar = pg.ColorBarItem(
                values=tuple(self._cax), colorMap=to_pyqtgraph(blueblackred())
            )
            self._colorbar.setImageItem(self._image, insert_in=self._plot)

            # Traces: one line per condition, plus markers for t=0 and the selected time.
            self._trace_plot = self._glw.addPlot(row=0, col=2)
            self._trace_plot.setLabel("bottom", "Time from event (s)")
            self._trace_plot.setLabel("left", "Activity")
            self._trace_plot.showGrid(x=True, y=True, alpha=0.2)
            self._trace_curves = [
                self._trace_plot.plot(pen=pg.mkPen(tuple((c * 255).astype(int)), width=1))
                for c in self._colors
            ]
            self._zero_line = pg.InfiniteLine(
                pos=0.0, angle=90, pen=pg.mkPen("k", style=QtCore.Qt.DashLine)
            )
            self._time_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("k", width=2))
            self._trace_plot.addItem(self._zero_line)
            self._trace_plot.addItem(self._time_line)

            # Tuning curve.
            self._tc_plot = self._glw.addPlot(row=0, col=3)
            self._tc_plot.setLabel("bottom", "Condition")
            self._tc_plot.setLabel("left", "Activity")
            self._tc_plot.showGrid(x=True, y=True, alpha=0.2)
            self._tc_curve = self._tc_plot.plot(
                pen=pg.mkPen("k", width=1), symbol="o", symbolSize=6, symbolBrush="k"
            )
            self._tc_marker = pg.ScatterPlotItem(
                size=14, symbol="star", pen=pg.mkPen("k"), brush=pg.mkBrush("k")
            )
            self._tc_plot.addItem(self._tc_marker)
            if not self._numeric_labels:
                axis = self._tc_plot.getAxis("bottom")
                ticks = zip(self._cond_x, self._conditions, strict=True)
                axis.setTicks([[(x, str(c)) for x, c in ticks]])

            self._glw.ci.layout.setColumnStretchFactor(0, 2)
            self._glw.ci.layout.setColumnStretchFactor(1, 2)
            self._glw.ci.layout.setColumnStretchFactor(2, 2)
            self._glw.ci.layout.setColumnStretchFactor(3, 2)

            self._status = QtWidgets.QLabel()
            layout.addWidget(self._status)
            hint = QtWidgets.QLabel(
                "click any panel · arrows: time/condition · wheel: time · ijkl: pixel · "
                "p: play · f/s: speed · -/=: color scale · alt+arrows: rotate/flip · r: ROI"
            )
            hint.setStyleSheet("color: gray;")
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self._image.scene().sigMouseClicked.connect(self._on_scene_click)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            # Keys work from anywhere in the window, not only while this widget has focus.
            install_hotkeys(self, self._handle_key)

        # ---------------------------------------------------------------- computation

        def _brain_image(self) -> np.ndarray:
            if self._stack_cond != self._cond_idx:
                self._stack = self._flat_u @ self._avg_v[self._cond_idx]
                self._stack_cond = self._cond_idx
            return self._stack[:, self._time_idx].reshape(self.shape)

        def _recompute_traces(self) -> None:
            """(nCond, nTime) traces for the current pixel, or the ROI mean if one is active.

            One einsum over the small averaged V — never touches the full movie.
            """
            if self.roi is not None and self.roi["mask"].any():
                weights = self._flat_u[self.roi["mask"].reshape(-1)].mean(axis=0)
            else:
                y, x = self._pixel
                weights = self._flat_u[y * self.shape[1] + x]
            self._traces = np.einsum("s,csw->cw", weights, self._avg_v)

        # ---------------------------------------------------------------- interaction

        def _on_scene_click(self, event):
            pos = event.scenePos()
            if self._plot.sceneBoundingRect().contains(pos):
                pt = self._plot.vb.mapSceneToView(pos)
                dx, dy = int(np.floor(pt.x())), int(np.floor(pt.y()))
                dh, dw = self._orient.display_shape(self.shape)
                if 0 <= dy < dh and 0 <= dx < dw:
                    self._pixel = self._orient.to_data(dy, dx, self.shape)
                    self._recompute_traces()
                    self._refresh_all()
            elif self._trace_plot.sceneBoundingRect().contains(pos):
                x = self._trace_plot.vb.mapSceneToView(pos).x()
                self._time_idx = int(np.argmin(np.abs(self._win_samps - x)))
                self._refresh_all()
            elif self._tc_plot.sceneBoundingRect().contains(pos):
                x = self._tc_plot.vb.mapSceneToView(pos).x()
                self._cond_idx = int(np.argmin(np.abs(self._cond_x - x)))
                self._refresh_all()

        def wheelEvent(self, event):
            steps = event.angleDelta().y() / 120.0
            self._step_time(int(-steps) or (-1 if steps > 0 else 1))

        def _step_time(self, delta: int) -> None:
            self._time_idx = int(np.clip(self._time_idx + delta, 0, self._n_time - 1))
            self._refresh_all()

        def keyPressEvent(self, event):
            # text_entry_focused: a cutoff box that ignores a key lets it propagate here,
            # where acting on it would fire hotkeys while the user is typing a number.
            if text_entry_focused(self) or not self._handle_key(event.key(), event.modifiers()):
                super().keyPressEvent(event)

        def _handle_key(self, key, mods) -> bool:
            """Act on a hotkey; True if consumed. See install_hotkeys."""

            if mods & QtCore.Qt.AltModifier:
                if key == QtCore.Qt.Key_Right:
                    self._orient.rotate(-1)
                elif key == QtCore.Qt.Key_Left:
                    self._orient.rotate(1)
                elif key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                    self._orient.toggle_flip()
                else:
                    return False
                self._refresh_all()
                self._plot.vb.autoRange()
                return True

            if key == QtCore.Qt.Key_Right:
                self._step_time(1)
            elif key == QtCore.Qt.Key_Left:
                self._step_time(-1)
            elif key == QtCore.Qt.Key_Up:
                self._cond_idx = (self._cond_idx + 1) % self._n_cond  # wraps, as in the MATLAB
                self._refresh_all()
            elif key == QtCore.Qt.Key_Down:
                self._cond_idx = (self._cond_idx - 1) % self._n_cond
                self._refresh_all()
            elif key in (
                QtCore.Qt.Key_I,
                QtCore.Qt.Key_K,
                QtCore.Qt.Key_J,
                QtCore.Qt.Key_L,
            ):
                d_row, d_col = {
                    QtCore.Qt.Key_I: (_PIXEL_STEP, 0),
                    QtCore.Qt.Key_K: (-_PIXEL_STEP, 0),
                    QtCore.Qt.Key_J: (0, -_PIXEL_STEP),
                    QtCore.Qt.Key_L: (0, _PIXEL_STEP),
                }[key]
                self._pixel = self._orient.step_on_screen(*self._pixel, self.shape, d_row, d_col)
                self._recompute_traces()
                self._refresh_all()
            elif key == QtCore.Qt.Key_P:
                self.toggle_play()
            elif key == QtCore.Qt.Key_F:
                self._rate *= 2
                self._update_status()
            elif key == QtCore.Qt.Key_S:
                self._rate = max(1, -(-self._rate // 2))  # ceil, so 1 is the floor
                self._update_status()
            elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
                self._scale_cax(0.75)
            elif key in (QtCore.Qt.Key_Equal, QtCore.Qt.Key_Plus):
                self._scale_cax(1.25)
            elif key == QtCore.Qt.Key_R:
                self.toggle_roi()
            else:
                return False
            return True

        def toggle_play(self) -> None:
            self._playing = not self._playing
            if self._playing:
                self._timer.start()
            else:
                self._timer.stop()
            self._update_status()

        def _on_tick(self) -> None:
            self._time_idx = (self._time_idx + self._rate) % self._n_time
            self._refresh_all()

        def _scale_cax(self, factor: float) -> None:
            self._cax = [c * factor for c in self._cax]
            self._refresh_all()

        # ---------------------------------------------------------------- ROI

        def toggle_roi(self) -> None:
            """Add or remove a draggable polygon ROI; while present, traces are its mean."""
            if self._roi is not None:
                self._plot.removeItem(self._roi)
                self._roi = None
                self.roi = None
                self._recompute_traces()
                self._refresh_all()
                return

            dh, dw = self._orient.display_shape(self.shape)
            cy, cx = dh / 2.0, dw / 2.0
            r = max(3.0, min(dh, dw) / 6.0)
            pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
            self._roi = pg.PolyLineROI(pts, closed=True, pen=pg.mkPen((0, 255, 0), width=2))
            self._roi.setZValue(20)
            self._plot.addItem(self._roi)
            self._roi.sigRegionChanged.connect(self._on_roi_changed)
            self._on_roi_changed()

        def _on_roi_changed(self, *_):
            if self._roi is None:
                return
            # Handle positions are ROI-local; add the ROI's own position to get scene/display.
            origin = self._roi.pos()
            verts = np.array(
                [
                    [h.pos().x() + origin.x(), h.pos().y() + origin.y()]
                    for h in self._roi.getHandles()
                ]
            )
            display_shape = self._orient.display_shape(self.shape)
            mask_display = polygon_mask(verts, display_shape)
            mask = self._orient.unapply(mask_display)
            self.roi = {
                "mask": mask,
                "n_pixels": int(mask.sum()),
                "vertices": verts,
                "traces": None,  # filled in below once the traces are recomputed
                "win_samps": self._win_samps,
                "conditions": self._conditions,
            }
            self._recompute_traces()
            self.roi["traces"] = self._traces.copy()
            self._refresh_all()

        # ---------------------------------------------------------------- rendering

        def _refresh_all(self) -> None:
            self._image.setImage(
                self._orient.apply(self._brain_image()), autoLevels=False, levels=self._cax
            )
            self._colorbar.setLevels(tuple(self._cax))
            dy, dx = self._orient.to_display(*self._pixel, self.shape)
            self._marker.setData([dx + 0.5], [dy + 0.5])
            self._marker.setVisible(self.roi is None)

            for c, curve in enumerate(self._trace_curves):
                curve.setData(self._win_samps, self._traces[c])
                pen = pg.mkPen(
                    tuple((self._colors[c] * 255).astype(int)),
                    width=3 if c == self._cond_idx else 1,
                )
                curve.setPen(pen)
            self._time_line.setPos(float(self._win_samps[self._time_idx]))
            self._trace_plot.setYRange(*self._cax, padding=0)
            self._trace_plot.setXRange(
                float(self._win_samps[0]), float(self._win_samps[-1]), padding=0.02
            )

            tc = self._traces[:, self._time_idx]
            self._tc_curve.setData(self._cond_x, tc)
            self._tc_marker.setData([self._cond_x[self._cond_idx]], [tc[self._cond_idx]])
            self._tc_plot.setYRange(*self._cax, padding=0)
            if self._n_cond > 1:
                mid = (self._cond_x[-1] + self._cond_x[0]) / 2.0
                span = (self._cond_x[-1] - self._cond_x[0]) * 1.1
                self._tc_plot.setXRange(mid - span / 2, mid + span / 2, padding=0)

            self._update_status()

        def _update_status(self) -> None:
            cond = self._conditions[self._cond_idx]
            cond_str = f"{cond:g}" if self._numeric_labels else str(cond)
            bits = []
            if self.roi is not None:
                bits.append(f"ROI ({self.roi['n_pixels']} px)")
            else:
                bits.append(f"pixel ({self._pixel[0]}, {self._pixel[1]})")
            bits.append(f"t = {self._win_samps[self._time_idx]:+.3f} s")
            bits.append(f"condition {cond_str}")
            bits.append(f"scale +/-{self._cax[1]:.3g}")
            if self._playing:
                bits.append(f"PLAYING x{self._rate}")
            elif self._rate != 1:
                bits.append(f"rate x{self._rate}")
            if self._orient.rot or self._orient.flip:
                bits.append(
                    f"rot {self._orient.rot * 90}deg{' flipped' if self._orient.flip else ''}"
                )
            self._status.setText("  |  ".join(bits))

        def closeEvent(self, event):
            self._timer.stop()
            super().closeEvent(event)

        # ---------------------------------------------------------------- programmatic API

        def set_pixel(self, y: int, x: int) -> None:
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise IndexError(f"pixel {(y, x)} outside image of shape {self.shape}")
            self._pixel = (int(y), int(x))
            self._recompute_traces()
            self._refresh_all()

        @property
        def pixel(self):
            return self._pixel

        @property
        def time_index(self):
            return self._time_idx

        @property
        def condition_index(self):
            return self._cond_idx

        @property
        def traces(self) -> np.ndarray:
            """(nConditions, nWindow) traces currently plotted."""
            return self._traces

        @property
        def tuning_curve(self) -> np.ndarray:
            """(nConditions,) values at the selected time — the right-hand panel."""
            return self._traces[:, self._time_idx]

        @property
        def brain_image(self) -> np.ndarray:
            return self._brain_image()

    return PixelTuningCurveViewer


_CLASS = None


def _get_class():
    global _CLASS
    if _CLASS is None:
        _CLASS = _build()
    return _CLASS


def __getattr__(name):
    if name == "PixelTuningCurveViewer":
        return _get_class()
    raise AttributeError(name)


def pixel_tuning_curve_viewer(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    event_times: np.ndarray,
    event_labels: np.ndarray,
    calc_win: tuple[float, float] = (-0.5, 1.5),
    block: bool = True,
):
    """Open the tuning viewer. Equivalent to ``pixelTuningCurveViewerSVD(U,V,t,times,labels,win)``.

    Parameters
    ----------
    u : (Ypix, Xpix, nSV)
    v : (nSV, nFrames)
    t : (nFrames,) frame times, same clock as ``event_times``
    event_times : (nEvents,)
    event_labels : (nEvents,) condition per event. Numeric labels become the tuning-curve
        x-axis; string labels are spaced evenly and shown as tick labels.
    calc_win : (start, stop) seconds relative to each event.
    """
    app = ensure_app()
    viewer = _get_class()(u, v, t, event_times, event_labels, calc_win)
    viewer.resize(1500, 620)
    viewer.show()
    viewer.setFocus()
    run_app(app, block)
    return viewer
