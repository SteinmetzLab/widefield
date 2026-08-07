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
``e``                        cycle shaded s.e.m.: selected condition / all / none
ctrl + scroll                step time (a plain scroll zooms, as pyqtgraph normally does)
==========================  ============================================================

Differences, all deliberate: ``i``/``j``/``k``/``l`` and the arrow keys move relative to the
screen rather than the data, so they stay correct after a rotation (the MATLAB's ``i``/``k`` are
data-space and invert visually once rotated). The ROI is a live, draggable polygon whose result
stays available as :attr:`roi` instead of being dumped into the base workspace by ``assignin``.
"""

from __future__ import annotations

import numpy as np

from widefield.colormaps import blueblackred, condition_colors, to_pyqtgraph
from widefield.events import event_locked_avg_svd, peri_event_series
from widefield.gui._common import (
    Orientation,
    ensure_app,
    install_hotkeys,
    make_bandpass_control,
    polygon_mask,
    require_qt,
    run_app,
    text_entry_focused,
    window_title,
)
from widefield.svd import flatten_u

__all__ = ["pixel_tuning_curve_viewer"]

_PIXEL_STEP = 5  # matches the MATLAB's i/j/k/l step
# Shaded-error modes cycled by the 'e' key.
_SEM_NONE, _SEM_SELECTED, _SEM_ALL = 0, 1, 2
_SEM_NAMES = {
    _SEM_NONE: "no s.e.m.",
    _SEM_SELECTED: "s.e.m. on selected",
    _SEM_ALL: "s.e.m. on all",
}


def _nanmean(a, axis):
    """nanmean without the all-NaN-slice warning (the result is still NaN)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis):
    """Sample s.d. (ddof=1) ignoring NaN, quietly."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis, ddof=1)


_TIMER_MS = 100  # MATLAB timer Period 0.1 s


def _build():
    pg, QtCore, _QtGui, QtWidgets = require_qt()

    class PixelTuningCurveViewer(QtWidgets.QWidget):
        def __init__(
            self,
            u,
            v,
            t,
            event_times,
            event_labels,
            calc_win,
            upsample=4,
            session=None,
            parent=None,
        ):
            super().__init__(parent)
            self._u = np.asarray(u)
            self.shape = (int(self._u.shape[0]), int(self._u.shape[1]))
            # Kept so the traces can be rebuilt per pixel with per-event spread, and so the
            # band-pass control can re-derive everything from the unfiltered components.
            self._v_raw = np.asarray(v, dtype=np.float32)
            self._v = self._v_raw
            self._t = np.asarray(t, dtype=float).ravel()
            self._event_times = np.asarray(event_times, dtype=float).ravel()
            self._event_labels = np.asarray(event_labels).ravel()
            self._calc_win = tuple(calc_win)
            self._upsample = int(upsample)
            self._fs = float(1.0 / np.median(np.diff(self._t))) if self._t.size > 1 else 1.0
            self._sem_mode = 1  # 0 = none, 1 = selected condition only, 2 = all

            self._recompute_average()
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
            self._sem: np.ndarray | None = None  # (nCond, nWindow) s.e.m. across events

            self.setWindowTitle(window_title("Pixel tuning curve (SVD)", session))
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
            # One shaded s.e.m. band per condition, drawn under the lines. FillBetweenItem
            # needs two curves, so each band keeps its own (hidden) upper/lower pair.
            self._sem_bands = []
            for c in self._colors:
                # A *transparent* pen, not pen=None: with no pen pyqtgraph never builds the
                # PlotDataItem's internal curve path, so FillBetweenItem.updatePath finds no
                # polygons and silently fills nothing. This is why the bands were invisible.
                invisible = pg.mkPen((0, 0, 0, 0))
                lo = self._trace_plot.plot(pen=invisible)
                hi = self._trace_plot.plot(pen=invisible)
                rgb = tuple((c * 255).astype(int))
                # 50% alpha in the trace's own color: dark enough to read the mean line
                # through, bright enough to see against the dark background.
                band = pg.FillBetweenItem(lo, hi, brush=pg.mkBrush((*rgb, 128)))
                band.setZValue(-10)  # behind the mean lines
                self._trace_plot.addItem(band)
                self._sem_bands.append((band, lo, hi))
            self._zero_line = pg.InfiniteLine(
                pos=0.0, angle=90, pen=pg.mkPen((160, 160, 160), style=QtCore.Qt.DashLine)
            )
            # White, not black: the plot background is dark, so a black cursor vanished.
            self._time_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("w", width=2))
            self._trace_plot.addItem(self._zero_line)
            self._trace_plot.addItem(self._time_line)

            # Tuning curve.
            self._tc_plot = self._glw.addPlot(row=0, col=3)
            self._tc_plot.setLabel("bottom", "Condition")
            self._tc_plot.setLabel("left", "Activity")
            self._tc_plot.showGrid(x=True, y=True, alpha=0.2)
            # The connecting line stays neutral; the points carry the condition colors so the
            # right panel reads against the middle one at a glance.
            self._tc_curve = self._tc_plot.plot(pen=pg.mkPen((150, 150, 150), width=1))
            self._tc_points = pg.ScatterPlotItem(pxMode=True)
            self._tc_points.setZValue(15)
            self._tc_plot.addItem(self._tc_points)
            # Vertical mean +/- s.e.m. bars on each condition, unconnected.
            self._tc_errors = pg.ErrorBarItem(pen=pg.mkPen((200, 200, 200), width=1), beam=0.0)
            self._tc_plot.addItem(self._tc_errors)

            if not self._numeric_labels:
                axis = self._tc_plot.getAxis("bottom")
                ticks = zip(self._cond_x, self._conditions, strict=True)
                axis.setTicks([[(x, str(c)) for x, c in ticks]])

            self._glw.ci.layout.setColumnStretchFactor(0, 2)
            self._glw.ci.layout.setColumnStretchFactor(1, 2)
            self._glw.ci.layout.setColumnStretchFactor(2, 2)
            self._glw.ci.layout.setColumnStretchFactor(3, 2)

            self.bandpass = make_bandpass_control(self._v_raw, self._fs, self._on_filtered)
            layout.addWidget(self.bandpass)

            self._status = QtWidgets.QLabel()
            layout.addWidget(self._status)
            hint = QtWidgets.QLabel(
                "click any panel · arrows: time/condition · ctrl+wheel: time · ijkl: pixel · "
                "p: play · f/s: speed · -/=: color scale · alt+arrows: rotate/flip · "
                "r: ROI · e: s.e.m."
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

        def _recompute_average(self) -> None:
            """Event-locked average of the components. Rebuilt when the band-pass changes.

            ``keep_peri=False``: the per-event spread is computed from the *pixel's* timecourse
            instead (see :meth:`_recompute_traces`), which is exact and costs kilobytes, whereas
            holding every event in component space would be hundreds of MB on an opto session.
            """
            self._avg = event_locked_avg_svd(
                self._v,
                self._t,
                self._event_times,
                self._event_labels,
                self._calc_win,
                upsample=self._upsample,
                keep_peri=False,
            )

        def _weights(self) -> np.ndarray:
            """Spatial weights of the current pixel, or the ROI mean."""
            if self.roi is not None and self.roi["mask"].any():
                return self._flat_u[self.roi["mask"].reshape(-1)].mean(axis=0)
            y, x = self._pixel
            return self._flat_u[y * self.shape[1] + x]

        def _recompute_traces(self) -> None:
            """Per-condition mean traces and their s.e.m. across events, for the current pixel.

            The mean could come straight from the averaged components (one einsum), but the
            s.e.m. cannot — a spread across events needs the individual events. Projecting the
            pixel onto ``V`` first and windowing *that* gives both, and is far cheaper than
            keeping every event in component space: one (nEvents, nWindow) matrix instead of
            (nSV, nEvents, nWindow).

            The mean is identical either way, because projection and averaging are both linear.
            """
            weights = self._weights()
            trace = weights @ self._v[: self._nsv]
            peri, _ = peri_event_series(
                trace,
                self._t,
                self._event_times,
                self._calc_win,
                upsample=self._upsample,
            )
            # peri rows follow sorted event order, matching self._avg.sorted_labels.
            labels = self._avg.sorted_labels
            n_cond, n_win = self._n_cond, self._win_samps.size
            self._traces = np.empty((n_cond, n_win))
            self._sem = np.empty((n_cond, n_win))
            for c, label in enumerate(self._conditions):
                block = peri[labels == label]
                self._traces[c] = _nanmean(block, axis=0)
                n = np.sum(np.isfinite(block), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sd = _nanstd(block, axis=0)
                    self._sem[c] = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)

        def _on_filtered(self, filtered_v, _description) -> None:
            """Band-passed components: redo the event-locked average and everything after it."""
            self._v = np.asarray(filtered_v, dtype=np.float32)
            self._recompute_average()
            self._avg_v = np.ascontiguousarray(self._avg.avg_v[:, : self._nsv, :], dtype=np.float32)
            self._stack = None
            self._stack_cond = None
            self._recompute_traces()
            self._refresh_all()

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
            """Ctrl+wheel steps time; a plain wheel is left to pyqtgraph so it can zoom."""
            if not (event.modifiers() & QtCore.Qt.ControlModifier):
                super().wheelEvent(event)
                return
            steps = event.angleDelta().y() / 120.0
            self._step_time(int(-steps) or (-1 if steps > 0 else 1))
            event.accept()

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
            elif key == QtCore.Qt.Key_E:
                self._sem_mode = (self._sem_mode + 1) % 3
                self._refresh_all()
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

                band, lo, hi = self._sem_bands[c]
                show = self._sem_mode == _SEM_ALL or (
                    self._sem_mode == _SEM_SELECTED and c == self._cond_idx
                )
                if show and self._sem is not None:
                    lo.setData(self._win_samps, self._traces[c] - self._sem[c])
                    hi.setData(self._win_samps, self._traces[c] + self._sem[c])
                band.setVisible(bool(show))
            self._time_line.setPos(float(self._win_samps[self._time_idx]))
            self._trace_plot.setYRange(*self._cax, padding=0)
            self._trace_plot.setXRange(
                float(self._win_samps[0]), float(self._win_samps[-1]), padding=0.02
            )

            tc = self._traces[:, self._time_idx]
            self._tc_curve.setData(self._cond_x, tc)
            # Each point in its condition's color; the selected one becomes a larger star.
            self._tc_points.setData(
                x=self._cond_x,
                y=tc,
                symbol=["star" if c == self._cond_idx else "o" for c in range(self._n_cond)],
                size=[18 if c == self._cond_idx else 8 for c in range(self._n_cond)],
                brush=[
                    pg.mkBrush(tuple((self._colors[c] * 255).astype(int)))
                    for c in range(self._n_cond)
                ],
                pen=[
                    (
                        pg.mkPen("w", width=2)
                        if c == self._cond_idx
                        else pg.mkPen(tuple((self._colors[c] * 255).astype(int)))
                    )
                    for c in range(self._n_cond)
                ],
            )
            if self._sem is not None:
                self._tc_errors.setData(
                    x=self._cond_x,
                    y=tc,
                    top=self._sem[:, self._time_idx],
                    bottom=self._sem[:, self._time_idx],
                )
                self._tc_errors.setVisible(self._sem_mode != _SEM_NONE)
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
            bits.append(_SEM_NAMES[self._sem_mode])
            if self._upsample > 1:
                bits.append(f"{self._upsample}x upsampled")
            if getattr(self, "bandpass", None) is not None:
                if self.bandpass.description != "unfiltered":
                    bits.append(self.bandpass.description)
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
    upsample: int = 4,
    session=None,
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
    upsample : sample the peri-event window this many times finer than the frame rate.
        Event times are jittered relative to frames, so interpolating each event onto a
        dense grid before averaging recovers sub-frame detail. 1 reproduces the MATLAB.
    """
    app = ensure_app()
    viewer = _get_class()(
        u, v, t, event_times, event_labels, calc_win, upsample=upsample, session=session
    )
    viewer.resize(1500, 620)
    viewer.show()
    viewer.setFocus()
    run_app(app, block)
    return viewer
