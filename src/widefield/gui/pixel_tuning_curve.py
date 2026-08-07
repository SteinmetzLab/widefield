"""Peri-event tuning viewer. Port of ``pixelTuningCurveViewerSVD.m``, extended.

Four linked panels answering four questions about the same event-locked data:

==================  =========================================================================
brain image         what the cortex is doing at the selected time, for the selected condition
condition averages  how the selected *pixel* responds over time, one line per condition
tuning curve        how that pixel's response at the selected *time* varies with condition
single trials       every individual trial behind the selected condition's average
==================  =========================================================================

Laid out 2x2, with the two time-axis panels stacked in the right-hand column and their x-axes
linked, so zooming the averages zooms the trials with them.

The MATLAB has the first three in a row. The fourth exists because a condition average is not
evidence on its own: one trial that wandered off is enough to move a 90-trial mean far enough to
invent a whole effect, and there is no way to see that from the average. So the viewer also lets
you swap the average for a **median** (``m``), and click any single trial to send *that trial's*
movie to the brain panel.

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

Added here
----------
==========================  ============================================================
``m``                        mean +/- s.e.m.  <->  median + 95% CI
``e``                        cycle the shaded band: selected condition / all / none
click a single trial         select it; the brain panel shows that trial alone
``[`` / ``]``                previous / next trial of the selected condition
``esc`` or ``a``             back to the condition average
ctrl + scroll                step time (a plain scroll zooms, as pyqtgraph normally does)
==========================  ============================================================

Differences, all deliberate: ``i``/``j``/``k``/``l`` and the arrow keys move relative to the
screen rather than the data, so they stay correct after a rotation (the MATLAB's ``i``/``k`` are
data-space and invert visually once rotated). The ROI is a live, draggable polygon whose result
stays available as :attr:`roi` instead of being dumped into the base workspace by ``assignin``.

The band-pass control here high-passes **causally** (forwards only), unlike the continuous
viewers. Zero-phase filtering runs backwards as well, which pushes part of every response back
past ``t = 0``: the pre-event baselines then fan out by condition, in proportion to the response
each condition eventually produces, and look like a genuine anticipatory effect. Filtering
forwards only cannot move anything backwards. The cost is that the high-pass adds its own phase
lag to the response — modest at the cutoffs used here, and preferable to a fabricated baseline.
See :func:`widefield.svd.bandpass_filt`.

One thing the ``m`` toggle does *not* change is the brain image, which stays a mean over the
condition's trials. A pixelwise median image cannot be built from the component medians — the
median does not commute with ``U @ V`` — so it would mean reconstructing every trial at full
resolution for every displayed time point, which is seconds per frame rather than the ~8 ms that
makes time-scrubbing feel live. Select the offending trial instead: that shows its movie exactly.
"""

from __future__ import annotations

import numpy as np

from widefield.colormaps import blueblackred, condition_colors, to_pyqtgraph
from widefield.events import event_locked_avg_svd, peri_event_components, peri_event_series
from widefield.gui._common import (
    Orientation,
    ensure_app,
    install_hotkeys,
    install_wheel,
    make_bandpass_control,
    polygon_mask,
    require_qt,
    run_app,
    text_entry_focused,
    window_title,
)
from widefield.stats import BAND_LABEL, MEAN, MEDIAN, trial_summary
from widefield.svd import flatten_u

__all__ = ["pixel_tuning_curve_viewer"]

_PIXEL_STEP = 5  # matches the MATLAB's i/j/k/l step
# Shaded-band modes cycled by the 'e' key.
_SEM_NONE, _SEM_SELECTED, _SEM_ALL = 0, 1, 2
_SEM_NAMES = {_SEM_NONE: "no band", _SEM_SELECTED: "band on selected", _SEM_ALL: "band on all"}

_TIMER_MS = 100  # MATLAB timer Period 0.1 s
# How close a click must be, as a fraction of the plot's visible y-range, to count as landing
# *on* a trace rather than merely inside the panel. Without it, every click in the trials panel
# would select whichever trace happened to be nearest, however far away.
_PICK_TOLERANCE = 0.05


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
            statistic=MEAN,
            cax=None,
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
            self._sem_mode = _SEM_SELECTED
            if statistic not in (MEAN, MEDIAN):
                raise ValueError(f"statistic must be {MEAN!r} or {MEDIAN!r}; got {statistic!r}")
            self._statistic = statistic

            # Events sorted by time, matching the order event_locked_avg_svd works in; the
            # permutation is kept so a selected trial can be named by its original index.
            self._order = np.argsort(self._event_times, kind="stable")
            self._sorted_times = self._event_times[self._order]

            self._recompute_average()
            self._win_samps = self._avg.win_samps
            self._conditions = self._avg.conditions
            self._n_cond = int(self._conditions.size)
            self._n_time = int(self._win_samps.size)
            if self._n_time == 0:
                raise ValueError(f"calc_win {calc_win} contains no samples")

            # Row indices into the peri-event matrix, per condition. Computed once: the labels
            # never change, only the pixel they are evaluated at.
            labels = self._avg.sorted_labels
            self._cond_rows = [np.nonzero(labels == c)[0] for c in self._conditions]

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
            # Reconstructed image stack for whatever the brain panel is showing, built on demand.
            # The peri-event window is short (tens of samples), so a whole condition costs
            # nPix * nWindow * 4 bytes and makes playback and time-scrubbing free instead of a
            # GEMV per step. Exactly one stack is held — a condition average or a single trial —
            # keyed by which, so switching rebuilds rather than accumulating.
            self._stack: np.ndarray | None = None
            self._stack_key: tuple | None = None

            self._orient = Orientation()
            self._pixel = (self.shape[0] // 2, self.shape[1] // 2)
            self._time_idx = 0
            self._cond_idx = 0
            self._trial_idx: int | None = None  # position within the selected condition
            self._cax = self._auto_cax() if cax is None else [-abs(float(cax)), abs(float(cax))]
            self._rate = 1
            self._playing = False
            self._roi = None
            self.roi: dict | None = None
            self._peri: np.ndarray | None = None  # (nEvents, nWindow) at the current pixel
            self._lo: np.ndarray | None = None  # (nCond, nWindow) lower band edge
            self._hi: np.ndarray | None = None

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

            self._build_brain_panel(pg)
            self._build_average_panel(pg, QtCore)
            self._build_tuning_panel(pg)
            self._build_trial_panel(pg, QtCore)

            # The two time-axis panels share a column and an x-axis; zoom one, both follow.
            self._trial_plot.setXLink(self._trace_plot)
            self._glw.ci.layout.setColumnStretchFactor(0, 1)
            self._glw.ci.layout.setColumnStretchFactor(1, 1)
            self._glw.ci.layout.setRowStretchFactor(0, 1)
            self._glw.ci.layout.setRowStretchFactor(1, 1)

            # causal_highpass: this is the event-locked viewer, so a zero-phase high-pass would
            # push part of each response backwards past t = 0 and separate the pre-event
            # baselines by condition. Forwards-only cannot do that. Not a knob, because the
            # zero-phase answer here is simply wrong and nobody should have to know why.
            self.bandpass = make_bandpass_control(
                self._v_raw, self._fs, self._on_filtered, causal_highpass=True
            )
            layout.addWidget(self.bandpass)

            self._status = QtWidgets.QLabel()
            layout.addWidget(self._status)
            hint = QtWidgets.QLabel(
                "click any panel · arrows: time/condition · ctrl+wheel: time · ijkl: pixel · "
                "p: play · f/s: speed · -/=: color scale · alt+arrows: rotate/flip · r: ROI · "
                "m: mean/median · e: band · click a trial to isolate it · [ ]: step trials · "
                "esc: back to the average"
            )
            hint.setStyleSheet("color: gray;")
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self._image.scene().sigMouseClicked.connect(self._on_scene_click)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            # Keys and ctrl+wheel work from anywhere in the window, not only while this widget
            # has focus — and, for the wheel, not only when the cursor is off the plots.
            install_hotkeys(self, self._handle_key)
            install_wheel(self, self._handle_wheel)

        def _build_brain_panel(self, pg):
            self._plot = self._glw.addPlot(row=0, col=0)
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

        def _build_average_panel(self, pg, QtCore):
            """One line per condition, plus a shaded band and markers for t=0 and the cursor."""
            self._trace_plot = self._glw.addPlot(row=0, col=1)
            self._trace_plot.setTitle("Condition averages")
            self._trace_plot.setLabel("bottom", "Time from event (s)")
            self._trace_plot.setLabel("left", "Activity")
            self._trace_plot.showGrid(x=True, y=True, alpha=0.2)
            self._trace_curves = [
                self._trace_plot.plot(pen=pg.mkPen(tuple((c * 255).astype(int)), width=1))
                for c in self._colors
            ]
            # One shaded band per condition, drawn under the lines. FillBetweenItem needs two
            # curves, so each band keeps its own (hidden) upper/lower pair.
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

        def _build_tuning_panel(self, pg):
            self._tc_plot = self._glw.addPlot(row=1, col=0)
            self._tc_plot.setTitle("Tuning curve")
            self._tc_plot.setLabel("bottom", "Condition")
            self._tc_plot.setLabel("left", "Activity")
            self._tc_plot.showGrid(x=True, y=True, alpha=0.2)
            # The connecting line stays neutral; the points carry the condition colors so this
            # panel reads against the averages at a glance.
            self._tc_curve = self._tc_plot.plot(pen=pg.mkPen((150, 150, 150), width=1))
            self._tc_points = pg.ScatterPlotItem(pxMode=True)
            self._tc_points.setZValue(15)
            self._tc_plot.addItem(self._tc_points)
            # Vertical bars spanning the band on each condition, unconnected.
            self._tc_errors = pg.ErrorBarItem(pen=pg.mkPen((200, 200, 200), width=1), beam=0.0)
            self._tc_plot.addItem(self._tc_errors)

            if not self._numeric_labels:
                axis = self._tc_plot.getAxis("bottom")
                ticks = zip(self._cond_x, self._conditions, strict=True)
                axis.setTicks([[(x, str(c)) for x, c in ticks]])

        def _build_trial_panel(self, pg, QtCore):
            """Every trial of the selected condition, as one item plus one for the selected trial.

            Two curve items, not one per trial. A condition can hold hundreds of trials — 202 on
            a 2220-event opto session — and the traces are all the same color, so joining them
            with NaN into a single ``connect="finite"`` curve draws the lot in one pass and keeps
            redraw cost flat in the trial count. Only the highlighted trial needs its own item.
            """
            self._trial_plot = self._glw.addPlot(row=1, col=1)
            self._trial_plot.setLabel("bottom", "Time from event (s)")
            self._trial_plot.setLabel("left", "Activity")
            self._trial_plot.showGrid(x=True, y=True, alpha=0.2)
            self._trial_curves = self._trial_plot.plot(
                pen=pg.mkPen((255, 255, 255, 70), width=1), connect="finite"
            )
            self._trial_highlight = self._trial_plot.plot(pen=pg.mkPen("w", width=2.5))
            self._trial_highlight.setZValue(5)
            self._trial_mean = self._trial_plot.plot(pen=pg.mkPen("w", width=2.5))
            self._trial_mean.setZValue(10)
            self._trial_zero = pg.InfiniteLine(
                pos=0.0, angle=90, pen=pg.mkPen((160, 160, 160), style=QtCore.Qt.DashLine)
            )
            self._trial_time = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("w", width=2))
            self._trial_plot.addItem(self._trial_zero)
            self._trial_plot.addItem(self._trial_time)

        # ---------------------------------------------------------------- computation

        def _stack_for_display(self) -> np.ndarray:
            """(nPix, nWindow) reconstruction of whatever the brain panel should be showing."""
            key = (self._cond_idx, self._trial_idx)
            if self._stack_key != key:
                if self._trial_idx is None:
                    peri_v = self._avg_v[self._cond_idx]
                else:
                    peri_v, _ = peri_event_components(
                        self._v[: self._nsv],
                        self._t,
                        self.trial_time,
                        self._calc_win,
                        upsample=self._upsample,
                    )
                    peri_v = np.ascontiguousarray(peri_v, dtype=np.float32)
                self._stack = self._flat_u @ peri_v
                self._stack_key = key
            return self._stack

        def _brain_image(self) -> np.ndarray:
            return self._stack_for_display()[:, self._time_idx].reshape(self.shape)

        def _auto_cax(self) -> list[float]:
            """Symmetric color scale taken from the data, unless the caller named one.

            A fixed default cannot serve both callers: dF/F lives near +/-0.4, while the
            hemodynamically corrected components of a real session run past +/-60. At the wrong
            scale the image is uniform gray and every trace is pinned off the top and bottom of
            its panel — a scale problem that looks like a data problem, and one you can only fix
            by holding down a key.

            So: the 99.5th percentile of ``|value|``, high enough to ignore the occasional hot
            pixel and still contain the response. Taken over **every** condition, not just the
            one selected at startup — up/down cycles conditions and the scale does not follow
            them, so a scale fitted to the weakest one would clip the strongest. Affordable
            because it runs on a few thousand pixels rather than all of them; the estimate is a
            percentile, and a percentile does not need every sample.
            """
            step = max(1, self._flat_u.shape[0] // 5000)
            u_sample = self._flat_u[::step]
            values = np.concatenate(
                [np.abs(u_sample @ self._avg_v[c]).ravel() for c in range(self._n_cond)]
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return [-1.0, 1.0]
            m = float(np.percentile(finite, 99.5))
            return [-m, m] if np.isfinite(m) and m > 0 else [-1.0, 1.0]

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
            """Per-trial traces at the current pixel, and each condition's center and band.

            The center could come straight from the averaged components (one einsum), but the
            spread could not — that needs the individual trials. Projecting the pixel onto ``V``
            first and windowing *that* gives both, and is far cheaper than keeping every event in
            component space: one (nEvents, nWindow) matrix instead of (nSV, nEvents, nWindow).
            It is also what feeds the single-trial panel.

            In mean mode the result is identical to averaging the components, because projection
            and averaging are both linear. In median mode it is not — and could not be, since the
            median of a projection is not the projection of medians. This is the honest version:
            the median of what this pixel actually did on each trial.
            """
            weights = self._weights()
            trace = weights @ self._v[: self._nsv]
            # Sorted times, not the caller's order: peri_event_series returns rows in the order
            # it is given, while the labels these rows are grouped by come from
            # event_locked_avg_svd, which sorts by time. Passing the unsorted times paired every
            # trial with somebody else's condition whenever the caller's events were not already
            # in time order — invisible on an opto session, where they always are.
            self._peri, _ = peri_event_series(
                trace,
                self._t,
                self._sorted_times,
                self._calc_win,
                upsample=self._upsample,
            )
            n_cond, n_win = self._n_cond, self._win_samps.size
            self._traces = np.empty((n_cond, n_win))
            self._lo = np.empty((n_cond, n_win))
            self._hi = np.empty((n_cond, n_win))
            for c, rows in enumerate(self._cond_rows):
                center, lo, hi = trial_summary(self._peri[rows], self._statistic)
                self._traces[c], self._lo[c], self._hi[c] = center, lo, hi

        def _on_filtered(self, filtered_v, _description) -> None:
            """Band-passed components: redo the event-locked average and everything after it."""
            self._v = np.asarray(filtered_v, dtype=np.float32)
            self._recompute_average()
            self._avg_v = np.ascontiguousarray(self._avg.avg_v[:, : self._nsv, :], dtype=np.float32)
            self._stack = None
            self._stack_key = None
            self._recompute_traces()
            self._refresh_all()

        # ---------------------------------------------------------------- trial selection

        @property
        def trial_index(self) -> int | None:
            """Which trial of the selected condition the brain panel is showing, or None."""
            return self._trial_idx

        @property
        def n_trials(self) -> int:
            """Trials in the selected condition."""
            return int(self._cond_rows[self._cond_idx].size)

        @property
        def trial_time(self) -> float:
            """Event time of the selected trial. Raises if the average is being shown."""
            if self._trial_idx is None:
                raise ValueError("no single trial is selected")
            return float(self._sorted_times[self._cond_rows[self._cond_idx][self._trial_idx]])

        @property
        def trial_event(self) -> int:
            """Index of the selected trial in the *original* event arrays, for cross-referencing."""
            if self._trial_idx is None:
                raise ValueError("no single trial is selected")
            return int(self._order[self._cond_rows[self._cond_idx][self._trial_idx]])

        @property
        def trial_traces(self) -> np.ndarray:
            """(nTrials, nWindow) individual trials of the selected condition, as plotted."""
            return self._peri[self._cond_rows[self._cond_idx]]

        def select_trial(self, index: int | None) -> None:
            """Show one trial of the selected condition in the brain panel; None for the average."""
            if index is not None:
                n = self.n_trials
                if not (0 <= int(index) < n):
                    raise IndexError(f"trial {index} outside 0..{n - 1} for this condition")
                index = int(index)
            self._trial_idx = index
            self._refresh_all()

        def _step_trial(self, delta: int) -> None:
            n = self.n_trials
            if n == 0:
                return
            # Stepping from the average starts at the first trial, so ']' is a way in.
            current = -1 if self._trial_idx is None else self._trial_idx
            self._trial_idx = int(np.clip(current + delta, 0, n - 1))
            self._refresh_all()

        def _set_condition(self, index: int) -> None:
            """Change condition, dropping any trial selection — its index means nothing here."""
            self._cond_idx = int(index) % self._n_cond
            self._trial_idx = None
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
                self._click_averages(self._trace_plot.vb.mapSceneToView(pos))
            elif self._trial_plot.sceneBoundingRect().contains(pos):
                self._click_trials(self._trial_plot.vb.mapSceneToView(pos))
            elif self._tc_plot.sceneBoundingRect().contains(pos):
                x = self._tc_plot.vb.mapSceneToView(pos).x()
                self._set_condition(int(np.argmin(np.abs(self._cond_x - x))))

        def _click_averages(self, pt) -> None:
            """Set the time, and go back to the condition average — clicking a line picks it."""
            self._time_idx = int(np.argmin(np.abs(self._win_samps - pt.x())))
            self._trial_idx = None
            hit = self._nearest_trace(self._trace_plot, self._traces, pt)
            if hit is not None:
                self._cond_idx = hit
            self._refresh_all()

        def _click_trials(self, pt) -> None:
            """Set the time, and select (or deselect) the trial nearest the click."""
            self._time_idx = int(np.argmin(np.abs(self._win_samps - pt.x())))
            hit = self._nearest_trace(self._trial_plot, self.trial_traces, pt)
            if hit is not None:
                # Clicking the highlighted trial again releases it — a toggle, so the panel does
                # not become a trap you can only leave with a key.
                self._trial_idx = None if hit == self._trial_idx else hit
            self._refresh_all()

        def _nearest_trace(self, plot, curves, pt) -> int | None:
            """Row of ``curves`` passing closest to ``pt``, or None if the click missed them all.

            Distance is measured in y at the clicked time and scaled by the panel's visible
            y-range, so "close" means close *on screen* however the axes happen to be zoomed.
            """
            if curves is None or len(curves) == 0:
                return None
            col = int(np.argmin(np.abs(self._win_samps - pt.x())))
            values = np.asarray(curves)[:, col]
            if not np.isfinite(values).any():
                return None
            lo, hi = plot.vb.viewRange()[1]
            span = abs(hi - lo)
            if span <= 0:
                return None
            with np.errstate(invalid="ignore"):
                distance = np.abs(values - pt.y())
            distance = np.where(np.isfinite(distance), distance, np.inf)
            best = int(np.argmin(distance))
            return best if distance[best] <= _PICK_TOLERANCE * span else None

        def wheelEvent(self, event):
            """Ctrl+wheel steps time; a plain wheel is left to pyqtgraph so it can zoom.

            In practice :func:`install_wheel` handles the event long before it could reach here —
            pyqtgraph's view consumes it on the way up. This override only covers a wheel event
            delivered straight to the container.
            """
            if not self._handle_wheel(event.angleDelta().y() / 120.0, event.modifiers()):
                super().wheelEvent(event)
                return
            event.accept()

        def _handle_wheel(self, steps, mods) -> bool:
            """Act on a wheel notch; True if consumed. See install_wheel."""
            if not (mods & QtCore.Qt.ControlModifier):
                return False  # plain wheel: pyqtgraph's zoom
            # Scrolling down (negative) moves forward in time, matching the MATLAB, which counts
            # VerticalScrollCount positive downwards. Round to at least one step so a trackpad's
            # fractional notches still move.
            n = int(round(steps)) or (1 if steps > 0 else -1)
            self._step_time(-n)
            return True

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
                self._set_condition(self._cond_idx + 1)  # wraps, as in the MATLAB
            elif key == QtCore.Qt.Key_Down:
                self._set_condition(self._cond_idx - 1)
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
            elif key == QtCore.Qt.Key_M:
                self.toggle_statistic()
            elif key == QtCore.Qt.Key_BracketRight:
                self._step_trial(1)
            elif key == QtCore.Qt.Key_BracketLeft:
                self._step_trial(-1)
            elif key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_A):
                self.select_trial(None)
            else:
                return False
            return True

        def toggle_statistic(self) -> None:
            """Swap mean +/- s.e.m. for median + 95% CI, or back."""
            self._statistic = MEDIAN if self._statistic == MEAN else MEAN
            self._recompute_traces()
            self._refresh_all()

        @property
        def statistic(self) -> str:
            return self._statistic

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
            self._refresh_brain()
            self._refresh_averages()
            self._refresh_trials()
            self._refresh_tuning()
            self._update_status()

        def _refresh_brain(self) -> None:
            self._image.setImage(
                self._orient.apply(self._brain_image()), autoLevels=False, levels=self._cax
            )
            self._colorbar.setLevels(tuple(self._cax))
            dy, dx = self._orient.to_display(*self._pixel, self.shape)
            self._marker.setData([dx + 0.5], [dy + 0.5])
            self._marker.setVisible(self.roi is None)
            if self._trial_idx is None:
                title = f"Condition {self._condition_label()} — mean of {self.n_trials} trials"
            else:
                title = (
                    f"Condition {self._condition_label()} — trial {self._trial_idx + 1}"
                    f"/{self.n_trials} alone (event {self.trial_event}, "
                    f"t = {self.trial_time:.2f} s)"
                )
            self._plot.setTitle(title)

        def _refresh_averages(self) -> None:
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
                if show and self._lo is not None:
                    lo.setData(self._win_samps, self._lo[c])
                    hi.setData(self._win_samps, self._hi[c])
                band.setVisible(bool(show))
            self._time_line.setPos(float(self._win_samps[self._time_idx]))
            self._trace_plot.setYRange(*self._cax, padding=0)
            self._trace_plot.setXRange(
                float(self._win_samps[0]), float(self._win_samps[-1]), padding=0.02
            )
            self._trace_plot.setTitle(f"Condition averages ({self._center_label()})")

        def _refresh_trials(self) -> None:
            """Redraw the single-trial panel for the selected condition."""
            block = self.trial_traces
            n_trials, n_win = block.shape
            if n_trials:
                # One polyline for all of them: x repeats the window, y carries a NaN between
                # trials so connect="finite" lifts the pen. x stays finite throughout, because a
                # NaN there would drop the point instead of just breaking the line.
                xs = np.concatenate(
                    [
                        np.broadcast_to(self._win_samps, (n_trials, n_win)),
                        np.full((n_trials, 1), self._win_samps[-1]),
                    ],
                    axis=1,
                ).ravel()
                ys = np.concatenate([block, np.full((n_trials, 1), np.nan)], axis=1).ravel()
                self._trial_curves.setData(xs, ys)
            else:
                self._trial_curves.setData([], [])

            self._trial_mean.setData(self._win_samps, self._traces[self._cond_idx])
            self._trial_mean.setPen(
                pg.mkPen(tuple((self._colors[self._cond_idx] * 255).astype(int)), width=2.5)
            )
            if self._trial_idx is not None:
                self._trial_highlight.setData(self._win_samps, block[self._trial_idx])
            self._trial_highlight.setVisible(self._trial_idx is not None)
            self._trial_time.setPos(float(self._win_samps[self._time_idx]))

            # Individual trials are much noisier than their average, so this panel scales to its
            # own data rather than to the color scale — otherwise almost every trace would run
            # off the top and bottom. Robust limits, so one wild trial does not flatten the rest.
            if n_trials and np.isfinite(block).any():
                lo, hi = np.nanpercentile(block, [1.0, 99.0])
                pad = 0.15 * max(hi - lo, 1e-9)
                self._trial_plot.setYRange(lo - pad, hi + pad, padding=0)
            self._trial_plot.setTitle(
                f"Single trials — condition {self._condition_label()} ({n_trials})"
            )

        def _refresh_tuning(self) -> None:
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
            if self._lo is not None:
                # ErrorBarItem wants lengths, not edges, and cannot draw a NaN one — a condition
                # with too few trials for a CI simply gets no bar.
                top = np.nan_to_num(self._hi[:, self._time_idx] - tc, nan=0.0)
                bottom = np.nan_to_num(tc - self._lo[:, self._time_idx], nan=0.0)
                self._tc_errors.setData(x=self._cond_x, y=tc, top=top, bottom=bottom)
                self._tc_errors.setVisible(self._sem_mode != _SEM_NONE)
            self._tc_plot.setYRange(*self._cax, padding=0)
            if self._n_cond > 1:
                mid = (self._cond_x[-1] + self._cond_x[0]) / 2.0
                span = (self._cond_x[-1] - self._cond_x[0]) * 1.1
                self._tc_plot.setXRange(mid - span / 2, mid + span / 2, padding=0)
            self._tc_plot.setTitle(
                f"Tuning curve at t = {self._win_samps[self._time_idx]:+.3f} s "
                f"({self._center_label()})"
            )

        def _condition_label(self) -> str:
            cond = self._conditions[self._cond_idx]
            return f"{cond:g}" if self._numeric_labels else str(cond)

        def _center_label(self) -> str:
            band = BAND_LABEL[self._statistic]
            return f"mean +/- {band}" if self._statistic == MEAN else f"median + {band}"

        def _update_status(self) -> None:
            bits = []
            if self.roi is not None:
                bits.append(f"ROI ({self.roi['n_pixels']} px)")
            else:
                bits.append(f"pixel ({self._pixel[0]}, {self._pixel[1]})")
            bits.append(f"t = {self._win_samps[self._time_idx]:+.3f} s")
            bits.append(f"condition {self._condition_label()}")
            if self._trial_idx is None:
                bits.append(f"average of {self.n_trials} trials")
            else:
                bits.append(
                    f"TRIAL {self._trial_idx + 1}/{self.n_trials} (event {self.trial_event})"
                )
            bits.append(self._center_label())
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
            """(nConditions, nWindow) center estimates currently plotted."""
            return self._traces

        @property
        def band(self) -> tuple[np.ndarray, np.ndarray]:
            """(lo, hi) edges of the shaded band, each (nConditions, nWindow)."""
            return self._lo, self._hi

        @property
        def _sem(self) -> np.ndarray:
            """Half-width of the band. Exactly the s.e.m. in mean mode; kept for that reading."""
            return (self._hi - self._lo) / 2.0

        @property
        def tuning_curve(self) -> np.ndarray:
            """(nConditions,) values at the selected time — the tuning panel."""
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
    statistic: str = MEAN,
    cax: float | None = None,
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
    statistic : ``"mean"`` (+/- s.e.m.) or ``"median"`` (+ 95% CI). Toggled live with ``m``.
    cax : half-width of the symmetric color scale. ``None`` reads it off the data, which is
        what you want unless you are comparing sessions and need them on identical scales.
    """
    app = ensure_app()
    viewer = _get_class()(
        u,
        v,
        t,
        event_times,
        event_labels,
        calc_win,
        upsample=upsample,
        statistic=statistic,
        cax=cax,
        session=session,
    )
    viewer.resize(1400, 900)
    viewer.show()
    viewer.setFocus()
    run_app(app, block)
    return viewer
