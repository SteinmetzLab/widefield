"""Movie viewer with synchronized traces. Port of ``movieWithTracesSVD.m``.

Plays the reconstructed movie next to a scrolling window of whatever traces you hand it (wheel
speed, pupil, licks, a V component, ...) plus the timecourse of every pixel you have clicked, so
you can see an activity pattern and read off what the animal was doing at that moment. Optional
auxiliary video panels put the eye/face cameras alongside.

MATLAB parity
-------------
==========================  ============================================================
``p``                        play / pause
``r``                        start / stop recording (needs ``movie_save_path``)
up / down                    double / halve playback rate
``b``                        jump back 20 frames x rate
click                        move the selected pixel
**ctrl+click**               add another pixel (each gets its own color)
``c``                        clear all but the last pixel
``-`` / ``=``                shrink / grow the color scale (stays centered on zero)
alt + left/right             rotate 90 degrees
alt + up/down                flip vertically
==========================  ============================================================

Adding a pixel is ctrl+click rather than the MATLAB's right-click: in pyqtgraph right-click
opens the view's own context menu, so the two actions fought each other.

Additions over the MATLAB: a scrub slider and frame/time readout, ``home`` to jump to the start,
a **temporal band-pass** (type cutoffs in Hz; filtering ``V`` filters every pixel, so it is cheap
to re-apply), a **Follow** toggle so a manual zoom on the trace plots survives playback instead
of being reset every frame, and ``nsv_display`` to cap the components used per frame when
full-rank playback can't keep up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from widefield.colormaps import blueblackred, to_pyqtgraph
from widefield.gui._common import (
    Orientation,
    ensure_app,
    install_hotkeys,
    require_qt,
    run_app,
    text_entry_focused,
)
from widefield.svd import bandpass_filt, flatten_u

__all__ = ["Trace", "AuxVideo", "movie_with_traces"]

_WINDOW_SECONDS = 10.0  # trace window width; MATLAB's windowSize
_TIMER_MS = 100  # MATLAB timer Period 0.1 s
_JUMP_BACK_FRAMES = 20  # MATLAB's 'b'
# Frames reconstructed per batch during playback. At 512x512x2000 a block costs
# nPix * block * 4 bytes (~33 MB at 32 frames) and turns a GEMV into a GEMM.
_PREFETCH_BLOCK = 32

# MATLAB's default axes ColorOrder, so multi-pixel colors match a MATLAB figure.
_PIXEL_COLORS = np.array(
    [
        [0.0000, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ]
)
# The MATLAB cycles through only the first 5 (nColors = 5) even though 7 are defined.
_N_CYCLE = 5


@dataclass
class Trace:
    """One trace to plot alongside the movie.

    ``t`` and ``v`` need not match the imaging clock's sampling — they are plotted on their own
    time base, which is the point (wheel encoders and cameras run at different rates).
    ``lims`` fixes the y-range; without it the range is taken from the data once.
    """

    t: np.ndarray
    v: np.ndarray
    name: str = ""
    lims: tuple[float, float] | None = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float).ravel()
        self.v = np.asarray(self.v, dtype=float).ravel()
        if self.t.size != self.v.size:
            raise ValueError(
                f"trace {self.name!r}: t has {self.t.size} samples, v has {self.v.size}"
            )


@dataclass
class AuxVideo:
    """An auxiliary video panel.

    ``render`` is called as ``render(image_item, time, data)`` whenever the frame changes and is
    responsible for putting something in the panel — the same contract as the MATLAB's function
    handle, so an existing frame-grabber wraps directly.
    """

    render: object
    data: object = None
    name: str = ""
    extra: dict = field(default_factory=dict)


def _as_traces(traces) -> list[Trace]:
    """Accept Trace objects, dicts, or None — dicts keep parity with the MATLAB struct array."""
    if traces is None:
        return []
    out = []
    for item in traces:
        if isinstance(item, Trace):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
                Trace(
                    t=item["t"],
                    v=item["v"],
                    name=item.get("name", ""),
                    lims=item.get("lims"),
                )
            )
        else:
            raise TypeError(f"traces must be Trace or dict, got {type(item).__name__}")
    return out


def _build():
    pg, QtCore, QtGui, QtWidgets = require_qt()

    class MovieWithTracesViewer(QtWidgets.QWidget):
        def __init__(
            self,
            u,
            v,
            t=None,
            traces=None,
            movie_save_path=None,
            aux_videos=None,
            nsv_display=None,
            cax=(-0.4, 0.4),
            parent=None,
        ):
            super().__init__(parent)
            self._u = np.asarray(u)
            self._v = np.asarray(v)
            self.shape = (int(self._u.shape[0]), int(self._u.shape[1]))
            self._n_frames = int(self._v.shape[1])
            self._t = (
                np.arange(self._n_frames, dtype=float)
                if t is None
                else np.asarray(t, dtype=float).ravel()
            )
            if self._t.size != self._n_frames:
                raise ValueError(f"t has {self._t.size} samples but V has {self._n_frames} frames")

            nsv = min(self._u.shape[-1], self._v.shape[0])
            if nsv_display is not None:
                nsv = min(nsv, int(nsv_display))
            # Reconstructing a frame is a matrix product against flat U, and it is the only hot
            # path during playback. Two layout choices earn most of the speed:
            #   * flat U C-contiguous float32 (copy=False, so free for arrays off disk),
            #   * V Fortran-ordered so that V[:, frame] is a *contiguous* vector. With a C-order
            #     V that column is strided and BLAS copies it on every single frame, which
            #     measured 10.7 ms/frame versus 7.9 ms on a real 512x512 session.
            self._flat_u = flatten_u(self._u[..., :nsv]).astype(np.float32, copy=False)
            # Keep the unfiltered components: the band-pass control re-derives _v32 from these,
            # so filtering is always a single pass from the original rather than compounding.
            self._v_raw = np.asarray(self._v[:nsv], dtype=np.float64)
            self._v32 = np.asfortranarray(self._v[:nsv], dtype=np.float32)
            self._block_start = -1
            self._block: np.ndarray | None = None
            self._fs = float(1.0 / np.median(np.diff(self._t))) if self._t.size > 1 else 1.0

            self._traces = _as_traces(traces)
            self._aux = list(aux_videos or [])
            self._orient = Orientation()
            self._frame = 0
            self._rate = 1
            self._playing = False
            self._cax = [float(cax[0]), float(cax[1])]
            self._pixels: list[tuple[int, int]] = [(self.shape[0] // 2, self.shape[1] // 2)]
            self._pixel_traces: list[np.ndarray] = []

            self._writer = None
            self._movie_save_path = movie_save_path
            self._recording = False

            self.setWindowTitle("Movie with traces (SVD)")
            self._build_ui(pg, QtWidgets)
            self._recompute_pixel_traces()

            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(_TIMER_MS)
            self._timer.timeout.connect(self._on_tick)

            self._refresh(full=True)

        # ---------------------------------------------------------------- construction

        def _build_ui(self, pg, QtWidgets):
            root = QtWidgets.QVBoxLayout(self)

            controls = QtWidgets.QHBoxLayout()
            self._play_btn = QtWidgets.QPushButton("play")
            self._play_btn.setCheckable(True)
            self._play_btn.setFixedWidth(70)
            self._play_btn.toggled.connect(self._on_play_toggled)
            controls.addWidget(self._play_btn)

            self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self._slider.setRange(0, max(0, self._n_frames - 1))
            self._slider.valueChanged.connect(self._on_slider)
            controls.addWidget(self._slider, stretch=1)

            self._readout = QtWidgets.QLabel()
            self._readout.setMinimumWidth(260)
            controls.addWidget(self._readout)

            # Follow: keep the trace window centerd on the current frame. Any manual zoom or pan
            # switches it off so the user's view survives playback (see _on_manual_range).
            self._follow_chk = QtWidgets.QCheckBox("Follow")
            self._follow_chk.setChecked(True)
            self._follow_chk.setToolTip(
                "Keep the trace window centered on the current frame.\n"
                "Zooming or panning a trace plot turns this off; re-check it to resume."
            )
            self._follow_chk.toggled.connect(self._on_follow_toggled)
            controls.addWidget(self._follow_chk)

            if self._movie_save_path is not None:
                self._rec_btn = QtWidgets.QPushButton("record")
                self._rec_btn.setCheckable(True)
                self._rec_btn.setFixedWidth(80)
                self._rec_btn.toggled.connect(self._set_recording)
                controls.addWidget(self._rec_btn)
            root.addLayout(controls)

            # Temporal band-pass. Filtering V filters every pixel, so this is cheap enough to
            # re-apply interactively.
            filt_row = QtWidgets.QHBoxLayout()
            filt_row.addWidget(QtWidgets.QLabel("Band-pass — high-pass (Hz):"))
            self._hp_edit = QtWidgets.QLineEdit("0")
            self._hp_edit.setFixedWidth(70)
            self._hp_edit.setToolTip("Lower cutoff in Hz. 0 means no high-pass.")
            filt_row.addWidget(self._hp_edit)
            filt_row.addWidget(QtWidgets.QLabel("low-pass (Hz):"))
            self._lp_edit = QtWidgets.QLineEdit("inf")
            self._lp_edit.setFixedWidth(70)
            self._lp_edit.setToolTip(
                f"Upper cutoff in Hz. inf means no low-pass (Nyquist is " f"{self._fs / 2:.2f} Hz)."
            )
            filt_row.addWidget(self._lp_edit)
            apply_btn = QtWidgets.QPushButton("Apply")
            apply_btn.setFixedWidth(70)
            apply_btn.clicked.connect(self._apply_filter)
            filt_row.addWidget(apply_btn)
            reset_btn = QtWidgets.QPushButton("Reset")
            reset_btn.setFixedWidth(70)
            reset_btn.clicked.connect(self._reset_filter)
            filt_row.addWidget(reset_btn)
            self._filt_label = QtWidgets.QLabel("unfiltered")
            self._filt_label.setStyleSheet("color: gray;")
            filt_row.addWidget(self._filt_label)
            filt_row.addStretch(1)
            # Enter in either box applies, which is what you expect after typing a number.
            self._hp_edit.returnPressed.connect(self._apply_filter)
            self._lp_edit.returnPressed.connect(self._apply_filter)
            root.addLayout(filt_row)

            self._glw = pg.GraphicsLayoutWidget()
            root.addWidget(self._glw, stretch=1)

            # Column 0: the movie. Columns 1..: stacked traces, then any aux videos.
            self._plot = self._glw.addPlot(row=0, col=0, rowspan=max(1, len(self._traces) + 1))
            self._plot.setAspectLocked(True)
            self._plot.invertY(True)
            self._plot.hideAxis("bottom")
            self._plot.hideAxis("left")
            self._image = pg.ImageItem()
            self._image.setLookupTable(to_pyqtgraph(blueblackred()).getLookupTable(nPts=256))
            self._plot.addItem(self._image)
            self._markers = pg.ScatterPlotItem(size=12, symbol="o", pen=pg.mkPen("k", width=1))
            self._markers.setZValue(10)
            self._plot.addItem(self._markers)
            self._colorbar = pg.ColorBarItem(
                values=tuple(self._cax), colorMap=to_pyqtgraph(blueblackred())
            )
            self._colorbar.setImageItem(self._image, insert_in=self._plot)

            n_rows = len(self._traces) + 1
            self._trace_plots = []
            self._trace_curves = []
            self._trace_marks = []
            for i in range(n_rows):
                p = self._glw.addPlot(row=i, col=1)
                p.showGrid(x=True, y=True, alpha=0.15)
                p.getAxis("left").setWidth(50)
                if i < n_rows - 1:
                    p.hideAxis("bottom")
                    p.setTitle(self._traces[i].name or None, size="9pt")
                    self._trace_curves.append([p.plot(pen=pg.mkPen("w", width=1))])
                else:
                    p.setLabel("bottom", "Time (s)")
                    p.setTitle("Selected pixels", size="9pt")
                    self._trace_curves.append([])  # one curve per selected pixel, added lazily
                mark = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y", style=QtCore.Qt.DashLine))
                p.addItem(mark)
                self._trace_marks.append(mark)
                # Manual zoom/pan on any trace plot pins the view (turns Follow off).
                p.vb.sigRangeChangedManually.connect(self._on_manual_range)
                self._trace_plots.append(p)
            # Traces share a time axis; linking means zooming one zooms all.
            for p in self._trace_plots[1:]:
                p.setXLink(self._trace_plots[0])

            self._aux_items = []
            for j, aux in enumerate(self._aux):
                p = self._glw.addPlot(row=0, col=2 + j, rowspan=n_rows)
                p.setAspectLocked(True)
                p.invertY(True)
                p.hideAxis("bottom")
                p.hideAxis("left")
                p.setTitle(aux.name or f"Aux {j}", size="9pt")
                item = pg.ImageItem()
                p.addItem(item)
                self._aux_items.append(item)

            self._glw.ci.layout.setColumnStretchFactor(0, 3)
            self._glw.ci.layout.setColumnStretchFactor(1, 2)

            hint = QtWidgets.QLabel(
                "p: play · up/down: rate · b: back · click: move pixel · "
                "ctrl+click: add pixel · c: clear · -/=: color scale · alt+arrows: rotate/flip"
                + ("  · r: record" if self._movie_save_path else "")
            )
            hint.setStyleSheet("color: gray;")
            hint.setWordWrap(True)
            root.addWidget(hint)

            self._image.scene().sigMouseClicked.connect(self._on_click)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            # Route keys from anywhere in the window (image, buttons, slider) to _handle_key.
            install_hotkeys(self, self._handle_key)

        # ---------------------------------------------------------------- computation

        def _frame_image(self) -> np.ndarray:
            """Reconstruct the current frame, batching a block ahead while playing.

            One frame is a GEMV; a block of frames is a GEMM, which BLAS runs far closer to peak
            (measured 0.53 ms/frame in blocks of 32 versus 7.9 ms one at a time — about 15x
            MATLAB's per-frame rate). Blocks are only worth it when frames are consumed in
            order, so scrubbing and clicking stay on the single-frame path rather than paying
            for 31 frames nobody asked for.
            """
            f = self._frame
            if (
                self._block is not None
                and self._block_start <= f < self._block_start + self._block.shape[1]
            ):
                return self._block[:, f - self._block_start].reshape(self.shape)

            if self._playing:
                start = f
                stop = min(start + _PREFETCH_BLOCK, self._n_frames)
                self._block = self._flat_u @ self._v32[:, start:stop]
                self._block_start = start
                return self._block[:, 0].reshape(self.shape)

            self._block = None
            self._block_start = -1
            return (self._flat_u @ self._v32[:, f]).reshape(self.shape)

        def _recompute_pixel_traces(self) -> None:
            self._pixel_traces = [
                self._flat_u[y * self.shape[1] + x] @ self._v32 for y, x in self._pixels
            ]

        # ---------------------------------------------------------------- interaction

        def _on_click(self, event):
            pt = self._plot.vb.mapSceneToView(event.scenePos())
            dx, dy = int(np.floor(pt.x())), int(np.floor(pt.y()))
            dh, dw = self._orient.display_shape(self.shape)
            if not (0 <= dy < dh and 0 <= dx < dw):
                return
            pixel = self._orient.to_data(dy, dx, self.shape)
            # Ctrl+click adds a pixel. The MATLAB used right-click, but in pyqtgraph right-click
            # is the view's own context menu (autoscale, export, ...), so the two fought: you got
            # a menu *and* a new point on every attempt.
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self._pixels.append(pixel)  # keep the old one, as the MATLAB does
            elif event.button() == QtCore.Qt.LeftButton:
                self._pixels[-1] = pixel
            else:
                return  # right-click: leave it to the context menu
            self._recompute_pixel_traces()
            self._refresh()

        def keyPressEvent(self, event):
            # text_entry_focused: a cutoff box that ignores a key lets it propagate here,
            # where acting on it would fire hotkeys while the user is typing a number.
            if text_entry_focused(self) or not self._handle_key(event.key(), event.modifiers()):
                super().keyPressEvent(event)

        def _handle_key(self, key, mods) -> bool:
            """Act on a hotkey. Returns True if it was consumed.

            Split out of ``keyPressEvent`` so :func:`install_hotkeys` can call it for key presses
            delivered to any child widget — otherwise the shortcuts only work while the top-level
            widget holds focus, which it loses as soon as you click the image or a button.
            """
            if mods & QtCore.Qt.AltModifier:
                if key == QtCore.Qt.Key_Right:
                    self._orient.rotate(-1)
                elif key == QtCore.Qt.Key_Left:
                    self._orient.rotate(1)
                elif key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                    self._orient.toggle_flip()
                else:
                    return False
                self._refresh(full=True)
                return True

            if key == QtCore.Qt.Key_P:
                self._play_btn.setChecked(not self._play_btn.isChecked())
            elif key == QtCore.Qt.Key_Up:
                self._rate *= 2
                self._refresh()
            elif key == QtCore.Qt.Key_Down:
                self._rate = max(1, self._rate // 2)
                self._refresh()
            elif key == QtCore.Qt.Key_B:
                self.set_frame(self._frame - _JUMP_BACK_FRAMES * self._rate)
            elif key == QtCore.Qt.Key_Home:
                self.set_frame(0)
            elif key == QtCore.Qt.Key_C:
                self._pixels = [self._pixels[-1]]
                self._recompute_pixel_traces()
                self._rebuild_pixel_curves()
                self._refresh()
            elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
                self._scale_cax(0.75)
            elif key in (QtCore.Qt.Key_Equal, QtCore.Qt.Key_Plus):
                self._scale_cax(1.25)
            elif key == QtCore.Qt.Key_R and self._movie_save_path is not None:
                self._rec_btn.setChecked(not self._rec_btn.isChecked())
            else:
                return False
            return True

        # ---------------------------------------------------------------- band-pass filter

        @staticmethod
        def _parse_cutoff(text: str, default: float) -> float:
            text = text.strip().lower()
            if text in ("", "none"):
                return default
            if text in ("inf", "+inf", "infinity"):
                return np.inf
            return float(text)

        def _apply_filter(self) -> None:
            """Band-pass V and rebuild everything derived from it."""
            try:
                hp = self._parse_cutoff(self._hp_edit.text(), 0.0)
                lp = self._parse_cutoff(self._lp_edit.text(), np.inf)
                filtered = bandpass_filt(self._v_raw, self._fs, highpass=hp, lowpass=lp)
            except ValueError as exc:
                self._filt_label.setText(f"invalid: {exc}")
                self._filt_label.setStyleSheet("color: #cc4444;")
                return

            self._v32 = np.asfortranarray(filtered, dtype=np.float32)
            self._invalidate_frames()
            self._recompute_pixel_traces()

            nyquist = self._fs / 2.0
            active_hp = hp > 0
            active_lp = np.isfinite(lp) and lp < nyquist
            if active_hp and active_lp:
                desc = f"band-pass {hp:g}–{lp:g} Hz"
            elif active_hp:
                desc = f"high-pass {hp:g} Hz"
            elif active_lp:
                desc = f"low-pass {lp:g} Hz"
            else:
                desc = "unfiltered"
            self._filt_label.setText(f"{desc} (Butterworth order 3, zero-phase)")
            self._filt_label.setStyleSheet("color: gray;")
            self._refresh()

        def _reset_filter(self) -> None:
            self._hp_edit.setText("0")
            self._lp_edit.setText("inf")
            self._apply_filter()

        def _invalidate_frames(self) -> None:
            """Drop the prefetched reconstruction block after V changes underneath it."""
            self._block = None
            self._block_start = -1

        # ---------------------------------------------------------------- follow / zoom

        def _on_manual_range(self, *_):
            """A manual zoom or pan means the user wants that view kept, so stop following."""
            if self._follow_chk.isChecked():
                self._follow_chk.setChecked(False)  # triggers _on_follow_toggled

        def _on_follow_toggled(self, on: bool) -> None:
            if on:
                self._refresh()  # snap straight back to the sliding window

        def _scale_cax(self, factor: float) -> None:
            # Stays symmetric about zero — signed dF/F must keep zero at the colormap's black.
            self._cax = [c * factor for c in self._cax]
            self._refresh()

        def _on_play_toggled(self, on: bool) -> None:
            self._playing = on
            self._play_btn.setText("pause" if on else "play")
            if on:
                self._timer.start()
            else:
                self._timer.stop()
            self._refresh()

        def _on_slider(self, value: int) -> None:
            if value != self._frame:
                self.set_frame(value)

        def _on_tick(self) -> None:
            # MATLAB restarts from the first frame when it runs past the end
            # (``if currentFrame > size(V,2); currentFrame = 1;``) rather than wrapping modulo,
            # so at rate > 1 the remainder is discarded. Note this differs from the tuning
            # viewer, which really is modulo — the two MATLAB files disagree, and so do we.
            nxt = self._frame + self._rate
            self.set_frame(0 if nxt >= self._n_frames else nxt)
            if self._recording:
                self._grab_frame()

        def set_frame(self, frame: int) -> None:
            self._frame = int(np.clip(frame, 0, self._n_frames - 1))
            self._refresh()

        # ---------------------------------------------------------------- recording

        def _set_recording(self, on: bool) -> None:
            if on and self._writer is None:
                try:
                    import imageio.v2 as imageio
                except ImportError:
                    self._recording = False
                    self._rec_btn.setChecked(False)
                    self._readout.setText("recording needs: pip install 'imageio[ffmpeg]'")
                    return
                self._writer = imageio.get_writer(str(self._movie_save_path), fps=35)
            self._recording = on
            self._rec_btn.setText("RECORDING" if on else "record")

        def _grab_frame(self) -> None:
            """Append the current widget appearance to the movie file."""
            if self._writer is None:
                return
            img = self.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB888)
            w, h = img.width(), img.height()
            ptr = img.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes())
            # Rows are padded to a 4-byte boundary; bytesPerLine may exceed 3 * width.
            arr = arr.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
            self._writer.append_data(arr.copy())

        def closeEvent(self, event):
            self._timer.stop()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            super().closeEvent(event)

        # ---------------------------------------------------------------- rendering

        def _rebuild_pixel_curves(self) -> None:
            """Match the number of curves in the bottom panel to the number of pixels."""
            plot = self._trace_plots[-1]
            curves = self._trace_curves[-1]
            while len(curves) > len(self._pixels):
                plot.removeItem(curves.pop())
            while len(curves) < len(self._pixels):
                color = _PIXEL_COLORS[len(curves) % _N_CYCLE]
                curves.append(plot.plot(pen=pg.mkPen(tuple((color * 255).astype(int)), width=2)))

        def _refresh(self, full: bool = False) -> None:
            self._image.setImage(
                self._orient.apply(self._frame_image()), autoLevels=False, levels=self._cax
            )
            self._colorbar.setLevels(tuple(self._cax))

            display = [self._orient.to_display(y, x, self.shape) for y, x in self._pixels]
            brushes = [
                pg.mkBrush(tuple((_PIXEL_COLORS[i % _N_CYCLE] * 255).astype(int)))
                for i in range(len(self._pixels))
            ]
            self._markers.setData(
                [d[1] + 0.5 for d in display], [d[0] + 0.5 for d in display], brush=brushes
            )

            now = float(self._t[self._frame])
            half = _WINDOW_SECONDS / 2.0
            lo, hi = now - half, now + half

            # While following, the plotted slice is just the visible window — cheap, and it keeps
            # the traces scrolling. Once the user has zoomed we hand over the whole trace so
            # panning around does not run off the end of the data.
            follow = self._follow_chk.isChecked()

            for i, trace in enumerate(self._traces):
                if follow:
                    sel = slice(*np.searchsorted(trace.t, [lo, hi]))
                    if trace.lims is not None:
                        self._trace_plots[i].setYRange(*trace.lims, padding=0)
                else:
                    sel = slice(None)
                self._trace_curves[i][0].setData(trace.t[sel], trace.v[sel])

            self._rebuild_pixel_curves()
            sel = slice(*np.searchsorted(self._t, [lo, hi])) if follow else slice(None)
            # strict: _rebuild_pixel_curves just synced the curves to self._pixels, and the
            # traces were computed from the same list, so a length mismatch is a real bug.
            for curve, pt in zip(self._trace_curves[-1], self._pixel_traces, strict=True):
                curve.setData(self._t[sel], pt[sel])

            if follow:
                self._trace_plots[-1].setYRange(*self._cax, padding=0)
                self._trace_plots[0].setXRange(lo, hi, padding=0)
            for mark in self._trace_marks:
                mark.setPos(now)  # the time cursor tracks the frame either way

            aux_errors = []
            for item, aux in zip(self._aux_items, self._aux, strict=True):
                try:
                    aux.render(item, now, aux.data)
                except Exception as exc:  # a broken aux renderer must not kill playback
                    aux_errors.append(f"aux {aux.name!r} failed: {exc}")

            if self._slider.value() != self._frame:
                blocked = self._slider.blockSignals(True)
                self._slider.setValue(self._frame)
                self._slider.blockSignals(blocked)

            bits = [
                f"frame {self._frame + 1}/{self._n_frames}",
                f"t = {now:.3f} s",
                f"rate x{self._rate}",
                f"scale +/-{self._cax[1]:.3g}",
                f"{len(self._pixels)} px",
            ]
            if self._recording:
                bits.append("RECORDING")
            # Aux failures go last so the status line does not silently clobber them — a broken
            # camera panel would otherwise fail invisibly at 10 refreshes a second.
            bits.extend(aux_errors)
            self._readout.setText("  |  ".join(bits))

            if full:
                self._plot.vb.autoRange()

        # ---------------------------------------------------------------- programmatic API

        def add_pixel(self, y: int, x: int) -> None:
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise IndexError(f"pixel {(y, x)} outside image of shape {self.shape}")
            self._pixels.append((int(y), int(x)))
            self._recompute_pixel_traces()
            self._refresh()

        def set_pixel(self, y: int, x: int) -> None:
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise IndexError(f"pixel {(y, x)} outside image of shape {self.shape}")
            self._pixels[-1] = (int(y), int(x))
            self._recompute_pixel_traces()
            self._refresh()

        @property
        def pixels(self):
            return list(self._pixels)

        @property
        def frame(self):
            return self._frame

        @property
        def rate(self):
            return self._rate

        @property
        def frame_image(self) -> np.ndarray:
            return self._frame_image()

    return MovieWithTracesViewer


_CLASS = None


def _get_class():
    global _CLASS
    if _CLASS is None:
        _CLASS = _build()
    return _CLASS


def __getattr__(name):
    if name == "MovieWithTracesViewer":
        return _get_class()
    raise AttributeError(name)


def movie_with_traces(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray | None = None,
    traces=None,
    movie_save_path=None,
    aux_videos=None,
    nsv_display: int | None = None,
    cax: tuple[float, float] = (-0.4, 0.4),
    block: bool = True,
):
    """Open the movie viewer. Equivalent to ``movieWithTracesSVD(U, V, t, traces, path[, auxVid])``.

    Parameters
    ----------
    u, v : the SVD movie. ``t`` defaults to frame indices if omitted.
    traces : sequence of :class:`Trace` (or dicts with ``t``/``v``/``name``/``lims``).
    movie_save_path : enables the record button. Writing needs ``pip install 'imageio[ffmpeg]'``.
    aux_videos : sequence of :class:`AuxVideo` for extra panels (eye/face cameras).
    nsv_display : cap components used per reconstructed frame. Lower it if full-rank playback
        stutters; the image gets smoother, not wrong.
    cax : initial color limits. The default assumes dF/F — run :func:`widefield.dff_from_svd`
        first, or widen this, for raw components.
    """
    app = ensure_app()
    viewer = _get_class()(
        u,
        v,
        t=t,
        traces=traces,
        movie_save_path=movie_save_path,
        aux_videos=aux_videos,
        nsv_display=nsv_display,
        cax=cax,
    )
    viewer.resize(1400, 780)
    viewer.show()
    viewer.setFocus()
    run_app(app, block)
    return viewer
