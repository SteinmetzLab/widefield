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
up / down                    double / halve playback **speed** (see below)
``b``                        jump back half a second (x speed)
left / right                 step one frame (10 with shift); pauses playback
click                        move the selected pixel
**ctrl+click**               add another pixel (each gets its own color)
``c``                        clear all but the last pixel
``-`` / ``=``                shrink / grow the color scale (stays centered on zero)
alt + left/right             rotate 90 degrees
alt + up/down                flip vertically
==========================  ============================================================

Adding a pixel is ctrl+click rather than the MATLAB's right-click: in pyqtgraph right-click
opens the view's own context menu, so the two actions fought each other.

Playback is driven by the **wall clock**, not one frame per timer tick. The MATLAB advanced a
fixed number of frames per 100 ms tick, which caps playback at 10 fps and means a big window or a
slow machine silently plays the recording in slow motion. Here the frame is computed from elapsed
time, so frames are *dropped* when rendering cannot keep up and the movie always runs at the
requested speed. ``speed`` is a float multiplier (1.0 = real time), so slow motion works too --
the MATLAB's integer frame-step could only ever skip.

The readout shows the frames actually drawn per second, so a slow render is visible rather than
mysterious. Rendering cost is dominated by blitting the upscaled image: on a 2540x1360 window a
560x560 movie draws at ~8 fps, a half-size window at ~22 fps, and ``use_opengl=True`` buys about
a third on top.

Additions over the MATLAB: a scrub slider and frame/time readout, ``home`` to jump to the start,
a **temporal band-pass** (type cutoffs in Hz; filtering ``V`` filters every pixel, so it is cheap
to re-apply), a **Follow** toggle so a manual zoom on the trace plots survives playback instead
of being reset every frame, and ``nsv_display`` to cap the components used per frame when
full-rank playback can't keep up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from widefield.colormaps import blueblackred, to_pyqtgraph
from widefield.gui._common import (
    Orientation,
    ensure_app,
    install_hotkeys,
    make_bandpass_control,
    require_qt,
    run_app,
    text_entry_focused,
    window_title,
)
from widefield.svd import flatten_u

log = logging.getLogger(__name__)

__all__ = ["Trace", "AuxVideo", "movie_with_traces"]

_WINDOW_SECONDS = 10.0  # trace window width; MATLAB's windowSize
# Poll rate, not frame rate. The MATLAB used a 100 ms timer and advanced one frame per tick,
# which caps playback at 10 fps however fast the machine is. We poll often and let the wall
# clock decide which frame to show, so 60 Hz is just 'check frequently enough'.
_TIMER_MS = 16
_JUMP_BACK_SECONDS = 0.5  # MATLAB's 'b' jumped 20 frames; express it as time instead
# Frames reconstructed per batch during playback. At 512x512x2000 a block costs
# nPix * block * 4 bytes (~33 MB at 32 frames) and turns a GEMV into a GEMM.
_PREFETCH_BLOCK = 32
# The trace panels are redrawn at most this often while playing. Their repaint costs about
# as much as the movie's (a large pyqtgraph scene is a large CPU blit), and a scrolling
# 10 s window conveys nothing extra above ~12 Hz -- so decoupling them roughly doubles the
# movie's frame rate. Paused, they always update.
_TRACE_HZ = 12.0

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


def _thin(curve) -> None:
    """Let pyqtgraph draw a curve at screen resolution instead of point-by-point.

    A 10 s window of a 2 kHz Timeline channel is ~2000 points drawn into ~1000 px, and
    stroking that vector path dominated the frame time: on a 2540x1360 window the trace
    panels cost ~60 ms/frame, more than the movie itself. Peak-mode downsampling keeps the
    visible extremes (so brief spikes still show) while cutting the point count to what the
    panel can actually resolve; clipToView drops everything outside the visible x-range.
    """
    curve.setDownsampling(auto=True, method="peak")
    curve.setClipToView(True)


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
            use_opengl=False,
            session=None,
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
            # Left in the input dtype (float32 off disk) rather than promoted: bandpass_filt
            # promotes internally anyway, and forcing float64 here doubled the resident size to
            # 311 MB on a 92-minute session for no change in the result.
            self._v_raw = np.asarray(self._v[:nsv])
            self._v32 = np.asfortranarray(self._v[:nsv], dtype=np.float32)
            self._block_start = -1
            self._block: np.ndarray | None = None
            self._last_trace_draw = 0.0
            self._fs = float(1.0 / np.median(np.diff(self._t))) if self._t.size > 1 else 1.0

            self._traces = _as_traces(traces)
            self._aux = list(aux_videos or [])
            self._orient = Orientation()
            self._frame = 0
            self._speed = 1.0  # playback speed multiplier; 1.0 is real time
            self._playing = False
            self._play_t0 = 0.0
            self._play_frame0 = 0
            self._fps_t0 = 0.0
            self._fps_count = 0
            self._cax = [float(cax[0]), float(cax[1])]
            self._pixels: list[tuple[int, int]] = [(self.shape[0] // 2, self.shape[1] // 2)]
            self._pixel_traces: list[np.ndarray] = []

            self._writer = None
            self._movie_save_path = movie_save_path
            self._recording = False
            self._use_opengl = bool(use_opengl)

            self.setWindowTitle(window_title("Movie with traces (SVD)", session))
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
            # Monospace and fixed-width fields: the readout updates every frame, and
            # with a proportional font a change in one field (frame number growing a
            # digit, "17 fps" appearing) shifted all the others sideways and made the
            # whole row unreadable while playing.
            mono = QtGui.QFont("Consolas")
            mono.setStyleHint(QtGui.QFont.Monospace)
            mono.setPointSize(9)
            self._readout.setFont(mono)
            self._readout.setMinimumWidth(560)
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

            self._fast_chk = QtWidgets.QCheckBox("Fast (GPU)")
            self._fast_chk.setToolTip(
                "Draw the movie as a GPU texture instead of through pyqtgraph.\n"
                "Much faster on a large window (~60 fps vs ~14), but no pan/zoom,\n"
                "colorbar or axes while it is on. Toggle with g."
            )
            self._fast_chk.toggled.connect(self._on_fast_toggled)
            controls.addWidget(self._fast_chk)

            if self._movie_save_path is not None:
                self._rec_btn = QtWidgets.QPushButton("record")
                self._rec_btn.setCheckable(True)
                self._rec_btn.setFixedWidth(80)
                self._rec_btn.toggled.connect(self._set_recording)
                controls.addWidget(self._rec_btn)
            root.addLayout(controls)

            self.bandpass = make_bandpass_control(self._v_raw, self._fs, self._on_filtered)
            root.addWidget(self.bandpass)

            body = QtWidgets.QHBoxLayout()
            root.addLayout(body, stretch=1)

            # --- image area: a stack of the normal pyqtgraph view and a GPU-texture view ---
            self._img_stack = QtWidgets.QStackedWidget()
            body.addWidget(self._img_stack, stretch=3)

            self._img_glw = pg.GraphicsLayoutWidget()
            if self._use_opengl:
                # Note this is NOT a texture path: pyqtgraph's useOpenGL only retargets QPainter
                # at a GL-backed surface, so every item is still rasterized on the CPU. Measured
                # ~20% (88 -> 72 ms/frame) on a 2540x1360 window. The real fix is the fast view.
                try:
                    self._img_glw.useOpenGL(True)
                except Exception:  # pragma: no cover - depends on the machine
                    log.debug("OpenGL unavailable; using the raster path", exc_info=True)
            self._img_stack.addWidget(self._img_glw)

            self._plot = self._img_glw.addPlot(row=0, col=0)
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

            self._fast_view = self._make_fast_view(pg, QtCore, QtWidgets)
            self._img_stack.addWidget(self._fast_view if self._fast_view else QtWidgets.QWidget())

            self._glw = pg.GraphicsLayoutWidget()
            if self._use_opengl:
                # Worth as much here as on the image: the trace panels are their own large
                # scene, and on a 2540x1360 window moving them to GL took playback from
                # 9.9 to 16.3 fps.
                try:
                    self._glw.useOpenGL(True)
                except Exception:  # pragma: no cover - depends on the machine
                    log.debug("OpenGL unavailable for the trace panels", exc_info=True)
            body.addWidget(self._glw, stretch=2)

            n_rows = len(self._traces) + 1
            self._trace_plots = []
            self._trace_curves = []
            self._trace_marks = []
            for i in range(n_rows):
                p = self._glw.addPlot(row=i, col=0)
                p.showGrid(x=True, y=True, alpha=0.15)
                p.getAxis("left").setWidth(50)
                if i < n_rows - 1:
                    p.hideAxis("bottom")
                    p.setTitle(self._traces[i].name or None, size="9pt")
                    curve = p.plot(pen=pg.mkPen("w", width=1))
                    _thin(curve)
                    self._trace_curves.append([curve])
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
                p = self._glw.addPlot(row=0, col=1 + j, rowspan=n_rows)
                p.setAspectLocked(True)
                p.invertY(True)
                p.hideAxis("bottom")
                p.hideAxis("left")
                p.setTitle(aux.name or f"Aux {j}", size="9pt")
                item = pg.ImageItem()
                p.addItem(item)
                self._aux_items.append(item)

            hint = QtWidgets.QLabel(
                "p: play · up/down: speed · left/right: step frame · b: back · "
                "click: move pixel · "
                "ctrl+click: add pixel · c: clear · -/=: color scale · alt+arrows: rotate/flip"
                " · g: fast GPU view" + ("  · r: record" if self._movie_save_path else "")
            )
            hint.setStyleSheet("color: gray;")
            hint.setWordWrap(True)
            root.addWidget(hint)

            self._image.scene().sigMouseClicked.connect(self._on_click)
            if self._fast_view is not None:
                self._fast_image.mousePressEvent = self._fast_mouse_press
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            # Route keys from anywhere in the window (image, buttons, slider) to _handle_key.
            install_hotkeys(self, self._handle_key)

        # ---------------------------------------------------------------- fast (GPU) view

        def _make_fast_view(self, pg, QtCore, QtWidgets):
            """A GPU-texture image view, or None if pyqtgraph/OpenGL cannot provide one.

            pyqtgraph's ImageItem does the colormap lookup *and* the upscale on the CPU; measured
            on a 2540x1360 window that is 70 ms/frame against 34 ms for an empty view. A
            RawImageGLWidget uploads the frame as a texture and lets the GPU sample it, which
            measured 16.6 ms - i.e. vsync-locked 60 fps. That is the same trick MATLAB's renderer
            uses when you replace CData.

            The cost is that it is a bare image: no ViewBox, so no pan/zoom, no axes, no colorbar.
            Markers are stamped into the pixels instead (see :meth:`_fast_rgba`).
            """
            try:
                from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
            except ImportError:  # pragma: no cover - depends on the install
                log.debug("RawImageGLWidget unavailable", exc_info=True)
                return None

            try:
                view = RawImageGLWidget(smooth=False)
            except Exception:  # pragma: no cover - depends on drivers
                log.debug("could not create a GL image widget", exc_info=True)
                return None

            # RawImageGLWidget blits the texture across its whole rect, so a square movie in a
            # tall panel would be stretched. Keep it aspect-correct by sizing it ourselves.
            class _AspectBox(QtWidgets.QWidget):
                def __init__(self, child, aspect):
                    super().__init__()
                    self._child = child
                    self._aspect = aspect
                    child.setParent(self)

                def resizeEvent(self, event):
                    w, h = self.width(), self.height()
                    cw, ch = (
                        (int(h * self._aspect), h)
                        if w / max(h, 1) > self._aspect
                        else (
                            w,
                            int(w / self._aspect),
                        )
                    )
                    self._child.setGeometry((w - cw) // 2, (h - ch) // 2, cw, ch)
                    super().resizeEvent(event)

            ypix, xpix = self.shape
            box = _AspectBox(view, xpix / ypix)
            box.setStyleSheet("background: black;")
            self._fast_image = view
            # 256 x 4 uint8 colormap, applied by us so we can stamp markers before upload.
            table = to_pyqtgraph(blueblackred()).getLookupTable(nPts=256, alpha=True)
            self._fast_lut = np.ascontiguousarray(table, dtype=np.uint8)
            self._fast_idx = None
            return box

        def _on_fast_toggled(self, on: bool) -> None:
            if on and self._fast_view is None:
                self._fast_chk.setChecked(False)
                self._readout.setText("GPU view unavailable (needs OpenGL)")
                return
            self._img_stack.setCurrentIndex(1 if on else 0)
            self._refresh()

        @property
        def fast_mode(self) -> bool:
            return bool(self._fast_chk.isChecked())

        def _fast_rgba(self, oriented: np.ndarray) -> np.ndarray:
            """Colormap the frame ourselves and stamp the pixel markers into it.

            Doing the lookup here rather than handing float32 + levels to the widget costs the
            same (~4 ms) but lets the markers survive, which the GL view could not otherwise
            overlay.
            """
            lo, hi = self._cax
            scale = 255.0 / (hi - lo) if hi > lo else 0.0
            if self._fast_idx is None or self._fast_idx.shape != oriented.shape:
                self._fast_idx = np.empty(oriented.shape, dtype=np.uint8)
            tmp = (oriented - lo) * scale
            np.clip(tmp, 0, 255, out=tmp)
            np.copyto(self._fast_idx, tmp, casting="unsafe")
            rgba = self._fast_lut[self._fast_idx]

            # Bounds come from the buffer itself, not from self.shape: they normally agree, but
            # clipping against the array we are actually writing to cannot go out of range.
            dh, dw = rgba.shape[0], rgba.shape[1]
            for i, (y, x) in enumerate(self._pixels):
                dy, dx = self._orient.to_display(y, x, self.shape)
                if not (0 <= dy < dh and 0 <= dx < dw):
                    continue
                color = np.append((_PIXEL_COLORS[i % _N_CYCLE] * 255).astype(np.uint8), 255)
                for oy in range(-4, 5):  # a small cross, clipped at the edges
                    yy = dy + oy
                    if 0 <= yy < dh:
                        rgba[yy, dx] = color
                for ox in range(-4, 5):
                    xx = dx + ox
                    if 0 <= xx < dw:
                        rgba[dy, xx] = color
            return rgba

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

        def _fast_mouse_press(self, event):
            """Map a click on the GL view to a data pixel (it fills its widget exactly)."""
            w, h = self._fast_image.width(), self._fast_image.height()
            if w <= 0 or h <= 0:
                return
            dh, dw = self._orient.display_shape(self.shape)
            dx = int(event.position().x() / w * dw)
            dy = int(event.position().y() / h * dh)
            if not (0 <= dy < dh and 0 <= dx < dw):
                return
            pixel = self._orient.to_data(dy, dx, self.shape)
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self._pixels.append(pixel)
            elif event.button() == QtCore.Qt.LeftButton:
                self._pixels[-1] = pixel
            else:
                return
            self._recompute_pixel_traces()
            self._refresh()

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
                self.set_speed(self._speed * 2)
            elif key == QtCore.Qt.Key_Down:
                self.set_speed(self._speed / 2)
            elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                # Step frames. Stepping implies you want to look at something, so it
                # pauses rather than fighting playback for the next tick.
                step = 10 if mods & QtCore.Qt.ShiftModifier else 1
                if key == QtCore.Qt.Key_Left:
                    step = -step
                if self._playing:
                    self._play_btn.setChecked(False)
                self.set_frame(self._frame + step)
            elif key == QtCore.Qt.Key_B:
                back = max(1, int(round(_JUMP_BACK_SECONDS * self._fs * self._speed)))
                self.set_frame(self._frame - back)
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
            elif key == QtCore.Qt.Key_G:
                self._fast_chk.setChecked(not self._fast_chk.isChecked())
            elif key == QtCore.Qt.Key_R and self._movie_save_path is not None:
                self._rec_btn.setChecked(not self._rec_btn.isChecked())
            else:
                return False
            return True

        # ---------------------------------------------------------------- band-pass filter

        def _on_filtered(self, filtered_v, _description) -> None:
            """Swap in band-passed components and rebuild everything derived from them."""
            self._v32 = np.asfortranarray(filtered_v, dtype=np.float32)
            self._invalidate_frames()
            self._recompute_pixel_traces()
            self._refresh()

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
                self._anchor_playback()
                self._timer.start()
            else:
                self._timer.stop()
            self._refresh()

        def _on_slider(self, value: int) -> None:
            if value != self._frame:
                self.set_frame(value)

        def _anchor_playback(self) -> None:
            """Peg wall-clock time to the current frame; playback is measured from here."""
            self._play_t0 = time.perf_counter()
            self._play_frame0 = self._frame
            self._fps_t0 = self._play_t0
            self._fps_count = 0

        def _on_tick(self) -> None:
            """Advance to whichever frame the wall clock says we should be showing.

            Deliberately *not* "advance N frames per tick", which is what the MATLAB does. That
            couples playback speed to how fast frames can be drawn, so a big window or a slow
            machine plays the recording in slow motion — on a 2540x1360 window a 560x560 movie
            renders at ~12 fps, which for a 35 Hz recording is 0.35x real time and reads as
            "broken" rather than "slow".

            Computing the target frame from elapsed time instead means frames are *dropped* when
            rendering cannot keep up, and the movie always plays at the requested speed. This is
            what every video player does.
            """
            elapsed = time.perf_counter() - self._play_t0
            target = self._play_frame0 + elapsed * self._fs * self._speed
            if target >= self._n_frames:
                # MATLAB restarts at the first frame rather than wrapping modulo; keep that, and
                # re-peg the clock so the next lap is timed from the restart.
                self._frame = 0
                self._anchor_playback()
            else:
                self._frame = int(target)
            self._refresh()

            self._fps_count += 1
            if self._recording:
                self._grab_frame()

        def _achieved_fps(self) -> float | None:
            """Frames actually drawn per second, over the last stretch of playback."""
            if not self._playing or self._fps_count < 3:
                return None
            dt = time.perf_counter() - self._fps_t0
            if dt < 0.4:  # too short a sample to be meaningful
                return None
            fps = self._fps_count / dt
            if dt > 2.0:  # restart the window so the number tracks rather than averaging forever
                self._fps_t0 = time.perf_counter()
                self._fps_count = 0
            return fps

        def set_frame(self, frame: int) -> None:
            self._frame = int(np.clip(frame, 0, self._n_frames - 1))
            if self._playing:
                self._anchor_playback()  # a manual jump re-pegs the clock
            self._refresh()

        def set_speed(self, speed: float) -> None:
            """Playback speed multiplier. 1.0 is real time; clamped to a sane range."""
            self._speed = float(np.clip(speed, 1 / 32, 64))
            if self._playing:
                self._anchor_playback()
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
                curve = plot.plot(pen=pg.mkPen(tuple((color * 255).astype(int)), width=2))
                _thin(curve)
                curves.append(curve)

        def _refresh(self, full: bool = False) -> None:
            oriented = self._orient.apply(self._frame_image())
            if self.fast_mode and self._fast_view is not None:
                self._fast_image.setImage(self._fast_rgba(oriented))
            else:
                self._image.setImage(oriented, autoLevels=False, levels=self._cax)
                self._colorbar.setLevels(tuple(self._cax))

            if not self.fast_mode:
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

            # Redrawing the trace panels costs about as much as the movie itself, and they do
            # not need the movie's frame rate. Skip them on most ticks while playing.
            now_wall = time.perf_counter()
            draw_traces = not self._playing or (now_wall - self._last_trace_draw) >= 1.0 / _TRACE_HZ
            if draw_traces:
                self._last_trace_draw = now_wall

            if draw_traces:
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
                # strict: _rebuild_pixel_curves just synced the curves to self._pixels, and
                # the traces came from the same list, so a mismatch is a real bug.
                for curve, pt in zip(self._trace_curves[-1], self._pixel_traces, strict=True):
                    curve.setData(self._t[sel], pt[sel])

                if follow:
                    self._trace_plots[-1].setYRange(*self._cax, padding=0)
                    self._trace_plots[0].setXRange(lo, hi, padding=0)
                for mark in self._trace_marks:
                    mark.setPos(now)  # the time cursor tracks the frame

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

            # Every field is padded to a constant width so nothing shifts as values
            # change; the fps slot shows dashes rather than vanishing when unmeasured.
            width = len(str(self._n_frames))
            achieved = self._achieved_fps()
            fps_text = f"{achieved:4.0f}" if achieved is not None else "  --"
            bits = [
                f"frame {self._frame + 1:>{width}}/{self._n_frames}",
                f"t {now:9.3f} s",
                f"speed {self._speed:>6g}x",
                f"scale +/-{self._cax[1]:<8.3g}",
                f"{len(self._pixels):>3} px",
                # Surfaced so slow rendering is visible rather than mysterious: if this
                # sits well below fs * speed, frames are being dropped to hold speed.
                f"{fps_text} fps drawn",
            ]
            if self._recording:
                bits.append("RECORDING")
            # Aux failures go last so the status line does not silently clobber them — a broken
            # camera panel would otherwise fail invisibly at 10 refreshes a second.
            bits.extend(aux_errors)
            self._readout.setText("  |  ".join(bits))

            if full and not self.fast_mode:
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
        def speed(self) -> float:
            """Playback speed multiplier (1.0 = real time)."""
            return self._speed

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
    use_opengl: bool = False,
    session=None,
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
    use_opengl : render through OpenGL. Roughly 33% faster on a large window, at the cost of
        needing working drivers; falls back to the raster path if unavailable.
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
        use_opengl=use_opengl,
        session=session,
    )
    viewer.resize(1400, 780)
    viewer.show()
    viewer.setFocus()
    run_app(app, block)
    return viewer
