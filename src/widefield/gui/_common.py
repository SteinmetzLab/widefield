"""Shared building blocks for the viewers.

The one genuinely fiddly piece is :class:`Orientation`. The MATLAB viewers let you rotate and
flip the brain image with alt+arrows (mice are not mounted consistently), and they *also* remap
the plain arrow keys so that "right" always means right *on screen* regardless of rotation. The
MATLAB does this by circular-shifting a key list against the axes' ``View`` property.

Rather than replicate that shift, we keep the cursor in data coordinates and convert to and
from display coordinates around it. Moving in screen space is then screen-correct by
construction, and the mapping is unit-testable without a running GUI — which the MATLAB version
is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Orientation",
    "polygon_mask",
    "require_qt",
    "ensure_app",
    "run_app",
    "clamp",
    "install_hotkeys",
    "text_entry_focused",
]


def polygon_mask(vertices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of the pixels inside a polygon. Equivalent to MATLAB's ``roipoly``.

    ``vertices`` is ``(N, 2)`` as ``(x, y)`` in pixel coordinates; ``shape`` is ``(Ypix, Xpix)``.
    A pixel is inside if its *center* is, by the even-odd rule.

    Implemented with a vectorized ray cast rather than via matplotlib or scikit-image: it keeps
    the ROI feature inside the base install, and it is fast enough (one pass per edge over the
    whole image — a few ms at 512x512).
    """
    vertices = np.asarray(vertices, dtype=float)
    ypix, xpix = int(shape[0]), int(shape[1])
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(f"vertices must be (N, 2) as (x, y); got shape {vertices.shape}")
    if vertices.shape[0] < 3:
        return np.zeros((ypix, xpix), dtype=bool)

    yy, xx = np.mgrid[0:ypix, 0:xpix]
    px = xx + 0.5
    py = yy + 0.5

    inside = np.zeros((ypix, xpix), dtype=bool)
    x0, y0 = vertices[:, 0], vertices[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1, strict=True):
        if ay == by:  # horizontal edges never count as crossings under the even-odd rule
            continue
        straddles = (py >= min(ay, by)) & (py < max(ay, by))
        # x of the edge at this row's height
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (px < x_at)
    return inside


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


@dataclass
class Orientation:
    """Rotation/flip state of a displayed image, and the coordinate maps that go with it.

    ``rot`` counts 90-degree counter-clockwise steps (0-3), matching ``np.rot90``. ``flip``
    mirrors vertically *after* rotation, which is how MATLAB's ``View(2) < 0`` behaves.

    Data coordinates are always ``(row, col)`` into the original ``(Ypix, Xpix)`` image;
    display coordinates index the array actually shown.
    """

    rot: int = 0
    flip: bool = False

    def rotate(self, steps: int) -> None:
        self.rot = (self.rot + steps) % 4

    def toggle_flip(self) -> None:
        self.flip = not self.flip

    def reset(self) -> None:
        self.rot, self.flip = 0, False

    # -- array ---------------------------------------------------------------------

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return ``image`` rotated/flipped for display. Operates on the first two axes."""
        out = np.rot90(image, self.rot, axes=(0, 1)) if self.rot else image
        if self.flip:
            out = out[::-1]
        return out

    def unapply(self, displayed: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`apply` — take a display-space array back to data orientation.

        Used for ROI masks, which are naturally drawn in display space but must be indexed
        against the data.
        """
        out = displayed[::-1] if self.flip else displayed
        return np.rot90(out, -self.rot, axes=(0, 1)) if self.rot else out

    def display_shape(self, shape: tuple[int, int]) -> tuple[int, int]:
        ypix, xpix = int(shape[0]), int(shape[1])
        return (xpix, ypix) if self.rot % 2 else (ypix, xpix)

    # -- coordinates ---------------------------------------------------------------

    def to_display(self, y: int, x: int, shape: tuple[int, int]) -> tuple[int, int]:
        """Map a data pixel to its display position."""
        h, w = int(shape[0]), int(shape[1])
        for _ in range(self.rot):
            # np.rot90 once: display[i, j] = data[j, w - 1 - i]  =>  (y, x) -> (w - 1 - x, y)
            y, x, h, w = w - 1 - x, y, w, h
        if self.flip:
            y = h - 1 - y
        return y, x

    def to_data(self, dy: int, dx: int, shape: tuple[int, int]) -> tuple[int, int]:
        """Map a display position back to a data pixel. Inverse of :meth:`to_display`."""
        dh, dw = self.display_shape(shape)
        if self.flip:
            dy = dh - 1 - dy
        for _ in range(self.rot):
            # invert (y, x) -> (w - 1 - x, y), i.e. (dy, dx) -> (dx, dh - 1 - dy)
            dy, dx, dh, dw = dx, dh - 1 - dy, dw, dh
        return dy, dx

    def step_on_screen(
        self, y: int, x: int, shape: tuple[int, int], d_row: int, d_col: int
    ) -> tuple[int, int]:
        """Move a data pixel by a *screen-space* offset, clamped to the image.

        ``d_row``/``d_col`` are in display space (``d_row = +1`` is one pixel down the screen),
        so arrow keys behave the same whatever the rotation — the MATLAB's key-remapping
        behavior, without the key remapping.
        """
        dh, dw = self.display_shape(shape)
        dy, dx = self.to_display(y, x, shape)
        dy = clamp(dy + d_row, 0, dh - 1)
        dx = clamp(dx + d_col, 0, dw - 1)
        return self.to_data(dy, dx, shape)


# --------------------------------------------------------------------------- Qt helpers


def require_qt():
    """Import the Qt/pyqtgraph stack, with an actionable message if the extra is missing."""
    try:
        import pyqtgraph as pg
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "The widefield viewers need the GUI extra. Install it with:\n"
            "    pip install 'widefield[gui]'"
        ) from exc
    # numpy's (row, col) is (y, x); without this pyqtgraph transposes every image.
    pg.setConfigOptions(imageAxisOrder="row-major")
    return pg, QtCore, QtGui, QtWidgets


def text_entry_focused(widget) -> bool:
    """True if a text-entry descendant of ``widget`` currently holds focus.

    Both key-delivery paths need this guard. :func:`install_hotkeys` consults it, but a text
    entry that *ignores* a key (a digit box handed ``p``, or any key at all while the window is
    not activated) lets Qt propagate the event up to the container and call its ``keyPressEvent``
    directly — so the container's own handler has to check as well, or typing in a cutoff box
    fires the hotkeys anyway.

    Uses ``widget.focusWidget()`` rather than ``QApplication.focusWidget()``: the latter is
    ``None`` whenever the window is not activated.
    """
    _pg, _QtCore, _QtGui, QtWidgets = require_qt()
    focused = widget.focusWidget()
    return isinstance(
        focused, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QTextEdit)
    )


def install_hotkeys(widget, handler):
    """Route key presses anywhere inside ``widget`` to ``handler(key, modifiers) -> bool``.

    Without this, hotkeys only work while the top-level widget happens to hold focus. Clicking
    the image gives focus to the pyqtgraph view, clicking Play gives it to the button, and both
    swallow keys before they reach the container — so ``-``/``=`` would appear to work only right
    after touching a button, which is exactly the bug this fixes.

    One filter instance is installed on ``widget`` and on each descendant widget that exists at
    call time. It is deliberately *not* installed on the ``QApplication``: an app-wide filter is
    invoked for every event delivered to any object, and each invocation crosses the C++/Python
    boundary. With several viewers open that turned widget construction from ~0.1 s into ~3.5 s,
    because every filter saw every other widget's paint and layout events.

    Text-entry widgets are skipped, or typing "0.5" into a cutoff box would trigger whatever
    ``-`` and ``=`` are bound to.

    Returns the filter object; it is parented to ``widget``, so it lives exactly as long as the
    viewer and needs no manual cleanup.
    """
    _pg, QtCore, _QtGui, QtWidgets = require_qt()

    text_entry = (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QTextEdit)

    class _HotkeyFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() != QtCore.QEvent.Type.KeyPress:
                return False
            if isinstance(obj, text_entry) or text_entry_focused(widget):
                return False
            if handler(event.key(), event.modifiers()):
                event.accept()
                return True  # consumed: don't let a slider also act on the arrow key
            return False

    filt = _HotkeyFilter(widget)
    widget.installEventFilter(filt)
    for child in widget.findChildren(QtWidgets.QWidget):
        child.installEventFilter(filt)
    return filt


def ensure_app():
    """Return the running ``QApplication``, creating one if needed.

    Reuses an existing instance so a viewer opened from an IPython session with the Qt event
    loop already integrated does not create a second application (which would crash).
    """
    _, _, _, QtWidgets = require_qt()
    app = QtWidgets.QApplication.instance()
    return app or QtWidgets.QApplication([])


def run_app(app, block: bool) -> None:
    """Start the event loop, unless we're inside an interactive session that already runs one.

    In IPython with ``%gui qt`` the loop is already spinning and calling ``exec()`` would hang
    the prompt, so ``block=True`` from a script blocks and from a live kernel does not.
    """
    if not block:
        return
    try:  # already-integrated event loop?
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None and getattr(ip, "active_eventloop", None):
            return
    except ImportError:
        pass
    app.exec()
