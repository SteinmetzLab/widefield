"""The numerics must stay importable and usable with no Qt installed.

This is a real deployment constraint, not hygiene: batch preprocessing runs on headless machines
where PySide6 is not installed, and DataBrowser's CI has an equivalent `core-portable` job. The
check works by making Qt imports fail, so it is meaningful even on a machine that has Qt.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest

BLOCKED = ("PySide6", "pyqtgraph", "PyQt5", "PyQt6", "matplotlib")

CORE_MODULES = [
    "widefield",
    "widefield.svd",
    "widefield.events",
    "widefield.correlation",
    "widefield.hemo",
    "widefield.compress",
    "widefield.io",
    "widefield.signals",
    "widefield.colormaps",
    "widefield.utils",
]


@pytest.fixture
def no_qt(monkeypatch):
    """Make any Qt/matplotlib import raise, and drop the modules from the cache."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"{root} is blocked by the portability test")
        return real_import(name, *args, **kwargs)

    # Only the widefield modules are evicted, so they genuinely re-import under the guard.
    # Qt itself is left in sys.modules on purpose: pytest-qt holds live references to it and
    # processes events in its setup hook, so removing it segfaults the interpreter. The guard
    # runs ahead of the module cache, so a blocked import still fails even though it is loaded.
    for mod in list(sys.modules):
        if mod.startswith("widefield"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded)
    yield


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports_without_qt(no_qt, module):
    importlib.import_module(module)


def test_numerics_actually_run_without_qt(no_qt):
    """Importing is not enough — the math must work too."""
    wf = importlib.import_module("widefield")
    rng = np.random.default_rng(0)
    u = rng.standard_normal((6, 5, 4))
    v = rng.standard_normal((4, 200))
    t = np.arange(200) / 35.0

    frame = wf.svd_frame_reconstruct(u, v[:, 3])
    assert frame.shape == (6, 5)

    corr = wf.SeedCorrelation(u, v).map((2, 2))
    assert corr.shape == (6, 5)

    avg = wf.event_locked_avg_svd(v, t, np.array([1.0, 2.0]), np.array([0, 1]), (-0.1, 0.2))
    assert avg.avg_v.shape[0] == 2

    r = wf.hemo_correct_nonlocal(v, v * 0.5 + rng.standard_normal(v.shape) * 0.1)
    assert r.v_corrected.shape == v.shape


def test_colormaps_available_without_matplotlib(no_qt):
    cm = importlib.import_module("widefield.colormaps")
    assert cm.blueblackred().shape == (101, 3)
    assert cm.to_lookup_table(cm.blueblackred()).shape == (256, 3)


def test_gui_package_imports_without_qt(no_qt):
    """`import widefield.gui` must also be safe — Qt is only touched when a viewer is opened."""
    importlib.import_module("widefield.gui")


def test_opening_a_viewer_without_qt_gives_an_actionable_error(no_qt):
    gui = importlib.import_module("widefield.gui")
    with pytest.raises(ImportError, match=r"widefield\[gui\]"):
        gui.pixel_correlation_viewer(np.zeros((4, 4, 2)), np.zeros((2, 10)))
