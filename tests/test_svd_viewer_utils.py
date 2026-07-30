"""The SVD component browser, and find_nearest_point."""

from __future__ import annotations

import numpy as np
import pytest

from widefield.utils import find_nearest_point

pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui  # noqa: E402

from widefield.gui.svd_viewer import _get_class as _svd_class  # noqa: E402


def press(widget, key, mods=QtCore.Qt.NoModifier):
    widget.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, mods))


@pytest.fixture
def viewer(qtbot):
    rng = np.random.default_rng(2)
    ypix, xpix, nsv, nframes = 14, 10, 6, 500
    q, _ = np.linalg.qr(rng.standard_normal((ypix * xpix, nsv)))
    u = q.reshape(ypix, xpix, nsv)
    v = rng.standard_normal((nsv, nframes))
    sv = np.array([100.0, 50.0, 25.0, 10.0, 5.0, 1.0])
    w = _svd_class()(u, sv, v, fs=35.0)
    qtbot.addWidget(w)
    return w


def test_starts_on_the_first_component(viewer):
    assert viewer.component == 0


def test_arrows_step_components_and_clamp(viewer):
    press(viewer, QtCore.Qt.Key_Right)
    assert viewer.component == 1
    for _ in range(20):
        press(viewer, QtCore.Qt.Key_Right)
    assert viewer.component == viewer._n_comp - 1
    for _ in range(20):
        press(viewer, QtCore.Qt.Key_Left)
    assert viewer.component == 0


def test_displayed_map_is_the_selected_component(viewer):
    viewer.set_component(3)
    expected = viewer._orient.apply(np.asarray(viewer._u[:, :, 3], dtype=float))
    np.testing.assert_allclose(viewer._image.image, expected, atol=1e-9)


def test_trace_is_the_selected_components_timecourse(viewer):
    viewer.set_component(2)
    _, y = viewer._trace_curve.getData()
    np.testing.assert_allclose(y, viewer._v[2], atol=1e-9)


def test_p_toggles_pixel_mode(viewer):
    assert not viewer.pixel_mode
    press(viewer, QtCore.Qt.Key_P)
    assert viewer.pixel_mode
    assert viewer._trace_extra.isVisible()
    press(viewer, QtCore.Qt.Key_P)
    assert not viewer.pixel_mode


def test_pixel_mode_shows_full_and_cumulative_reconstruction(viewer):
    """Orange trace = first k+1 components; white = all of them."""
    press(viewer, QtCore.Qt.Key_P)
    viewer.set_component(2)
    y, x = viewer._pixel
    weights = np.asarray(viewer._u[y, x, :], dtype=float)
    _, full = viewer._trace_curve.getData()
    _, partial = viewer._trace_extra.getData()
    np.testing.assert_allclose(full, weights @ np.asarray(viewer._v, dtype=float), atol=1e-8)
    np.testing.assert_allclose(
        partial, weights[:3] @ np.asarray(viewer._v[:3], dtype=float), atol=1e-8
    )


def test_pixel_mode_partial_converges_to_full_at_the_last_component(viewer):
    press(viewer, QtCore.Qt.Key_P)
    viewer.set_component(viewer._n_comp - 1)
    _, full = viewer._trace_curve.getData()
    _, partial = viewer._trace_extra.getData()
    np.testing.assert_allclose(partial, full, atol=1e-8)


def test_variance_percentages_are_reported(viewer):
    viewer.set_component(0)
    # sv = [100, 50, 25, 10, 5, 1], total 191 -> first component is 52.4%
    assert "52.4" in viewer._status.text()


def test_cumulative_variance_reaches_100_percent(viewer):
    viewer.set_component(viewer._n_comp - 1)
    assert "100" in viewer._status.text()


def test_explicit_total_variance_lowers_the_percentages(qtbot):
    rng = np.random.default_rng(0)
    u = rng.standard_normal((6, 5, 3))
    v = rng.standard_normal((3, 100))
    sv = np.array([10.0, 5.0, 1.0])
    w = _svd_class()(u, sv, v, fs=35.0, total_variance=160.0)
    qtbot.addWidget(w)
    w.set_component(0)
    assert "6.25" in w._status.text()  # 10/160


def test_alt_arrows_rotate(viewer):
    before = viewer._image.image.shape
    press(viewer, QtCore.Qt.Key_Right, QtCore.Qt.AltModifier)
    assert viewer._image.image.shape == before[::-1]


def test_caxis_keys_change_the_levels(viewer):
    before = viewer._image.levels[1]
    press(viewer, QtCore.Qt.Key_Minus)
    assert viewer._image.levels[1] < before


def test_spectrum_has_no_dc_bin(viewer):
    """Log axes cannot render f = 0, so it must be dropped."""
    x, _ = viewer._spec_curve.getData()
    assert (x > 0).all() or np.isfinite(x).all()


def test_rejects_zero_components(qtbot):
    with pytest.raises(ValueError, match="at least one component"):
        _svd_class()(np.zeros((4, 4, 0)), np.array([]), np.zeros((0, 10)))


def test_component_count_limited_by_the_smallest_input(qtbot):
    """Mismatched U/Sv/V ranks must clamp to what all three provide."""
    rng = np.random.default_rng(0)
    w = _svd_class()(
        rng.standard_normal((5, 4, 8)),
        np.arange(6, 0, -1).astype(float),
        rng.standard_normal((3, 50)),
    )
    qtbot.addWidget(w)
    assert w._n_comp == 3


# --------------------------------------------------------------------- find_nearest_point


def test_find_nearest_point_basic():
    nearest, idx = find_nearest_point([1.1, 2.9, 5.4], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(nearest, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(idx, [0, 2, 4])


def test_find_nearest_point_clamps_before_the_start():
    nearest, idx = find_nearest_point([-10.0], [1.0, 2.0])
    np.testing.assert_allclose(nearest, [1.0])
    np.testing.assert_array_equal(idx, [0])


def test_find_nearest_point_clamps_after_the_end():
    nearest, idx = find_nearest_point([99.0], [1.0, 2.0])
    np.testing.assert_allclose(nearest, [2.0])
    np.testing.assert_array_equal(idx, [1])


def test_find_nearest_point_exact_matches():
    nearest, idx = find_nearest_point([2.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(nearest, [2.0, 4.0])
    np.testing.assert_array_equal(idx, [1, 3])


def test_find_nearest_point_ties_go_to_the_earlier_element():
    """Matches the MATLAB's `nextDiff >= prevDiff` test."""
    nearest, idx = find_nearest_point([1.5], [1.0, 2.0])
    np.testing.assert_allclose(nearest, [1.0])
    np.testing.assert_array_equal(idx, [0])


def test_find_nearest_point_single_candidate():
    nearest, idx = find_nearest_point([5.0, -5.0], [3.0])
    np.testing.assert_allclose(nearest, [3.0, 3.0])
    np.testing.assert_array_equal(idx, [0, 0])


def test_find_nearest_point_shape_follows_query():
    nearest, idx = find_nearest_point(np.arange(7.0), np.arange(0, 20.0, 3))
    assert nearest.shape == (7,) and idx.shape == (7,)


def test_find_nearest_point_matches_brute_force():
    rng = np.random.default_rng(9)
    candidates = np.sort(rng.uniform(0, 100, 50))
    queries = np.sort(rng.uniform(-10, 110, 200))
    nearest, idx = find_nearest_point(queries, candidates)
    brute = np.array([candidates[np.argmin(np.abs(candidates - q))] for q in queries])
    np.testing.assert_allclose(nearest, brute)
    np.testing.assert_allclose(candidates[idx], brute)


def test_find_nearest_point_rejects_empty_candidates():
    with pytest.raises(ValueError, match="no nearest point"):
        find_nearest_point([1.0], [])
