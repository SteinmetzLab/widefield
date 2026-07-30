"""Orientation coordinate maps — pure logic, no Qt needed.

The MATLAB viewers rotate/flip the image with alt+arrows and keep plain arrows screen-relative.
Getting that wrong is easy and shows up as "clicking here highlights a pixel over there", so
every orientation is checked exhaustively against the actual displayed array.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from widefield.gui._common import Orientation, clamp

ALL_ORIENTATIONS = list(itertools.product(range(4), [False, True]))
SHAPE = (5, 3)  # deliberately non-square so transposes can't hide


def _data(shape=SHAPE):
    return np.arange(shape[0] * shape[1]).reshape(shape)


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_display_shape_matches_the_actual_array(rot, flip):
    o = Orientation(rot, flip)
    assert o.apply(_data()).shape == o.display_shape(SHAPE)


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_to_display_lands_on_the_same_value(rot, flip):
    """The mapping must agree with what the user actually sees, for every pixel."""
    o = Orientation(rot, flip)
    data = _data()
    shown = o.apply(data)
    for y in range(SHAPE[0]):
        for x in range(SHAPE[1]):
            dy, dx = o.to_display(y, x, SHAPE)
            assert shown[dy, dx] == data[y, x], f"rot={rot} flip={flip} at ({y},{x})"


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_to_data_is_the_inverse_of_to_display(rot, flip):
    o = Orientation(rot, flip)
    for y in range(SHAPE[0]):
        for x in range(SHAPE[1]):
            assert o.to_data(*o.to_display(y, x, SHAPE), SHAPE) == (y, x)


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_every_display_pixel_maps_back_into_range(rot, flip):
    o = Orientation(rot, flip)
    dh, dw = o.display_shape(SHAPE)
    seen = set()
    for dy in range(dh):
        for dx in range(dw):
            y, x = o.to_data(dy, dx, SHAPE)
            assert 0 <= y < SHAPE[0] and 0 <= x < SHAPE[1]
            seen.add((y, x))
    assert len(seen) == SHAPE[0] * SHAPE[1], "mapping is not a bijection"


def test_identity_orientation_is_a_no_op():
    o = Orientation()
    data = _data()
    np.testing.assert_array_equal(o.apply(data), data)
    assert o.to_display(2, 1, SHAPE) == (2, 1)


def test_rotate_wraps_at_four():
    o = Orientation()
    o.rotate(5)
    assert o.rot == 1
    o.rotate(-1)
    assert o.rot == 0


def test_rotate_four_times_returns_to_start():
    data = _data()
    o = Orientation()
    for _ in range(4):
        o.rotate(1)
    np.testing.assert_array_equal(o.apply(data), data)


def test_reset_clears_rotation_and_flip():
    o = Orientation(3, True)
    o.reset()
    assert (o.rot, o.flip) == (0, False)


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_step_on_screen_moves_down_on_screen(rot, flip):
    """+1 in d_row must move one row *down in the displayed image*, whatever the orientation."""
    o = Orientation(rot, flip)
    data = _data()
    shown = o.apply(data)
    start_dy, start_dx = 1, 1
    y, x = o.to_data(start_dy, start_dx, SHAPE)

    ny, nx = o.step_on_screen(y, x, SHAPE, 1, 0)
    ndy, ndx = o.to_display(ny, nx, SHAPE)
    assert (ndy, ndx) == (start_dy + 1, start_dx)
    assert shown[ndy, ndx] == data[ny, nx]


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_step_on_screen_moves_right_on_screen(rot, flip):
    o = Orientation(rot, flip)
    y, x = o.to_data(1, 1, SHAPE)
    ny, nx = o.step_on_screen(y, x, SHAPE, 0, 1)
    assert o.to_display(ny, nx, SHAPE) == (1, 2)


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_step_on_screen_clamps_at_the_edges(rot, flip):
    """Walking off the edge must stop, not wrap or go out of bounds."""
    o = Orientation(rot, flip)
    dh, dw = o.display_shape(SHAPE)
    y, x = o.to_data(0, 0, SHAPE)
    for _ in range(dh + dw + 5):
        y, x = o.step_on_screen(y, x, SHAPE, -1, -1)
    assert o.to_display(y, x, SHAPE) == (0, 0)

    y, x = o.to_data(dh - 1, dw - 1, SHAPE)
    for _ in range(dh + dw + 5):
        y, x = o.step_on_screen(y, x, SHAPE, 1, 1)
    assert o.to_display(y, x, SHAPE) == (dh - 1, dw - 1)


@pytest.mark.parametrize("rot,flip", ALL_ORIENTATIONS)
def test_step_never_leaves_the_image(rot, flip):
    o = Orientation(rot, flip)
    rng = np.random.default_rng(0)
    y, x = 2, 1
    for _ in range(200):
        d_row, d_col = rng.integers(-7, 8, size=2)
        y, x = o.step_on_screen(y, x, SHAPE, int(d_row), int(d_col))
        assert 0 <= y < SHAPE[0] and 0 <= x < SHAPE[1]


def test_rot90_convention_matches_numpy():
    """Pin the convention: rot counts np.rot90 (counter-clockwise) steps."""
    data = _data()
    np.testing.assert_array_equal(Orientation(1, False).apply(data), np.rot90(data, 1))


def test_flip_is_applied_after_rotation():
    """Order matters: flip-then-rotate differs from rotate-then-flip."""
    data = _data()
    np.testing.assert_array_equal(Orientation(1, True).apply(data), np.rot90(data, 1)[::-1])


def test_non_square_shapes_transpose_on_odd_rotations():
    assert Orientation(1).display_shape((5, 3)) == (3, 5)
    assert Orientation(2).display_shape((5, 3)) == (5, 3)


def test_apply_preserves_trailing_axes():
    """Rotating a movie stack must rotate frames, not shuffle time."""
    stack = np.arange(5 * 3 * 4).reshape(5, 3, 4)
    out = Orientation(1).apply(stack)
    assert out.shape == (3, 5, 4)
    for f in range(4):
        np.testing.assert_array_equal(out[:, :, f], np.rot90(stack[:, :, f], 1))


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert clamp(2, 0, 3) == 2
