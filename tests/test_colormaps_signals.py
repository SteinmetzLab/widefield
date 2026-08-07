"""Colormap tables and Schmitt-trigger detection, both checked against MATLAB."""

from __future__ import annotations

import numpy as np
import pytest

from widefield import colormaps as cm
from widefield.signals import schmitt, schmitt_times

# ------------------------------------------------------------------------- colormaps


def test_blueblackred_matches_matlab(ref):
    got = cm.blueblackred()
    assert got.shape == ref["cmap_blueblackred"].shape == (101, 3)
    np.testing.assert_allclose(got, ref["cmap_blueblackred"], atol=1e-12)


def test_redblackblue_matches_matlab(ref):
    np.testing.assert_allclose(cm.redblackblue(), ref["cmap_redblackblue"], atol=1e-12)


def test_blue_white_red_matches_matlab(ref):
    np.testing.assert_allclose(cm.blue_white_red(), ref["cmap_BlueWhiteRed"], atol=1e-12)


def test_red_white_blue_matches_matlab(ref):
    np.testing.assert_allclose(cm.red_white_blue(), ref["cmap_RedWhiteBlue"], atol=1e-12)


def test_copper_matches_matlab(ref):
    np.testing.assert_allclose(cm.copper(4), ref["cmap_copper4"], atol=1e-9)


def test_blueblackred_is_black_at_the_middle():
    """Zero activity must render black — that is the point of this map."""
    m = cm.blueblackred()
    np.testing.assert_allclose(m[m.shape[0] // 2], [0, 0, 0], atol=1e-12)


def test_blueblackred_is_the_reverse_of_redblackblue():
    np.testing.assert_allclose(cm.blueblackred(), cm.redblackblue()[::-1], atol=1e-12)


def test_diverging_maps_are_in_gamut():
    for table in (cm.blueblackred(), cm.blue_white_red(), cm.red_white_blue()):
        assert table.min() >= 0.0 and table.max() <= 1.0


def test_blue_white_red_is_white_in_the_middle():
    m = cm.blue_white_red()
    np.testing.assert_allclose(m[m.shape[0] // 2], [1, 1, 1], atol=1e-12)


def test_condition_colors_matlab_scheme_reverses_copper_channels():
    np.testing.assert_allclose(
        cm.condition_colors(4, scheme="matlab"), cm.copper(4)[:, ::-1], atol=1e-12
    )


def test_matlab_condition_colors_start_at_pure_black():
    """Why 'matlab' is not the default: its first entry is invisible on a dark background."""
    np.testing.assert_allclose(cm.condition_colors(4, scheme="matlab")[0], [0, 0, 0], atol=1e-12)


def test_bright_condition_colors_are_all_visible_on_black():
    """Every entry must have real luminance, or a condition vanishes against the background."""
    colors = cm.condition_colors(11)
    lum = colors @ [0.2126, 0.7152, 0.0722]  # Rec. 709 relative luminance
    assert lum.min() > 0.15, f"dimmest condition color has luminance {lum.min():.3f}"
    assert colors.max() <= 1.0 and colors.min() >= 0.0


def test_bright_condition_colors_are_distinguishable():
    """Adjacent conditions must differ enough to tell apart."""
    colors = cm.condition_colors(11)
    gaps = np.linalg.norm(np.diff(colors, axis=0), axis=1)
    assert gaps.min() > 0.05


def test_condition_colors_single_condition():
    assert cm.condition_colors(1).shape == (1, 3)


def test_condition_colors_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown scheme"):
        cm.condition_colors(4, scheme="rainbow")


def test_hsv_to_rgb_primaries():
    got = cm.hsv_to_rgb(np.array([0.0, 1 / 3, 2 / 3]), 1.0, 1.0)
    np.testing.assert_allclose(got, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-9)


def test_hsv_to_rgb_zero_saturation_is_gray():
    got = cm.hsv_to_rgb(np.linspace(0, 1, 7), 0.0, 0.6)
    np.testing.assert_allclose(got, 0.6, atol=1e-9)


def test_lookup_table_is_uint8_and_right_length():
    lut = cm.to_lookup_table(cm.blueblackred(), n=256)
    assert lut.shape == (256, 3) and lut.dtype == np.uint8


def test_lookup_table_preserves_endpoints():
    table = cm.blueblackred()
    lut = cm.to_lookup_table(table, n=64)
    np.testing.assert_allclose(lut[0] / 255.0, table[0], atol=1 / 255)
    np.testing.assert_allclose(lut[-1] / 255.0, table[-1], atol=1 / 255)


def test_copper_rejects_zero_length():
    with pytest.raises(ValueError, match="m must be"):
        cm.copper(0)


# ------------------------------------------------------------------------- schmitt


def test_schmitt_times_match_matlab(ref):
    flips, up, down = schmitt_times(ref["t"], ref["schmitt_sig"], ref["schmitt_thresh"])
    np.testing.assert_allclose(flips, ref["schmitt_flipTimes"], atol=1e-12)
    np.testing.assert_allclose(up, ref["schmitt_flipUp"], atol=1e-12)
    np.testing.assert_allclose(down, ref["schmitt_flipDown"], atol=1e-12)


def test_schmitt_states_are_plus_minus_one():
    sig = np.array([0.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    y = schmitt(sig, (-0.5, 0.5))
    assert set(np.unique(y)).issubset({-1.0, 1.0, 0.0})


def test_schmitt_leading_zeros_before_first_crossing():
    """Before either threshold is met the state is undefined and reported as 0."""
    sig = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    y = schmitt(sig, (-0.5, 0.5))
    assert (y[:3] == 0).all() and y[3] == 1.0


def test_schmitt_ignores_noise_inside_the_hysteresis_band():
    """The reason for a Schmitt trigger: wobble between the thresholds must not flip state."""
    sig = np.concatenate([np.ones(5), np.full(5, 0.1), np.ones(5)])
    y = schmitt(sig, (-0.5, 0.5))
    assert (y == 1.0).all()


def test_schmitt_detects_a_real_square_wave():
    sig = np.tile(np.concatenate([np.ones(10), -np.ones(10)]), 5)
    t = np.arange(sig.size, dtype=float)
    flips, up, down = schmitt_times(t, sig, (-0.5, 0.5))
    # 5 cycles: 4 rising edges after the first, 5 falling
    assert up.size == 4 and down.size == 5
    assert flips.size == up.size + down.size
    np.testing.assert_array_equal(flips, np.sort(flips))


def test_schmitt_scalar_hysteresis_derives_thresholds_from_range():
    sig = np.concatenate([np.zeros(5), np.full(5, 10.0)])
    y = schmitt(sig, 0.5)  # thresholds at 2.5 and 7.5
    assert y[0] == -1.0 and y[-1] == 1.0


def test_schmitt_min_width_rejects_a_single_sample_glitch():
    sig = np.concatenate([-np.ones(20), [1.0], -np.ones(20)])
    t = np.arange(sig.size, dtype=float)
    without = schmitt_times(t, sig, (-0.5, 0.5))[0]
    with_min = schmitt_times(t, sig, (-0.5, 0.5), min_width=3)[0]
    assert without.size > with_min.size


def test_schmitt_empty_input():
    assert schmitt(np.array([])).size == 0


def test_schmitt_constant_signal_never_flips():
    flips, up, down = schmitt_times(np.arange(10.0), np.ones(10), (-0.5, 0.5))
    assert flips.size == 0 and up.size == 0 and down.size == 0


def test_schmitt_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="exceeds high"):
        schmitt(np.arange(10.0), (5.0, 1.0))


def test_schmitt_rejects_out_of_range_hysteresis():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        schmitt(np.arange(10.0), 1.5)


def test_schmitt_times_rejects_length_mismatch():
    with pytest.raises(ValueError, match="samples"):
        schmitt_times(np.arange(5.0), np.arange(6.0), (0.0, 1.0))
