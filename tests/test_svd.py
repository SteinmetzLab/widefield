"""Validate widefield.svd against the MATLAB golden reference, plus behavioral edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from widefield.svd import (
    bin_image,
    change_u,
    detrend_and_filt,
    dff_from_svd,
    flatten_u,
    hp_filt,
    pixel_timecourse,
    subsample_shift,
    svd_frame_reconstruct,
)

# --------------------------------------------------------------------- reconstruction


def test_reconstruct_matches_matlab(ref):
    got = svd_frame_reconstruct(ref["U"], ref["V"])
    assert got.shape == ref["recon_all"].shape
    np.testing.assert_allclose(got, ref["recon_all"], rtol=1e-10, atol=1e-10)


def test_reconstruct_single_frame_matches_matlab(ref):
    # MATLAB indexes from 1: its V(:, 7) is our v[:, 6].
    got = svd_frame_reconstruct(ref["U"], ref["V"][:, 6])
    assert got.shape == ref["recon_frame7"].shape
    np.testing.assert_allclose(got, ref["recon_frame7"], rtol=1e-10, atol=1e-10)


def test_reconstruct_keeps_axis_for_2d_v(ref):
    """A (nSV, 1) V keeps its frame axis; only a 1-D V collapses it."""
    assert svd_frame_reconstruct(ref["U"], ref["V"][:, 6:7]).ndim == 3
    assert svd_frame_reconstruct(ref["U"], ref["V"][:, 6]).ndim == 2


def test_reconstruct_truncates_u_to_v_rank(small_uv):
    """corr's V has fewer components than blue's U — reconstruction must not error."""
    u, v, _ = small_uv
    frame = svd_frame_reconstruct(u, v[:2])
    assert frame.shape == (u.shape[0], u.shape[1], v.shape[1])
    # identical to explicitly truncating U
    np.testing.assert_allclose(frame, svd_frame_reconstruct(u[..., :2], v[:2]))


def test_reconstruct_rejects_bad_shapes(small_uv):
    u, v, _ = small_uv
    with pytest.raises(ValueError, match="U must be"):
        svd_frame_reconstruct(u[..., 0], v)
    with pytest.raises(ValueError, match="V must be"):
        svd_frame_reconstruct(u, v[None])


def test_pixel_timecourse_matches_matlab(ref):
    pixel = tuple(int(p) - 1 for p in ref["corr_pixel"])  # MATLAB 1-based -> 0-based
    got = pixel_timecourse(ref["U"], ref["V"], pixel)
    np.testing.assert_allclose(got, ref["pixel_trace"], rtol=1e-10, atol=1e-10)


def test_pixel_timecourse_equals_full_reconstruction(small_uv):
    """The O(nSV*T) shortcut must agree with reconstructing everything and indexing."""
    u, v, _ = small_uv
    full = svd_frame_reconstruct(u, v)
    np.testing.assert_allclose(pixel_timecourse(u, v, (3, 4)), full[3, 4, :], atol=1e-12)


def test_flatten_u_is_contiguous_even_from_a_strided_view(small_uv):
    u, _, _ = small_uv
    flat = flatten_u(u[::2, ::2, :])
    assert flat.flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------- basis change / dF/F


def test_change_u_matches_matlab(ref):
    got = change_u(ref["U"], ref["V"], ref["newU_in"])
    np.testing.assert_allclose(got, ref["changeU_out"], rtol=1e-9, atol=1e-9)


def test_change_u_roundtrip_is_identity_for_same_basis(small_uv):
    u, v, _ = small_uv
    np.testing.assert_allclose(change_u(u, v, u), v, atol=1e-10)


def test_change_u_rejects_pixel_count_mismatch(small_uv):
    u, v, _ = small_uv
    with pytest.raises(ValueError, match="same pixel count"):
        change_u(u, v, u[:-1])


def test_dff_from_svd_matches_matlab(ref):
    new_u, new_v = dff_from_svd(ref["U"], ref["V"], ref["meanImage"])
    np.testing.assert_allclose(new_u, ref["dff_U"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(new_v, ref["dff_V"], rtol=1e-8, atol=1e-8)


def test_dff_returns_u_unchanged(ref):
    new_u, _ = dff_from_svd(ref["U"], ref["V"], ref["meanImage"])
    np.testing.assert_array_equal(new_u, ref["U"])


def test_dff_is_zero_mean_in_time(ref):
    """dF/F subtracts the temporal mean, so every component must average to ~0."""
    _, new_v = dff_from_svd(ref["U"], ref["V"], ref["meanImage"])
    np.testing.assert_allclose(new_v.mean(axis=1), 0.0, atol=1e-9)


def test_dff_soft_norm_default_is_median_of_mean_image(ref):
    _, auto = dff_from_svd(ref["U"], ref["V"], ref["meanImage"])
    _, explicit = dff_from_svd(
        ref["U"], ref["V"], ref["meanImage"], soft_norm=float(np.median(ref["meanImage"]))
    )
    np.testing.assert_allclose(auto, explicit, atol=1e-12)


def test_dff_larger_soft_norm_shrinks_the_signal(ref):
    """Sanity check on the soft-norm knob: more softening => smaller dF/F amplitudes."""
    _, small = dff_from_svd(ref["U"], ref["V"], ref["meanImage"], soft_norm=1.0)
    _, large = dff_from_svd(ref["U"], ref["V"], ref["meanImage"], soft_norm=1e6)
    assert np.abs(large).max() < np.abs(small).max()


def test_dff_rejects_mean_image_of_wrong_size(ref):
    with pytest.raises(ValueError, match="pixels"):
        dff_from_svd(ref["U"], ref["V"], ref["meanImage"][:-1, :])


# --------------------------------------------------------------------- temporal filters


def test_hp_filt_matches_matlab(ref):
    # float32 output (MATLAB casts to single), so compare at single precision.
    for cutoff, key in ((0.01, "hpFilt_0p01"), (0.5, "hpFilt_0p5")):
        got = hp_filt(ref["V"], ref["Fs"], cutoff)
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, ref[key], rtol=1e-4, atol=1e-4)


def test_hp_filt_removes_a_constant_offset(ref):
    """A DC-only component must come out ~0 after high-passing."""
    v = np.ones((1, 2000)) * 7.0
    got = hp_filt(v, ref["Fs"], 0.01)
    assert np.abs(got).max() < 1e-3


def test_detrend_and_filt_matches_matlab(ref):
    got = detrend_and_filt(ref["V"], ref["Fs"])
    np.testing.assert_allclose(got, ref["detrendAndFilt"], rtol=1e-8, atol=1e-8)


def test_detrend_and_filt_suppresses_heartbeat_band(ref):
    """The 9-14 Hz notch should knock down an 11 Hz tone far more than a 2 Hz one."""
    fs = 35.0
    n = 4000
    t = np.arange(n) / fs
    slow = np.sin(2 * np.pi * 2.0 * t)[None, :]
    heart = np.sin(2 * np.pi * 11.0 * t)[None, :]

    # Compare late samples only: the causal filter has a startup transient.
    keep = slice(n // 2, None)
    slow_gain = np.std(detrend_and_filt(slow, fs)[0, keep]) / np.std(slow[0, keep])
    heart_gain = np.std(detrend_and_filt(heart, fs)[0, keep]) / np.std(heart[0, keep])
    assert heart_gain < 0.1 * slow_gain


# --------------------------------------------------------------------- subsample shift


@pytest.mark.parametrize("p,q,key", [(1, 2, "sss_1_2"), (1, 4, "sss_1_4"), (3, 4, "sss_3_4")])
def test_subsample_shift_matches_matlab(ref, p, q, key):
    np.testing.assert_allclose(subsample_shift(ref["V"], p, q), ref[key], rtol=1e-10, atol=1e-10)


def test_subsample_shift_preserves_last_sample(ref):
    """The whole point of the interp1 rewrite: the final sample is carried through exactly."""
    got = subsample_shift(ref["V"], 1, 2)
    np.testing.assert_array_equal(got[:, -1], ref["V"][:, -1])


def test_subsample_shift_is_exact_on_a_linear_ramp():
    """Linear interpolation of a straight line is exact — a half-sample delay shifts by 0.5."""
    v = np.arange(50, dtype=float)[None, :] * 3.0
    got = subsample_shift(v, 1, 2)
    np.testing.assert_allclose(got[0, :-1], v[0, :-1] + 1.5, atol=1e-12)


def test_subsample_shift_preserves_dc():
    v = np.full((2, 40), 4.2)
    np.testing.assert_allclose(subsample_shift(v, 1, 2), 4.2, atol=1e-12)


def test_subsample_shift_no_nans_for_subsample_delays(ref):
    assert np.isfinite(subsample_shift(ref["V"], 1, 2)).all()


def test_subsample_shift_rejects_bad_args(ref):
    with pytest.raises(ValueError, match="positive"):
        subsample_shift(ref["V"], -1, 2)
    with pytest.raises(ValueError, match="V must be"):
        subsample_shift(ref["V"][0], 1, 2)


# --------------------------------------------------------------------- spatial binning


@pytest.mark.parametrize("factor,key", [(2, "binImage_b2"), (4, "binImage_b4")])
def test_bin_image_matches_matlab(ref, factor, key):
    got = bin_image(ref["binImage_in"], factor)
    assert got.shape == ref[key].shape
    np.testing.assert_allclose(got, ref[key], rtol=1e-10, atol=1e-10)


def test_bin_image_2d_input_stays_2d(ref):
    got = bin_image(ref["binImage_in"][:, :, 0], 2)
    assert got.ndim == 2
    np.testing.assert_allclose(got, ref["binImage_b2"][:, :, 0], atol=1e-10)


def test_bin_image_factor_one_is_a_copy(ref):
    img = ref["binImage_in"]
    got = bin_image(img, 1)
    np.testing.assert_array_equal(got, img)
    assert got is not img


def test_bin_image_b2_coincides_with_block_mean(ref):
    """At B=2 the conv2-'same' offset happens to select exactly the 2x2 blocks."""
    img = ref["binImage_in"][:, :, 0]
    y, x = img.shape
    block_mean = img[: y // 2 * 2, : x // 2 * 2].reshape(y // 2, 2, x // 2, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(bin_image(img, 2), block_mean, atol=1e-10)


def test_bin_image_b4_is_not_a_block_mean(ref):
    """At B=4 it emphatically is not — the window straddles blocks and edges are zero-padded.

    This is the guard that matters: if someone "simplifies" bin_image to reshape-and-mean,
    B=4 silently changes by >100% and this fails. Values are pinned to the MATLAB reference.
    """
    img = ref["binImage_in"][:, :, 0]
    got = bin_image(img, 4)
    # ceil(N/B) per axis, not floor: the trailing partial window is kept.
    assert got.shape == (3, 3)
    block_mean = img[:8, :8].reshape(2, 4, 2, 4).mean(axis=(1, 3))
    assert not np.allclose(got[:2, :2], block_mean)
    # first cell averages rows/cols -1..2 with zero padding
    assert got[0, 0] == pytest.approx(7.875)
    assert block_mean[0, 0] == pytest.approx(20.5)


def test_bin_image_rejects_bad_factor(ref):
    with pytest.raises(ValueError, match="bin_factor"):
        bin_image(ref["binImage_in"], 0)
