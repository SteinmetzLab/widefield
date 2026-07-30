"""Validate seed-pixel correlation against MATLAB and against a brute-force computation."""

from __future__ import annotations

import numpy as np
import pytest

from widefield.correlation import SeedCorrelation, correlation_map_raw
from widefield.svd import svd_frame_reconstruct


def test_cov_v_matches_matlab(ref):
    sc = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64)
    np.testing.assert_allclose(sc.cov_v, ref["corr_covV"], rtol=1e-10, atol=1e-10)


def test_variance_image_matches_matlab(ref):
    """Compare as an *image*, not as a flat vector.

    MATLAB's ``reshape(U, P, [])`` flattens pixels column-major; numpy flattens row-major. The
    two flat ``varP`` vectors are therefore permutations of each other while every image
    derived from them is identical — so the meaningful comparison undoes MATLAB's ordering
    with an order='F' reshape.
    """
    sc = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64)
    shape = (int(ref["Ypix"]), int(ref["Xpix"]))
    matlab_image = np.reshape(ref["corr_varP"], shape, order="F")
    np.testing.assert_allclose(sc.variance_image, matlab_image, rtol=1e-9, atol=1e-9)


def test_var_p_is_row_major_flat(ref):
    """Pin the documented convention: var_p[y * Xpix + x] is pixel (y, x)."""
    sc = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64)
    xpix = int(ref["Xpix"])
    for y, x in ((0, 0), (2, 3), (5, 4)):
        assert sc.var_p[y * xpix + x] == pytest.approx(sc.variance_image[y, x])


def test_seed_map_matches_matlab(ref):
    sc = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64)
    pixel = tuple(int(p) - 1 for p in ref["corr_pixel"])
    np.testing.assert_allclose(sc.map(pixel), ref["corr_map"], rtol=1e-9, atol=1e-9)


def test_seed_map_normalize_by_max_matches_matlab(ref):
    """The viewer's 'V' key: normalize by the global max variance instead of each pixel's."""
    sc = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64)
    pixel = tuple(int(p) - 1 for p in ref["corr_pixel"])
    got = sc.map(pixel, normalize_by_max=True)
    np.testing.assert_allclose(got, ref["corr_map_max"], rtol=1e-9, atol=1e-9)


def test_seed_map_float32_is_close_enough_to_float64(ref):
    """float32 is the default for speed; it must not visibly change a [-1, 1] map."""
    pixel = tuple(int(p) - 1 for p in ref["corr_pixel"])
    f32 = SeedCorrelation(ref["U"], ref["V"], dtype=np.float32).map(pixel)
    f64 = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64).map(pixel)
    assert np.abs(f32 - f64).max() < 1e-5


def test_seed_of_itself_is_unity(ref):
    sc = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64)
    pixel = (5, 4)
    assert sc.map(pixel)[pixel] == pytest.approx(1.0, abs=1e-9)


def test_seed_map_is_bounded(ref):
    sc = SeedCorrelation(ref["U"], ref["V"], dtype=np.float64)
    m = sc.map((3, 2))
    assert m.min() >= -1.0 - 1e-9 and m.max() <= 1.0 + 1e-9


def test_svd_shortcut_equals_brute_force_correlation(small_uv):
    """The whole trick: the SVD route must equal correlating the reconstructed movie."""
    u, v, _ = small_uv
    movie = svd_frame_reconstruct(u, v)
    pixel = (4, 3)
    fast = SeedCorrelation(u, v, dtype=np.float64).map(pixel)
    slow = correlation_map_raw(movie, pixel)
    np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_svd_shortcut_equals_numpy_corrcoef(small_uv):
    """Belt and braces: also check against np.corrcoef on the reconstructed pixel traces."""
    u, v, _ = small_uv
    movie = svd_frame_reconstruct(u, v)
    ypix, xpix, _ = movie.shape
    flat = movie.reshape(ypix * xpix, -1)
    expected = np.corrcoef(flat)[4 * xpix + 3].reshape(ypix, xpix)
    got = SeedCorrelation(u, v, dtype=np.float64).map((4, 3))
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_max_components_truncates(small_uv):
    u, v, _ = small_uv
    sc = SeedCorrelation(u, v, max_components=2, dtype=np.float64)
    assert sc.n_components == 2
    np.testing.assert_allclose(
        sc.map((4, 3)), SeedCorrelation(u[..., :2], v[:2], dtype=np.float64).map((4, 3)), atol=1e-12
    )


def test_handles_v_with_fewer_components_than_u(small_uv):
    """The corr channel case: V truncated relative to U."""
    u, v, _ = small_uv
    sc = SeedCorrelation(u, v[:3], dtype=np.float64)
    assert sc.n_components == 3
    assert sc.map((4, 3)).shape == u.shape[:2]


def test_variance_image_shape_and_positivity(small_uv):
    u, v, _ = small_uv
    img = SeedCorrelation(u, v, dtype=np.float64).variance_image
    assert img.shape == u.shape[:2]
    assert (img >= 0).all()


def test_chunked_variance_matches_unchunked(monkeypatch, small_uv):
    """var_p is accumulated in row chunks to bound memory; chunking must be invisible."""
    import widefield.correlation as corr_mod

    u, v, _ = small_uv
    big = SeedCorrelation(u, v, dtype=np.float64).var_p
    monkeypatch.setattr(corr_mod, "_VAR_CHUNK", 3)
    small = SeedCorrelation(u, v, dtype=np.float64).var_p
    np.testing.assert_allclose(big, small, atol=1e-12)


def test_dead_pixel_gives_zero_not_nan():
    """A pixel with no variance would divide by zero; it must come out 0, not NaN."""
    u = np.zeros((3, 3, 2))
    u[0, 0, 0] = 1.0
    u[1, 1, 1] = 1.0
    v = np.random.default_rng(0).standard_normal((2, 100))
    m = SeedCorrelation(u, v, dtype=np.float64).map((0, 0))
    assert np.isfinite(m).all()
    assert m[2, 2] == 0.0


def test_out_of_bounds_pixel_raises(small_uv):
    u, v, _ = small_uv
    sc = SeedCorrelation(u, v)
    with pytest.raises(IndexError, match="outside image"):
        sc.map((99, 0))
    with pytest.raises(IndexError, match="outside image"):
        sc.map((0, -1))


def test_correlation_map_raw_rejects_2d(small_uv):
    with pytest.raises(ValueError, match="must be"):
        correlation_map_raw(np.zeros((4, 4)), (0, 0))
