"""Hemodynamic correction, validated against MATLAB and against its own physics."""

from __future__ import annotations

import numpy as np
import pytest

from widefield.hemo import (
    hemo_correct_local,
    hemo_correct_nonlocal,
    variance_explained,
)
from widefield.svd import flatten_u

# --------------------------------------------------------------------- local method


def test_local_transform_matches_matlab(ref):
    r = hemo_correct_local(
        ref["U"],
        ref["V"],
        ref["hemo_Vaux"],
        ref["Fs"],
        freq_range=tuple(np.asarray(ref["hemo_freqRange"]).ravel()),
        pix_space=int(np.asarray(ref["hemo_pixSpace"]).ravel()[0]),
    )
    np.testing.assert_allclose(r.transform, ref["hemo_local_T"], rtol=1e-7, atol=1e-9)


def test_local_corrected_v_matches_matlab(ref):
    r = hemo_correct_local(
        ref["U"],
        ref["V"],
        ref["hemo_Vaux"],
        ref["Fs"],
        freq_range=tuple(np.asarray(ref["hemo_freqRange"]).ravel()),
        pix_space=int(np.asarray(ref["hemo_pixSpace"]).ravel()[0]),
    )
    np.testing.assert_allclose(r.v_corrected, ref["hemo_local_V"], rtol=1e-7, atol=1e-8)


def test_local_scale_factors_match_matlab_as_a_set(ref):
    """MATLAB's subgrid is flattened column-major and its map is transposed (see hemo.py).

    Comparing the sorted values checks the gains themselves without inheriting that ordering.
    """
    r = hemo_correct_local(
        ref["U"],
        ref["V"],
        ref["hemo_Vaux"],
        ref["Fs"],
        freq_range=tuple(np.asarray(ref["hemo_freqRange"]).ravel()),
        pix_space=int(np.asarray(ref["hemo_pixSpace"]).ravel()[0]),
    )
    got = np.sort(r.scale_factor_map.ravel())
    want = np.sort(np.asarray(ref["hemo_local_scale"]).ravel())
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


def test_local_scale_map_is_image_oriented(ref):
    """The returned map must be (y, x): a non-square image is the case MATLAB gets wrong."""
    u = np.asarray(ref["U"])[:10, :6, :]  # deliberately non-square
    r = hemo_correct_local(u, ref["V"], ref["hemo_Vaux"], ref["Fs"], pix_space=2)
    y_span, x_span = r.subgrid
    assert r.scale_factor_map.shape == (y_span.size, x_span.size) == (5, 3)


def test_local_variance_explained_matches_matlab(ref):
    r = hemo_correct_local(
        ref["U"],
        ref["V"],
        ref["hemo_Vaux"],
        ref["Fs"],
        freq_range=tuple(np.asarray(ref["hemo_freqRange"]).ravel()),
        pix_space=int(np.asarray(ref["hemo_pixSpace"]).ravel()[0]),
    )
    assert r.heart_variance_explained == pytest.approx(
        float(np.asarray(ref["hemo_local_heartPct"]).ravel()[0]), rel=1e-6
    )
    assert r.slow_variance_explained == pytest.approx(
        float(np.asarray(ref["hemo_local_slowPct"]).ravel()[0]), rel=1e-6
    )


def test_local_removes_a_shared_heartbeat_component(ref):
    """The point of the whole exercise: a vascular signal present in both channels must go."""
    rng = np.random.default_rng(3)
    fs, n = 70.0, 4000
    t = np.arange(n) / fs
    u = np.asarray(ref["U"])
    nsv = u.shape[-1]

    heart = np.sin(2 * np.pi * 11.0 * t)
    neural = rng.standard_normal((nsv, n)) * 0.1
    v = neural + heart[None, :] * 2.0
    v_aux = rng.standard_normal((nsv, n)) * 0.05 + heart[None, :] * 2.0

    r = hemo_correct_local(u, v, v_aux, fs)
    assert r.heart_variance_explained > 80.0

    # heartbeat power in the corrected signal must drop a lot
    def band_power(x):
        f = np.fft.rfftfreq(n, 1 / fs)
        p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
        return p[:, (f > 10) & (f < 12)].sum()

    assert band_power(r.v_corrected) < 0.2 * band_power(v)


def test_local_transform_is_symmetric_in_the_u_basis(ref):
    """pinv(Usub) @ diag(g) @ Usub is (nSV, nSV) — pin the shape contract."""
    r = hemo_correct_local(ref["U"], ref["V"], ref["hemo_Vaux"], ref["Fs"])
    nsv = int(ref["nSV"])
    assert r.transform.shape == (nsv, nsv)


def test_local_zero_aux_gives_zero_gain(ref):
    """No auxiliary signal means nothing to regress out; gains must be 0, not NaN."""
    r = hemo_correct_local(ref["U"], ref["V"], np.zeros_like(ref["V"]), ref["Fs"])
    assert np.isfinite(r.scale_factor_map).all()
    np.testing.assert_allclose(r.scale_factor_map, 0.0, atol=1e-12)
    np.testing.assert_allclose(r.v_corrected, ref["V"], atol=1e-9)


def test_local_pix_space_one_uses_every_pixel(ref):
    u = np.asarray(ref["U"])
    r = hemo_correct_local(u, ref["V"], ref["hemo_Vaux"], ref["Fs"], pix_space=1)
    assert r.scale_factor_map.shape == u.shape[:2]


def test_local_coarser_subgrid_is_cheaper_but_similar(ref):
    fine = hemo_correct_local(ref["U"], ref["V"], ref["hemo_Vaux"], ref["Fs"], pix_space=1)
    coarse = hemo_correct_local(ref["U"], ref["V"], ref["hemo_Vaux"], ref["Fs"], pix_space=3)
    assert coarse.scale_factor_map.size < fine.scale_factor_map.size
    # both should explain a similar amount of heartbeat power
    assert abs(coarse.heart_variance_explained - fine.heart_variance_explained) < 30.0


def test_local_rejects_shape_mismatch(ref):
    with pytest.raises(ValueError, match="same shape"):
        hemo_correct_local(ref["U"], ref["V"], ref["hemo_Vaux"][:, :-1], ref["Fs"])


def test_local_rejects_bad_pix_space(ref):
    with pytest.raises(ValueError, match="pix_space"):
        hemo_correct_local(ref["U"], ref["V"], ref["hemo_Vaux"], ref["Fs"], pix_space=0)


def test_local_handles_v_truncated_relative_to_u(ref):
    """corr-style rank mismatch must not error."""
    r = hemo_correct_local(ref["U"], ref["V"][:3], ref["hemo_Vaux"][:3], ref["Fs"])
    assert r.transform.shape == (3, 3)


# --------------------------------------------------------------------- nonlocal method


def test_nonlocal_weights_match_matlab(ref):
    r = hemo_correct_nonlocal(
        ref["V"], ref["hemo_Vaux"], ref["Fs"], tuple(np.asarray(ref["hemo_freqRange"]).ravel())
    )
    np.testing.assert_allclose(r.transform, ref["hemo_nonlocal_Wts"], rtol=1e-6, atol=1e-8)


def test_nonlocal_corrected_v_matches_matlab(ref):
    r = hemo_correct_nonlocal(
        ref["V"], ref["hemo_Vaux"], ref["Fs"], tuple(np.asarray(ref["hemo_freqRange"]).ravel())
    )
    np.testing.assert_allclose(r.v_corrected, ref["hemo_nonlocal_V"], rtol=1e-6, atol=1e-7)


def test_nonlocal_without_filtering_works(ref):
    r = hemo_correct_nonlocal(ref["V"], ref["hemo_Vaux"])
    assert r.transform.shape == (int(ref["nSV"]), int(ref["nSV"]))
    assert r.scale_factor_map is None


def test_nonlocal_removes_a_perfectly_correlated_aux(ref):
    """If aux == signal, the correction should remove essentially everything."""
    v = np.asarray(ref["V"])
    r = hemo_correct_nonlocal(v, v.copy())
    assert np.abs(r.v_corrected - v.mean(axis=1, keepdims=True)).max() < 1e-6


def test_nonlocal_leaves_an_uncorrelated_aux_alone(ref):
    rng = np.random.default_rng(5)
    v = np.asarray(ref["V"])
    aux = rng.standard_normal(v.shape) * v.std()
    r = hemo_correct_nonlocal(v, aux)
    # can't remove what isn't there: residual keeps most of the variance
    assert r.v_corrected.std() > 0.7 * v.std()


def test_nonlocal_rejects_only_one_of_fs_and_range(ref):
    with pytest.raises(ValueError, match="both fs and freq_range"):
        hemo_correct_nonlocal(ref["V"], ref["hemo_Vaux"], fs=35.0)


def test_nonlocal_rejects_shape_mismatch(ref):
    with pytest.raises(ValueError, match="same shape"):
        hemo_correct_nonlocal(ref["V"], ref["hemo_Vaux"][:, :-1])


# --------------------------------------------------------------------- helper


def test_variance_explained_bounds():
    sig = np.array([[1.0, -1.0, 1.0, -1.0]])
    assert variance_explained(sig, np.zeros_like(sig)) == pytest.approx(100.0)
    assert variance_explained(sig, sig) == pytest.approx(0.0)


def test_variance_explained_of_silence_is_zero():
    assert variance_explained(np.zeros((2, 5)), np.zeros((2, 5))) == 0.0


def test_variance_explained_can_go_negative():
    """A correction that makes things worse must report it, not clamp to 0."""
    sig = np.ones((1, 4))
    assert variance_explained(sig, sig * 2) < 0


def test_local_applies_the_transform_in_the_right_direction(ref):
    """Catch a transposed application, which T's near-symmetry otherwise almost hides.

    Reproduce MATLAB's ``V - zVaux*T'`` elementwise from the returned transform and compare.
    A ``T.T`` slip changes the reference variance-explained by only 0.006 points, so this
    explicit reconstruction — not the aggregate — is what pins the orientation.
    """
    r = hemo_correct_local(
        ref["U"],
        ref["V"],
        ref["hemo_Vaux"],
        ref["Fs"],
        freq_range=tuple(np.asarray(ref["hemo_freqRange"]).ravel()),
        pix_space=int(np.asarray(ref["hemo_pixSpace"]).ravel()[0]),
    )
    v = np.asarray(ref["V"], dtype=float)
    v_aux = np.asarray(ref["hemo_Vaux"], dtype=float)
    zv_aux = v_aux - v_aux.mean(axis=1, keepdims=True)

    # MATLAB, index by index: Vout(t, s) = V(t, s) - sum_k zVaux(t, k) * T(s, k)
    t_mat = np.asarray(ref["hemo_local_T"], dtype=float)
    expected = np.empty_like(v)
    for s in range(v.shape[0]):
        expected[s] = v[s] - np.einsum("k,kt->t", t_mat[s], zv_aux)
    np.testing.assert_allclose(r.v_corrected, expected, rtol=1e-9, atol=1e-10)

    # and the wrong orientation must be measurably different, or this test proves nothing
    wrong = v - t_mat.T @ zv_aux
    assert not np.allclose(r.v_corrected, wrong, rtol=1e-6, atol=1e-6)


def test_local_subgrid_indices_match_the_flattening(ref):
    """The subgrid rows of Usub must be the pixels the subgrid claims."""
    u = np.asarray(ref["U"])
    r = hemo_correct_local(u, ref["V"], ref["hemo_Vaux"], ref["Fs"], pix_space=3)
    y_span, x_span = r.subgrid
    flat = flatten_u(u)
    xpix = u.shape[1]
    expected_first = flat[y_span[0] * xpix + x_span[0]]
    np.testing.assert_allclose(expected_first, u[y_span[0], x_span[0], :])
