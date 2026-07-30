"""SVD compression, validated against MATLAB's get_svdcomps on the same binary movie.

``U`` and ``V`` are only defined up to a per-component sign (eigenvectors are), so the
cross-language comparisons are on sign-invariant quantities: the eigenvalues, the total variance,
and the reconstruction ``U @ V``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from widefield.compress import iter_raw_frames, svd_compress
from widefield.svd import svd_frame_reconstruct

DATA_DIR = Path(__file__).parent / "data"
RAW = DATA_DIR / "raw_movie.bin"
SVD_REF = DATA_DIR / "matlab_svd_reference.mat"


@pytest.fixture(scope="module")
def sref():
    scipy_io = pytest.importorskip("scipy.io")
    if not (RAW.exists() and SVD_REF.exists()):
        pytest.skip("regenerate with tests/matlab_ref/gen_svd_reference.m")
    raw = scipy_io.loadmat(str(SVD_REF))
    out = {k: v for k, v in raw.items() if not k.startswith("__")}
    for key in ("svd_Ly", "svd_Lx", "svd_nFrames", "svd_NavgFramesSVD", "svd_nSVD"):
        out[key] = int(np.asarray(out[key]).ravel()[0])
    out["svd_totalVar"] = float(np.asarray(out["svd_totalVar"]).ravel()[0])
    out["svd_crop_totalVar"] = float(np.asarray(out["svd_crop_totalVar"]).ravel()[0])
    out["svd_Sv"] = np.asarray(out["svd_Sv"]).ravel()
    out["svd_crop_Sv"] = np.asarray(out["svd_crop_Sv"]).ravel()
    return out


@pytest.fixture(scope="module")
def compressed(sref):
    return svd_compress(
        RAW,
        sref["svd_mimg"],
        n_svd=sref["svd_nSVD"],
        n_avg_frames_svd=sref["svd_NavgFramesSVD"],
        dtype=np.uint16,
    )


# --------------------------------------------------------------------- frame reading


def test_raw_frames_read_in_the_matlab_layout(sref):
    """Each frame is column-major on disk; getting this wrong transposes the whole movie."""
    batches = list(iter_raw_frames(RAW, sref["svd_Ly"], sref["svd_Lx"], np.uint16))
    total = sum(b.shape[2] for _, b in batches)
    assert total == sref["svd_nFrames"]
    # The mean over all frames must match the mean image MATLAB computed from the same file.
    stack = np.concatenate([b for _, b in batches], axis=2)
    np.testing.assert_allclose(stack.mean(axis=2), sref["svd_mimg"], rtol=1e-5, atol=1e-3)


def test_raw_frames_batches_are_contiguous_and_ordered(sref):
    starts = [s for s, _ in iter_raw_frames(RAW, sref["svd_Ly"], sref["svd_Lx"], np.uint16)]
    assert starts == sorted(starts)
    assert starts[0] == 0


def test_raw_frames_respects_cancellation(sref):
    got = list(
        iter_raw_frames(RAW, sref["svd_Ly"], sref["svd_Lx"], np.uint16, should_cancel=lambda: True)
    )
    assert got == []


# --------------------------------------------------------------------- vs MATLAB


def test_total_variance_matches_matlab(compressed, sref):
    assert compressed.total_var == pytest.approx(sref["svd_totalVar"], rel=1e-5)


def test_eigenvalues_match_matlab(compressed, sref):
    """The leading eigenvalues carry essentially all the variance and must agree."""
    got, want = compressed.sv, sref["svd_Sv"]
    assert got.shape == want.shape
    # The trailing components are noise-level and numerically ill-determined; check the ones
    # that actually matter (the movie is rank 5 plus noise).
    np.testing.assert_allclose(got[:6], want[:6], rtol=1e-4)


def test_eigenvalues_are_descending(compressed):
    assert np.all(np.diff(compressed.sv) <= 1e-6)


def test_reconstruction_matches_matlab(compressed, sref):
    """Sign-invariant comparison: the reconstructed frame must match MATLAB's."""
    got = svd_frame_reconstruct(compressed.u, compressed.v[:, 9])
    want = np.asarray(sref["svd_recon_frame10"])
    assert got.shape == want.shape
    # tolerance relative to the frame's own scale
    scale = np.abs(want).max()
    np.testing.assert_allclose(got, want, atol=0.02 * scale)


def test_shapes_match_matlab(compressed, sref):
    assert compressed.u.shape == tuple(np.asarray(sref["svd_U"]).shape)
    assert compressed.v.shape == tuple(np.asarray(sref["svd_V"]).shape)


def test_crop_and_roi_match_matlab(sref):
    """MATLAB's yrange/xrange are 1-based inclusive; convert before comparing."""
    ys = np.asarray(sref["svd_crop_yrange"]).ravel().astype(int) - 1
    xs = np.asarray(sref["svd_crop_xrange"]).ravel().astype(int) - 1
    roi = np.asarray(sref["svd_crop_roi"], dtype=bool)

    r = svd_compress(
        RAW,
        sref["svd_mimg"],
        n_svd=sref["svd_nSVD"],
        n_avg_frames_svd=sref["svd_NavgFramesSVD"],
        yrange=slice(ys[0], ys[-1] + 1),
        xrange=slice(xs[0], xs[-1] + 1),
        roi=roi,
    )
    assert r.u.shape == tuple(np.asarray(sref["svd_crop_U"]).shape)
    assert r.total_var == pytest.approx(sref["svd_crop_totalVar"], rel=1e-5)
    np.testing.assert_allclose(r.sv[:6], sref["svd_crop_Sv"][:6], rtol=1e-4)

    got = svd_frame_reconstruct(r.u, r.v[:, 9])
    want = np.asarray(sref["svd_crop_recon_frame10"])
    np.testing.assert_allclose(got, want, atol=0.02 * np.abs(want).max())


def test_roi_zeroes_outside_pixels(sref):
    """Pixels outside the ROI must contribute nothing, so their components stay ~0."""
    roi = np.zeros((sref["svd_Ly"], sref["svd_Lx"]), dtype=bool)
    roi[4:10, 3:8] = True
    r = svd_compress(RAW, sref["svd_mimg"], n_svd=5, n_avg_frames_svd=100, roi=roi)
    outside = r.u[~roi]
    assert np.abs(outside).max() < 1e-6


# --------------------------------------------------------------------- properties


def test_u_columns_are_unit_norm(compressed):
    """MATLAB's normc: each spatial component is normalised."""
    flat = compressed.u.reshape(-1, compressed.u.shape[-1])
    norms = np.linalg.norm(flat, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_low_rank_movie_is_captured_by_few_components(compressed):
    """The fixture movie is rank 5 + noise, so 5 components should dominate the variance."""
    frac = compressed.sv[:5].sum() / compressed.sv.sum()
    assert frac > 0.95


def test_reconstruction_recovers_the_movie(sref):
    """End-to-end: reconstruct and compare with the raw frames the file actually holds."""
    r = svd_compress(RAW, sref["svd_mimg"], n_svd=15, n_avg_frames_svd=300)
    batches = list(iter_raw_frames(RAW, sref["svd_Ly"], sref["svd_Lx"], np.uint16))
    stack = np.concatenate([b for _, b in batches], axis=2)
    centred = stack - np.asarray(sref["svd_mimg"])[:, :, None]

    recon = svd_frame_reconstruct(r.u, r.v[:, :50])
    residual = centred[:, :, :50] - recon
    # 15 components on a rank-5 + noise movie should leave only the noise floor.
    assert residual.std() < 0.25 * centred[:, :, :50].std()


def test_total_var_equals_sum_of_all_eigenvalues(sref):
    """Documented property: keep everything and sum(sv) is the total variance."""
    r = svd_compress(RAW, sref["svd_mimg"], n_svd=10_000, n_avg_frames_svd=60)
    assert r.sv.sum() == pytest.approx(r.total_var, rel=1e-3)


def test_v_covers_every_frame(sref):
    r = svd_compress(RAW, sref["svd_mimg"], n_svd=5, n_avg_frames_svd=100)
    assert r.v.shape[1] == sref["svd_nFrames"]


def test_n_svd_capped_by_available_frames(sref):
    """MATLAB reserves 2 of the averaged frames; asking for more must clamp, not crash."""
    r = svd_compress(RAW, sref["svd_mimg"], n_svd=500, n_avg_frames_svd=20)
    assert r.u.shape[-1] <= 20 - 2


def test_rejects_non_2d_mean_image(sref):
    with pytest.raises(ValueError, match="must be 2-D"):
        svd_compress(RAW, np.zeros((4, 4, 2)))


def test_rejects_mismatched_roi(sref):
    with pytest.raises(ValueError, match="roi shape"):
        svd_compress(
            RAW, sref["svd_mimg"], n_svd=5, n_avg_frames_svd=100, roi=np.zeros((3, 3), bool)
        )


def test_rejects_too_few_averaged_frames(sref):
    with pytest.raises(ValueError, match="averaged frames"):
        svd_compress(RAW, sref["svd_mimg"], n_svd=5, n_avg_frames_svd=1)


def test_rejects_empty_movie(tmp_path, sref):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="nothing to compress"):
        svd_compress(empty, sref["svd_mimg"])
