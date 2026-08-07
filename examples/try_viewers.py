"""Open all four viewers on one session, for kicking the tyres.

    python examples/try_viewers.py                       # default session on the server
    python examples/try_viewers.py --session "Y:\\Subjects\\ZYE_0057\\2022-01-10\\1"
    python examples/try_viewers.py --demo                 # synthetic data, no server needed
    python examples/try_viewers.py --nsv 500              # more components (slower load)
    python examples/try_viewers.py --only movie           # just one viewer (movie/corr/tuning/svd)

Windows are tiled across the screen. Each is independent; the process exits when you close them
all. Keyboard shortcuts are listed along the bottom of each window and in the module docstrings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import widefield as wf
from widefield.gui._common import ensure_app
from widefield.gui.movie import Trace
from widefield.gui.movie import _get_class as movie_class
from widefield.gui.pixel_correlation import _get_class as corr_class
from widefield.gui.pixel_tuning_curve import _get_class as tuning_class
from widefield.gui.svd_viewer import _get_class as svd_class

DEFAULT_SESSION = r"Y:\Subjects\AB_0004\2021-03-24\1"
N_CONDITIONS = 4

log = logging.getLogger("try_viewers")


# --------------------------------------------------------------------------- data loading


def load_sv(session: Path, n_components: int):
    """Real singular values + total variance from ``blue/dataSummary.mat``, if present.

    Without them the component browser can only show percentages of the variance *retained*,
    which understates how much of the movie the leading components actually explain.
    """
    path = session / "blue" / "dataSummary.mat"
    if not path.exists():
        return None, None
    try:
        from scipy.io import loadmat

        m = loadmat(str(path))
        sv = np.asarray(m["Sv"], dtype=float).ravel()[:n_components]
        total = float(np.asarray(m["totalVar"]).ravel()[0])
        return sv, total
    except Exception as exc:  # a missing/odd dataSummary must not stop the demo
        log.warning("could not read %s (%s); falling back to component variance", path.name, exc)
        return None, None


def load_timeline_trace(session: Path, name: str):
    """Load a Timeline analogue channel as ``(t, v)``.

    The ``*.timestamps_Timeline.npy`` files are a 2-point linear map from sample index to
    Timeline seconds (``[[i0, t0], [i1, t1]]``), not a per-sample vector, so the time base is
    interpolated rather than read.
    """
    raw = session / f"{name}.raw.npy"
    ts = session / f"{name}.timestamps_Timeline.npy"
    if not (raw.exists() and ts.exists()):
        return None
    try:
        v = np.asarray(np.load(raw, mmap_mode="r")).ravel().astype(float)
        m = np.asarray(np.load(ts)).reshape(-1, 2)
        idx, times = m[:, 0], m[:, 1]
        t = np.interp(np.arange(v.size), idx, times)
        return t, v
    except Exception as exc:
        log.warning("could not read Timeline channel %r (%s)", name, exc)
        return None


def laser_trace(session: Path):
    """A 0/power step trace of laser pulses, for opto sessions. ``None`` if not one.

    Built from ``laserOnTimes``/``laserOffTimes`` (+ ``laserPowers`` when present) as an explicit
    staircase, so the movie viewer's scrolling window shows exactly when the laser was on and how
    hard, against the activity it drove.
    """
    on_p, off_p = session / "laserOnTimes.npy", session / "laserOffTimes.npy"
    if not (on_p.exists() and off_p.exists()):
        return None
    try:
        on = np.asarray(np.load(on_p)).ravel()
        off = np.asarray(np.load(off_p)).ravel()
        n = min(on.size, off.size)
        on, off = on[:n], off[:n]
        power_p = session / "laserPowers.npy"
        power = np.asarray(np.load(power_p)).ravel()[:n] if power_p.exists() else np.ones(n)
        # Four points per pulse: rise, hold, fall, off. Sorted because a plot needs monotone x.
        t = np.concatenate([on, on, off, off])
        v = np.concatenate([np.zeros(n), power, power, np.zeros(n)])
        order = np.argsort(t, kind="stable")
        return t[order], v[order], n
    except Exception as exc:
        log.warning("could not read laser times (%s)", exc)
        return None


def stim_times_from_photodiode(session: Path, min_gap_s: float = 1.0, cap: int = 300):
    """Detect stimulus onsets from the photodiode with a Schmitt trigger.

    Returns ``None`` if the trace is missing or the result looks implausible, so the caller can
    fall back to invented times. Real condition labels would come from the trials ALF, which this
    session does not have — see :func:`make_events`.
    """
    got = load_timeline_trace(session, "photodiode")
    if got is None:
        return None
    t, v = got
    lo, hi = np.percentile(v, [20, 80])
    if hi - lo < 1e-6:  # flat trace: no stimuli, or the channel was not recorded
        return None
    flips, up, _down = wf.schmitt_times(t, v, (lo, hi))
    if up.size < 10:
        return None
    # Keep only onsets separated by at least min_gap_s: the photodiode also flips on every
    # stimulus frame, and we want trial starts.
    keep = [up[0]]
    for x in up[1:]:
        if x - keep[-1] >= min_gap_s:
            keep.append(x)
    onsets = np.array(keep)
    if onsets.size < 10:
        return None
    if onsets.size > cap:  # evenly thin, keeping the spread across the session
        onsets = onsets[np.linspace(0, onsets.size - 1, cap).astype(int)]
    return onsets


def laser_events(session: Path):
    """Laser onsets with power as the condition label. ``None`` unless this is an opto session.

    This gives the tuning viewer *real* conditions: response versus laser power, which is an
    actual tuning curve rather than a shape-check. Preferred over the photodiode route wherever
    it is available.
    """
    on_p, pw_p = session / "laserOnTimes.npy", session / "laserPowers.npy"
    if not (on_p.exists() and pw_p.exists()):
        return None
    try:
        on = np.asarray(np.load(on_p)).ravel()
        power = np.asarray(np.load(pw_p)).ravel()
        n = min(on.size, power.size)
        return on[:n], power[:n]
    except Exception as exc:
        log.warning("could not read laser events (%s)", exc)
        return None


def make_events(session: Path, t: np.ndarray, rng):
    """Event times + condition labels for the tuning viewer, and a note on their provenance.

    Three sources in descending order of scientific meaning:
      1. laser onsets labelled by power (opto sessions) - a genuine tuning curve;
      2. photodiode onsets with cycled pseudo-labels - exercises the viewer, curve is meaningless;
      3. invented random times - last resort.
    """
    if session is not None:
        got = laser_events(session)
        if got is not None:
            onsets, labels = got
            keep = (onsets > t[0] + 1.0) & (onsets < t[-1] - 2.0)
            onsets, labels = onsets[keep], labels[keep]
            if onsets.size >= 10:
                n_levels = np.unique(labels).size
                return (
                    onsets,
                    labels,
                    f"laser onsets ({onsets.size}), {n_levels} real power levels",
                )

    onsets = stim_times_from_photodiode(session) if session is not None else None
    if onsets is not None:
        onsets = onsets[(onsets > t[0] + 1.0) & (onsets < t[-1] - 2.0)]
    if onsets is None or onsets.size < 10:
        onsets = np.sort(rng.uniform(t[0] + 1.0, t[-1] - 2.0, 200))
        source = "invented (random) times"
    else:
        source = f"photodiode onsets ({onsets.size})"
    # No trials ALF here, so there are no real condition labels. Cycling through four
    # pseudo-conditions exercises the viewer; the tuning curve is meaningless by design.
    labels = np.arange(onsets.size) % N_CONDITIONS / (N_CONDITIONS - 1.0)
    return onsets, labels, source + ", PSEUDO-conditions"


def demo_data():
    """Synthetic session with a couple of traveling blobs, so the movie has visible structure."""
    rng = np.random.default_rng(0)
    ypix, xpix, nsv, nframes, fs = 128, 128, 60, 4000, 35.0
    t = np.arange(nframes) / fs

    yy, xx = np.mgrid[0:ypix, 0:xpix]
    movie = np.zeros((ypix, xpix, 0), dtype=np.float32)
    # Build a low-rank movie directly in U/V form rather than pixel space (which would be 260 MB).
    centers = [(40, 40), (90, 45), (60, 95), (30, 100)]
    freqs = [0.20, 0.35, 0.11, 0.55]
    u_list, v_list = [], []
    for (cy, cx), f in zip(centers, freqs, strict=True):
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 14.0**2))
        u_list.append(blob.astype(np.float32))
        v_list.append((np.sin(2 * np.pi * f * t) * 0.3).astype(np.float32))
    u = np.stack(u_list, axis=2)
    v = np.stack(v_list, axis=0)
    # Pad out to nsv with low-amplitude noise components so the browser has something to page.
    u_noise = rng.standard_normal((ypix, xpix, nsv - u.shape[2])).astype(np.float32) * 0.02
    v_noise = rng.standard_normal((nsv - v.shape[0], nframes)).astype(np.float32) * 0.05
    u = np.ascontiguousarray(np.concatenate([u, u_noise], axis=2))
    v = np.ascontiguousarray(np.concatenate([v, v_noise], axis=0))
    del movie

    sv = np.array([np.var(row) * np.sum(u[..., i] ** 2) for i, row in enumerate(v)])
    order = np.argsort(sv)[::-1]
    return u[..., order], v[order], t, fs, sv[order], float(sv.sum())


# --------------------------------------------------------------------------- layout


def tile(windows, app):
    """Lay the windows out on the primary screen: one fills it, two split it, more go 2x2."""
    geo = app.primaryScreen().availableGeometry()
    n = len(windows)
    cols = 1 if n == 1 else 2
    rows = 1 if n <= 2 else 2
    w, h = geo.width() // cols, geo.height() // rows
    for i, win in enumerate(windows):
        col, row = i % cols, i // cols
        win.resize(w - 20, h - 40)
        win.move(geo.x() + col * w + 10, geo.y() + row * h + 10)
        win.show()
        win.raise_()


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=DEFAULT_SESSION, help="session folder on the server")
    ap.add_argument("--demo", action="store_true", help="synthetic data; no server needed")
    ap.add_argument("--nsv", type=int, default=200, help="components to load (default 200)")
    ap.add_argument("--calc-win", type=float, nargs=2, default=(-0.3, 0.8), metavar=("T0", "T1"))
    ap.add_argument(
        "--no-opengl",
        action="store_true",
        help="disable OpenGL rendering in the movie viewer (it is on by default, and is worth "
        "about a third of the frame rate on a large window)",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        choices=["svd", "corr", "tuning", "movie"],
        help="open only these viewers (default: all four)",
    )
    ap.add_argument(
        "--no-exec",
        action="store_true",
        help="build everything and exit without entering the event loop (smoke test)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(0)
    app = ensure_app()

    if args.demo:
        u, v, t, fs, sv, total_var = demo_data()
        dff_u, dff_v = u, v  # already dF/F-scaled
        session = None
        extra_traces = []
        log.info("demo data: %s px, %d components, %d frames", u.shape[:2], v.shape[0], v.shape[1])
    else:
        session = Path(args.session)
        log.info("loading %s (nsv=%d) ...", session, args.nsv)
        corr = wf.load_uvt(session, nsv=args.nsv, channel="corr")
        u, v, t, fs = corr.u, corr.v, corr.t, corr.fs
        log.info(
            "  %s px, %d components, %d frames, fs=%.3f Hz", u.shape[:2], v.shape[0], v.shape[1], fs
        )

        sv, total_var = load_sv(session, v.shape[0])
        if sv is None:
            sv, total_var = v.var(axis=1), None

        # The movie viewer's default color scale assumes dF/F, so give it dF/F. Skipped
        # entirely when the movie viewer is not being opened — it is the expensive step.
        dff_u = dff_v = t_movie = None
        if args.only is None or "movie" in args.only:
            blue = wf.load_uvt(session, nsv=args.nsv, channel="blue")
            log.info("  computing dF/F ...")
            dff_u, dff_v = wf.dff_from_svd(blue.u, blue.v, blue.mean_image)
            # float32 halves the resident size; on a 92-minute session dff_v is 311 MB in
            # float64 and the viewer only ever displays it.
            dff_v = np.asarray(dff_v, dtype=np.float32)
            t_movie = blue.t

        extra_traces = []
        for name, label in (("rotaryEncoder", "wheel"), ("photodiode", "photodiode")):
            got = load_timeline_trace(session, name)
            if got is not None:
                # Decimate: these run at ~2 kHz for 11 minutes and only ~10 s is ever on screen.
                tt, vv = got
                extra_traces.append(Trace(t=tt[::10], v=vv[::10], name=label))
        laser = laser_trace(session)
        if laser is not None:
            lt, lv, n_pulses = laser
            extra_traces.append(Trace(t=lt, v=lv, name=f"laser ({n_pulses} pulses)"))
            log.info("  opto session: %d laser pulses", n_pulses)
        log.info("  behavioral traces: %s", [tr.name for tr in extra_traces] or "none found")

    wanted = set(args.only) if args.only else {"svd", "corr", "tuning", "movie"}

    # Only the tuning viewer needs events, and finding them reads a ~90 MB photodiode trace.
    event_times = event_labels = None
    event_source = "not needed"
    if "tuning" in wanted:
        event_times, event_labels, event_source = make_events(session, t, rng)
        log.info("  tuning events: %s", event_source)

    windows = []

    if "svd" in wanted:
        log.info("building SVD component browser ...")
        w_svd = svd_class()(u, sv, v, fs=fs, total_variance=total_var, session=session)
        w_svd.setWindowTitle("1. " + w_svd.windowTitle())
        windows.append(w_svd)

    if "corr" in wanted:
        log.info("building correlation viewer (precomputing covariance) ...")
        w_corr = corr_class()(u, v, t=t, session=session)
        w_corr.setWindowTitle("2. " + w_corr.windowTitle())
        windows.append(w_corr)

    if "tuning" in wanted:
        log.info("building tuning viewer (event-locked average) ...")
        w_tun = tuning_class()(
            u, v, t, event_times, event_labels, tuple(args.calc_win), session=session
        )
        w_tun.setWindowTitle(f"3. {w_tun.windowTitle()}  [{event_source}]")
        windows.append(w_tun)

    if "movie" in wanted:
        log.info("building movie viewer ...")
        # Take the reference trace from the arrays actually displayed (blue dF/F), not from the
        # corr channel used by the other viewers — otherwise the trace and the movie are different
        # data on slightly different clocks, which is needlessly confusing.
        t_disp = t if args.demo else t_movie
        centre = (dff_u.shape[0] // 2, dff_u.shape[1] // 2)
        pixel_trace = wf.pixel_timecourse(dff_u, dff_v, centre)
        traces = [
            Trace(t=t_disp, v=pixel_trace, name=f"center pixel {centre} dF/F"),
            *extra_traces,
        ]
        w_movie = movie_class()(
            dff_u,
            dff_v,
            t=t_disp,
            traces=traces,
            use_opengl=not args.no_opengl,
            session=session,
        )
        w_movie.setWindowTitle("4. " + w_movie.windowTitle())
        windows.append(w_movie)

    tile(windows, app)

    TIPS = {
        "svd": ["SVD viewer      left/right to page components; p then click for pixel mode"],
        "corr": [
            "Correlation     just move the mouse (hover is on); h toggles it; v for",
            "                variance normalization; try a 1 Hz high-pass",
        ],
        "tuning": [
            "Tuning          click any of the four panels; m for median instead of mean;",
            "                click a single trial (bottom right) to send just that trial",
            "                to the brain panel; [ ] step trials; esc back to the average",
        ],
        "movie": [
            "Movie           p to play; ctrl+click to add pixels; -/= color scale;",
            "                type band-pass cutoffs in Hz; Follow keeps a zoom on playback",
        ],
    }
    log.info("")
    log.info(
        "%d window%s open. Things worth trying:", len(windows), "" if len(windows) == 1 else "s"
    )
    for key in ("svd", "corr", "tuning", "movie"):
        if key in wanted:
            for line in TIPS[key]:
                log.info("  %s", line)
    log.info("  %s", "any             alt+arrows to rotate/flip; hotkeys work with any focus")
    log.info("")
    log.info("Close the window%s to exit.", "" if len(windows) == 1 else "s")
    if args.no_exec:
        log.info("(--no-exec: built OK, exiting without showing anything)")
        return 0
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
