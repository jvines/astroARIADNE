"""SPHEREx DR1 per-star spectro-photometry for the librarian.

Extracts a 0.75-5 um spectrum from the public DR1 spectral images, or from
QR3 where DR1 has no planes (see RELEASES), one LVF wavelength per
overlapping exposure, and bins it into ARIADNE's SPHEREX_* super-bands.
Validated on QR2 against 2MASS/WISE on GJ 1214 and TRAPPIST-1 (band ratios
0.94-1.07; see notebooks/spherex_extract.py for the standalone validation
tool); IRSA retired the QR2 SIA collection in Nov 2026.

Method per exposure: 3-px aperture + 10-16 px background annulus on the IBE
cutout (subpixel-weighted, numpy-only -- the SPHEREx PSF holds 99.6% of its
energy inside 3 px, zone spread 0.2%, so no aperture correction applies);
MJy/sr -> uJy via the 6.2" pixel solid angle; wavelength from the
science-grade per-detector CWAVE array at the star's RAW detector pixel (NOT
the L2 WCS-WAVE table, which the mission designates visualization-only -- see
_cwave). Exposures with any bad flag inside the aperture are dropped (bits
1 = OVERFLOW per the official SDT guidance and 21 = benign frame-wide marker
are ignored), then a 4-sigma MAD clip per wavelength bin.

Binned results are cached per source id under
~/.astroARIADNE/spherex/<CACHE_VERSION>/ so a star is only ever extracted once.
"""
import logging
import os
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


def _ssl_context():
    """TLS context for every IRSA / S3 request in this module.

    certifi's bundle when importable: on 2026-11-12 IRSA moved to an SSL.com chain
    that older system bundles (macOS /etc/ssl/cert.pem) lack, and a uv-managed
    Python verifies against the system bundle -- every IRSA read then failed over
    to S3 and the calibration download failed outright."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()

OMEGA = (6.2 * 4.84813681e-6) ** 2          # sr per 6.2" SPHEREx pixel
AP_R, ANN_IN, ANN_OUT = 3.0, 10.0, 16.0
IGNORE_BITS = (1, 21)
C_UM = 2.99792458e14
# Bumped whenever the extraction changes NUMERICALLY, so two solutions can never
# be mixed inside one corpus. v1 (unversioned, ~/.astroARIADNE/spherex/*.csv) was
# written with the celestial SIP leaking into the wavelength lookup (see
# _alt_wcs): its lambda column is biased ~0.005 um blueward. v2 fixed that but
# still used the visualization-grade WCS-WAVE table; v3 takes lambda from the
# science-grade per-detector CWAVE product (_cwave) -- bump again if
# CAL_WCS_VERSION changes, it sets the wavelengths. v4 moved from QR2 to DR1/QR3
# data and the cal-swcs-v5 CWAVE (+0.006 um against v4 at 4.7 um). Old versions
# are left in place -- ignored, not deleted -- so their results stay reproducible.
CACHE_VERSION = 'v4'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'spherex',
                         CACHE_VERSION)
# Withheld until the raw grid reconvolution covers them at warm nodes (the
# cache-derived columns only reach 4.63 um for most grids).
WITHHELD = ('SPHEREX_4650', 'SPHEREX_4900')
# Usable point-source window (Vega Ks): saturation flags begin ~9 Vega,
# all-sky 5-sigma depth ~16.5 (mission paper, 2 um).
KS_BRIGHT, KS_FAINT = 8.5, 16.5
MIN_EXP_PER_BAND = 3
ABS_CAL_FLOOR = 0.02                        # fractional


def _superband_windows():
    """SPHEREX_* -> (lo_um, hi_um) from the filter library (single source of
    truth: custom_filters.h5 via phot_utils)."""
    from ..phot_utils import _CUSTOM_LIB
    out = {}
    for name, filt in _CUSTOM_LIB.items():
        if not name.startswith('SPHEREX_') or name in WITHHELD:
            continue
        wl = filt.wavelength.to('um').value
        tr = np.asarray(filt.transmit, float)
        on = wl[tr > 0.5]
        out[name] = (float(on.min()), float(on.max()))
    return out


def _aperture_sum(img, x, y, r, oversamp=5):
    """Subpixel-weighted circular aperture sum (numpy only). Accurate to
    <<1% at r=3 px with 5x oversampling; photutils-free by design (the
    ariadne env pins numpy)."""
    ny, nx = img.shape
    x0, x1 = int(max(0, np.floor(x - r - 1))), int(min(nx, np.ceil(x + r + 2)))
    y0, y1 = int(max(0, np.floor(y - r - 1))), int(min(ny, np.ceil(y + r + 2)))
    sub = (np.arange(oversamp) + 0.5) / oversamp - 0.5
    xs = np.arange(x0, x1)[:, None] + sub[None, :]      # (npx, os)
    ys = np.arange(y0, y1)[:, None] + sub[None, :]
    dx2 = (xs - x) ** 2                                  # (npx, os)
    dy2 = (ys - y) ** 2
    inside = (dy2[:, None, :, None] + dx2[None, :, None, :]) <= r * r
    frac = inside.mean(axis=(2, 3))                      # (npy, npx)
    return float(np.sum(img[y0:y1, x0:x1] * frac)), frac.sum()


def _annulus_median(img, x, y, r_in, r_out):
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - x, yy - y)
    # finite-only: a single NaN pixel (detector edge / unobserved region)
    # otherwise poisons np.median and silently drops the whole exposure --
    # this killed the SPHEREX_2750/3050 super-bands (the 2MASS-WISE gap)
    # for SCR 1845 despite 4 good exposures per window.
    m = (rr >= r_in) & (rr <= r_out) & np.isfinite(img)
    if m.sum() < 10:
        return np.nan
    return float(np.median(img[m]))


# SPHEREx L2 headers carry SIP coefficients for the CELESTIAL solution (CTYPE
# 'RA---TAN-SIP') alongside two lettered solutions we need: 'A' ("0-based active
# pixel coordinates" -- cutout pixel -> raw detector pixel) and 'W' (the
# visualization-grade WAVE-TAB lookup). astropy's SIP reader is not
# alternate-key aware -- it reads the unkeyed A_/B_ coefficients whatever `key`
# is asked for -- so building EITHER off the same header picks the celestial
# distortion up and (a) logs a long "Inconsistent SIP distortion" record per
# exposure (astropy.wcs, via log.info: a log record, not a warning, so no
# warnings filter silences it; ~150-230 exposures per star) and (b) applies the
# polynomial to a pixel frame it has no business touching. Both solutions are
# defined on RAW detector pixels -- where the photon physically lands -- so the
# distortion must not be applied: on real QR2 cutouts the leak displaced the
# wavelength by a median -0.005 um (max 0.014), ~11% of a SPHEREx resolution
# element, systematically blueward.
# This is IRSA's own prescription -- "the wavelength per pixel is a property of
# the detector-filter combination and is independent of optical distortion in
# the telescope [...] so we turn the SIP distortion off for this WCS"
# (irsa-tutorials spherex-intro / spherex-cutouts, which do
# `spectral_wcs.sip = None`); the open astropy issue is #13105. We drop the
# keywords from a header copy rather than nulling `.sip` afterwards: the INFO is
# logged inside WCS.__init__, so post-hoc nulling fixes the numbers but still
# prints, and silencing it would mean muting astropy's logger.
_SIP_KW = re.compile(r'^(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)$')


def _alt_wcs(hdr, hdul, key):
    """An alternate-key WCS off a SPHEREx header, with the celestial SIP
    stripped. Both lettered solutions we use need this -- 'W' (WAVE-TAB) and
    'A' ("0-based active pixel coordinates")."""
    from astropy.wcs import WCS
    clean = hdr.copy()
    # Dedup the keywords first: `del header[kw]` drops EVERY card with that name
    # and then raises KeyError, so a duplicated SIP card would blow up on the
    # second pass -- and _fetch_one's bare except would swallow it, costing the
    # star every exposure with no log line. (astropy's own _read_sip_kw dedups
    # through a set for exactly this reason.)
    for k in {m.group() for m in map(_SIP_KW.match, list(clean)) if m}:
        del clean[k]
    return WCS(clean, fobj=hdul, key=key)


def _raw_pixel(hdr, hdul, x, y):
    """Cutout pixel -> 0-based RAW detector pixel, via the header's own 'A'
    solution (WCSNAMEA = "0-based active pixel coordinates"). IRSA re-references
    it to the cutout origin, so this is the mission's declared mapping onto the
    2040^2 detector grid the calibration arrays are indexed on."""
    return (float(v) for v in
            _alt_wcs(hdr, hdul, 'A').wcs_pix2world(np.array([[x, y]]), 0)[0])


# --------------------------------------------------------------------------- #
# Science-grade wavelength: the per-detector CWAVE calibration array
# --------------------------------------------------------------------------- #
# The WCS-WAVE table carried by each L2 file is explicitly NOT for science:
# "The fidelity of the WCS-WAVE lookup table is intended for visualization
# purposes. For science analysis, use the CWAVE and CBAND extensions in the
# Spectral WCS calibration product" (IRSA SPHEREx data-products page; ExpSupp
# v1.6 4.3.1). So we index CWAVE at the star's raw detector pixel instead.
# One 33 MB product per detector, six detectors, fetched once and reused.
# DR1 and QR3 ship the same product (identical bytes under dr1/ and qr3/).
CAL_WCS_VERSION = 'cal-swcs-v5-2026-191'
CAL_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'spherex',
                       'cal', CAL_WCS_VERSION)
_CAL_NAME = 'spectral_wcs_D{det}_spx_' + CAL_WCS_VERSION + '.fits'
_CAL_URL = ('https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/spectral_wcs/'
            + CAL_WCS_VERSION + '/{det}/' + _CAL_NAME)
_CAL_TIMEOUT = 300      # s, 33 MB


class CalibrationUnavailable(RuntimeError):
    """The detector's CWAVE product could not be fetched.

    Raised rather than swallowed: falling back to WAVE-TAB would silently mix
    two wavelength solutions in one corpus, and returning "no exposures" would
    cache an empty result and mark the star as having no SPHEREx data forever.
    """


_CAL_LOCKS = {}
_CAL_LOCKS_GUARD = None


def _cal_lock(det):
    """One lock per detector, so N workers hitting a cold cache download the
    33 MB product once rather than N times."""
    global _CAL_LOCKS_GUARD
    import threading
    if _CAL_LOCKS_GUARD is None:
        _CAL_LOCKS_GUARD = threading.Lock()
    with _CAL_LOCKS_GUARD:
        return _CAL_LOCKS.setdefault(det, threading.Lock())


@lru_cache(maxsize=None)
def _cwave(det):
    """Per-pixel central wavelength (um) for one detector, 2040x2040.

    Cached on disk under CAL_DIR and in memory per process. The download is
    serialised per detector and staged through a per-writer temp file, then
    renamed: concurrent first touches cost one download, a reader never sees a
    partial product, and a second writer cannot pull the temp file out from
    under a first (which it could when the temp name was keyed on the pid
    alone, since threads share it).
    """
    from astropy.io import fits
    import threading
    det = int(det)
    path = os.path.join(CAL_DIR, _CAL_NAME.format(det=det))
    if not os.path.exists(path):
        with _cal_lock(det):
            if not os.path.exists(path):             # another worker may have won
                os.makedirs(CAL_DIR, exist_ok=True)
                tmp = '%s.%d.%d.part' % (path, os.getpid(), threading.get_ident())
                try:
                    with urllib.request.urlopen(_CAL_URL.format(det=det),
                                                timeout=_CAL_TIMEOUT,
                                                context=_SSL_CTX) as r, \
                            open(tmp, 'wb') as fh:
                        shutil.copyfileobj(r, fh)
                    os.replace(tmp, path)
                except Exception as e:               # noqa: BLE001
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise CalibrationUnavailable(
                        'SPHEREx spectral-WCS product for detector %d (%s): %s: %s'
                        % (det, CAL_WCS_VERSION, type(e).__name__, e)) from e
    try:
        with fits.open(path) as h:
            return np.asarray(h['CWAVE'].data, np.float32)
    except Exception as e:                               # noqa: BLE001
        raise CalibrationUnavailable(
            'unreadable CWAVE product %s: %s: %s'
            % (path, type(e).__name__, e)) from e


def _bilinear(a, x, y):
    """a[y, x] at fractional 0-based indices, or NaN off the grid.

    Half a pixel of overhang is clamped rather than rejected: the first and last
    pixel CENTRES sit at 0 and n-1, so a source legitimately on an edge row can
    land a hair outside (a star on detector row 0 comes back as -2e-12), and
    NaN there would silently drop the exposure."""
    ny, nx = a.shape
    if not (-0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5):
        return float('nan')
    x = min(max(x, 0.0), nx - 1.0)
    y = min(max(y, 0.0), ny - 1.0)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    fx, fy = x - x0, y - y0
    # CWAVE ships float32; weight in float64 so the result does not depend on
    # the array's storage dtype.
    c00, c01 = float(a[y0, x0]), float(a[y0, x1])
    c10, c11 = float(a[y1, x0]), float(a[y1, x1])
    return (c00 * (1 - fx) * (1 - fy) + c01 * fx * (1 - fy) +
            c10 * (1 - fx) * fy + c11 * fx * fy)


def _measure_arrays(img, var, flg, x, y, lam):
    """The photometry itself, on pixels already in hand.

    Split out of `_measure_cutout` so the local-file path and the ranged
    remote path (`_measure_remote`) run the SAME science code -- the two
    are asserted bit-identical in the tests. `x`/`y` are the star's position
    within the given window; `lam` its CWAVE wavelength.
    """
    ny, nx = img.shape
    if not (ANN_OUT <= x <= nx - ANN_OUT - 1 and
            ANN_OUT <= y <= ny - ANN_OUT - 1):
        return None
    ignore = sum(1 << b for b in IGNORE_BITS)
    for b in range(32):
        if b not in IGNORE_BITS and (flg >> b & 1).mean() > 0.99:
            ignore |= 1 << b                 # auto-detected universal marker
    yy, xx = np.mgrid[0:ny, 0:nx]
    near = np.hypot(xx - x, yy - y) <= AP_R + 0.5
    if ((flg & ~ignore)[near] != 0).any():
        return None
    bkg = _annulus_median(img, x, y, ANN_IN, ANN_OUT)
    if not np.isfinite(bkg):
        return None
    s, area = _aperture_sum(img, x, y, AP_R)
    v, _ = _aperture_sum(var, x, y, AP_R)
    f = (s - bkg * area) * OMEGA * 1e12      # MJy/sr -> uJy
    e = np.sqrt(max(v, 0.0)) * OMEGA * 1e12
    if not (np.isfinite(f) and f > 0):
        return None
    if not np.isfinite(lam):
        # A star off the calibrated grid yields NaN rather than an exception. A
        # NaN lambda used to be cached and then matched no super-band and no MAD
        # bin -- a silent drop. Reject it so the exposure is accounted for.
        return None
    return lam, f, e


# --------------------------------------------------------------------------- #
# Crowded fields: forced ePSF photometry of the target and its neighbours
# --------------------------------------------------------------------------- #
# The 3-px aperture (18.6") collects every source inside it. Over the ARIADNE II
# M-dwarf survey (1288 stars, mostly low-latitude Sco-Cen fields) the
# SPHEREx-minus-2MASS offset is predicted star by star by the 2MASS sources in
# the aperture (correlation 0.84; 93% of the >0.5 mag offsets): the photometry
# is right for the aperture, but the aperture is not the star. Where catalogued
# neighbours are predicted to put more than CONTAM_TRIGGER of the target's flux
# into the aperture, each exposure is instead fitted with the mission's R7 ePSF
# at the target AND neighbour positions (fixed), amplitudes and a constant local
# background free, inverse-variance weighted -- IRSA's own forced-photometry
# design (Explanatory Supplement v2.0, 3.2). Everywhere else the aperture
# result stands, bit-identical to before. The ePSF already includes the pixel
# response, so it is sampled directly at native pixel-centre offsets; the archive
# recommends it for QR2 images too ("safe to just exchange the PSFs").
EPSF_VERSION = '2026-191'
EPSF_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'spherex',
                        'cal', 'epsf')
_EPSF_NAME = 'epsf_D{det}_spx_cal-epsf-{v}-' + EPSF_VERSION + '.fits'
_EPSF_URL = ('https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com/qr3/epsf/'
             'cal-epsf-{v}-' + EPSF_VERSION + '/{det}/' + _EPSF_NAME)
CONTAM_TRIGGER = 0.02     # predicted neighbour flux / target flux in the aperture
NB_RADIUS = 111.0         # arcsec: catalogue cone, out to the background annulus (+2 px)
ANN_TRIGGER = 5.0         # 2MASS flux in the background annulus / target flux
TMASS_EPOCH = 2000.0      # mean 2MASS epoch, for matching 2MASS to PM-propagated Gaia
R_FIT = 4.0               # native px: pixels fitted around the target
MERGE_PX = 0.5            # neighbours closer than this cannot be separated from the target
NB_MIN_FRAC = 0.01        # a neighbour is fitted only if >1% of the target's flux reaches the fit region
NB_MAX = 20               # at most this many fitted neighbours (brightest contributions first)
CORE_PX = 2.0             # each fitted neighbour's core (px) joins the fit pixels
NB_SIGMA = 3.0            # arcsec: Gaussian edge of the aperture in the contamination prediction
PSF_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'spherex',
                             'psf-v2')   # v2: DR1/QR3 data, cal-swcs-v5 (see CACHE_VERSION)


def _epsf_version(det):
    return 'v2' if int(det) == 3 else 'v1'   # detector 3 was re-zoned (11 x 41)


@lru_cache(maxsize=None)
def _epsf(det):
    """(xcenter, ycenter, cube, oversampling) of the detector's R7 ePSF zones.

    One ~8 MB product per detector, fetched once (serialised like the CWAVE
    product) and cached on disk under EPSF_DIR."""
    from astropy.io import fits
    import threading
    det = int(det)
    name = _EPSF_NAME.format(det=det, v=_epsf_version(det))
    path = os.path.join(EPSF_DIR, name)
    if not os.path.exists(path):
        with _cal_lock(('epsf', det)):
            if not os.path.exists(path):
                os.makedirs(EPSF_DIR, exist_ok=True)
                tmp = '%s.%d.%d.part' % (path, os.getpid(), threading.get_ident())
                try:
                    with urllib.request.urlopen(
                            _EPSF_URL.format(det=det, v=_epsf_version(det)),
                            timeout=_CAL_TIMEOUT, context=_SSL_CTX) as r, open(tmp, 'wb') as fh:
                        shutil.copyfileobj(r, fh)
                    os.replace(tmp, path)
                except Exception as e:               # noqa: BLE001
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise CalibrationUnavailable(
                        'SPHEREx ePSF product for detector %d: %s: %s'
                        % (det, type(e).__name__, e)) from e
    with fits.open(path) as h:
        t = h['EPSF']
        ovs = int(t.header.get('OVSMPX', 5))
        return (np.asarray(t.data['XCENTER'], float), np.asarray(t.data['YCENTER'], float),
                np.asarray(t.data['EPSF'], float), ovs)


def _epsf_pix(det, xa, ya, dx, dy):
    """Fraction of a unit-flux source falling in native pixels at pixel-centre
    offsets (dx, dy) from it, for a source on raw-detector position (xa, ya):
    nearest ePSF zone, bilinear on the oversampled grid. The oversampled array
    is normalised to unit sum, so native-pixel samples carry ovs**2 (checked:
    native samples sum to 0.0399 = 1/25 at ovs 5, and the rendered N_eff
    reproduces the table's NEFF_MEAN to 1-4%)."""
    from scipy.ndimage import map_coordinates
    xc, yc, cube, ovs = _epsf(det)
    k = int(np.argmin((xc - xa) ** 2 + (yc - ya) ** 2))
    c = (cube.shape[-1] - 1) / 2.0
    return ovs * ovs * map_coordinates(cube[k], [c + np.asarray(dy) * ovs, c + np.asarray(dx) * ovs],
                                       order=1, mode='constant', cval=0.0)


def _epsf_reach(det):
    """Half-extent of the ePSF stamp in native pixels."""
    _, _, cube, ovs = _epsf(det)
    return (cube.shape[-1] - 1) / 2.0 / ovs


def _psf_flux(img, var, flg, x, y, xa, ya, det, others):
    """Forced ePSF photometry of the target at (x, y) in the window, with the
    neighbours `others` [(x, y), ...] (window pixels) fitted simultaneously.

    Positions fixed; one amplitude per source plus a constant background,
    weighted by 1/VARIANCE over unflagged pixels within R_FIT of the target.
    The reject rule is the aperture path's (any bad flag within AP_R + 0.5 px
    drops the exposure), so both paths keep the same exposures. Returns
    (flux_uJy, err_uJy) with the error scaled by sqrt(reduced chi2) when the
    model fits worse than the variance says, or None."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ignore = sum(1 << b for b in IGNORE_BITS)
    for b in range(32):
        if b not in IGNORE_BITS and (flg >> b & 1).mean() > 0.99:
            ignore |= 1 << b
    rr = np.hypot(xx - x, yy - y)
    if ((flg & ~ignore)[rr <= AP_R + 0.5] != 0).any():
        return None
    # Neighbours closer than MERGE_PX to an already-kept source are folded in
    # (two PSFs at one position are degenerate); each kept neighbour's core
    # joins the fit pixels, so its amplitude is measured, not guessed from wings.
    srcs = [(x, y)]
    for sx, sy in others:
        if all(np.hypot(sx - px, sy - py) >= MERGE_PX for px, py in srcs):
            srcs.append((sx, sy))
    region = rr <= R_FIT
    for sx, sy in srcs[1:]:
        region |= np.hypot(xx - sx, yy - sy) <= CORE_PX
    m = region & np.isfinite(img) & np.isfinite(var) & (var > 0) \
        & ((flg & ~ignore) == 0)
    if m.sum() < 15 + len(srcs):
        return None
    cols = [_epsf_pix(det, xa, ya, xx[m] - sx, yy[m] - sy) for sx, sy in srcs]
    A = np.column_stack(cols + [np.ones(int(m.sum()))])
    sw = 1.0 / np.sqrt(var[m])
    Aw, bw = A * sw[:, None], img[m] * sw
    try:
        coef = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        cov = np.linalg.inv(Aw.T @ Aw)
    except np.linalg.LinAlgError:
        return None
    dof = max(int(m.sum()) - A.shape[1], 1)
    chi2 = float(np.sum((bw - Aw @ coef) ** 2) / dof)
    f = coef[0] * OMEGA * 1e12
    e = np.sqrt(max(cov[0, 0], 0.0) * max(chi2, 1.0)) * OMEGA * 1e12
    if not (np.isfinite(f) and f > 0 and np.isfinite(e)):
        return None
    return f, e


def neighbours(ra, dec, source_id, epoch=2025.9, radius=NB_RADIUS):
    """Catalogued sources around the target, cached per source id.

    Gaia DR3 propagated to `epoch` plus 2MASS PSC sources without a Gaia
    counterpart within 2". Each entry is {ra, dec, sep (arcsec), ks}, with ks the
    2MASS Ks of a matched source or NaN. The target itself (nearest source
    within 2" of ra/dec) is excluded, and sources within MERGE_PX of it are
    dropped as unresolvable (they stay in the target's flux, as they would in
    any photometry at SPHEREx resolution) -- the count is kept under
    'n_merged'. Each entry also carries Gaia G (NaN for 2MASS-only sources).
    Returns (list, n_merged)."""
    import json
    cdir = os.path.join(os.path.dirname(CACHE_DIR), 'neighbours-v3')
    os.makedirs(cdir, exist_ok=True)
    cache = os.path.join(cdir, f'{source_id}.json')
    if os.path.exists(cache):
        d = json.load(open(cache))
        return d['neighbours'], d['n_merged']
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from ._cds import MirroredVizier as Vizier   # CDS outages: VizieR mirror
    c = SkyCoord(ra * u.deg, dec * u.deg)
    g = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Gmag'], row_limit=-1).query_region(
        c, radius=radius * u.arcsec, catalog='I/355/gaiadr3')
    t = Vizier(columns=['RAJ2000', 'DEJ2000', 'Kmag'], row_limit=-1).query_region(
        c, radius=radius * u.arcsec, catalog='II/246/out')
    src, at2mass = [], []
    if len(g):
        dt = epoch - 2016.0
        for r in g[0]:
            pmra = float(r['pmRA']) if np.isfinite(r['pmRA']) else 0.0
            pmde = float(r['pmDE']) if np.isfinite(r['pmDE']) else 0.0
            ra0, de = float(r['RA_ICRS']), float(r['DE_ICRS'])
            cosd = np.cos(np.radians(de))
            src.append([ra0 + pmra * dt / 3.6e6 / cosd, de + pmde * dt / 3.6e6, np.nan,
                        float(r['Gmag']) if np.isfinite(r['Gmag']) else np.nan])
            # 2MASS positions are of epoch ~2000: match there, or a fast mover
            # (VB 10: 1.4"/yr) meets its own 2MASS entry as a phantom neighbour.
            at2mass.append((ra0 + pmra * (TMASS_EPOCH - 2016.0) / 3.6e6 / cosd,
                            de + pmde * (TMASS_EPOCH - 2016.0) / 3.6e6))
    tm = [] if not len(t) else [(float(r['RAJ2000']), float(r['DEJ2000']),
                                 float(r['Kmag']) if np.isfinite(r['Kmag']) else np.nan) for r in t[0]]
    if src and tm:
        cs = SkyCoord([a[0] for a in at2mass] * u.deg, [a[1] for a in at2mass] * u.deg)
        for tra, tde, tk in tm:
            sep = SkyCoord(tra * u.deg, tde * u.deg).separation(cs).arcsec
            i = int(np.argmin(sep))
            if sep[i] < 2.0:
                src[i][2] = tk
            else:
                src.append([tra, tde, tk, np.nan])
    elif tm:
        src = [list(x) + [np.nan] for x in tm]
    out, n_merged = [], 0
    if src:
        seps = SkyCoord([s[0] for s in src] * u.deg, [s[1] for s in src] * u.deg).separation(c).arcsec
        it = int(np.argmin(seps)) if seps.min() < 2.0 else -1
        for i, (s, sp) in enumerate(zip(src, seps)):
            if i == it:
                continue
            if sp < MERGE_PX * 6.15:
                n_merged += 1
                continue
            out.append(dict(ra=s[0], dec=s[1], sep=float(sp), ks=s[2], g=s[3]))
    json.dump(dict(neighbours=out, n_merged=n_merged), open(cache, 'w'))
    return out, n_merged


def fit_neighbours(nbrs, target_ks):
    """The neighbours worth a free amplitude in the forced fit: those sending
    more than NB_MIN_FRAC of the target's flux into its fit region (R_FIT),
    brightest first, at most NB_MAX. Brightness is 2MASS Ks, or Gaia G - 2.5 for
    sources 2MASS does not resolve (a typical G-Ks of field K/M stars). The rest
    are faint enough for the local background to absorb. In the densest fields
    a 45" cone holds hundreds of catalogued sources; fitting all of them against
    ~50 pixels is underdetermined and the target flux comes out unphysical."""
    from scipy.special import erf
    r_fit = R_FIT * 6.15
    scored = []
    for n in nbrs:
        k = n['ks'] if np.isfinite(n['ks']) else (n.get('g', np.nan) - 2.5)
        if not np.isfinite(k):
            continue
        w = 0.5 * (1.0 + erf((r_fit - n['sep']) / (np.sqrt(2.0) * NB_SIGMA)))
        fr = w * 10 ** (-0.4 * (k - target_ks))
        if fr > NB_MIN_FRAC:
            scored.append((fr, n))
    scored.sort(key=lambda t: -t[0])
    return [n for _, n in scored[:NB_MAX]]


def annulus_flux(nbrs, target_ks):
    """2MASS Ks flux of catalogued sources in the background annulus (+-2 px),
    relative to the target's. A crowded annulus raises the median background
    and the aperture is over-subtracted: over the survey's aperture-clean stars
    the SPHEREx-2MASS offset grows from +0.05 (below 5x) to +0.08 (5-10x),
    +0.12 (10-20x) and +0.15 (>20x), and the forced ePSF fit, which fits its
    background locally with the neighbours modelled, removes it."""
    lo, hi = (ANN_IN - 2) * 6.15, (ANN_OUT + 2) * 6.15
    return float(sum(10 ** (-0.4 * (n['ks'] - target_ks)) for n in nbrs
                     if np.isfinite(n['ks']) and lo <= n['sep'] <= hi))


def predicted_contamination(nbrs, target_ks):
    """Predicted flux of catalogued neighbours inside the 3-px aperture, as a
    fraction of the target's (2MASS Ks; neighbours without Ks are not counted).
    The aperture edge is softened by a Gaussian of NB_SIGMA: calibrated on the
    survey, it tracks the measured SPHEREx-2MASS offsets best (corr 0.84)."""
    from scipy.special import erf
    ap = AP_R * 6.2
    frac = 0.0
    for n in nbrs:
        if np.isfinite(n['ks']):
            w = 0.5 * (1.0 + erf((ap - n['sep']) / (np.sqrt(2.0) * NB_SIGMA)))
            frac += w * 10 ** (-0.4 * (n['ks'] - target_ks))
    return float(frac)


def _star_in_window(hdr, hdul, ra, dec):
    """`(x, y, xa, ya, det)`: the star's position in the window this header
    describes, and on the raw detector grid."""
    from astropy.wcs import WCS
    x, y = (float(v) for v in WCS(hdr).world_to_pixel_values(ra, dec))
    xa, ya = _raw_pixel(hdr, hdul, x, y)
    return x, y, xa, ya, int(hdr["DETECTOR"])


def _measure_cutout(path, ra, dec, nbrs=None):
    """One exposure -> (wavelength_um, flux_uJy, err_uJy) or None.

    The local-file path: reads a whole cutout off disk. Retained as the
    reference implementation the ranged path is validated against, and as the
    fallback when `USE_RANGED_READS` is off. With `nbrs` (a crowded field) the
    flux comes from the forced ePSF fit instead of the aperture."""
    from astropy.io import fits
    with fits.open(path) as h:
        img = np.asarray(h["IMAGE"].data, float)
        var = np.asarray(h["VARIANCE"].data, float)
        flg = np.asarray(h["FLAGS"].data, int)
        hdr = h["IMAGE"].header
        x, y, xa, ya, det = _star_in_window(hdr, h, ra, dec)
        others = None if nbrs is None else \
            [_star_in_window(hdr, h, n['ra'], n['dec'])[:2] for n in nbrs]
    lam = _bilinear(_cwave(det), xa, ya)
    if others is None:
        return _measure_arrays(img, var, flg, x, y, lam)
    return _psf_measure(img, var, flg, x, y, xa, ya, det, others, lam)


# Network timeouts: a single hung IRSA request must never block the star (and
# hence a whole bulk chunk -- the pool waits for ALL jobs to finish).
_CUTOUT_TIMEOUT = 60    # s, per cutout download (urlopen connect + each read)
_SIA_TIMEOUT = 120      # s, IRSA SIA query


# ── transport: ranged reads of the static L2 file ───────────────────────────
# The IRSA cutout service hands back a 5.09 MB MEF of which 4.93 MB is a PSF
# cube we never open, and it is the fragile part of the system: measured 503
# rates of 0% at 2 concurrent requests but 37.5% at 16, against 0% for the
# static file at the same concurrency. So we read the static L2 file directly
# and pull only the ~41 detector rows the aperture and annulus touch: 1.0 MB and
# 2 round trips per exposure instead of 5.09 MB, measured 4.1x faster and
# bit-identical (see tests).
#
# Two hosts serve the identical bytes, and the SIA access_url maps to the S3 key
# by a pure prefix swap. They differ in ONE way that matters: IRSA honours
# multi-range requests (206 multipart/byteranges), S3 silently IGNORES them and
# returns the WHOLE 71.6 MB object -- 14x worse than the cutout it replaces. So
# the multi-range shape is per-host, never assumed.
_L2_PREFIX = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
_L2_HOSTS = (
    ("irsa.ipac.caltech.edu", "/ibe/data/spherex/", True),
    ("nasa-irsa-spherex.s3.us-east-1.amazonaws.com", "/", False),
)
_BLOCK = 2880           # FITS block
_WIN = 39               # window side, = the 240 arcsec cutout at 6.15"/px
_FITS_DTYPE = {-64: ">f8", -32: ">f4", 64: ">i8", 32: ">i4", 16: ">i2", 8: "u1"}
_RANGE_TIMEOUT = 60
_RANGE_RETRIES = 3      # a swallowed 503 is a silently lost exposure


class _Reader:
    """Keep-alive ranged reader over one L2 host, with retry and host failover.

    NOT thread-safe (http.client connections are not): the pool gives each
    worker thread its own. Reuse matters -- a fresh connection costs a measured
    ~0.30 s of TCP+TLS, which at 2 requests per exposure would dominate.
    """

    def __init__(self, host_index=0):
        self._i = host_index
        self._conn = None

    @property
    def multirange(self):
        return _L2_HOSTS[self._i][2]

    def _connect(self):
        import http.client
        host = _L2_HOSTS[self._i][0]
        self._conn = http.client.HTTPSConnection(host, timeout=_RANGE_TIMEOUT,
                                                 context=_SSL_CTX)

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:                            # noqa: BLE001
                pass
            self._conn = None

    def get(self, key, ranges):
        """`ranges` = [(start, end), ...] inclusive. Returns [bytes, ...].

        Falls back to sequential single-range GETs on a host that does not do
        multipart, and to the next host after exhausting retries here.
        """
        import time
        last = None
        for host_try in range(len(_L2_HOSTS)):
            path = _L2_HOSTS[self._i][1] + key
            for attempt in range(_RANGE_RETRIES):
                try:
                    if self.multirange and len(ranges) > 1:
                        return self._multi(path, ranges)
                    return [self._one(path, a, b) for a, b in ranges]
                except Exception as e:                   # noqa: BLE001
                    last = e
                    self.close()
                    if attempt + 1 < _RANGE_RETRIES:
                        time.sleep(0.4 * (attempt + 1))
            self._i = (self._i + 1) % len(_L2_HOSTS)     # try the other host
            self.close()
        raise last

    def _request(self, path, hdr):
        if self._conn is None:
            self._connect()
        self._conn.request("GET", path, headers=hdr)
        r = self._conn.getresponse()
        body = r.read()
        if r.status not in (200, 206):
            raise OSError(f"HTTP {r.status} for {path}")
        return r, body

    def _one(self, path, a, b):
        r, body = self._request(path, {"Range": f"bytes={a}-{b}"})
        if r.status == 200 and len(body) > (b - a + 1) * 2:
            # the host ignored Range and sent the whole object
            raise OSError("Range ignored by host")
        return body

    def _multi(self, path, ranges):
        spec = ",".join(f"{a}-{b}" for a, b in ranges)
        r, body = self._request(path, {"Range": f"bytes={spec}"})
        ctype = r.getheader("Content-Type", "")
        if "multipart" not in ctype:
            raise OSError("multi-range not honoured")
        parts = _split_multipart(body, ctype)
        # Key each part by the offset in its OWN Content-Range: servers are not
        # required to return the parts in the order requested.
        return [parts[a] for a, _ in ranges]


def _split_multipart(body, ctype):
    """`{start_offset: payload}` from a multipart/byteranges body."""
    bound = ("--" + ctype.split("boundary=")[1].strip()).encode()
    out = {}
    for part in body.split(bound):
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep or b"Content-Range" not in head:
            continue
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-range"):
                start = int(line.split(b"bytes")[1].strip().split(b"-")[0])
                out[start] = data[:-2] if data.endswith(b"\r\n") else data
    return out


def _data_span(h):
    """Bytes of data (padded to whole FITS blocks) following a header."""
    if h.get("NAXIS", 0) == 0:
        return 0
    n = abs(h["BITPIX"]) // 8
    for i in range(1, h["NAXIS"] + 1):
        n *= h[f"NAXIS{i}"]
    n += h.get("PCOUNT", 0)
    return ((n + _BLOCK - 1) // _BLOCK) * _BLOCK


def _header_at(reader, key, off, nblk=12):
    """Parse the FITS header starting at `off`; returns (Header, data_offset)."""
    from astropy.io import fits
    raw = b""
    while True:
        want = nblk * _BLOCK
        raw += reader.get(key, [(off + len(raw), off + len(raw) + want - 1)])[0]
        for i in range(0, len(raw), 80):
            if raw[i:i + 8] == b"END     ":
                n = ((i + 80 + _BLOCK - 1) // _BLOCK) * _BLOCK
                return fits.Header.fromstring(raw[:n].decode("ascii")), off + n
        if len(raw) > 400 * _BLOCK:
            raise OSError(f"no END card in {key}")


# The HDU order is PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE -- note
# FLAGS BEFORE VARIANCE. Offsets are stable per (processing version, detector)
# but NOT across them: an l2b-v24 IMAGE header runs one 2880-byte block longer
# than an l2b-v20 one, and it varies detector to detector within a version. So
# the layout is cached per variant and re-validated on every use against the
# IMAGE header we have to read anyway; a mismatch just re-walks.
_LAYOUTS = {}
_LAYOUT_LOCK = None


def _variant(key):
    parts = key.split("/")
    return tuple(parts[3:5]) if len(parts) > 4 else (key,)


class UnsupportedLayout(RuntimeError):
    """An L2 plane is stored in a way the ranged reader cannot decode.

    Raised, not swallowed: a per-exposure failure on EVERY exposure would
    otherwise be cached as "this star has no SPHEREx data".
    """


def _tile_layout(h, data_off):
    """None for a plain image HDU; for a tile-compressed one, what the ranged
    reader needs to decode whole-row tiles.

    DR1/QR3 store FLAGS RICE_1-compressed, one 2040-px tile per row (QR2 stored
    it as a plain int32 image): a BINTABLE of (nbytes, heap offset) descriptors
    whose bytes live in the heap. Any other compressed layout is refused.
    """
    if not h.get("ZIMAGE"):
        return None
    tform = str(h.get("TFORM1", ""))
    m = re.match(r"^1?([PQ])B", tform)
    zpar = {h.get(f"ZNAME{i}"): h.get(f"ZVAL{i}") for i in range(1, 10)
            if h.get(f"ZNAME{i}")}
    if (h.get("ZCMPTYPE") != "RICE_1" or h.get("TFIELDS") != 1 or not m
            or h.get("ZTILE2") != 1 or h.get("ZTILE1") != h.get("ZNAXIS1")
            or h.get("ZBITPIX") not in (8, 16, 32) or "ZSCALE" in h):
        raise UnsupportedLayout(
            "compressed %s: ZCMPTYPE=%s ZTILE=%sx%s TFORM1=%s ZBITPIX=%s"
            % (h.get("EXTNAME"), h.get("ZCMPTYPE"), h.get("ZTILE1"),
               h.get("ZTILE2"), tform, h.get("ZBITPIX")))
    return dict(rowbytes=int(h["NAXIS1"]),
                heap=data_off + int(h.get("THEAP", h["NAXIS1"] * h["NAXIS2"])),
                desc=">i4" if m.group(1) == "P" else ">i8",
                blocksize=int(zpar.get("BLOCKSIZE", 32)),
                bytepix=int(zpar.get("BYTEPIX", abs(h["ZBITPIX"]) // 8)),
                width=int(h["ZNAXIS1"]))


def _decode_tile_rows(reader, key, z, desc_buf):
    """The row tiles whose descriptors are `desc_buf`, decoded to a (rows, width)
    array -- one more round trip, for their compressed bytes in the heap."""
    from astropy.io.fits.hdu.compressed._codecs import Rice1
    d = np.frombuffer(desc_buf, dtype=z["desc"]).reshape(-1, 2).astype(np.int64)
    n, o = d[:, 0], d[:, 1]
    if (n <= 0).any():
        raise UnsupportedLayout(f"empty compressed tile in {key}")
    lo, hi = int(o.min()), int((o + n).max())
    heap = np.frombuffer(
        reader.get(key, [(z["heap"] + lo, z["heap"] + hi - 1)])[0], dtype=np.uint8)
    codec = Rice1(blocksize=z["blocksize"], bytepix=z["bytepix"],
                  tilesize=z["width"])
    rows = [np.asarray(codec.decode(heap[a - lo:a - lo + b])) for a, b in zip(o, n)]
    out = np.vstack(rows)
    if out.shape[1] != z["width"]:
        raise UnsupportedLayout(f"decoded tile width {out.shape[1]} in {key}")
    return out


def _hdu_offsets(reader, key):
    """`{EXTNAME: data_offset}` for IMAGE / FLAGS / VARIANCE, plus the IMAGE
    header. One round trip when the variant's layout is already known."""
    global _LAYOUT_LOCK
    if _LAYOUT_LOCK is None:
        import threading
        _LAYOUT_LOCK = threading.Lock()
    v = _variant(key)
    with _LAYOUT_LOCK:
        known = _LAYOUTS.get(v)
    if known is not None:
        ih, iend = _header_at(reader, key, known["image_hdr"])
        if iend == known["IMAGE"][0]:             # layout still valid
            return ih, known
    prim, doff = _header_at(reader, key, 0)
    image_hdr = doff + _data_span(prim)
    ih, ioff = _header_at(reader, key, image_hdr)
    # (data offset, BITPIX): the pixel type is read from the file, never assumed
    # -- QR2 ships float32/int32 today, and a silently mis-cast slab would be
    # numerically wrong rather than an error.
    # The third element is the tile layout of a compressed plane (_tile_layout).
    if _tile_layout(ih, ioff) is not None:
        raise UnsupportedLayout(f"compressed IMAGE in {key}")
    offs = {"image_hdr": image_hdr, "IMAGE": (ioff, ih["BITPIX"], None)}
    o = ioff + _data_span(ih)
    for _ in range(4):
        hh, dd = _header_at(reader, key, o)
        offs[hh.get("EXTNAME", "?")] = (dd, hh["BITPIX"], _tile_layout(hh, dd))
        o = dd + _data_span(hh)
        if {"FLAGS", "VARIANCE"} <= set(offs):
            break
    with _LAYOUT_LOCK:
        _LAYOUTS[v] = offs
    return ih, offs


# The 11 super-bands cover 1.795 um of the 0.75-5 um range, so most exposures
# fall between windows and are discarded by superband_mags after being paid for.
# Skipping them changes results only through the MAD clip, whose 0.15 um bins can
# straddle a window edge: measured over 509 real cached stars, 1.86% of band
# slots move at all and 0.60% by more than 5 mmag -- inside the 2% (~22 mmag)
# absolute-calibration floor. Set ARIADNE_SPHEREX_ALL_BANDS=1 to keep them.
SKIP_OUT_OF_BAND = not os.environ.get("ARIADNE_SPHEREX_ALL_BANDS")


@lru_cache(maxsize=1)
def _superband_edges():
    return tuple(sorted(_superband_windows().values()))


def _in_any_superband(lam):
    if not np.isfinite(lam):
        return False
    return any(lo <= lam <= hi for lo, hi in _superband_edges())


def l2_key(access_url):
    """SIA `access_url` -> the path both L2 hosts serve it under."""
    return access_url[len(_L2_PREFIX):] if access_url.startswith(_L2_PREFIX) \
        else access_url


def _psf_measure(img, var, flg, x, y, xa, ya, det, others, lam):
    """`_measure_arrays`' contract, with the flux from the forced ePSF fit."""
    ny, nx = img.shape
    if not (ANN_OUT <= x <= nx - ANN_OUT - 1 and ANN_OUT <= y <= ny - ANN_OUT - 1):
        return None                        # same window guard as the aperture path
    if not np.isfinite(lam):
        return None
    r = _psf_flux(img, var, flg, x, y, xa, ya, det, others)
    return None if r is None else (lam, r[0], r[1])


def _measure_remote(reader, key, ra, dec, nbrs=None):
    """One exposure by ranged reads -- bit-identical to `_measure_cutout`.

    Two round trips: the IMAGE header (needed for this exposure's pointing), then
    one bundled request for the three row slabs the aperture and annulus touch.
    """
    ih, offs = _hdu_offsets(reader, key)
    nx, ny = ih["NAXIS1"], ih["NAXIS2"]
    x, y, xa, ya, det = _star_in_window(ih, None, ra, dec)
    # Crop to the SAME window the cutout service would have returned. It is not
    # cosmetic: _measure_arrays derives the universal-flag marker from the
    # fraction of flagged pixels in its window, so handing it a full-width slab
    # would change `ignore` and hence the reject decisions.
    x0, y0 = int(round(xa)) - _WIN // 2, int(round(ya)) - _WIN // 2
    if not (0 <= x0 and x0 + _WIN <= nx and 0 <= y0 and y0 + _WIN <= ny):
        return None                     # the edge guard, before fetching pixels
    lam = _bilinear(_cwave(det), xa, ya)
    if SKIP_OUT_OF_BAND and not _in_any_superband(lam):
        # 65% of exposures land between the super-band windows: they are
        # measured, cached, and then silently discarded by superband_mags. The
        # header we just read gives the EXACT wavelength, so we can drop them
        # before paying for the megabyte of pixels -- no footprint-based
        # prediction, no guard band, no chance of skipping a usable exposure.
        return None
    planes = ("IMAGE", "FLAGS", "VARIANCE")
    spans, kinds = [], []
    for name in planes:
        off, bitpix, z = offs[name]
        # a compressed plane's rows are its tile descriptors (decoded below)
        row = z["rowbytes"] if z is not None else nx * (abs(bitpix) // 8)
        spans.append((off + y0 * row, off + (y0 + _WIN) * row - 1))
        kinds.append(z if z is not None else _FITS_DTYPE[bitpix])
    bufs = reader.get(key, spans)

    def arr(buf, kind):
        if isinstance(kind, dict):
            a = _decode_tile_rows(reader, key, kind, buf)
        else:
            a = np.frombuffer(buf, dtype=kind).reshape(_WIN, nx)
        return np.asarray(a[:, x0:x0 + _WIN], float)

    img, flg, var = (arr(b, k) for b, k in zip(bufs, kinds))
    if nbrs is None:
        return _measure_arrays(img, var, flg.astype(int), xa - x0, ya - y0, lam)
    others = []
    for n in nbrs:
        _, _, oxa, oya, _ = _star_in_window(ih, None, n['ra'], n['dec'])
        others.append((oxa - x0, oya - y0))
    return _psf_measure(img, var, flg.astype(int), xa - x0, ya - y0, xa, ya, det,
                        others, lam)


# Ranged reads are the default; set ARIADNE_SPHEREX_CUTOUTS=1 to fall back to the
# IRSA cutout service (the two are asserted bit-identical, so this is an escape
# hatch for a bad day at the archive, not a science switch).
USE_RANGED_READS = not os.environ.get("ARIADNE_SPHEREX_CUTOUTS")


class _ReaderPool:
    """A fixed set of keep-alive readers, leased one per in-flight exposure.

    Explicitly pooled rather than thread-local: `extract_spectrum` builds a
    fresh ThreadPoolExecutor for every star, so thread-local connections would
    be abandoned (and their sockets left to the garbage collector) 17,839 times
    over a catalog run.
    """

    def __init__(self, n):
        import queue
        self._q = queue.LifoQueue()
        self._all = [_Reader() for _ in range(max(1, n))]
        for r in self._all:
            self._q.put(r)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for r in self._all:
            r.close()

    def lease(self):
        return _Lease(self._q)


class _Lease:
    def __init__(self, q):
        self._q = q

    def __enter__(self):
        self._r = self._q.get()
        return self._r

    def __exit__(self, *exc):
        self._q.put(self._r)
        return False


def _fetch_one(args):
    url, ra, dec, tmpdir, i = args[:5]
    pool = args[5] if len(args) > 5 else None
    nbrs = args[6] if len(args) > 6 else None
    if USE_RANGED_READS:
        try:
            if pool is None:
                with _ReaderPool(1) as p, p.lease() as rd:
                    return _measure_remote(rd, l2_key(url), ra, dec, nbrs)
            with pool.lease() as rd:
                return _measure_remote(rd, l2_key(url), ra, dec, nbrs)
        except (CalibrationUnavailable, UnsupportedLayout):
            raise
        except Exception as e:                           # noqa: BLE001
            # Both hosts and every retry failed. Say so: a swallowed failure is
            # indistinguishable from a legitimate reject and quietly shrinks the
            # star's exposure count.
            logger.warning("SPHEREx: ranged read failed for %s (%s: %s)",
                           url.rsplit("/", 1)[-1], type(e).__name__, e)
            return None
    path = os.path.join(tmpdir, f"spx_{i:04d}.fits")
    try:
        req = f"{url}?center={ra},{dec}d&size=240arcsec"
        for attempt in range(_RANGE_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=_CUTOUT_TIMEOUT,
                                            context=_SSL_CTX) as resp, \
                        open(path, "wb") as fh:
                    fh.write(resp.read())
                break
            except Exception:                            # noqa: BLE001
                # The cutout service returns 503 under concurrency (0% at 2
                # concurrent, 37.5% at 16). Retrying is the difference between a
                # slow exposure and a silently missing one.
                if attempt + 1 == _RANGE_RETRIES:
                    raise
                import time
                time.sleep(0.4 * (attempt + 1))
        return _measure_cutout(path, ra, dec, nbrs)
    except CalibrationUnavailable:
        raise            # never degrade to "this star has no SPHEREx data"
    except Exception as e:                               # noqa: BLE001
        logger.warning("SPHEREx: cutout failed for %s (%s: %s)",
                       url.rsplit("/", 1)[-1], type(e).__name__, e)
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# IRSA SIA collections in order of preference, each with the mid-epoch of its
# observations (decimal year) for proper-motion propagation. DR1 spans
# 2025.31-2026.35, so its mid-epoch keeps every exposure within +-0.52 yr
# (<< 1 px for any ordinary PM); QR3 (2026.40-2026.75) is queried only when DR1
# returns no planes or fails. QR2 left SIA in Nov 2026.
RELEASES = (('spherex_dr1', 2025.83), ('spherex_qr3', 2026.57))
EPOCH = RELEASES[0][1]


def _query_planes(ra, dec, epoch=EPOCH, position=None):
    """`(urls, collection, ra, dec)` from the first release with planes.

    `position(epoch) -> (ra, dec)`, if given, re-propagates the target to a
    fallback release's epoch; without it (ra, dec) is used as is. No planes in
    any release returns `([], None, ra, dec)`; if every release FAILED, the
    first error is raised.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.irsa import Irsa
    Irsa.TIMEOUT = _SIA_TIMEOUT      # bound the SIA query (no indefinite hang)
    errors = []
    for collection, rel_epoch in RELEASES:
        r_ra, r_dec = ra, dec
        if position is not None and abs(rel_epoch - epoch) > 0.05:
            r_ra, r_dec = (float(v) for v in position(rel_epoch))
        try:
            res = Irsa.query_sia(
                pos=(SkyCoord(ra=r_ra * u.deg, dec=r_dec * u.deg), 10 * u.arcsec),
                collection=collection)
        except Exception as e:                       # noqa: BLE001
            logger.warning('SPHEREx: SIA query of %s failed: %s', collection, e)
            errors.append(e)
            continue
        urls = [str(u_) for u_ in res['access_url']]
        logger.info('SPHEREx: %d overlapping %s planes', len(urls), collection)
        if urls:
            return urls, collection, r_ra, r_dec
    if len(errors) == len(RELEASES):
        raise errors[0]
    return [], None, ra, dec


def extract_spectrum(ra, dec, source_id, threads=4, ks=None, epoch=EPOCH, meta=None,
                     position=None):
    """Per-exposure (lam, flux_uJy, err_uJy) arrays for a PM-propagated
    position; cached per source_id.

    (ra, dec) is the target at `epoch` (DR1's mid-epoch); `position(epoch)`
    re-propagates it if the QR3 fallback is used (see _query_planes). No planes
    in any release is NOT cached -- an empty answer from IRSA is how QR2's
    retirement first showed up, and caching it would hide the star's SPHEREx
    data forever.

    With the target's 2MASS `ks`, the catalogued neighbours decide the method
    first: if they are predicted to put more than CONTAM_TRIGGER of the
    target's flux into the aperture, or more than ANN_TRIGGER times it into the
    background annulus, every exposure is measured by forced ePSF photometry of
    target + neighbours (cached separately under PSF_CACHE_DIR); otherwise the
    aperture extraction runs unchanged. Without `ks` there is no prediction and
    the aperture is used. `meta`, if a dict, receives {method, contamination,
    annulus_flux, n_neighbours, n_merged}."""
    method, contam, ann, nbrs, n_merged = 'aperture', None, None, None, 0
    if ks is not None and np.isfinite(ks):
        allnb, n_merged = neighbours(ra, dec, source_id, epoch=epoch)
        contam = predicted_contamination(allnb, float(ks))
        ann = annulus_flux(allnb, float(ks))
        if contam > CONTAM_TRIGGER or ann > ANN_TRIGGER:
            method = 'psf'
            nbrs = fit_neighbours(allnb, float(ks))
    if isinstance(meta, dict):
        meta.update(method=method, contamination=contam, annulus_flux=ann,
                    n_neighbours=None if nbrs is None else len(nbrs), n_merged=n_merged,
                    release=None)
    cache_dir = PSF_CACHE_DIR if method == 'psf' else CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{source_id}.csv')
    if os.path.exists(cache):
        arr = np.loadtxt(cache, delimiter=',', ndmin=2)
        if arr.size:
            return arr[:, 0], arr[:, 1], arr[:, 2]
        return np.array([]), np.array([]), np.array([])

    urls, release, ra, dec = _query_planes(ra, dec, epoch=epoch, position=position)
    if isinstance(meta, dict):
        meta['release'] = release
    if not urls:
        logger.warning('SPHEREx: no planes in %s for source %s; not cached',
                       '/'.join(c for c, _ in RELEASES), source_id)
        return np.array([]), np.array([]), np.array([])
    import contextlib
    import tempfile
    rows = []
    # One keep-alive reader per worker, closed with the pool.
    pool_cm = _ReaderPool(threads) if USE_RANGED_READS else contextlib.nullcontext()
    with tempfile.TemporaryDirectory(prefix='spherex_') as tmpdir, pool_cm as pool:
        jobs = [(u_, ra, dec, tmpdir, i, pool, nbrs) for i, u_ in enumerate(urls)]
        with ThreadPoolExecutor(threads) as ex:
            for r in ex.map(_fetch_one, jobs):
                if r is not None:
                    rows.append(r)
    if not rows:
        np.savetxt(cache, np.empty((0, 3)), delimiter=',')
        return np.array([]), np.array([]), np.array([])
    arr = np.array(rows)
    # 4-sigma MAD clip per 0.15 um bin (junk exposures the flags miss)
    lam, f, e = arr.T
    keep = np.ones(len(lam), bool)
    for lo in np.arange(0.7, 5.1, 0.15):
        m = (lam >= lo) & (lam < lo + 0.15)
        if m.sum() >= 4:
            med = np.median(f[m])
            sig = 1.4826 * np.median(np.abs(f[m] - med))
            keep[m] &= np.abs(f[m] - med) < 4 * max(sig, 1e-3 * med)
    arr = arr[keep]
    np.savetxt(cache, arr, delimiter=',')
    return arr[:, 0], arr[:, 1], arr[:, 2]


def superband_mags(lam, f, e):
    """Bin per-exposure fluxes into SPHEREX_* Vega mags.

    Returns {band: (mag, mag_err, n_exposures)}. Median flux per window;
    error = MAD-scatter/sqrt(n) with a 2% absolute-calibration floor.
    """
    from ..phot_utils import flux_to_mag
    out = {}
    for name, (lo, hi) in sorted(_superband_windows().items()):
        m = (lam >= lo) & (lam <= hi)
        n = int(m.sum())
        if n < MIN_EXP_PER_BAND:
            continue
        fmu = float(np.median(f[m]))
        if not (np.isfinite(fmu) and fmu > 0):
            continue
        scatter = 1.4826 * float(np.median(np.abs(f[m] - fmu)))
        fer = max(scatter / np.sqrt(n), ABS_CAL_FLOOR * fmu)
        lam_c = 0.5 * (lo + hi)
        flam = fmu * 1e-29 * C_UM / lam_c ** 2          # erg/cm2/s/um
        flam_e = fer * 1e-29 * C_UM / lam_c ** 2
        mag, mag_e = flux_to_mag(flam, flam_e, name)
        if np.isfinite(mag) and np.isfinite(mag_e):
            out[name] = (float(mag), float(mag_e), n)
    return out
