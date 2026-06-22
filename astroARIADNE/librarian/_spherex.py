"""SPHEREx QR2 per-star spectro-photometry for the librarian.

Extracts a 0.75-5 um spectrum from the public QR2 weekly spectral images
(one LVF wavelength per overlapping exposure; ~150-230 clean exposures per
star) and bins it into ARIADNE's SPHEREX_* super-bands. Validated against
2MASS/WISE on GJ 1214 and TRAPPIST-1 (band ratios 0.94-1.07; see
notebooks/spherex_extract.py for the standalone validation tool).

Method per exposure: 3-px aperture + 10-16 px background annulus on the IBE
cutout (subpixel-weighted, numpy-only -- the SPHEREx PSF holds 99.6% of its
energy inside 3 px, zone spread 0.2%, so no aperture correction applies);
MJy/sr -> uJy via the 6.2" pixel solid angle; wavelength from the per-pixel
WCS-WAVE solution. Exposures with any bad flag inside the aperture are
dropped (bits 1 = OVERFLOW per the official SDT guidance and 21 = benign
frame-wide marker are ignored), then a 4-sigma MAD clip per wavelength bin.

Binned results are cached per source id under ~/.astroARIADNE/spherex/ so a
star is only ever extracted once.
"""
import logging
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

OMEGA = (6.2 * 4.84813681e-6) ** 2          # sr per 6.2" SPHEREx pixel
AP_R, ANN_IN, ANN_OUT = 3.0, 10.0, 16.0
IGNORE_BITS = (1, 21)
C_UM = 2.99792458e14
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'spherex')
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


def _measure_cutout(path, ra, dec):
    """One exposure -> (wavelength_um, flux_uJy, err_uJy) or None."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path) as h:
        img = np.asarray(h['IMAGE'].data, float)
        var = np.asarray(h['VARIANCE'].data, float)
        flg = np.asarray(h['FLAGS'].data, int)
        w = WCS(h['IMAGE'].header)
        x, y = (float(v) for v in w.world_to_pixel_values(ra, dec))
        lam = float(WCS(h['IMAGE'].header, fobj=h, key='W')
                    .pixel_to_world_values(x, y)[0])
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
    return lam, f, e


def _fetch_one(args):
    url, ra, dec, tmpdir, i = args
    path = os.path.join(tmpdir, f'spx_{i:04d}.fits')
    try:
        urllib.request.urlretrieve(
            f'{url}?center={ra},{dec}d&size=240arcsec', path)
        return _measure_cutout(path, ra, dec)
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def extract_spectrum(ra, dec, source_id, threads=4):
    """Per-exposure (lam, flux_uJy, err_uJy) arrays for a PM-propagated
    position; cached per source_id."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{source_id}.csv')
    if os.path.exists(cache):
        arr = np.loadtxt(cache, delimiter=',', ndmin=2)
        if arr.size:
            return arr[:, 0], arr[:, 1], arr[:, 2]
        return np.array([]), np.array([]), np.array([])

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.irsa import Irsa
    res = Irsa.query_sia(
        pos=(SkyCoord(ra=ra * u.deg, dec=dec * u.deg), 10 * u.arcsec),
        collection='spherex_qr2')
    urls = [str(u_) for u_ in res['access_url']]
    logger.info('SPHEREx: %d overlapping QR2 planes', len(urls))
    import tempfile
    rows = []
    with tempfile.TemporaryDirectory(prefix='spherex_') as tmpdir:
        jobs = [(u_, ra, dec, tmpdir, i) for i, u_ in enumerate(urls)]
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
