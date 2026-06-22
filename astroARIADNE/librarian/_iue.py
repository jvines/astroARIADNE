"""IUE/INES ultraviolet-spectra ingestion for the librarian.

GALEX is a photon-counting microchannel-plate imager and saturates for bright
sources (the 10%-local-nonlinearity points are FUV = 13.73, NUV = 13.85 AB;
Morrissey et al. 2007, ApJS 173, 682), so the very stars whose SEDs most need a
UV anchor -- hot stars and white-dwarf companions -- are exactly the ones GALEX
either never measured or measured into saturation. IUE observed ~10^4 bright
stars at 1150-3350 A and the reprocessed INES low-resolution, large-aperture
spectra are absolute-flux-calibrated, so they let us *synthesise* the GALEX FUV
and NUV magnitudes a saturated catalogue cannot supply.

Archive route (verified live 2026-06): MAST IUE holdings via
``astroquery.mast.Observations``. We take the SCIENCE "VO spectral container"
products (``*mxlo_vo.fits`` -- one BinTableHDU with WAVE [A], FLUX and SIGMA
[erg/cm^2/s/A] and an INES QUALITY bitmask), restricted to LOW-DISP exposures
through the LARGE aperture (``APERTURE`` keyword LARGE or BOTH; the SMALL
aperture loses light and biases the absolute flux low). These are the
INES-rebinned products the task asks for; the INES CGI at
http://sdc.cab.inta-csic.es/ines/ serves the same spectra but the MAST route is
the more robust, scriptable one today.

Camera split / splice: SWP covers 1150-1980 A, LWP/LWR cover 1850-3350 A. Within
a camera channel (SWP, or LW = LWP+LWR) multiple epochs are resampled onto a
common grid and **median-combined**, with the cross-epoch scatter (1.4826*MAD)
as the error so an intrinsically variable star inflates its own bars. The SWP
and LW combined spectra are then spliced into one: SWP below the overlap, LW
above it, and an inverse-variance average across the 1851-1978 A overlap so the
join is continuous.

Absolute calibration: the INES / IUE-NEWSIPS low-resolution large-aperture flux
scale is good to ~3-5% (Gonzalez-Riestra et al. 2001, the INES system;
absolute calibration of Massa & Fitzpatrick 2000). We carry a 5% absolute-
calibration floor on every synthesised band, added in quadrature with the
photon/scatter error.

Per-source products are cached under ~/.astroARIADNE/iue/ keyed by position so a
star is only ever fetched once.
"""
import logging
import os
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'iue')
MAST_DL = 'https://mast.stsci.edu/api/v0.1/Download/file?uri='
C_AA = 2.99792458e18                 # speed of light, A/s
ABS_CAL_FLOOR = 0.05                 # INES absolute-cal floor, fractional
T_FRAC = 0.01                        # band support := transmission > 1% of peak
MAX_GAP_AA = 60.0                    # largest tolerated hole in a band's support
SW_LW_SPLIT = 1900.0                 # nominal SWP/LW crossover (inside overlap)
# Channel -> the cameras that feed it.
LW_CAMERAS = ('LWP', 'LWR')
GALEX_BANDS = ('GALEX_FUV', 'GALEX_NUV')


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_path(ra, dec):
    return os.path.join(CACHE_DIR, f'{ra:+09.5f}{dec:+09.5f}.csv')


def _save_cache(ra, dec, wave, flux, err):
    os.makedirs(CACHE_DIR, exist_ok=True)
    arr = (np.empty((0, 3)) if wave is None or not len(wave)
           else np.column_stack([wave, flux, err]))
    np.savetxt(_cache_path(ra, dec), arr, delimiter=',')


def _load_cache(ra, dec):
    """Return (wave, flux, err) or None (no IUE data) or False (no cache)."""
    path = _cache_path(ra, dec)
    if not os.path.exists(path):
        return False
    arr = np.loadtxt(path, delimiter=',', ndmin=2)
    if not arr.size:
        return None
    return arr[:, 0], arr[:, 1], arr[:, 2]


# --------------------------------------------------------------------------- #
# MAST fetch
# --------------------------------------------------------------------------- #
def _read_vo_spectrum(path):
    """One INES ``*mxlo_vo.fits`` -> (wave_A, flux, sigma, quality, aperture).

    Bad samples (non-finite or non-positive flux/sigma, or any nonzero INES
    QUALITY flag) are returned as NaN in flux: a nonzero flag marks reseaux,
    microphonics, ITF-extrapolation or saturation, and for the bright stars
    this module targets the saturation flags are exactly the ones that would
    bias a synthesised magnitude, so we drop every flagged sample and rely on
    the per-band coverage guard to reject a band the flags gutted.
    """
    from astropy.io import fits
    with fits.open(path) as h:
        ap = str(h[0].header.get('APERTURE',
                                 h[1].header.get('APERTURE', ''))).upper()
        d = h[1].data
        w = np.asarray(d['WAVE']).ravel().astype(float)
        f = np.asarray(d['FLUX']).ravel().astype(float)
        s = np.asarray(d['SIGMA']).ravel().astype(float)
        q = (np.asarray(d['QUALITY']).ravel().astype(int)
             if 'QUALITY' in d.columns.names else np.zeros(w.size, int))
    bad = ~np.isfinite(f) | ~np.isfinite(s) | (f <= 0) | (s <= 0) | (q != 0)
    f = f.copy()
    f[bad] = np.nan
    order = np.argsort(w)
    return w[order], f[order], s[order], q[order], ap


def _query_lowres(ra, dec, radius_arcsec):
    """Download every LOW-DISP large-aperture INES spectrum near a position.

    Returns {'SWP': [(w, f, s), ...], 'LW': [...]} (LW = LWP + LWR), already
    quality-masked. Empty channels are omitted.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Observations
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    obs = Observations.query_criteria(
        coordinates=coord, radius=radius_arcsec * u.arcsec,
        obs_collection='IUE', dataproduct_type='spectrum')
    if obs is None or not len(obs):
        return {}
    obs = obs[obs['filters'] == 'LOW DISP']
    out = {}
    import tempfile
    with tempfile.TemporaryDirectory(prefix='iue_') as tmp:
        for row in obs:
            cam = str(row['instrument_name'])
            channel = 'LW' if cam in LW_CAMERAS else cam
            if channel not in ('SWP', 'LW'):
                continue
            try:
                prod = Observations.get_product_list(row)
            except Exception as e:                       # noqa: BLE001
                logger.debug('IUE product list failed for %s: %s',
                             row['obs_id'], e)
                continue
            uri = None
            for p in prod:
                if str(p['productFilename']).endswith('mxlo_vo.fits'):
                    uri = str(p['dataURI'])
                    break
            if uri is None:
                continue
            dest = os.path.join(tmp, uri.split('/')[-1])
            try:
                urllib.request.urlretrieve(MAST_DL + uri, dest)
                w, f, s, q, ap = _read_vo_spectrum(dest)
            except Exception as e:                       # noqa: BLE001
                logger.debug('IUE download/read failed (%s): %s', uri, e)
                continue
            finally:
                try:
                    os.remove(dest)
                except OSError:
                    pass
            if ap not in ('LARGE', 'BOTH'):              # SMALL loses light
                continue
            if np.isfinite(f).sum() < 10:
                continue
            out.setdefault(channel, []).append((w, f, s))
    return out


# --------------------------------------------------------------------------- #
# Combination
# --------------------------------------------------------------------------- #
def _median_combine(spectra):
    """Median-combine epochs of one channel onto a common grid.

    Each (wave, flux, sigma) is resampled (linear, NaN outside coverage) onto
    the densest epoch's grid; flux is the per-pixel median across epochs and
    the error is the cross-epoch scatter (1.4826*MAD, so a variable star widens
    its own bars), floored at the median per-point SIGMA / sqrt(N).
    """
    spectra = [s for s in spectra if np.isfinite(s[1]).any()]
    if not spectra:
        return None
    ref = max(spectra, key=lambda s: np.isfinite(s[1]).sum())
    grid = ref[0]
    fstack, sstack = [], []
    for w, f, s in spectra:
        good = np.isfinite(f)
        if good.sum() < 2:
            continue
        fi = np.interp(grid, w[good], f[good], left=np.nan, right=np.nan)
        si = np.interp(grid, w[good], s[good], left=np.nan, right=np.nan)
        # never extrapolate flux past this epoch's own coverage
        outside = (grid < w[good].min()) | (grid > w[good].max())
        fi[outside] = np.nan
        si[outside] = np.nan
        fstack.append(fi)
        sstack.append(si)
    if not fstack:
        return None
    fstack = np.array(fstack)
    sstack = np.array(sstack)
    with np.errstate(all='ignore'):
        flux = np.nanmedian(fstack, axis=0)
        n = np.isfinite(fstack).sum(axis=0)
        mad = np.nanmedian(np.abs(fstack - flux), axis=0)
        scatter = 1.4826 * mad
        psig = np.nanmedian(sstack, axis=0) / np.sqrt(np.maximum(n, 1))
        err = np.where(n >= 2, np.maximum(scatter, psig),
                       np.nanmedian(sstack, axis=0))
    keep = np.isfinite(flux) & (flux > 0)
    if keep.sum() < 10:
        return None
    return grid[keep], flux[keep], err[keep]


def _merge_sw_lw(sw, lw):
    """Splice the SWP and LW combined spectra into one continuous spectrum.

    SWP supplies wavelengths below the overlap, LW above it, and the two are
    inverse-variance averaged across the overlap (here LW is resampled onto the
    SWP grid) so the join carries no step.
    """
    if sw is None:
        return lw
    if lw is None:
        return sw
    ws, fs, es = sw
    wl, fl, el = lw
    ov_lo = max(ws.min(), wl.min())
    ov_hi = min(ws.max(), wl.max())
    if ov_lo >= ov_hi:                                   # disjoint: concatenate
        w = np.concatenate([ws, wl])
        f = np.concatenate([fs, fl])
        e = np.concatenate([es, el])
        o = np.argsort(w)
        return w[o], f[o], e[o]
    # SWP-only blue, LW-only red, weighted mean across the overlap.
    blue = ws < ov_lo
    red = wl > ov_hi
    inov = (ws >= ov_lo) & (ws <= ov_hi)
    fl_o = np.interp(ws[inov], wl, fl)
    el_o = np.interp(ws[inov], wl, el)
    sw_o, fs_o, es_o = ws[inov], fs[inov], es[inov]
    wlw = 1.0 / np.maximum(es_o, 1e-30) ** 2
    wlo = 1.0 / np.maximum(el_o, 1e-30) ** 2
    fo = (fs_o * wlw + fl_o * wlo) / (wlw + wlo)
    eo = 1.0 / np.sqrt(wlw + wlo)
    w = np.concatenate([ws[blue], sw_o, wl[red]])
    f = np.concatenate([fs[blue], fo, fl[red]])
    e = np.concatenate([es[blue], eo, el[red]])
    o = np.argsort(w)
    return w[o], f[o], e[o]


# --------------------------------------------------------------------------- #
# Public: fetch
# --------------------------------------------------------------------------- #
def fetch_spectrum(ra, dec, radius_arcsec=30):
    """Fetch the merged INES UV spectrum at a position.

    Returns ``(wave_A, flux_cgs, err_cgs)`` in erg/cm^2/s/A, or ``None`` when no
    usable IUE low-resolution large-aperture data exist. Cached per position
    under ~/.astroARIADNE/iue/.
    """
    cached = _load_cache(ra, dec)
    if cached is not False:
        return cached if cached is not None else None
    try:
        channels = _query_lowres(ra, dec, radius_arcsec)
    except Exception as e:                               # noqa: BLE001
        logger.warning('IUE/INES query failed: %s', e)
        return None
    sw = _median_combine(channels['SWP']) if 'SWP' in channels else None
    lw = _median_combine(channels['LW']) if 'LW' in channels else None
    merged = _merge_sw_lw(sw, lw)
    if merged is None:
        _save_cache(ra, dec, None, None, None)
        return None
    _save_cache(ra, dec, *merged)
    return merged


# --------------------------------------------------------------------------- #
# Public: synthesis
# --------------------------------------------------------------------------- #
def _band_support(band):
    """(lo_A, hi_A) where the GALEX transmission exceeds T_FRAC of its peak."""
    from ..phot_utils import _get_filter
    filt = _get_filter(band)
    wl = filt.wavelength.to('AA').value
    tr = np.asarray(filt.transmit, float)
    on = wl[tr > T_FRAC * tr.max()]
    return float(on.min()), float(on.max())


def _covered(wave, flux, lo, hi):
    """True if the band support [lo, hi] is fully spanned by good samples with
    no internal gap wider than MAX_GAP_AA."""
    good = np.isfinite(flux) & (flux > 0)
    w = np.sort(wave[good])
    if w.size < 4 or w[0] > lo or w[-1] < hi:
        return False
    inb = w[(w >= lo) & (w <= hi)]
    edges = np.concatenate([[lo], inb, [hi]])
    return bool(np.max(np.diff(edges)) <= MAX_GAP_AA)


def synth_galex(wave, flux, err, n_mc=256, seed=0):
    """Synthesise GALEX FUV/NUV AB photometry from an INES UV spectrum.

    ``wave`` in A, ``flux``/``err`` in erg/cm^2/s/A. Returns
    ``{'GALEX_FUV': (mag, mag_err), 'GALEX_NUV': (mag, mag_err)}`` containing
    only the bands whose transmission curve is fully inside the spectrum's
    coverage (so an LW-only spectrum yields no FUV). Magnitudes are rigorous
    photon-counting AB synthetic photometry through the pyphot GALEX curves
    (``get_flux`` / ``AB_zero_flux``, i.e. the proper pivot-wavelength AB
    zero-point); the flat-F_nu source returns its input AB mag to <0.1%. The
    flux error is propagated by Monte-Carlo through the same convolution and
    added in quadrature with the 5% INES absolute-calibration floor.
    """
    import astropy.units as u
    _flam = u.erg / u.s / u.cm ** 2 / u.AA
    from ..phot_utils import _get_filter
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    out = {}
    rng = np.random.default_rng(seed)
    for band in GALEX_BANDS:
        lo, hi = _band_support(band)
        if not _covered(wave, flux, lo, hi):
            continue
        good = np.isfinite(flux) & (flux > 0) & np.isfinite(err)
        w = wave[good]
        f = flux[good]
        e = err[good]
        order = np.argsort(w)
        w, f, e = w[order], f[order], e[order]
        filt = _get_filter(band)
        abzf = filt.AB_zero_flux.to('erg/s/cm**2/AA').value
        f_band = filt.get_flux(w * u.AA,
                               f * _flam).value
        if not (np.isfinite(f_band) and f_band > 0):
            continue
        draws = f[None, :] + e[None, :] * rng.standard_normal((n_mc, f.size))
        fb = np.array([filt.get_flux(w * u.AA,
                                     d * _flam).value
                       for d in draws])
        sig_f = float(np.nanstd(fb))
        frac = float(np.hypot(sig_f / f_band, ABS_CAL_FLOOR))
        mag = -2.5 * np.log10(f_band / abzf)
        mag_err = 1.0857 * frac
        if np.isfinite(mag) and np.isfinite(mag_err):
            out[band] = (float(mag), float(mag_err))
    return out
