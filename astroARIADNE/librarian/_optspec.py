"""Optical-spectrum synthetic photometry for the librarian (DESI MWS + SDSS).

Fills optical coverage gaps for *faint* stars (G > ~16) where the Gaia XP
gap-filler (`Librarian._fetch_xp_synth`) stops firing -- XP's continuous
spectrum enters its hockey-stick noise regime there, but DESI-DR1 (Milky Way
Survey) and SDSS/BOSS reach far fainter with flux-calibrated optical spectra.
This module is the SAME SHAPE as the XP filler: pull one spectrum per source,
synthesize standardized-band magnitudes by convolving with the pyphot/custom
band curves, and hand them back for the Librarian to gap-gate (the GAP/priority
logic lives in the wiring, not here -- exactly like XP).

Data come from the public SPARCL HTTP API (NOIRLab Astro Data Lab); no
``sparclclient`` dependency (the ariadne env pins numpy and sparclclient is not
installed). The API contract was reverse-engineered empirically against the
live service (2026-06):

* base ``https://astrosparcl.datalab.noirlab.edu/api`` (NOT ``/sparc`` -- that
  prefix is the web UI; the client library's ``/sparc`` path is a different
  deployment).
* ``POST /api/find/?limit=N`` with JSON body
  ``{"outfields": [...], "search": [[field, val...], [field, lo, hi], ...]}``
  -> JSON list whose element 0 is META and 1.. are the matched records. Box
  constraints are ``[field, lo, hi]``, categorical are ``[field, v1, v2, ...]``.
* ``POST /api/spectras/?include=...&format=json`` with body = a plain JSON list
  of sparcl_id strings -> JSON list (element 0 META, 1.. records). Each record
  carries ``flux`` (1e-17 erg/cm^2/s/AA), ``wavelength`` (AA, vacuum) and
  ``ivar`` (1e+34 erg^-2 cm^4 s^2 AA^2, i.e. inverse variance of the flux on
  the same 1e-17 scale -- so sigma_flux[1e-17] = 1/sqrt(ivar)).

Spectra are cached per source id under ~/.astroARIADNE/optspec/ (3-column CSV:
wavelength, flux, ivar) so a star is only ever fetched once -- mirroring
``_spherex.py``.
"""
import json
import logging
import os
import urllib.request
import urllib.error

import numpy as np

logger = logging.getLogger(__name__)

API_BASE = 'https://astrosparcl.datalab.noirlab.edu/api'
FIND_PATH = '/find/'
RETRIEVE_PATH = '/spectras/'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'optspec')

# Public SPARCL data-release identifiers (verified reachable without auth,
# 2026-06). BOSS-DR19 / SDSS-V are NOT public on SPARCL yet (the only BOSS
# releases are DR17 public and DR16 private); fetch_boss therefore serves
# BOSS-DR17, the newest reachable BOSS coadd -- see its docstring.
DR_DESI = 'DESI-DR1'
DR_BOSS = 'BOSS-DR17'

FLUX_SCALE = 1e-17          # record flux/ivar are on the 1e-17 erg/cm2/s/AA scale
_AA_PER_UM = 1e4            # erg/cm2/s/AA -> erg/cm2/s/um
_SUPPORT_FRAC = 1e-3        # band "support" = where transmission > frac * peak
_MIN_SAMPLES = 10           # spectrum pixels required inside a band's support
_MAX_MASKED_FRAC = 0.20     # skip a band if more in-band pixels than this have ivar<=0
# Absolute spectro-photometric calibration floor (mag), added in quadrature to
# the formal per-band error. PLACEHOLDER pending the per-band calibration
# campaign (see INTEGRATION_optspec.md): DESI/BOSS fibre spectro-photometry is
# good to ~2-3% in the optical but carries band-dependent residuals.
_CAL_FLOOR = 0.02

_HTTP_TIMEOUT = 180


# --------------------------------------------------------------------------- #
# SPARCL HTTP layer (single seam: tests stub _http_post_json)
# --------------------------------------------------------------------------- #
def _http_post_json(path, payload, qs=''):
    """POST ``payload`` as JSON to ``API_BASE + path + qs``; return parsed JSON.

    The one network seam in this module -- offline tests monkeypatch it.
    """
    url = API_BASE + path + qs
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json',
                 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def _sparcl_find(release, ra, dec, radius_arcsec):
    """Cone(box)-search a release for STAR spectra near (ra, dec).

    Returns a list of ``(sep_arcsec, sparcl_id)`` sorted nearest-first.
    """
    cosd = max(np.cos(np.radians(dec)), 1e-6)
    dra = radius_arcsec / 3600.0 / cosd
    ddec = radius_arcsec / 3600.0
    payload = {
        'outfields': ['sparcl_id', 'ra', 'dec'],
        'search': [['data_release', release], ['spectype', 'STAR'],
                   ['ra', ra - dra, ra + dra],
                   ['dec', dec - ddec, dec + ddec]],
    }
    recs = _http_post_json(FIND_PATH, payload, '?limit=50')
    out = []
    for r in recs[1:]:
        try:
            rr, dd = float(r['ra']), float(r['dec'])
        except (KeyError, TypeError, ValueError):
            continue
        sep = 3600.0 * np.hypot((rr - ra) * cosd, dd - dec)
        if sep <= radius_arcsec:
            out.append((sep, r['sparcl_id']))
    out.sort(key=lambda t: t[0])
    return out


def _sparcl_retrieve(sparcl_id):
    """Retrieve one spectrum by sparcl_id -> (wave_AA, flux_1e17, ivar) or None."""
    qs = '?include=sparcl_id,flux,wavelength,ivar&format=json'
    recs = _http_post_json(RETRIEVE_PATH, [sparcl_id], qs)
    if not recs:
        return None
    meta = recs[0] if isinstance(recs[0], dict) else {}
    status = meta.get('status', {})
    if isinstance(status, dict) and status.get('success') is False:
        logger.info('SPARCL retrieve failed: %s', status.get('info'))
        return None
    for rec in recs[1:]:
        if not isinstance(rec, dict) or 'flux' not in rec:
            continue
        wave = np.asarray(rec['wavelength'], float)
        flux = np.asarray(rec['flux'], float)
        ivar = np.asarray(rec['ivar'], float)
        if wave.size and wave.size == flux.size == ivar.size:
            order = np.argsort(wave)
            return wave[order], flux[order], ivar[order]
    return None


# --------------------------------------------------------------------------- #
# Fetchers (cached per source)
# --------------------------------------------------------------------------- #
def _query(release, ra, dec, radius_arcsec, source_id):
    """Fetch the nearest STAR spectrum in ``release``; cache per source_id.

    Returns ``(wave_AA, flux_1e17, ivar)`` or ``None``. An empty cache file
    records a confirmed no-spectrum result so the miss is not re-queried.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{source_id}.csv')
    if os.path.exists(cache):
        arr = np.loadtxt(cache, delimiter=',', ndmin=2)
        if arr.size:
            return arr[:, 0], arr[:, 1], arr[:, 2]
        return None

    try:
        hits = _sparcl_find(release, ra, dec, radius_arcsec)
    except (urllib.error.URLError, ValueError, KeyError) as e:
        logger.warning('SPARCL find failed for %s: %s', release, e)
        return None
    spec = None
    for _sep, sid in hits:
        try:
            spec = _sparcl_retrieve(sid)
        except (urllib.error.URLError, ValueError, KeyError) as e:
            logger.warning('SPARCL retrieve failed for %s: %s', sid, e)
            continue
        if spec is not None:
            break
    if spec is None:
        np.savetxt(cache, np.empty((0, 3)), delimiter=',')
        return None
    wave, flux, ivar = spec
    np.savetxt(cache, np.column_stack([wave, flux, ivar]), delimiter=',')
    return wave, flux, ivar


def _default_source_id(prefix, ra, dec):
    return f'{prefix}_{ra:.6f}{dec:+.6f}'


def fetch_desi(ra, dec, radius_arcsec=2.0, source_id=None):
    """DESI-DR1 (Milky Way Survey) optical spectrum at (ra, dec).

    Returns ``(wavelength_AA, flux_1e17_cgs, ivar)`` for the nearest DESI-DR1
    STAR spectrum within ``radius_arcsec``, or ``None``. Wavelengths are
    vacuum AA (3600-9824), flux is 1e-17 erg/cm^2/s/AA, ivar is the inverse
    variance of that flux. Cached per ``source_id`` (a Gaia id is ideal;
    defaults to a position key).
    """
    if source_id is None:
        source_id = _default_source_id('desi', ra, dec)
    return _query(DR_DESI, ra, dec, radius_arcsec, str(source_id))


def fetch_boss(ra, dec, radius_arcsec=2.0, source_id=None):
    """SDSS BOSS optical spectrum at (ra, dec).

    Returns ``(wavelength_AA, flux_1e17_cgs, ivar)`` for the nearest BOSS
    STAR spectrum within ``radius_arcsec``, or ``None``. Same units/contract
    as :func:`fetch_desi` (vacuum AA, ~3567-10400; 1e-17 erg/cm^2/s/AA).

    DR19 GAP (2026-06): SDSS-V DR19 BOSS spectra are NOT reachable
    programmatically without authentication -- SPARCL exposes only BOSS-DR17
    publicly (BOSS-DR16 is private; no DR19 release is registered), and the
    SDSS-V SAS requires credentials. This therefore serves **BOSS-DR17**, the
    newest publicly reachable BOSS coadd. When DR19 lands on SPARCL (or a
    public SAS path opens), only ``DR_BOSS`` needs updating.
    """
    if source_id is None:
        source_id = _default_source_id('boss', ra, dec)
    return _query(DR_BOSS, ra, dec, radius_arcsec, str(source_id))


# --------------------------------------------------------------------------- #
# Synthetic photometry
# --------------------------------------------------------------------------- #
def _band_support(filt, frac=_SUPPORT_FRAC):
    """(lo_AA, hi_AA) where the filter transmission exceeds ``frac * peak``."""
    wl = filt.wavelength.to('AA').magnitude
    tr = np.asarray(filt.transmit, float)
    on = wl[tr > frac * tr.max()]
    return float(on.min()), float(on.max())


def _trapz_weights(x):
    """Trapezoid quadrature weights for an ascending grid ``x``."""
    w = np.zeros_like(x, dtype=float)
    if x.size == 1:
        return w
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def synth_phot(wave_AA, flux, ivar, band_names):
    """Synthesize broadband magnitudes from a spectrum.

    Convolves the spectrum with each band's pyphot/custom curve using pyphot's
    own photon/energy-weighted ``get_flux`` (so the weighting convention is
    exact) and converts to a magnitude through ``phot_utils.flux_to_mag`` (so
    each band's AB/Vega zero point is exact). Only bands whose curve is FULLY
    inside the spectrum's wavelength coverage, with finite in-band flux and
    enough samples, are returned.

    Parameters
    ----------
    wave_AA : array
        Wavelength grid in Angstrom (ascending).
    flux : array
        Flux on the 1e-17 erg/cm^2/s/AA scale (the DESI/BOSS record units).
    ivar : array
        Inverse variance of ``flux`` (same 1e-17 scale); <=0 marks masked px.
    band_names : iterable of str
        Registry band names to attempt (the caller chooses which -- the
        gap/priority logic stays out of this module).

    Returns
    -------
    dict
        ``{band: (mag, mag_err)}`` for every band that passes the coverage and
        quality cuts. ``mag_err`` is the formal propagated error added in
        quadrature to ``_CAL_FLOOR``.
    """
    from pyphot.ezunits import unit
    from ..phot_utils import _get_filter, flux_to_mag

    wave = np.asarray(wave_AA, float)
    flux = np.asarray(flux, float)
    ivar = np.asarray(ivar, float)
    order = np.argsort(wave)
    wave, flux, ivar = wave[order], flux[order], ivar[order]
    wlo, whi = wave[0], wave[-1]

    phys = flux * FLUX_SCALE                       # erg/cm2/s/AA
    var = np.where(ivar > 0, 1.0 / np.maximum(ivar, 1e-300), np.nan)
    var_phys = var * FLUX_SCALE ** 2               # variance of phys flux

    out = {}
    for band in band_names:
        try:
            filt = _get_filter(band)
        except Exception:                          # noqa: BLE001 - unknown band
            continue
        try:
            s_lo, s_hi = _band_support(filt)
        except Exception:                          # noqa: BLE001
            continue
        # Coverage rule: the whole band curve must sit inside the spectrum.
        if not (s_lo >= wlo and s_hi <= whi):
            continue
        inb = (wave >= s_lo) & (wave <= s_hi)
        n = int(inb.sum())
        if n < _MIN_SAMPLES:
            continue
        if not np.all(np.isfinite(phys[inb])):
            continue
        masked = ivar[inb] <= 0
        if masked.mean() > _MAX_MASKED_FRAC:
            continue

        # Central value: pyphot's exact photon/energy-weighted mean f_lambda.
        val = filt.get_flux(wave * unit['AA'], phys * unit['flam'])
        f_aa = float(getattr(val, 'magnitude', getattr(val, 'value', val)))
        if not (np.isfinite(f_aa) and f_aa > 0):
            continue

        # Error: propagate the per-pixel variance through the SAME weighting
        # pyphot used (reinterp the passband onto the in-band grid).
        x = wave[inb]
        pb = filt.reinterp(x * unit['AA'])
        t = np.asarray(pb.transmit, float)
        g = (x * t) if 'photon' in filt.dtype else t
        w = _trapz_weights(x)
        gw = g * w
        denom = gw.sum()
        if not (np.isfinite(denom) and denom > 0):
            continue
        c = gw / denom                              # F_band = sum(c_i * F_i)
        vp = var_phys[inb].copy()
        if masked.any():                            # fill masked px with median
            med = np.nanmedian(vp)
            vp[~np.isfinite(vp)] = med if np.isfinite(med) else 0.0
        var_band = float(np.sum(c ** 2 * vp))
        f_aa_err = np.sqrt(max(var_band, 0.0))

        f_um = f_aa * _AA_PER_UM
        f_um_err = f_aa_err * _AA_PER_UM
        mag, mag_err = flux_to_mag(f_um, f_um_err, band)
        if not np.isfinite(mag):
            continue
        if not np.isfinite(mag_err):
            mag_err = _CAL_FLOOR
        mag_err = float(np.hypot(mag_err, _CAL_FLOOR))
        out[band] = (float(mag), mag_err)
    return out


def default_fill_bands(lo_um=0.38, hi_um=0.98):
    """Optical registry bands whose effective wavelength is in [lo_um, hi_um].

    Convenience list of fill candidates (SDSS griz, PS1 grizy, Johnson/Cousins
    BVRI, Stromgren vby, GaiaDR2v2, J-PLUS narrows). The wiring (not this
    module) decides which to actually request and how to gap-gate them.
    """
    from ..config import filter_names
    from ..phot_utils import get_effective_wavelength
    out = []
    for b in filter_names:
        try:
            lam = get_effective_wavelength(b)
        except Exception:                          # noqa: BLE001
            continue
        if np.isfinite(lam) and lo_um <= lam <= hi_um:
            out.append(str(b))
    return out
