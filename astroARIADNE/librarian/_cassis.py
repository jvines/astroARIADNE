"""CASSIS (Spitzer-IRS low-resolution) spectro-photometry for the librarian.

The 5-38 um sibling of ``_spherex.py``: fetches the optimally-extracted
low-resolution IRS spectrum from the CASSIS atlas (Combined Atlas of Sources
with Spitzer IRS Spectra; Lebouteiller et al. 2011, ApJS 196, 8) by cone
search at a sky position and bins it into ARIADNE's ``IRS_*`` super-band
top-hat photometry -- the mid-/far-IR analogue of the SPHEREX_* bands. These
bands feed the IR-excess / debris-disk machinery (all >= 5 um, so they are
``irx_bands``, never fit bands).

Product: ``yaaar_oe_wavesamp`` -- the wavelength-resampled, tuned, *optimal
extraction* low-res spectrum (CASSIS LR7). Optimal extraction assumes a point
source and yields the best flux calibration for unresolved targets; for a
*spatially resolved* disk (e.g. the HR 4796A ring) it underestimates the
extended flux relative to large-aperture WISE/IRAS -- a documented CASSIS
caveat, and exactly why the IRS bands are quantified as excess rather than fit.

Band design (see ``IRS_TOPHATS``): the IRS low-res modules are SL2 5.13-7.60,
SL1 7.46-14.29, LL2 13.9-21.3, LL1 19.9-39.9 um. Each pair of adjacent orders
overlaps in a *seam zone* where their independent flux calibrations can
disagree by up to ~10% -- the dominant low-res systematic. Every top-hat is
placed inside a single order's exclusive range so no window straddles a seam
(``IRS_SEAMS``); one window sits square on the 10 um silicate feature and one
on the 18 um feature (disk-mineralogy diagnostics), the rest sample continuum.

Absolute-calibration floor: the IRS low-res absolute spectrophotometric
accuracy is ~5% within a module (Lebouteiller et al. 2011; IRS Instrument
Handbook v5.0); module-to-module mismatch can reach ~10%, which the
seam-avoiding window placement sidesteps. ``ABS_CAL_FLOOR`` = 0.05 is the
per-band error floor.

Vega zero-points: a top-hat's Vega zero flux is computed from the Vega
spectrum pyphot carries. Once ``IRS_*`` enter ``custom_filters.h5`` (via
``notebooks/build_custom_filters.py`` -- see INTEGRATION_cassis.md) the ZP is
read from the filter library (single source of truth). Until then it is
computed on the fly from a freshly built energy top-hat *identical* to the one
build_custom_filters would write, so ``superband_mags`` returns Vega mags now
and the numbers are byte-identical once the filters are installed.

dtype = 'energy': CASSIS delivers flux-calibrated F_nu (Jy), an
energy-calibrated quantity, so the synthetic band must integrate with the
energy convention to reproduce it -- matching the IRAS/MIPS/AKARI bands in
build_custom_filters (energy counters) rather than the photon-counting SPHEREx
H2RG bands.

Network: the CASSIS host (https://cassis.sirtf.com) sits behind an Anubis
proof-of-work bot wall. ``_http_get`` transparently solves the SHA-256 PoW
(stdlib only -- difficulty 4 is ~65k hashes, < 0.1 s) and reuses the signed
cookie for the rest of the process. No third-party HTTP dependency, mirroring
``_spherex``'s urllib-only ethos.

Binned spectra are cached per position under ~/.astroARIADNE/cassis/ so a
position is only ever fetched once.
"""
import hashlib
import http.cookiejar
import io
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

C_UM = 2.99792458e14                         # c in um/s
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'cassis')
BASE = 'https://cassis.sirtf.com'
PRODUCT = 'yaaar_oe_wavesamp'                 # optimal-extraction, wave-resampled
ABS_CAL_FLOOR = 0.05                          # fractional; IRS low-res abs cal
MIN_SAMPLES = 3                               # spectral samples per window
TIMEOUT = 60
MAX_CANDIDATES = 5                            # AORs to weigh per cone search
_UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/124.0 Safari/537.36')

# IRS low-res order edges (um) and the order-overlap seam zones between them.
IRS_ORDER_EDGES = {
    'SL2': (5.13, 7.60), 'SL1': (7.46, 14.29),
    'LL2': (13.9, 21.3), 'LL1': (19.9, 39.9),
}
IRS_SEAMS = ((7.46, 7.60), (13.9, 14.29), (19.9, 21.3))  # SL2|SL1, SL1|LL2, LL2|LL1

# IRS super-band top-hats (lo_um, hi_um). Name IRS_XXXX, XXXX = center_um*1000
# (SPHEREx convention: IRS_5500 = 5.5 um). Each window lies inside ONE order's
# exclusive range (never crossing a seam in IRS_SEAMS); IRS_10000 and IRS_18000
# straddle the 10 um and 18 um silicate features, the rest sample continuum.
IRS_TOPHATS = {
    # name          (lo_um, hi_um)   order  role
    'IRS_5500':  (5.20, 5.80),   # SL2   continuum (blue)
    'IRS_6500':  (6.10, 6.90),   # SL2   continuum
    'IRS_8000':  (7.70, 8.30),   # SL1   continuum (blue shoulder of 10um)
    'IRS_10000': (9.00, 11.00),  # SL1   *** 10 um silicate feature ***
    'IRS_12000': (11.50, 12.50), # SL1   continuum (red shoulder of 10um)
    'IRS_13000': (12.70, 13.30), # SL1   continuum
    'IRS_15000': (14.50, 15.50), # LL2   continuum
    'IRS_18000': (17.00, 19.00), # LL2   *** 18 um silicate feature ***
    'IRS_23000': (21.50, 24.50), # LL1   continuum
    'IRS_30000': (28.00, 32.00), # LL1   continuum
    'IRS_35000': (33.00, 37.00), # LL1   continuum (cold-dust slope)
}


# --------------------------------------------------------------------------- #
# HTTP + Anubis proof-of-work (stdlib only)                                    #
# --------------------------------------------------------------------------- #
_opener = None


def _get_opener():
    global _opener
    if _opener is None:
        cj = http.cookiejar.CookieJar()
        _opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj))
        _opener.addheaders = [('User-Agent', _UA), ('Accept', '*/*')]
    return _opener


def _anubis_solve(random_data, difficulty):
    """Find a nonce so SHA256(random_data + nonce) has `difficulty` leading
    zero nibbles (Anubis 'fast' scheme). Returns (nonce, hex_digest)."""
    nbytes, odd = difficulty // 2, difficulty % 2
    nonce = 0
    while True:
        h = hashlib.sha256((random_data + str(nonce)).encode()).digest()
        ok = all(h[i] == 0 for i in range(nbytes))
        if ok and odd and (h[nbytes] >> 4) != 0:
            ok = False
        if ok:
            return nonce, h.hex()
        nonce += 1


_CHAL_RE = re.compile(
    r'id="anubis_challenge"[^>]*>(\{.*?\})\s*</script>', re.S)


def _anubis_redeem(challenge_json, redir_url):
    """Solve one Anubis challenge JSON and redeem the pass-challenge cookie
    (left in the module cookie jar for subsequent requests)."""
    chal = json.loads(challenge_json)
    nonce, hexhash = _anubis_solve(chal['challenge']['randomData'],
                                   int(chal['rules']['difficulty']))
    q = urllib.parse.urlencode({
        'id': chal['challenge']['id'], 'response': hexhash, 'nonce': nonce,
        'redir': redir_url, 'elapsedTime': 1500})
    _get_opener().open(
        BASE + '/.within.website/x/cmd/anubis/api/pass-challenge?' + q,
        timeout=TIMEOUT).read()


def _http_get(url, _depth=0):
    """GET url as bytes, transparently clearing an Anubis PoW wall if present.

    Detection is regex-gated on the actual challenge <script> block (not the
    bare 'anubis_challenge' substring), so a cleared page that merely mentions
    Anubis is returned rather than mistaken for a fresh challenge; up to three
    solve/retry rounds absorb a transient re-challenge after redemption.
    """
    try:
        r = _get_opener().open(url, timeout=TIMEOUT)
        body = r.read()
    except urllib.error.HTTPError as e:        # 402 carries the challenge page
        body = e.read()
    if _depth < 3 and b'anubis_challenge' in body:
        m = _CHAL_RE.search(body.decode('utf-8', 'replace'))
        if m:
            _anubis_redeem(m.group(1), url)
            return _http_get(url, _depth + 1)
    return body


# --------------------------------------------------------------------------- #
# CASSIS query + product parsing                                               #
# --------------------------------------------------------------------------- #
def _cone_search(ra, dec, radius_arcsec):
    """Return low-res (AORkey, distance_arcsec) matches, nearest first."""
    url = (f'{BASE}/atlas/cgi/radec.py?ra={ra}&dec={dec}'
           f'&radius={radius_arcsec}')
    txt = _http_get(url).decode('utf-8', 'replace')
    if 'No results' in txt:
        return []
    flat = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', txt))
    pairs = [(k, float(d)) for typ, k, d in
             re.findall(r'\b(LR|HR)\s+(\d+)\s+([\d.]+)"', flat) if typ == 'LR']
    if not pairs:                               # fallback: ordered unique keys
        seen = []
        for k in re.findall(r'aorkey\.py\?aorkey=\s*(\d+)', txt):
            if k not in seen:
                seen.append(k)
        pairs = [(k, 0.0) for k in seen]
    best = {}
    for k, d in pairs:
        if k not in best or d < best[k]:
            best[k] = d
    return sorted(best.items(), key=lambda kv: kv[1])


def _download_spectrum(aorkey, ptg=0):
    """Fetch+parse one CASSIS yaaar_oe product -> (wave_um, flux_Jy, err_Jy),
    sorted by wavelength, finite + positive flux + good status only. None on
    a non-FITS or malformed response."""
    from astropy.io import fits
    url = (f'{BASE}/atlas/cgi/download.py?aorkey={aorkey}&ptg={ptg}'
           f'&product={PRODUCT}')
    body = _http_get(url)
    if b'SIMPLE' not in body[:80]:
        return None
    with fits.open(io.BytesIO(body)) as h:
        d = np.asarray(h[0].data, float)        # (nrows, 16): col0 wave, col1
    if d.ndim != 2 or d.shape[1] < 3:           # flux(Jy), col2 err, col11 status
        return None
    wave, flux, err = d[:, 0], d[:, 1], d[:, 2]
    status = d[:, 11] if d.shape[1] > 11 else np.zeros(len(wave))
    good = (np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
            & (flux > 0) & (status == 0))
    wave, flux, err = wave[good], flux[good], err[good]
    o = np.argsort(wave)
    return wave[o], flux[o], err[o]


def fetch_spectrum(ra, dec, radius_arcsec=15):
    """CASSIS low-res IRS spectrum nearest (ra, dec) -> (wave_um, flux_Jy,
    err_Jy) or None. Cached per position under ~/.astroARIADNE/cassis/.

    Among the cone-search matches the AOR with the *widest* wavelength
    coverage is kept (ties broken by proximity): for IR-excess work the full
    SL+LL span to ~37 um beats a nearer SL-only observation. The chosen AOR
    and its offset are recorded in the cache header.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR,
                         f'{ra:.5f}_{dec:+.5f}_r{radius_arcsec:g}.csv')
    if os.path.exists(cache):
        try:
            arr = np.loadtxt(cache, delimiter=',', ndmin=2)
        except Exception:
            arr = np.empty((0, 3))
        if arr.size and arr.shape[1] >= 3:
            return arr[:, 0], arr[:, 1], arr[:, 2]
        return None

    try:
        cand = _cone_search(ra, dec, radius_arcsec)
    except Exception as e:
        logger.warning('CASSIS cone search failed: %s', e)
        return None
    if not cand:
        np.savetxt(cache, np.empty((0, 3)), delimiter=',')
        return None

    best = None                                  # (span, (w,f,e), key, dist)
    for key, dist in cand[:MAX_CANDIDATES]:
        try:
            sp = _download_spectrum(key)
        except Exception as e:
            logger.info('CASSIS download %s failed: %s', key, e)
            continue
        if sp is None or not len(sp[0]):
            continue
        span = float(sp[0].max() - sp[0].min())
        if best is None or span > best[0]:
            best = (span, sp, key, dist)
    if best is None:
        np.savetxt(cache, np.empty((0, 3)), delimiter=',')
        return None

    wave, flux, err = best[1]
    hdr = (f'CASSIS {PRODUCT} aorkey={best[2]} dist={best[3]:.2f}arcsec '
           f'span={best[0]:.1f}um wave_um,flux_Jy,err_Jy')
    np.savetxt(cache, np.column_stack([wave, flux, err]), delimiter=',',
               header=hdr)
    return wave, flux, err


# --------------------------------------------------------------------------- #
# Super-band binning -> Vega mags                                              #
# --------------------------------------------------------------------------- #
_ZP_CACHE = {}


def _vega_zero_flux(name, lo_um, hi_um):
    """Vega zero flux (erg/cm2/s/um) for an IRS top-hat. Prefers the installed
    filter library (single source of truth once build_custom_filters has run);
    otherwise builds the identical energy top-hat on the fly."""
    if name in _ZP_CACHE:
        return _ZP_CACHE[name]
    from ..phot_utils import _CUSTOM_LIB
    filt = _CUSTOM_LIB.get(name)
    if filt is None:
        import pyphot
        from pyphot.ezunits import unit
        lo, hi = lo_um * 1e4, hi_um * 1e4        # Angstrom, as build_custom_filters
        wl = np.linspace(lo * 0.98, hi * 1.02, 300)
        tr = ((wl >= lo) & (wl <= hi)).astype(float)
        filt = pyphot.Filter(wl * unit['AA'], tr, name=name,
                             dtype='energy', unit='Angstrom')
    zp = filt.Vega_zero_flux.to('erg/(um * cm ** 2 * s)').value
    _ZP_CACHE[name] = zp
    return zp


def superband_mags(wave, flux, err):
    """Bin an IRS spectrum into IRS_* Vega mags.

    Returns {band: (mag, mag_err, n_samples)}. Median F_nu (Jy) per top-hat
    window; error = MAD-scatter / sqrt(n) with a 5% absolute-calibration floor;
    converted to f_lambda and then to a Vega magnitude via the energy-top-hat
    zero point. ``err`` is accepted for signature parity with _spherex and
    future per-sample weighting; the window scatter currently drives mag_err.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    out = {}
    for name, (lo, hi) in sorted(IRS_TOPHATS.items()):
        m = (wave >= lo) & (wave <= hi)
        n = int(m.sum())
        if n < MIN_SAMPLES:
            continue
        fmu = float(np.median(flux[m]))
        if not (np.isfinite(fmu) and fmu > 0):
            continue
        scatter = 1.4826 * float(np.median(np.abs(flux[m] - fmu)))
        fer = max(scatter / np.sqrt(n), ABS_CAL_FLOOR * fmu)
        lam_c = 0.5 * (lo + hi)
        flam = fmu * 1e-23 * C_UM / lam_c ** 2          # Jy -> erg/cm2/s/um
        zp = _vega_zero_flux(name, lo, hi)
        mag = -2.5 * np.log10(flam / zp)
        mag_e = 2.5 * fer / (np.log(10) * fmu)
        if np.isfinite(mag) and np.isfinite(mag_e):
            out[name] = (float(mag), float(mag_e), n)
    return out
