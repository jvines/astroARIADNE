"""Offline unit tests for the CASSIS/IRS ingestion internals (_cassis.py).

Runs as a plain script (no pytest, no network):
    python tests/test_cassis.py
"""
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__

from astroARIADNE.librarian import _cassis  # noqa: E402
from astroARIADNE.librarian._cassis import (  # noqa: E402
    IRS_TOPHATS, IRS_SEAMS, IRS_ORDER_EDGES, ABS_CAL_FLOOR,
    superband_mags, fetch_spectrum, _anubis_solve, _vega_zero_flux)

C_UM = 2.99792458e14


def test_no_window_straddles_a_seam():
    """No IRS_TOPHATS window may cross an order seam (order-mismatch artefact),
    and every window must sit inside exactly one IRS module's exclusive range.
    """
    for name, (lo, hi) in IRS_TOPHATS.items():
        assert lo < hi, name
        # 1. disjoint from every order-overlap seam zone
        for a, b in IRS_SEAMS:
            assert not (lo < b and hi > a), \
                f'{name} ({lo}-{hi}) intersects seam {a}-{b}'
        # 2. contained in exactly one module range
        host = [m for m, (e0, e1) in IRS_ORDER_EDGES.items()
                if lo >= e0 and hi <= e1]
        assert len(host) == 1, f'{name} not in exactly one module: {host}'


def test_name_matches_center():
    """IRS_XXXX -> XXXX == center_um * 1000 (SPHEREx naming convention)."""
    for name, (lo, hi) in IRS_TOPHATS.items():
        center = 0.5 * (lo + hi)
        assert abs(int(name.split('_')[1]) - center * 1000) < 1e-6, name


def test_features_covered():
    """One window straddles the 10 um silicate peak, one the 18 um peak."""
    def covers(um):
        return any(lo <= um <= hi for lo, hi in IRS_TOPHATS.values())
    assert covers(10.0) and covers(18.0)


def test_superband_flat_fnu_recovers_known_mags():
    """A flat F_nu spectrum (constant Jy) must bin to the analytically known
    Vega mag in every window, with the error pinned at the abs-cal floor.

    A flat F_nu is NOT flat in f_lambda, so the mags are not round numbers;
    'known' means independently computable from the energy-top-hat zero point.
    """
    F0 = 0.5                                  # Jy, flat across 5-38 um
    wave = np.linspace(5.0, 38.0, 6000)
    flux = np.full_like(wave, F0)
    err = np.full_like(wave, 0.01 * F0)
    got = superband_mags(wave, flux, err)
    assert set(got) == set(IRS_TOPHATS), set(IRS_TOPHATS) - set(got)
    floor_mag_err = 2.5 * ABS_CAL_FLOOR / np.log(10)   # ~0.0543
    for name, (lo, hi) in IRS_TOPHATS.items():
        lam_c = 0.5 * (lo + hi)
        flam = F0 * 1e-23 * C_UM / lam_c ** 2
        zp = _vega_zero_flux(name, lo, hi)
        exp_mag = -2.5 * np.log10(flam / zp)
        mag, mag_err, n = got[name]
        assert abs(mag - exp_mag) < 1e-6, (name, mag, exp_mag)
        # flat spectrum -> zero scatter -> error is the abs-cal floor exactly
        assert abs(mag_err - floor_mag_err) < 1e-6, (name, mag_err)
        assert n >= 3


def test_superband_skips_thin_windows():
    """Windows with < MIN_SAMPLES spectral points are dropped, not faked."""
    wave = np.array([10.0, 10.5])             # only 2 pts, all in IRS_10000
    flux = np.array([0.3, 0.31])
    got = superband_mags(wave, flux, np.full(2, 0.01))
    assert 'IRS_10000' not in got and got == {}


def test_vega_zp_onthefly_matches_library_construction():
    """The on-the-fly energy top-hat ZP equals what an installed custom filter
    would give (same Vega, same top-hat, same energy convention) -- so mags are
    identical before and after build_custom_filters adds IRS_* to the library.
    """
    import pyphot
    from pyphot.ezunits import unit
    lo, hi = IRS_TOPHATS['IRS_18000']
    a, b = lo * 1e4, hi * 1e4
    wl = np.linspace(a * 0.98, b * 1.02, 300)
    tr = ((wl >= a) & (wl <= b)).astype(float)
    filt = pyphot.Filter(wl * unit['AA'], tr, name='IRS_18000',
                         dtype='energy', unit='Angstrom')
    lib_zp = filt.Vega_zero_flux.to('erg/(um * cm ** 2 * s)').value
    _cassis._ZP_CACHE.pop('IRS_18000', None)
    assert abs(_vega_zero_flux('IRS_18000', lo, hi) - lib_zp) < 1e-30


def test_anubis_solver():
    """The PoW solver returns a nonce whose SHA256(data+nonce) has the
    required leading zero nibbles (offline check of the bot-wall clearance)."""
    import hashlib
    data = 'deadbeefcafe'
    for diff in (1, 2, 3):
        nonce, hexd = _anubis_solve(data, diff)
        h = hashlib.sha256((data + str(nonce)).encode()).hexdigest()
        assert h == hexd
        assert h[:diff] == '0' * diff, (diff, h)


def test_cache_round_trip(monkeypatch_dir=None):
    """fetch_spectrum writes a per-position CSV and reads it back verbatim,
    without re-hitting the network on the second call (stubbed fetch)."""
    synthetic = (np.linspace(5.3, 37.0, 400),
                 np.linspace(0.2, 3.0, 400),
                 np.full(400, 0.02))
    calls = {'cone': 0, 'dl': 0}

    def fake_cone(ra, dec, radius):
        calls['cone'] += 1
        return [('3574272', 2.3), ('12687104', 1.63)]

    def fake_dl(aorkey, ptg=0):
        calls['dl'] += 1
        # the nearer SL-only AOR is narrower; widest-coverage must win
        if aorkey == '12687104':
            return (synthetic[0][:200], synthetic[1][:200], synthetic[2][:200])
        return synthetic

    orig_dir = _cassis.CACHE_DIR
    orig_cone, orig_dl = _cassis._cone_search, _cassis._download_spectrum
    with tempfile.TemporaryDirectory() as td:
        _cassis.CACHE_DIR = td
        _cassis._cone_search = fake_cone
        _cassis._download_spectrum = fake_dl
        try:
            w1, f1, e1 = fetch_spectrum(189.0042, -39.8696, radius_arcsec=15)
            assert calls['cone'] == 1 and calls['dl'] == 2     # weighed both
            # widest-coverage AOR (full 400-pt span) was kept
            assert len(w1) == 400
            files = os.listdir(td)
            assert len(files) == 1 and files[0].endswith('.csv')
            # second call: served from cache, no new network calls
            w2, f2, e2 = fetch_spectrum(189.0042, -39.8696, radius_arcsec=15)
            assert calls['cone'] == 1 and calls['dl'] == 2
            assert np.allclose(w1, w2) and np.allclose(f1, f2) \
                and np.allclose(e1, e2)
        finally:
            _cassis.CACHE_DIR = orig_dir
            _cassis._cone_search = orig_cone
            _cassis._download_spectrum = orig_dl


def test_cache_round_trip_no_results():
    """A 'no results' cone search caches an empty file and returns None,
    and the cached miss short-circuits the next call."""
    calls = {'cone': 0}

    def fake_cone(ra, dec, radius):
        calls['cone'] += 1
        return []

    orig_dir = _cassis.CACHE_DIR
    orig_cone = _cassis._cone_search
    with tempfile.TemporaryDirectory() as td:
        _cassis.CACHE_DIR = td
        _cassis._cone_search = fake_cone
        try:
            assert fetch_spectrum(0.0, 0.0) is None
            assert fetch_spectrum(0.0, 0.0) is None
            assert calls['cone'] == 1                # second call cached
        finally:
            _cassis.CACHE_DIR = orig_dir
            _cassis._cone_search = orig_cone


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v()
            print('ok', k)
    print('ALL PASS')
