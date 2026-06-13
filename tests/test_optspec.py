"""Offline unit tests for the optical-spectrum synthetic photometry module
(_optspec.py).

Network is stubbed (the single _http_post_json seam), so these exercise the
synthetic-photometry math, the band-coverage rule and the per-source cache
round-trip rather than the SPARCL archive.

Runs as a plain script (no pytest):
    python tests/test_optspec.py
"""
import os
import sys
import tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__

from astroARIADNE.librarian import _optspec  # noqa: E402
from astroARIADNE.phot_utils import flux_to_mag  # noqa: E402


def _flat_spectrum(c17=3.0e2, snr=50.0, lo=3000.0, hi=11000.0, n=16001):
    """Flat-F_lambda spectrum: flux constant at ``c17`` (1e-17 cgs)."""
    wave = np.linspace(lo, hi, n)
    flux = np.full_like(wave, c17)
    sigma = flux / snr
    ivar = 1.0 / sigma ** 2          # same 1e-17 scale as flux
    return wave, flux, ivar


def test_synth_phot_matches_analytic_AB_and_Vega():
    """On a flat F_lambda spectrum the photon/energy-weighted band flux is the
    constant itself, so the synthesized mag must equal flux_to_mag of that
    constant -- for both an AB band (SDSS_g) and a Vega band (STROMGREN_y)."""
    c17 = 3.0e2
    wave, flux, ivar = _flat_spectrum(c17=c17)
    res = _optspec.synth_phot(wave, flux, ivar, ['SDSS_g', 'STROMGREN_y'])
    f_um = c17 * _optspec.FLUX_SCALE * _optspec._AA_PER_UM   # erg/cm2/s/um
    for band in ('SDSS_g', 'STROMGREN_y'):
        assert band in res, f'{band} should pass coverage on 3000-11000 AA'
        mag, mag_err = res[band]
        mag_expect, _ = flux_to_mag(f_um, 0.0, band)
        assert abs(mag - mag_expect) < 1e-6, (band, mag, mag_expect)
        assert np.isfinite(mag_err) and mag_err >= _optspec._CAL_FLOOR


def test_coverage_rule_excludes_partial_bands():
    """A band whose curve runs off either edge of the spectrum is excluded;
    a band fully inside is kept."""
    # Spectrum 4000-6000 AA only.
    wave, flux, ivar = _flat_spectrum(lo=4000.0, hi=6000.0, n=4001)
    res = _optspec.synth_phot(
        wave, flux, ivar,
        ['SDSS_g',          # support 3680-5755 -> off the blue edge
         'SDSS_r',          # support 5405-7155 -> off the red edge
         'STROMGREN_y'])    # support 5150-5825 -> fully inside
    assert 'SDSS_g' not in res, 'band running off the blue edge was not excluded'
    assert 'SDSS_r' not in res, 'band running off the red edge was not excluded'
    assert 'STROMGREN_y' in res, 'fully-covered band was wrongly excluded'


def test_error_scales_with_snr():
    """Halving the S/N doubles the propagated flux error -> larger mag error."""
    wave, flux, ivar_hi = _flat_spectrum(snr=100.0)
    _, _, ivar_lo = _flat_spectrum(snr=25.0)
    e_hi = _optspec.synth_phot(wave, flux, ivar_hi, ['SDSS_r'])['SDSS_r'][1]
    e_lo = _optspec.synth_phot(wave, flux, ivar_lo, ['SDSS_r'])['SDSS_r'][1]
    assert e_lo > e_hi, (e_lo, e_hi)


def test_default_fill_bands_are_optical():
    bands = _optspec.default_fill_bands()
    assert 'SDSS_g' in bands and 'PS1_y' in bands and 'STROMGREN_v' in bands
    assert 'GaiaDR2v2_BP' in bands
    # nothing infrared / UV leaked in
    assert '2MASS_J' not in bands and 'GALEX_NUV' not in bands
    assert 'WISE_RSR_W1' not in bands


def test_cache_round_trip(tmpdir=None):
    """fetch_desi caches the spectrum; a second call hits the cache, not the
    network (the stub raises if called twice)."""
    calls = {'n': 0}

    def _stub(path, payload, qs=''):
        calls['n'] += 1
        if path == _optspec.FIND_PATH:
            return [{'META': 1},
                    {'sparcl_id': 'abc-123', 'ra': 180.0, 'dec': 0.0}]
        # retrieve
        wave = np.linspace(3600.0, 9800.0, 50)
        return [{'status': {'success': True}},
                {'sparcl_id': 'abc-123',
                 'wavelength': wave.tolist(),
                 'flux': np.full(50, 100.0).tolist(),
                 'ivar': np.full(50, 4.0).tolist()}]

    orig_post = _optspec._http_post_json
    orig_dir = _optspec.CACHE_DIR
    with tempfile.TemporaryDirectory(prefix='optspec_test_') as d:
        _optspec._http_post_json = _stub
        _optspec.CACHE_DIR = d
        try:
            r1 = _optspec.fetch_desi(180.0, 0.0, source_id='star1')
            n_after_first = calls['n']
            assert r1 is not None and len(r1[0]) == 50
            assert n_after_first >= 2, 'find + retrieve should both be called'
            # cache file written
            assert os.path.exists(os.path.join(d, 'star1.csv'))
            r2 = _optspec.fetch_desi(180.0, 0.0, source_id='star1')
            assert calls['n'] == n_after_first, 'second call must not hit network'
            assert np.allclose(r1[0], r2[0]) and np.allclose(r1[1], r2[1])
            assert np.allclose(r1[2], r2[2])
        finally:
            _optspec._http_post_json = orig_post
            _optspec.CACHE_DIR = orig_dir


def test_cache_records_empty_miss():
    """A confirmed no-spectrum result is cached as an empty file and not
    re-queried."""
    calls = {'n': 0}

    def _stub(path, payload, qs=''):
        calls['n'] += 1
        return [{'META': 1}]            # find returns no records

    orig_post = _optspec._http_post_json
    orig_dir = _optspec.CACHE_DIR
    with tempfile.TemporaryDirectory(prefix='optspec_test_') as d:
        _optspec._http_post_json = _stub
        _optspec.CACHE_DIR = d
        try:
            r1 = _optspec.fetch_boss(10.0, 10.0, source_id='miss1')
            assert r1 is None
            n1 = calls['n']
            r2 = _optspec.fetch_boss(10.0, 10.0, source_id='miss1')
            assert r2 is None
            assert calls['n'] == n1, 'cached miss should not re-query'
        finally:
            _optspec._http_post_json = orig_post
            _optspec.CACHE_DIR = orig_dir


def test_librarian_wiring_faint_gate_and_gap_rule():
    """_fetch_optspec only fires for G > 16, reuses the XP gap rule, and
    records fills in _xp_filled (so Fitter.spectra=False drops them)."""
    from astroARIADNE.librarian._api import Librarian

    def make_lib(gmag):
        lib = Librarian.__new__(Librarian)
        lib._gaia_id = 42
        lib._ignore = ()
        lib._pmra = lib._pmdec = None
        lib._gaia_ra = lib._gaia_dec = None
        lib.ra, lib.dec = 150.0, 2.0
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        lib._coord = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg)
        lib._magnitudes = {'Gaia_G': (gmag, 0.005),
                           'PS1_r': (gmag + 0.2, 0.01)}
        lib._xp_filled = []
        return lib

    calls = {'n': 0}

    def fake_fetch(ra, dec, radius_arcsec=2.0, source_id=None):
        calls['n'] += 1
        return (np.linspace(3700., 9800., 2000),
                np.full(2000, 5.0), np.full(2000, 1e4))

    def fake_synth(wave, flux, ivar, bands):
        return {'SDSS_g': (17.0, 0.01),    # gap -> fills
                'SDSS_r': (16.8, 0.01),    # blocked by observed PS1_r
                'SDSS_i': (16.6, 0.20)}    # too noisy -> rejected

    orig_fd, orig_sp = _optspec.fetch_desi, _optspec.synth_phot
    _optspec.fetch_desi = fake_fetch
    _optspec.synth_phot = fake_synth
    try:
        bright = make_lib(14.0)
        bright._fetch_optspec('DESI')
        assert calls['n'] == 0, 'fired for a bright star (XP territory)'

        faint = make_lib(17.0)
        faint._fetch_optspec('DESI')
        assert calls['n'] == 1
        assert 'SDSS_g' in faint._magnitudes
        assert 'SDSS_r' not in faint._magnitudes      # gap rule blocked it
        assert 'SDSS_i' not in faint._magnitudes      # error cut
        assert faint._xp_filled == ['SDSS_g']         # spectra-optout hook
        # the system floor rides on the fill
        assert faint._magnitudes['SDSS_g'][1] >= 0.07
    finally:
        _optspec.fetch_desi = orig_fd
        _optspec.synth_phot = orig_sp


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v()
            print('ok', k)
    print('ALL PASS')
