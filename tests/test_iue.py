"""Offline unit tests for the IUE/INES UV ingestion internals (_iue.py).

Runs as a plain script (no pytest, no network):
    python tests/test_iue.py
"""
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__

from astroARIADNE.librarian import _iue  # noqa: E402

C_AA = 2.99792458e18                     # A/s


def _flat_ab(wave, ab_mag):
    """f_lambda [erg/cm^2/s/A] of a source with constant AB magnitude."""
    fnu = 10 ** (-0.4 * (ab_mag + 48.6))             # erg/cm^2/s/Hz
    return fnu * C_AA / wave ** 2


def test_flat_ab_source_recovers_input_mag():
    """A flat-F_nu source must synthesise back to its input AB mag to <1%.

    1% in flux is ~0.011 mag; rigorous photon-counting AB photometry through
    the pyphot GALEX curves recovers it to <0.1% (the pivot-wavelength AB
    zero-point is exact for a flat source).
    """
    wave = np.linspace(1140.0, 3360.0, 6000)
    m_in = 12.0
    flux = _flat_ab(wave, m_in)
    err = 1e-3 * flux
    out = _iue.synth_galex(wave, flux, err, n_mc=64)
    assert set(out) == {'GALEX_FUV', 'GALEX_NUV'}, out
    for band, (mag, _) in out.items():
        assert abs(mag - m_in) < 0.011, (band, mag)


def test_coverage_rule_lw_only_has_no_fuv():
    """The FUV curve (~1350-1750 A) is not inside an LW-only spectrum
    (1851-3348 A), so FUV must be absent; a full SWP+LW spectrum yields both.
    """
    lw_wave = np.linspace(1851.0, 3348.0, 1200)
    lw = _iue.synth_galex(lw_wave, _flat_ab(lw_wave, 14.0),
                          1e-3 * _flat_ab(lw_wave, 14.0), n_mc=32)
    assert 'GALEX_FUV' not in lw, lw

    full_wave = np.linspace(1150.0, 3348.0, 2400)
    full = _iue.synth_galex(full_wave, _flat_ab(full_wave, 14.0),
                            1e-3 * _flat_ab(full_wave, 14.0), n_mc=32)
    assert 'GALEX_FUV' in full and 'GALEX_NUV' in full, full


def test_coverage_rule_internal_gap_rejected():
    """A saturation-style hole (NaN block) wider than MAX_GAP_AA inside the
    FUV support must reject the band."""
    wave = np.linspace(1150.0, 3348.0, 2400)
    flux = _flat_ab(wave, 10.0)
    err = 1e-3 * flux
    hole = (wave > 1450.0) & (wave < 1550.0)          # 100 A > MAX_GAP_AA
    flux = flux.copy()
    flux[hole] = np.nan
    out = _iue.synth_galex(wave, flux, err, n_mc=16)
    assert 'GALEX_FUV' not in out, out
    assert 'GALEX_NUV' in out, out


def test_cache_round_trip():
    """Saving then loading a spectrum returns it unchanged; an empty save is a
    'no IUE data' marker that loads back as None, and a missing file as False.
    """
    with tempfile.TemporaryDirectory() as tmp:
        old = _iue.CACHE_DIR
        _iue.CACHE_DIR = tmp
        try:
            assert _iue._load_cache(10.0, -20.0) is False
            w = np.linspace(1200.0, 3300.0, 500)
            f = _flat_ab(w, 13.0)
            e = 0.05 * f
            _iue._save_cache(10.0, -20.0, w, f, e)
            got = _iue._load_cache(10.0, -20.0)
            assert got is not False and got is not None
            assert np.allclose(got[0], w) and np.allclose(got[1], f)
            assert np.allclose(got[2], e)
            _iue._save_cache(30.0, 40.0, None, None, None)
            assert _iue._load_cache(30.0, 40.0) is None
        finally:
            _iue.CACHE_DIR = old


def test_swp_lw_splice_is_continuous():
    """Splicing SWP and LW spectra of the same continuum leaves no step.

    Both cameras sample a common smooth continuum across the 1851-1978 A
    overlap; the inverse-variance merge must reproduce that continuum (no jump)
    everywhere, including across the join.
    """
    def continuum(w):                                  # smooth, monotonic
        return 1e-12 * (2500.0 / w) ** 3

    sw_w = np.linspace(1150.0, 1980.0, 495)
    lw_w = np.linspace(1851.0, 3348.0, 563)
    rng = np.random.default_rng(1)
    sw = (sw_w, continuum(sw_w) * (1 + 0.01 * rng.standard_normal(sw_w.size)),
          0.01 * continuum(sw_w))
    lw = (lw_w, continuum(lw_w) * (1 + 0.01 * rng.standard_normal(lw_w.size)),
          0.01 * continuum(lw_w))
    w, f, e = _iue._merge_sw_lw(sw, lw)
    assert np.all(np.diff(w) > 0)                      # strictly increasing
    rel = np.abs(f - continuum(w)) / continuum(w)
    assert np.max(rel) < 0.05, np.max(rel)
    # no localised step at the splice: neighbouring points track the continuum
    ov = (w > 1851.0) & (w < 1978.0)
    assert np.max(rel[ov]) < 0.03, np.max(rel[ov])


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v()
            print('ok', k)
    print('ALL PASS')
