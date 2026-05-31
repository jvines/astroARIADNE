"""Gaia XP pseudo-band synthesis (ARIADNE II groundwork).

Offline-safe by default: the network-dependent tests are skipped unless
ARIADNE_TEST_NETWORK=1 (they hit the Gaia archive). The filter-construction and
unit logic run offline.

    python tests/test_xp.py            # offline checks only
    ARIADNE_TEST_NETWORK=1 python tests/test_xp.py   # + live Gaia retrieval
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import astroARIADNE as _aa
assert 'site-packages' not in _aa.__file__, (
    f'testing the installed build, not source: {_aa.__file__}')
from astroARIADNE.librarian import _xp


def test_band_set_is_resolution_safe_and_in_range():
    for name, c, fwhm in _xp.XP_PSEUDO_BANDS:
        assert fwhm >= 15.0, f'{name} FWHM {fwhm} < 15 nm (below XP resolution)'
        assert 336.0 < c < 1020.0, f'{name} center {c} outside XP coverage'


def test_filters_build_and_match_centers():
    filts = _xp._build_pseudo_filters()
    assert len(filts) == len(_xp.XP_PSEUDO_BANDS)
    for (name, c_nm, _), (fname, filt) in zip(_xp.XP_PSEUDO_BANDS, filts):
        assert fname == name
        leff_nm = filt.leff.magnitude * 1e3       # um -> nm
        assert abs(leff_nm - c_nm) < 2.0, f'{name} leff {leff_nm} != {c_nm}'


def test_flat_spectrum_recovers_constant():
    # a photon-counting band over a flat F_lambda must return that constant.
    # Units are micron throughout (matching the SED grid columns).
    from pyphot.ezunits import unit
    wl = np.linspace(0.30, 1.05, 7501)               # um
    F = np.full_like(wl, 3.0e-13)
    wq = unit.Quantity(wl, 'um')
    fq = unit.Quantity(F, 'erg/cm**2/s/um')
    for name, filt in _xp._build_pseudo_filters():
        r = filt.get_flux(wq, fq)
        # match the grid-builder convention: normalise to erg/cm^2/s/um
        v = float(r.to('erg/cm**2/s/um').magnitude) if hasattr(r, 'to') \
            else float(r)
        assert abs(v - 3.0e-13) / 3.0e-13 < 1e-6, name


def test_leff_matches_band_centers_in_um():
    # the UnitFilter must store leff in micron at the band center (guards the
    # ~1e4 mis-scaling that a plain Filter(unit='um') string produces).
    for (name, c_nm, _), (fname, filt) in zip(
            _xp.XP_PSEUDO_BANDS, _xp._build_pseudo_filters()):
        leff_um = filt.leff.magnitude
        assert abs(leff_um - c_nm * 1e-3) < 2e-3, (
            f'{name} leff {leff_um} um != center {c_nm*1e-3} um')


def test_live_retrieval_proxima():
    if os.environ.get('ARIADNE_TEST_NETWORK') != '1':
        print('  SKIP test_live_retrieval_proxima (set ARIADNE_TEST_NETWORK=1)')
        return
    out = _xp.xp_pseudo_fluxes(5853498713190525696)   # Proxima Cen
    assert out is not None
    names, fl, er = out
    assert len(names) == len(_xp.XP_PSEUDO_BANDS)
    assert np.all(np.isfinite(fl)) and np.all(fl > 0)
    assert np.all(er > 0)
    # M dwarf: NIR continuum >> blue (steep red SED)
    assert fl[-1] > 10 * fl[0]
    # bands rise monotonically-ish blue->red for an M dwarf continuum; the
    # NIR continuum anchor is the brightest band.
    d = dict(zip(names, fl))
    assert d['XP_NIRcont'] == max(d.values())


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS {fn.__name__}')
    print(f'ALL {len(fns)} XP TESTS PASSED')
