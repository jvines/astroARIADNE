"""Tests for the Koester DA white-dwarf grid.

Covers the built grid (schema + degenerate length-1 [Fe/H] axis), interpolation
through the ARIADNE path, and the white-dwarf prior block (uniform WD Teff/logg,
[Fe/H] pinned to the single dummy node).

Runs as a plain script (no pytest dep):
    python tests/test_koester.py
"""
import os
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
import astroARIADNE.fitter as fmod  # noqa: E402
from astroARIADNE.fitter import Fitter  # noqa: E402
from astroARIADNE.sed_library import get_interpolated_flux, load_grid  # noqa: E402
from astroARIADNE.config import (grid_wave_coverage, filter_names,  # noqa: E402
                                 irx_bands)

# The base (non-IR-excess) photosphere band set every grid must carry.
BASE_BANDS = set(filter_names) - set(irx_bands)


def test_grid_schema_and_degenerate_z():
    """HDF5 cube is (logg, teff, feh) with a length-1 feh axis and the standard
    43-band set; fully rectilinear (82x13x1) so no NaNs."""
    k = load_grid('koester')
    lg, tf, fe = k.index_columns
    # Koester (3 um WD) gains no IR band, so it carries exactly the base set.
    assert set(k.column_index) == BASE_BANDS
    assert len(fe) == 1 and float(fe[0]) == 0.0
    assert len(lg) == 13 and len(tf) == 82
    assert k.grid.shape == (13, 82, 1, 43)
    assert not np.isnan(k.grid).any()


def test_interpolation_finite_with_pinned_z():
    """Trilinear interpolation returns finite fluxes on the degenerate z-axis,
    both on a node and off-node, provided z is queried at exactly 0.0."""
    interp = load_grid('koester')
    bands = np.array(['SDSS_g', '2MASS_J', 'GALEX_NUV'])
    on = get_interpolated_flux(20000.0, 8.0, 0.0, bands, interp)
    off = get_interpolated_flux(21000.0, 8.1, 0.0, bands, interp)
    assert np.all(np.isfinite(np.asarray(on)))
    assert np.all(np.isfinite(np.asarray(off)))
    # z away from the single node -> out of bounds -> NaN (must stay pinned)
    bad = get_interpolated_flux(20000.0, 8.0, 0.5, bands, interp)
    assert np.all(np.isnan(np.asarray(bad)))


def test_grid_setter_loads_koester():
    f = Fitter()
    f.grid = 'koester'
    assert f._interpolator is not None


def test_wave_coverage_registered():
    assert grid_wave_coverage.get('koester') == 3.0


def _wd_fitter():
    """Minimal Fitter wired for the Koester prior block, with a faked star."""
    f = Fitter.__new__(Fitter)
    f.__dict__['_guard_active'] = False
    f._grid = 'koester'
    f._norm = False
    f.coordinator = np.zeros(6)
    f.fixed = np.zeros(6)
    f.star = types.SimpleNamespace(
        spectroscopic_params=None, rave_params=None,
        temp=25000.0, lum=1e-3, get_logg=False,
        filter_names=np.array(['GaiaDR2v2_G', '2MASS_J']),
        mags=np.array([13.0, 14.0]),
        filter_mask=np.array([True, True]),
        flux=np.array([1e-12, 1e-12]), flux_er=np.array([1e-13, 1e-13]),
        Av=0.0, Av_e=None, dist=100.0, dist_e=5.0,
    )
    fmod.order = np.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av'])
    return f


def test_wd_prior_block():
    f = _wd_fitter()
    d = f._default_priors()
    # Teff: uniform 5000-80000 K
    assert f.prior_sources['teff'] == 'koester_wd_uniform'
    assert abs(d['teff'].ppf(0.0) - 5000.0) < 1e-6
    assert abs(d['teff'].ppf(1.0) - 80000.0) < 1e-6
    # logg: uniform 6.5-9.5
    assert f.prior_sources['logg'] == 'koester_wd_uniform'
    assert abs(d['logg'].ppf(0.0) - 6.5) < 1e-6
    assert abs(d['logg'].ppf(1.0) - 9.5) < 1e-6
    # [Fe/H]: pinned to the single 0.0 node (fixed, index 2)
    assert f.prior_sources['z'] == 'koester_fixed_zero'
    assert d['z'] is None
    assert f.coordinator[2] == 1
    assert f.fixed[2] == 0.0
    # radius: degenerate WD range (must reach below the 0.05 R_sun MS floor)
    assert f.prior_sources['rad'] == 'koester_wd_uniform'
    assert abs(d['rad'].ppf(0.0) - 0.003) < 1e-6
    assert abs(d['rad'].ppf(1.0) - 0.05) < 1e-6


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('ok', fn.__name__)
    print('ALL PASS')
