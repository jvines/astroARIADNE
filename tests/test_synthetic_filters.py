"""User-specified synthetic filters from the spectra cache (#88).

Offline: a tiny fake spectra cache (blackbody-like spectra) stands in for the
3 GB real one; the band grid is replaced by the same convolution of that cache.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astropy.units as u
import extinction
import h5py
import numpy as np
import pyphot
import pytest

import astroARIADNE.config as config
import astroARIADNE.fitter as fitter_mod
from astroARIADNE import synth_phot as sp
from astroARIADNE.fitter import Fitter

W = np.linspace(0.3, 3.0, 1500)          # um
BANDS = ['2MASS_J', '2MASS_H']


def _bb(teff, w=W):
    """pi * B_lambda in erg s-1 cm-2 um-1 (a surface flux)."""
    h, c, k = 6.62607e-27, 2.99792458e10, 1.380649e-16
    lam = w * 1e-4                                   # cm
    b = 2 * h * c ** 2 / lam ** 5 / np.expm1(h * c / (lam * k * teff))
    return np.pi * b * 1e-4                          # per cm -> per um


def _write_cache(path):
    with h5py.File(path, 'w') as h:
        for name, teffs in (('btsettl', [3000., 3500., 4000., 4500.]),
                            ('btsettl_cool', [2500., 3000., 3500.])):
            t, g, z, f = [], [], [], []
            for tt in teffs:
                for lg in (4.0, 5.0):
                    t.append(tt), g.append(lg), z.append(0.0)
                    # cool group deliberately differs where nodes overlap
                    f.append(_bb(tt) * (1.0 if name == 'btsettl' else 0.5))
            grp = h.create_group(name)
            grp['wavelength'] = W
            grp['teff'], grp['logg'], grp['z'] = t, g, z
            grp['flux'] = np.array(f)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    p = str(tmp_path / 'spectra_cache.h5')
    _write_cache(p)
    monkeypatch.setattr(config, 'spectra_cache', p)
    return p


def _gauss(name, lam0, sig, dtype='photon'):
    wl = np.linspace(lam0 - 5 * sig, lam0 + 5 * sig, 200)
    tr = np.exp(-0.5 * ((wl - lam0) / sig) ** 2)
    return (name, wl, tr) if dtype == 'photon' else (name, wl, tr, dtype)


def _pyphot_flux(sf, f):
    return sf.filt.get_flux(
        (W * u.um).to('AA'),
        (f * 1e-4) * u.erg / u.s / u.cm ** 2 / u.AA).to(
        'erg/(s cm2 um)').value


@pytest.mark.parametrize('spec', ['2MASS_J', _gauss('g1', 1.6, 0.1),
                                  _gauss('g2', 1.2, 0.05, 'energy')])
def test_convolution_matches_pyphot(spec):
    sf = sp.resolve_filter(spec)
    F = np.array([_bb(3000.), _bb(4500.)])
    got = sp.convolve(F, *sp.coeff_matrix([sf], W))[:, 0]
    ref = np.array([_pyphot_flux(sf, f) for f in F])
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_nan_outside_cached_span():
    sf = sp.resolve_filter(_gauss('mid_ir', 10.0, 0.5))
    got = sp.convolve(np.array([_bb(3000.)]), *sp.coeff_matrix([sf], W))
    assert np.isnan(got).all()


def test_cube_prefers_btsettl_and_marks_missing_nodes(cache):
    sf = sp.resolve_filter(_gauss('g1', 1.6, 0.1))
    cube = sp.build_cube(cache, [sf])
    lg, tf, zz = cube.index_columns
    assert list(tf) == [2500., 3000., 3500., 4000., 4500.]
    own = sp.convolve(np.array([_bb(3000.)]), *sp.coeff_matrix([sf], W))[0, 0]
    # 3000 K exists in both groups: btsettl (factor 1.0) wins
    assert cube.grid[0, 1, 0, 0] == pytest.approx(own, rel=1e-12)
    # 2500 K has no btsettl node but a btsettl_cool one
    assert np.isfinite(cube.grid[0, 0, 0, 0])
    assert cube.grid.shape == (2, 5, 1, 1)


def _fitter(tmp_path, specs, monkeypatch, norm=False):
    monkeypatch.setattr(fitter_mod, 'filter_names', np.array(BANDS))
    filts = [sp.resolve_filter(b) for b in BANDS]
    band_cube = sp.build_cube(config.spectra_cache, filts)
    monkeypatch.setattr(Fitter, 'load_interpolator',
                        staticmethod(lambda model: band_cube))
    f = Fitter.__new__(Fitter)
    f.out_folder = str(tmp_path)
    f._norm = norm
    f._av_law = extinction.fitzpatrick99
    f.synthetic_filters = specs
    f.star = SimpleNamespace(filter_names=np.array(BANDS),
                             used_filters=np.array([1, 0]))
    return f


def _samples(n=400, seed=1):
    r = np.random.default_rng(seed)
    return {'teff': r.normal(3600, 60, n), 'logg': r.uniform(4.2, 4.8, n),
            'z': np.zeros(n), 'Av': r.uniform(0, 0.2, n),
            'dist': r.normal(20, 0.2, n), 'rad': r.normal(0.5, 0.01, n)}


def test_user_filter_table_and_files(cache, tmp_path, monkeypatch):
    specs = ['2MASS_H', _gauss('g_nir', 1.6, 0.1), 'GROUND_JOHNSON_V',
             _gauss('mid_ir', 10.0, 0.5)]
    f = _fitter(tmp_path, specs, monkeypatch)
    out = {}
    f._write_synthetic(out, _samples())
    txt = open(tmp_path / 'synthetic_user_filters.dat').read().splitlines()
    cols = [l for l in txt if l.startswith('# filter')][0][2:].split('\t')
    assert cols == ['filter', 'wavelength_um', 'flux', 'flux_err', 'mag_vega',
                    'mag_vega_err', 'mag_ab', 'mag_ab_err', 'source']
    rows = {l.split('\t')[0]: l.split('\t') for l in txt if l[0] != '#'}
    assert rows['2MASS_H'][-1] == 'grid'
    assert rows['g_nir'][-1] == 'spectra_cache'
    assert np.isfinite(float(rows['g_nir'][4])) and np.isfinite(float(rows['g_nir'][6]))
    assert np.isnan(float(rows['mid_ir'][2]))             # not spanned
    # Johnson_V (0.55 um) is inside the fake span and gets synthesized
    assert rows['GROUND_JOHNSON_V'][-1] == 'spectra_cache'
    # appended to the main tables with observed=0
    mags = open(tmp_path / 'synthetic_magnitudes.dat').read()
    gl = [l for l in mags.splitlines() if l.startswith('g_nir')][0].split('\t')
    assert gl[-1] == '0' and len(gl) == 7
    assert 'g_nir\t' in open(tmp_path / 'synthetic_fluxes.dat').read()
    assert 'mid_ir' not in mags
    assert set(out['synthetic_user_filters']) == {'2MASS_H', 'g_nir',
                                                  'GROUND_JOHNSON_V', 'mid_ir'}
    assert 'g_nir' in out['synthetic_photometry']


def test_cache_path_agrees_with_grid_path(cache, tmp_path, monkeypatch):
    """A tuple copy of 2MASS_J through the cube equals the band-grid value."""
    sf = sp.resolve_filter('2MASS_J')
    wl = sf.filt.wavelength.to('um').value
    tup = ('J_copy', wl, np.asarray(sf.filt.transmit))
    f = _fitter(tmp_path, ['2MASS_J', tup], monkeypatch)
    out = {}
    f._write_synthetic(out, _samples())
    u_ = out['synthetic_user_filters']
    assert u_['J_copy']['flux'] == pytest.approx(u_['2MASS_J']['flux'], rel=1e-8)
    assert u_['J_copy']['flux_err'] == pytest.approx(u_['2MASS_J']['flux_err'],
                                                     rel=1e-6)
    assert np.isfinite(u_['J_copy']['mag_ab'])


def test_missing_cache_warns_once_and_does_not_raise(tmp_path, monkeypatch,
                                                     caplog):
    # build with a cache first, then remove it
    p = str(tmp_path / 'c.h5')
    _write_cache(p)
    monkeypatch.setattr(config, 'spectra_cache', p)
    f = _fitter(tmp_path, ['2MASS_H', _gauss('g_nir', 1.6, 0.1)], monkeypatch)
    monkeypatch.setattr(config, 'spectra_cache', None)
    out = {}
    with caplog.at_level('WARNING'):
        f._write_synthetic(out, _samples())
    msgs = [r.getMessage() for r in caplog.records if 'spectra cache' in r.getMessage()]
    assert len(msgs) == 1 and 'fetch_spectra_cache()' in msgs[0]
    assert set(out['synthetic_user_filters']) == {'2MASS_H'}   # grid band only
    assert (tmp_path / 'synthetic_magnitudes.dat').exists()


def test_bad_spec_does_not_break_save(cache, tmp_path, monkeypatch, caplog):
    f = _fitter(tmp_path, [42, 'NOT_A_FILTER', _gauss('g_nir', 1.6, 0.1)],
                monkeypatch)
    out = {}
    with caplog.at_level('WARNING'):
        f._write_synthetic(out, _samples())
    assert caplog.text.count('entry skipped') == 2   # bad entries only
    assert set(out['synthetic_user_filters']) == {'g_nir'}
    # a failure inside the step must not raise either
    monkeypatch.setattr(Fitter, '_synthetic_user_filters',
                        lambda *a, **k: 1 / 0)
    f._write_synthetic({}, _samples())
