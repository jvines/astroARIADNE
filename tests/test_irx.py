"""Tests for reach-gated IR-excess quantification (Fitter._compute_irx).

Strategy under test (per design): IR excess is quantified only on bands every
in-play grid reaches -- single fit -> that grid's reach; BMA -> the intersection
of the ensemble grids' reach. Bands the star has but that lie beyond the common
reach are recorded plot-only (model NaN). Excess significance is
sigma = (obs - model) / sqrt(obs_err^2 + model_err^2) against the (evidence-
weighted, for BMA) posterior-predictive model flux.

Runs as a plain script (no pytest):
    python tests/test_irx.py
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
from astroARIADNE.sed_library import load_grid, model_grid  # noqa: E402
from astroARIADNE.config import filter_names  # noqa: E402
from extinction import fitzpatrick99  # noqa: E402

FN = np.asarray(filter_names)
W3 = int(np.where(FN == 'WISE_RSR_W3')[0][0])
THETA = dict(teff=4500., logg=4.6, z=0.0, dist=30., rad=0.6, Av=0.0)


def _avlaw(w, a, rv):
    return fitzpatrick99(w, a, rv)


def _grid_with_w3():
    """A phoenix grid extended with a finite WISE_RSR_W3 column (= W2's)."""
    g = load_grid('phoenix')
    w2 = g.column_index['WISE_RSR_W2']
    g.grid = np.concatenate([g.grid, g.grid[:, :, :, w2:w2 + 1]], axis=-1)
    g.column_index = dict(g.column_index)
    g.column_index['WISE_RSR_W3'] = g.grid.shape[-1] - 1
    g.finite_bands = frozenset(set(g.finite_bands) | {'WISE_RSR_W3'})
    return g


def _fitter(obs_w3, err_w3, irx=True):
    f = Fitter.__new__(Fitter)
    f.__dict__['_guard_active'] = False
    f._norm = False
    f.binary = False
    f._grid = 'phoenix'
    flux = np.zeros(len(FN)); flux[W3] = obs_w3
    ferr = np.zeros(len(FN)); ferr[W3] = err_w3
    wave = np.zeros(len(FN)); wave[W3] = 12.8
    f.star = types.SimpleNamespace(
        filter_names=FN, wave=wave, flux=flux, flux_er=ferr,
        irx_filter_mask=np.array([W3]) if irx else np.array([], int))
    return f


def _out(weights=None, originals=None):
    o = dict(best_fit_averaged=dict(THETA),
             posterior_samples={k: np.full(60, v) for k, v in THETA.items()})
    if weights is not None:
        o['weights'] = weights
        o['originals'] = originals
    return o


def _photo_w3(grid):
    """Photospheric model flux at W3 for the fixed THETA on a given grid."""
    f = _fitter(0, 0)
    theta = f._irx_theta(THETA)
    m = model_grid(theta, np.array(['WISE_RSR_W3']), np.array([12.8]),
                   grid, False, _avlaw)
    return float(np.asarray(m).ravel()[0])


# ------------------------------ tests ---------------------------------------
def test_no_irx_returns_empty():
    f = _fitter(1e-13, 1e-14, irx=False)
    assert f._compute_irx(_out(), ['phoenix'], _avlaw) == {}


def test_plot_only_when_grid_cannot_reach():
    # real phoenix stops at 5 um -> W3 is beyond reach -> plot-only, not fit.
    f = _fitter(1e-13, 1e-14)
    r = f._compute_irx(_out(), ['phoenix'], _avlaw)
    assert r['WISE_RSR_W3']['quantified'] is False
    assert np.isnan(r['WISE_RSR_W3']['model'])
    assert r['_summary']['plot_only'] == ['WISE_RSR_W3']
    assert r['_summary']['quantified'] == []


def test_quantified_excess_significance():
    g = _grid_with_w3()
    orig = fmod.load_grid
    fmod.load_grid = lambda name: g
    try:
        mod = _photo_w3(g)
        # no excess: obs == model -> sigma ~ 0
        r0 = _fitter(mod, mod * 0.1)._compute_irx(_out(), ['phoenix'], _avlaw)
        assert r0['WISE_RSR_W3']['quantified']
        assert abs(r0['WISE_RSR_W3']['sigma']) < 0.5
        assert r0['_summary']['significant'] == []
        # 2x excess with 10% errors -> sigma ~ 10, flagged significant
        r1 = _fitter(2 * mod, mod * 0.1)._compute_irx(_out(), ['phoenix'],
                                                      _avlaw)
        assert r1['WISE_RSR_W3']['sigma'] > 5
        assert r1['_summary']['significant'] == ['WISE_RSR_W3']
    finally:
        fmod.load_grid = orig


def test_bma_reach_is_intersection():
    """W3 is quantified only if EVERY ensemble grid reaches it."""
    g_w3 = _grid_with_w3()        # reaches W3
    g_no = load_grid('phoenix')   # does not
    orig = fmod.load_grid
    grids = ['phoenix', 'btsettl']
    weights = {'phoenix': 0.6, 'btsettl': 0.4}
    originals = {g: {k: np.full(60, v) for k, v in THETA.items()}
                 for g in grids}
    mod = _photo_w3(g_w3)
    try:
        # one member (btsettl) cannot reach W3 -> intersection excludes it
        fmod.load_grid = lambda n: g_w3 if n == 'phoenix' else g_no
        r = _fitter(2 * mod, mod * 0.1)._compute_irx(
            _out(weights, originals), grids, _avlaw)
        assert r['WISE_RSR_W3']['quantified'] is False

        # both members reach W3 -> quantified, excess recovered
        fmod.load_grid = lambda n: g_w3
        r = _fitter(2 * mod, mod * 0.1)._compute_irx(
            _out(weights, originals), grids, _avlaw)
        assert r['WISE_RSR_W3']['quantified']
        assert r['WISE_RSR_W3']['sigma'] > 5
    finally:
        fmod.load_grid = orig


def test_reliability_gating_respects_coverage():
    """grid_irx_reach drops physically-finite bands that fall beyond a grid's
    grid_wave_coverage cutoff (e.g. Coelho reaches ~8 um but is only trustworthy
    to 0.9 um, so its IRAC 5.8/8.0 are excluded from IRX)."""
    from astroARIADNE.sed_library import grid_irx_reach, load_grid
    from astroARIADNE.config import grid_wave_coverage
    from astroARIADNE.phot_utils import get_effective_wavelength
    for g in ['coelho', 'btsettl']:
        gi = load_grid(g)
        reach = grid_irx_reach(g, gi)
        assert reach <= set(gi.finite_bands)             # never exceeds physical
        cov = grid_wave_coverage.get(g)
        if cov is not None:
            assert all(get_effective_wavelength(b) <= cov for b in reach), g
    # a capped optical grid contributes no IR-excess band at all
    coelho = load_grid('coelho')
    from astroARIADNE.config import irx_bands
    assert not (grid_irx_reach('coelho', coelho) & set(irx_bands))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
    print('ALL PASS')
