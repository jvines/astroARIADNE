"""IR-excess error budgets ('covariant', 'independent', 'legacy') and the
selectable companion injection model.

'independent' adds each survey's absolute calibration in quadrature to the catalogue
error and widens the photosphere-model uncertainty to the equal-weight
inter-grid spread; 'legacy' reproduces the old 2% floor and the
evidence-weighted posterior-predictive spread exactly.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, REPO)
import astroARIADNE.fitter as fmod  # noqa: E402
from astroARIADNE.fitter import Fitter  # noqa: E402
from astroARIADNE.config import irx_bands  # noqa: E402
from astroARIADNE.sed_library import load_grid, model_grid  # noqa: E402
from astroARIADNE.phot_utils import get_effective_wavelength  # noqa: E402
from test_irx import (THETA, _avlaw, _fitter, _grid_with_w3, _out,  # noqa: E402
                      _photo_w3)


# ---------------------------------------------------------------- calibration
def test_calibration_table_covers_every_excess_band():
    """Every band that can be quantified as excess has a sourced calibration
    term: the irx_bands set and the 2.75-5 um block fit_irx can route."""
    routable = set(irx_bands) | set(Fitter._IRX_EXTRA_BANDS)
    missing = sorted(b for b in routable if b not in fmod._IRX_ABS_CAL)
    assert missing == [], missing
    v = np.array(list(fmod._IRX_ABS_CAL.values()))
    assert np.all((v >= 0.01) & (v <= 0.4))


def test_calibration_values():
    assert fmod._irx_abs_cal_frac('WISE_RSR_W3') == 0.045
    assert fmod._irx_abs_cal_frac('WISE_RSR_W4') == 0.057
    assert fmod._irx_abs_cal_frac('SPITZER_MIPS_70') == 0.05
    assert fmod._irx_abs_cal_frac('HERSCHEL_SPIRE_PSW') == 0.055
    assert fmod._irx_abs_cal_frac('SPHEREX_4900') == 0.022
    assert fmod._irx_abs_cal_frac('SPHEREX_3050') == 0.014
    # untabulated band (e.g. Ks routed by excess_from_um=2.2) -> 2% fallback
    assert fmod._irx_abs_cal_frac('2MASS_Ks') == fmod._IRX_ABS_CAL_FLOOR


def test_obs_error_full_vs_legacy():
    F = 1.0
    # tiny catalogue error: legacy floors at 2%, full adds 4.5% in quadrature
    assert fmod._irx_obs_error('WISE_RSR_W3', F, 1e-4, 'legacy') == 0.02
    assert np.isclose(fmod._irx_obs_error('WISE_RSR_W3', F, 1e-4, 'independent'),
                      np.hypot(1e-4, 0.045))
    # large catalogue error: legacy leaves it alone, full still adds the cal
    assert fmod._irx_obs_error('WISE_RSR_W3', F, 0.3, 'legacy') == 0.3
    assert np.isclose(fmod._irx_obs_error('WISE_RSR_W3', F, 0.3, 'independent'),
                      np.hypot(0.3, 0.045))
    # full is never smaller than the catalogue error
    for e in (1e-5, 0.01, 0.1, 1.0):
        assert fmod._irx_obs_error('IRAS_60', F, e, 'independent') >= e


# ---------------------------------------------------------------- model term
def test_model_variance_single_grid_is_posterior_predictive():
    assert fmod._irx_model_variance(4.0, [10.0], [4.0], [1.0], 10.0) == 4.0


def test_model_variance_peaked_weights_keep_grid_spread():
    """Evidence weights 0.999/0.001: the posterior-predictive spread ignores
    the disagreeing grid; the full term keeps the equal-weight spread."""
    mu = np.array([1.0, 1.2])
    var = np.array([1e-4, 1e-4])
    w = np.array([0.999, 0.001])
    mean = float(np.sum(w * mu))
    pp = float(np.sum(w * var) + np.sum(w * (mu - mean) ** 2))
    full = fmod._irx_model_variance(pp, mu, var, w, mean)
    expect = float(np.sum(w * var) + np.mean((mu - mean) ** 2))
    assert np.isclose(full, expect)
    assert full > 50 * pp
    # equal weights: never below the posterior-predictive variance
    w = np.array([0.5, 0.5]); mean = 1.1
    pp = float(np.sum(w * var) + np.sum(w * (mu - mean) ** 2))
    assert fmod._irx_model_variance(pp, mu, var, w, mean) >= pp
    # non-finite grids are ignored
    assert fmod._irx_model_variance(2.0, [np.nan], [np.nan], [1.0], 1.0) == 2.0


# ---------------------------------------------------------- _compute_irx path
def _with_grids(mapping, fn):
    orig = fmod.load_grid
    fmod.load_grid = lambda n: mapping[n]
    try:
        return fn()
    finally:
        fmod.load_grid = orig


def test_compute_irx_legacy_reproduces_old_budget():
    g = _grid_with_w3()
    mod = _photo_w3(g)

    def run(budget):
        f = _fitter(2 * mod, mod * 1e-3)
        f.irx_error_budget = budget
        return f._compute_irx(_out(), ['phoenix'], _avlaw)
    leg = _with_grids({'phoenix': g}, lambda: run('legacy'))['WISE_RSR_W3']
    full = _with_grids({'phoenix': g}, lambda: run('independent'))['WISE_RSR_W3']
    obs = 2 * mod
    assert np.isclose(leg['obs_err_eff'], max(mod * 1e-3, 0.02 * obs))
    assert np.isclose(full['obs_err_eff'], np.hypot(mod * 1e-3, 0.045 * obs))
    assert full['sigma'] < leg['sigma']
    assert full['model'] == leg['model']
    assert leg['cal_frac'] == 0.02 and full['cal_frac'] == 0.045


def test_compute_irx_default_is_covariant_and_rejects_unknown():
    g = _grid_with_w3()
    mod = _photo_w3(g)
    f = _fitter(2 * mod, mod * 0.1)        # built without __init__
    r = _with_grids({'phoenix': g},
                    lambda: f._compute_irx(_out(), ['phoenix'], _avlaw))
    assert r['_summary']['error_budget'] == 'covariant'
    f.irx_error_budget = 'bogus'
    with pytest.raises(ValueError):
        _with_grids({'phoenix': g},
                    lambda: f._compute_irx(_out(), ['phoenix'], _avlaw))


def test_compute_irx_full_adds_intergrid_spread():
    """Two grids whose W3 predictions differ by 30%, evidence 0.999/0.001:
    legacy model_err is ~0 (identical posterior samples), full carries the
    equal-weight spread (~15% of the model)."""
    ga = _grid_with_w3()
    gb = _grid_with_w3()
    gb.grid = gb.grid.copy()
    gb.grid[..., gb.column_index['WISE_RSR_W3']] *= 1.3
    grids = ['phoenix', 'btsettl']
    weights = {'phoenix': 0.999, 'btsettl': 0.001}
    originals = {g: {k: np.full(60, v) for k, v in THETA.items()}
                 for g in grids}
    mod = _photo_w3(ga)

    def run(budget):
        f = _fitter(2 * mod, mod * 0.05)
        f.irx_error_budget = budget
        return f._compute_irx(_out(weights, originals), grids, _avlaw)
    mp = {'phoenix': ga, 'btsettl': gb}
    leg = _with_grids(mp, lambda: run('legacy'))['WISE_RSR_W3']
    full = _with_grids(mp, lambda: run('independent'))['WISE_RSR_W3']
    assert full['model'] == leg['model']
    assert full['model_err_pp'] == leg['model_err']
    assert full['model_err'] > 0.1 * full['model']
    assert full['model_err'] > 5 * leg['model_err']
    assert np.isclose(full['model_err_grid'], full['model_err'], rtol=0.05)


# ---------------------------------------------------- covariant calibration
def test_cal_split_preserves_band_totals():
    """The split keeps each band's total calibration error, except where a
    source documents a different combination (SPIRE: 4% + 1.5% in
    quadrature instead of linearly)."""
    for b, (inst, a, r) in fmod._IRX_CAL_SPLIT.items():
        assert inst == fmod._irx_instrument(b)
        if inst == 'SPIRE':
            assert (a, r) == (0.04, 0.015)
            continue
        assert np.isclose(np.hypot(a, r), fmod._IRX_ABS_CAL[b], atol=6e-4), b
    assert fmod._IRX_CAL_SPLIT['WISE_RSR_W1'][1:] == (0.024, 0.0)
    assert fmod._IRX_CAL_SPLIT['IRAS_60'][1:] == (0.03, 0.05)
    assert fmod._IRX_CAL_SPLIT['SPHEREX_4900'][1:] == (0.02, 0.01)
    assert fmod._irx_instrument('2MASS_Ks') is None


def test_cal_offdiag_structure():
    bands = ['WISE_RSR_W3', 'WISE_RSR_W4', 'SPITZER_MIPS_24', '2MASS_Ks']
    F = np.array([2.0, 3.0, 5.0, 7.0])
    C = fmod._irx_cal_offdiag(bands, F)
    assert np.allclose(np.diag(C), 0)
    assert np.allclose(C, C.T)
    assert np.isclose(C[0, 1], 0.024 ** 2 * 2.0 * 3.0)
    assert C[0, 2] == 0 and C[1, 3] == 0 and C[2, 3] == 0


def _sel_inputs():
    from scipy.constants import h, c, k as kB
    names = np.array(['WISE_RSR_W2', 'WISE_RSR_W3', 'WISE_RSR_W4'])
    waves = np.array([get_effective_wavelength(b) for b in names])
    ll = waves * 1e-6
    B = (2 * h * c ** 2 / ll ** 5) / np.expm1(h * c / (ll * kB * 200.0)) * 1e-3
    return names, waves, 3e-17 * B


def test_model_selection_diagonal_cov_matches_diagonal_errors():
    """A diagonal excess_cov reproduces the diagonal-error evidences."""
    names, waves, Fd = _sel_inputs()
    E = 0.1 * Fd

    def sel(**kw):
        return fmod._excess_model_selection(names, waves, Fd, E, [], 0., 0.,
                                            30., 1.0, False, _avlaw, 5000.,
                                            **kw)
    a = sel()
    b = sel(excess_cov=np.diag(E ** 2))
    for k in ('p_none', 'p_disk', 'p_companion'):
        assert np.isclose(a[k] or 0.0, b[k] or 0.0, rtol=1e-8, atol=1e-300)
    assert np.isclose(a['disk']['t_dust'], b['disk']['t_dust'], rtol=1e-8)


def test_common_scale_offset_is_absorbed_by_covariance():
    """An excess that is just a common calibration offset of one instrument
    (every band 5% high) is much more probable as noise under the covariant
    budget than under the independent one."""
    names, waves, _ = _sel_inputs()
    obs = np.array([1.0, 0.5, 0.25])
    exc = 0.05 * obs                                  # +5% in all three
    a = 0.05
    E = np.hypot(0.01 * obs, a * obs)                 # same marginal errors
    Cind = np.diag(E ** 2)
    Ccov = Cind + a * a * (np.outer(obs, obs) - np.diag(obs ** 2))
    chi_ind = exc @ np.linalg.solve(Cind, exc)
    chi_cov = exc @ np.linalg.solve(Ccov, exc)
    assert chi_cov < 0.5 * chi_ind

    def p_none(C):
        return fmod._excess_model_selection(
            names, waves, exc, E, [], 0., 0., 30., 1.0, False, _avlaw, 5000.,
            excess_cov=C)['p_none']
    assert p_none(Ccov) > p_none(Cind)


def test_compute_irx_covariant_passes_full_matrix():
    """'covariant' hands the model selection a matrix whose off-diagonal is
    the instrument-level calibration term; 'independent' passes none."""
    seen = {}
    orig = fmod._excess_model_selection

    def spy(*a, **kw):
        seen['cov'] = kw.get('excess_cov')
        seen['err'] = np.asarray(a[3], float)
        return None
    g = _grid_with_w3()
    mod = _photo_w3(g)
    try:
        fmod._excess_model_selection = spy
        for budget in ('covariant', 'independent'):
            f = _fitter(2 * mod, mod * 0.05)
            # second WISE band: W4 reusing the W3 column
            W4 = int(np.where(np.asarray(f.star.filter_names)
                              == 'WISE_RSR_W4')[0][0])
            g.column_index['WISE_RSR_W4'] = g.column_index['WISE_RSR_W3']
            g.finite_bands = frozenset(set(g.finite_bands) | {'WISE_RSR_W4'})
            f.star.flux[W4] = 3 * mod
            f.star.flux_er[W4] = mod * 0.05
            f.star.wave[W4] = 22.0
            f.star.irx_filter_mask = np.array([f.star.irx_filter_mask[0], W4])
            f.irx_error_budget = budget
            f.star.dist = 30.0
            o = _out()
            o['best_fit_averaged']['lum'] = 0.2
            _with_grids({'phoenix': g},
                        lambda: f._compute_irx(o, ['phoenix'], _avlaw))
            if budget == 'independent':
                assert seen['cov'] is None
            else:
                C = seen['cov']
                assert np.allclose(np.diag(C), seen['err'] ** 2)
                assert np.isclose(C[0, 1], 0.024 ** 2 * (2 * mod) * (3 * mod))
    finally:
        fmod._excess_model_selection = orig


def test_fitter_accepts_irx_options():
    f = Fitter()
    assert f.irx_error_budget == 'covariant' and f.irx_companion_grids is None
    f.irx_error_budget = 'legacy'
    f.irx_companion_grids = ('atmo',)
    with pytest.raises(AttributeError):
        f.irx_error_budgets = 'covariant'                     # typo guard


# ------------------------------------------------------- injection model
_NAMES = np.array(['2MASS_J', '2MASS_Ks', 'WISE_RSR_W1', 'WISE_RSR_W2',
                   'WISE_RSR_W3', 'WISE_RSR_W4', 'HERSCHEL_SPIRE_PSW'])
_WAVE = np.array([get_effective_wavelength(b) for b in _NAMES])


def test_injection_default_is_sonora_and_differs_from_btsettl():
    f_s, used = fmod.companion_injection_flux(_NAMES, _WAVE, 1500., 0.1, 10.)
    assert used == 'sonora'
    f_b, used_b = fmod.companion_injection_flux(_NAMES, _WAVE, 1500., 0.1,
                                                10., model='btsettl')
    assert used_b == 'btsettl'
    assert np.all(f_s > 0) and np.all(f_b > 0)
    assert np.max(np.abs(f_s / f_b - 1)) > 0.1     # genuinely different model


def test_injection_btsettl_reproduces_old_driver():
    """model='btsettl' is the old injection: theta (T2, 5, 0, d, 1, 0)."""
    g = load_grid('btsettl')
    theta = np.array([1800., 5.0, 0.0, 12.0, 1.0, 0.0])
    old = np.asarray(model_grid(theta, _NAMES, _WAVE, g, False,
                                fmod.extinction.fitzpatrick99)).ravel()
    old = 0.12 ** 2 * np.where(np.isfinite(old) & (old > 0), old, 0.0)
    new, _ = fmod.companion_injection_flux(_NAMES, _WAVE, 1800., 0.12, 12.,
                                           model='btsettl')
    fin = old > 0
    assert np.allclose(new[fin], old[fin], rtol=1e-12)


def test_injection_falls_back_above_sonora():
    lo, hi, _, _ = fmod._grid_teff_range(load_grid('sonora'), 5.0, 0.0)
    assert (lo, hi) == (200.0, 2400.0)
    _, used = fmod.companion_injection_flux(_NAMES, _WAVE, 2500., 0.1, 10.)
    assert used == 'atmo'
    with pytest.raises(ValueError):
        fmod.companion_injection_flux(_NAMES, _WAVE, 2500., 0.1, 10.,
                                      fallback=None)
    with pytest.raises(ValueError):
        fmod.companion_injection_flux(_NAMES, _WAVE, 1500., 0.1, 10.,
                                      model='kurucz')


def test_injection_rj_fill_beyond_grid():
    """SPIRE 250 um is beyond Sonora's reach: filled by Rayleigh-Jeans from
    the reddest finite band, as the recovery does."""
    ig = load_grid('sonora')
    assert 'HERSCHEL_SPIRE_PSW' not in ig.finite_bands
    f, _ = fmod.companion_injection_flux(_NAMES, _WAVE, 1000., 0.1, 10.)
    i = list(_NAMES).index('HERSCHEL_SPIRE_PSW')
    assert f[i] > 0
    fin = [k for k, b in enumerate(_NAMES) if b in ig.finite_bands]
    k = max(fin, key=lambda m: _WAVE[m])
    assert f[i] < f[k]
    # scales as R2^2
    f2, _ = fmod.companion_injection_flux(_NAMES, _WAVE, 1000., 0.2, 10.)
    assert np.allclose(f2, 4 * f)
