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
        s1 = r1['_summary']
        assert s1['significant'] == ['WISE_RSR_W3']
        # aggregate stats: ONE significant band is a caution, not a detection
        # (a wrong far-IR crossmatch produces exactly this signature)
        assert s1['has_excess'] is False and s1['n_significant'] == 1
        assert s1['single_band_caution'] == 'WISE_RSR_W3'
        assert s1['excess_chi2'] > 25                 # sigma>5 -> chi2>25
        assert s1['disk'] is None                     # 1 band -> no selection
        assert r0['_summary']['has_excess'] is False
        assert r0['_summary']['single_band_caution'] is None
    finally:
        fmod.load_grid = orig


def test_blackbody_excess_recovers_temperature():
    """The debris-disk blackbody fit recovers an injected (T, L_dust/L_*)."""
    from scipy.constants import h, c, k as kB, sigma as sig
    from astroARIADNE.fitter import _fit_excess_blackbody

    def planck_per_um(lam_um, T):
        lam = np.asarray(lam_um) * 1e-6
        return (2 * h * c ** 2 / lam ** 5) / np.expm1(h * c / (lam * kB * T)) \
            * 1e-3                                     # erg/s/cm^2/um/sr

    T_true, Om, dist, lum = 200.0, 1e-16, 30.0, 1.0
    bands = np.array([12.0, 22.0, 23.8, 71.0])        # W3, W4, MIPS24, MIPS70
    F = Om * planck_per_um(bands, T_true)
    ld_ls = 4 * np.pi * (dist * 3.0856775815e18) ** 2 \
        * (Om * sig * T_true ** 4 / np.pi * 1e3) / (lum * 3.828e33)
    bb = _fit_excess_blackbody(bands, F, 0.05 * F, dist, lum)
    assert abs(bb['t_dust'] - T_true) / T_true < 0.05
    assert abs(bb['l_dust_over_lstar'] / ld_ls - 1) < 0.05
    assert bb['n_bands'] == 4
    # guards: a single band, or no positive excess, or no distance -> None
    assert _fit_excess_blackbody([22.], [1e-13], [1e-14], 30., 1.) is None
    assert _fit_excess_blackbody([12., 22.], [-1e-13, -1e-14],
                                 [1e-14, 1e-14], 30., 1.) is None
    assert _fit_excess_blackbody(bands, F, 0.05 * F, None, 1.) is None


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
    # coelho's reliability cap was lifted (its 0.13-10 um SED library matches
    # the dense grids to ~1-3% in the NIR/IRAC); its IRX reach is now its
    # finite IRAC bands, self-limited by the NaN guard at ~8 um.
    coelho = load_grid('coelho')
    from astroARIADNE.config import irx_bands
    reach = grid_irx_reach('coelho', coelho) & set(irx_bands)
    # Since the SPHEREx band extension (2026-06), coelho also reaches the
    # SPHEREx red super-bands inside its 10 um coverage; the raw rebuild adds
    # the IRS continuum bands below its ~8 um self-limit (IRS_5500/6500/8000).
    allowed = {'SPITZER_IRAC_58', 'SPITZER_IRAC_80',
               'SPHEREX_2750', 'SPHEREX_3050', 'SPHEREX_3350',
               'SPHEREX_3700', 'SPHEREX_4050', 'SPHEREX_4300',
               'IRS_5500', 'IRS_6500', 'IRS_8000'}
    assert reach <= allowed and {'SPITZER_IRAC_58', 'SPITZER_IRAC_80'} <= reach


def test_excess_model_selection():
    """Bayesian model selection on injected excesses (real grid IR bands):
    a cool companion, a debris disk, and pure noise each win the right model,
    with parameters recovered from the marginal posteriors."""
    from scipy.constants import h, c, k as kB
    from astroARIADNE.sed_library import load_grid, model_grid
    from astroARIADNE.phot_utils import get_effective_wavelength
    from astroARIADNE.fitter import _excess_model_selection
    g = load_grid('btsettl')
    ta = g.index_columns[1]
    names = np.array(['WISE_RSR_W3', 'AKARI_IRC_L18W', 'WISE_RSR_W4',
                      'SPITZER_MIPS_24', 'IRAS_60', 'SPITZER_MIPS_70',
                      'HERSCHEL_PACS_GREEN', 'SPITZER_MIPS_160'])
    waves = np.array([get_effective_wavelength(b) for b in names])
    dist = 30.0

    def sel(F, E):
        return _excess_model_selection(names, waves, F, E, [('btsettl', g)],
                                       0., 0., dist, 1.0, False, _avlaw, 6000.)

    # companion (Teff2 = 3000 K) -> P(companion) dominant, Teff2 recovered
    Fc = np.asarray(model_grid(np.array([3000., 4.8, 0., dist, 0.3, 0.]),
                               names, waves, g, False, _avlaw)).ravel()
    r = sel(Fc, 0.05 * Fc)
    assert r['excess_class'] == 'companion'
    assert r['p_companion'] > 0.9
    assert abs(r['companion']['teff2'] - 3000) < 300

    # debris disk (150 K) -> the SINGLE-T disk dominant (Occam suppresses the
    # nested two-temperature variant), T_dust recovered
    def planck_um(lam, T):
        ll = lam * 1e-6
        return (2 * h * c ** 2 / ll ** 5) / np.expm1(h * c / (ll * kB * T)) \
            * 1e-3
    Fd = 2e-16 * planck_um(waves, 150.0)
    r = sel(Fd, 0.05 * Fd)
    assert r['excess_class'] == 'disk'
    assert r['p_disk'] + r['p_disk2'] > 0.9
    assert r['p_disk'] > r['p_disk2']
    assert abs(r['disk']['t_dust'] - 150) < 25

    # two-temperature disk (300 K + 60 K belts) -> the 2-T variant wins and
    # recovers both components
    F2 = 1e-16 * planck_um(waves, 300.0) + 4e-15 * planck_um(waves, 60.0)
    r = sel(F2, 0.05 * F2)
    assert r['excess_class'] == 'disk'
    assert r['p_disk2'] > 0.9
    assert abs(r['disk2']['t_warm'] - 300) < 60
    assert abs(r['disk2']['t_cold'] - 60) < 20

    # pure noise (no excess) -> the Occam-penalised null wins, not a spurious
    # disk/companion (the whole point of evidence over chi^2)
    rng = np.random.default_rng(1)
    E = np.full(len(names), 1e-15)
    Fn = rng.normal(0, 1e-15, len(names))
    r = sel(Fn, E)
    assert r['excess_class'] == 'none'
    assert r['p_none'] > 0.5


def test_companion_radius_cap_degeneracy_floor():
    """The R2 <= R1 prior is floored at 0.12 Rsun: substellar companions are
    degeneracy-supported, so a T dwarf at ~0.10 Rsun is genuinely larger than
    a late-M primary (SCR 1845: literal R2 <= R1 = 0.07 excluded the true T6
    and flipped the verdict to 'disk')."""
    from astroARIADNE.sed_library import load_grid, model_grid
    from astroARIADNE.phot_utils import get_effective_wavelength
    from astroARIADNE.fitter import _excess_model_selection
    g = load_grid('btsettl')
    names = np.array(['WISE_RSR_W3', 'AKARI_IRC_L18W', 'WISE_RSR_W4',
                      'SPITZER_MIPS_24', 'IRAS_60', 'SPITZER_MIPS_70',
                      'HERSCHEL_PACS_GREEN', 'SPITZER_MIPS_160'])
    waves = np.array([get_effective_wavelength(b) for b in names])
    dist = 4.0
    # companion injected LARGER than the primary (0.10 vs 0.07 Rsun)
    Fc = np.asarray(model_grid(np.array([1000., 5.0, 0., dist, 0.10, 0.]),
                               names, waves, g, False, _avlaw)).ravel()
    r = _excess_model_selection(names, waves, Fc, 0.05 * Fc, [('btsettl', g)],
                                0., 0., dist, 2e-4, False, _avlaw, 2400.,
                                rad_primary=0.07)
    assert r['excess_class'] == 'companion'
    assert r['p_companion'] > 0.9
    assert abs(r['companion']['teff2'] - 1000) < 250
    assert abs(r['companion']['rad2'] - 0.10) < 0.02, \
        'R2 pegged at the primary radius: the degeneracy floor is not applied'


def test_fit_irx_band_routing():
    """fit_irx routing: the W2 block (>= 4.4 um) moves to the excess set by
    default; route_mid_ir widens to the full 2.75-5 um block; regular SED
    endpoints leave everything as fit bands. The user's Star object is
    untouched (shallow-copy contract)."""
    import types
    from astroARIADNE.fitter import Fitter

    def make_star():
        return types.SimpleNamespace(
            filter_names=np.array(['2MASS_Ks', 'WISE_RSR_W1',
                                   'SPITZER_IRAC_45', 'WISE_RSR_W2',
                                   'WISE_RSR_W3']),
            used_filters=np.array([1., 1., 1., 1., 0.]),
            irx_used_filters=np.array([0., 0., 0., 0., 1.]),
            filter_mask=np.array([0, 1, 2, 3]),
            irx_filter_mask=np.array([4]))

    # default IRX routing: W2 block only, W1 stays a fit anchor
    star = make_star()
    f = Fitter()
    f.star = star
    f._route_irx_extra(f._IRX_W2_BANDS, anchor_um=4.4)
    kept = f.star.filter_names[f.star.used_filters == 1]
    excess = f.star.filter_names[f.star.irx_used_filters == 1]
    assert list(kept) == ['2MASS_Ks', 'WISE_RSR_W1']
    assert set(excess) == {'SPITZER_IRAC_45', 'WISE_RSR_W2', 'WISE_RSR_W3'}
    assert star.used_filters.sum() == 4          # original untouched

    # route_mid_ir: the whole 2.75-5 um block goes too
    star = make_star()
    f = Fitter()
    f.star = star
    f._route_irx_extra()                         # defaults to the full block
    kept = f.star.filter_names[f.star.used_filters == 1]
    excess = f.star.filter_names[f.star.irx_used_filters == 1]
    assert list(kept) == ['2MASS_Ks']
    assert set(excess) == {'WISE_RSR_W1', 'SPITZER_IRAC_45', 'WISE_RSR_W2',
                           'WISE_RSR_W3'}
    assert star.used_filters.sum() == 4          # original untouched


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
    print('ALL PASS')
