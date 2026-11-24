"""IRX companion prior from Gaia astrometry (Fitter.irx_companion_prior) and
the joint-branch primary fill (Fitter.irx_primary_fill).

The real-data fixtures (tests/data/irx_ms_*.npz/.json) are the inputs of the
IRX model selection for two cal_run5 quick-check fits (2026-11-23): GJ 876
with an undetectable injected disk (T = 100 K, L/L* = 3e-5, no band above
3 sigma) and Gliese 1 (a clean null of the roster). Their recorded results
come from the cal_run4 code path (companion prior 'legacy', zero fill).
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE.fitter as fmod  # noqa: E402
from astroARIADNE.fitter import CompanionAstrometricPrior as Prior  # noqa: E402
from astroARIADNE.fitter import Fitter  # noqa: E402
from astroARIADNE.sed_library import load_grid  # noqa: E402
from extinction import fitzpatrick99  # noqa: E402

DATA = os.path.join(REPO, 'tests', 'data')
# GJ 876-like system: M4 at 4.7 pc, G = 8.9
GJ876 = dict(gmag=8.9, dist_pc=4.67, lum_sys=0.0135, teff_sys=3250.0,
             teff_sys_err=40.0)


# ------------------------------------------------------------- the prior
def test_weight_never_exceeds_one():
    for ruwe in (0.9, 1.2, 1.39, 2.0, None):
        p = Prior(ruwe=ruwe, **GJ876)
        for t2 in (900.0, 2000.0, 2600.0, 3150.0):
            q = np.geomspace(1e-4, 0.999, 50)
            assert np.all(p.ln_weight(t2, q, 3250.0) <= 1e-12)


def test_clean_astrometry_suppresses_bright_stellar_companion():
    p = Prior(ruwe=1.0, **dict(GJ876, lum_sys=None))   # astrometric term only
    assert p.astrometry == 'clean'
    fd = p.detection_probability(2600.0, [0.1, 0.46], 3250.0)
    assert np.all(fd > 0.98)                    # Gaia would have seen it
    lw = p.ln_weight(2600.0, [0.46], 3250.0)[0]
    assert np.isclose(lw, np.log(p.eps + (1 - p.eps) * (1 - fd[1])))
    assert lw >= np.log(p.eps) - 1e-12          # bounded by the reliability


def test_substellar_companion_is_not_constrained_by_astrometry():
    p = Prior(ruwe=1.0, **dict(GJ876, lum_sys=None))
    assert p.detection_probability(1500.0, [0.01], 3250.0)[0] == 0.0
    assert p.ln_weight(1500.0, [0.01], 3250.0)[0] == 0.0


def test_unknown_or_nss_astrometry_switches_term_off():
    for kw, state in ((dict(ruwe=None), 'unknown'),
                      (dict(ruwe=1.0, non_single_star=1), 'binary')):
        p = Prior(**kw, **dict(GJ876, lum_sys=None))
        assert p.astrometry == state
        assert p.detection_probability(2600.0, [0.3], 3250.0)[0] == 0.0
        assert p.ln_weight(2600.0, [0.3], 3250.0)[0] == 0.0


def test_excess_noise_is_ignored_unless_enabled():
    kw = dict(ruwe=1.0, aen_sig=50.0, **dict(GJ876, lum_sys=None))
    assert Prior(**kw).astrometry == 'clean'
    assert Prior(settings=dict(aen_sig_threshold=2.0), **kw).astrometry \
        == 'binary'


def test_elevated_ruwe_allows_more_wobble_and_is_continuous():
    base = dict(GJ876, lum_sys=None, dist_pc=60.0, gmag=12.0)
    fd = {r: Prior(ruwe=r, **base).detection_probability(2800.0, [0.3],
                                                         3250.0)[0]
          for r in (1.0, 1.399, 1.401, 3.0, 10.0)}
    assert Prior(ruwe=3.0, **base).astrometry == 'elevated'
    assert abs(fd[1.399] - fd[1.401]) < 1e-3     # no jump at the threshold
    assert fd[1.0] == fd[1.399]                   # clean: threshold wobble
    assert fd[3.0] < fd[1.401] and fd[10.0] < fd[3.0]


def test_detection_falls_with_distance():
    f = [Prior(ruwe=1.0, **dict(GJ876, lum_sys=None, dist_pc=d))
         .detection_probability(2800.0, [0.3], 3250.0)[0]
         for d in (5.0, 50.0, 500.0)]
    assert f[0] > f[1] > f[2]


def test_overluminosity_term():
    llms, _ = fmod._ms_logl(3250.0)
    q = np.array([0.01, 0.3, 0.99])
    # on the main sequence: brighter companions cost more
    p = Prior(ruwe=None, lum_sys=10 ** llms, teff_sys=3250.0,
              teff_sys_err=40.0, dist_pc=4.67)
    lw = p.ln_weight(3000.0, q, 3250.0)
    assert np.all(np.diff(lw) < 0) and lw[-1] < -1.0
    # 0.3 dex overluminous (an equal-light binary): a twin costs nothing
    p = Prior(ruwe=None, lum_sys=10 ** (llms + np.log10(2.0)),
              teff_sys=3250.0, teff_sys_err=40.0, dist_pc=4.67)
    assert np.allclose(p.ln_weight(3240.0, q, 3250.0), 0.0, atol=0.01)


def test_unknown_setting_raises():
    with pytest.raises(ValueError):
        Prior(ruwe=1.0, settings=dict(ruwe_treshold=1.2), **GJ876)


def test_apply_prior_makes_companion_strictly_fainter():
    p = Prior(ruwe=None, dist_pc=4.67)          # inactive: weight 1
    chi2 = np.zeros(3)
    r2sq = np.array([0.5, 1.0, 2.0])            # q = 0.5, 1, 2 at equal Teff
    out = fmod._apply_companion_prior(chi2, p, 3000.0, r2sq, 1.0, 3000.0)
    assert out[0] == 0.0 and np.isinf(out[1]) and np.isinf(out[2])


def test_fitter_options_validate():
    f = Fitter()
    assert f.irx_companion_prior == 'astrometric'
    assert f.irx_primary_fill == 'model'
    f.irx_companion_prior = 'legacy'
    assert f._irx_companion_prior_obj({}, {}) is None
    f.irx_companion_prior = 'bogus'
    with pytest.raises(ValueError):
        f._irx_companion_prior_obj({}, {})
    f.irx_primary_fill = 'bogus'
    with pytest.raises(ValueError):
        f._irx_primary_fill_mode()


# --------------------------------------------------- real-data model selection
def _load(name):
    meta = json.load(open(os.path.join(DATA, f'irx_ms_{name}.json')))
    arr = dict(np.load(os.path.join(DATA, f'irx_ms_{name}.npz')))
    cool = [(g, load_grid(g)) for g in meta['cool']]
    args = [arr.pop('names'), arr.pop('waves_um'), arr.pop('excess'),
            arr.pop('excess_err'), cool, meta['z'], meta['av'],
            meta['dist_pc'], meta['lum_lsun'], meta['use_norm'],
            fitzpatrick99, meta['teff_primary']]
    kw = dict(arr)
    for k in ('rad_primary', 'logg_primary', 'wd_host'):
        kw[k] = meta[k]
    kw['prim_ig'] = load_grid(meta['prim'])
    return args, kw, meta['result']


@pytest.mark.parametrize('name', ['gj876', 'gliese1'])
def test_legacy_settings_reproduce_cal_run4(name):
    args, kw, ref = _load(name)
    r = fmod._excess_model_selection(*args, comp_prior=None,
                                     primary_fill='zero', **kw)
    assert r['excess_class'] == ref['excess_class']
    for k in ('p_none', 'p_disk', 'p_disk2', 'p_companion'):
        assert np.isclose(r[k], ref[k], rtol=1e-8, atol=1e-300), k


@pytest.mark.parametrize('name', ['gj876', 'gliese1'])
def test_model_fill_restores_the_null(name):
    """Neither star has a significant excess. With the zero fill the null is
    rejected (ln p_none < -40: the re-fitted BT-Settl primary has no IRS flux);
    with the model fill it is the preferred hypothesis."""
    args, kw, ref = _load(name)
    assert np.log(ref['p_none']) < -40
    r = fmod._excess_model_selection(*args, comp_prior=None,
                                     primary_fill='model', **kw)
    assert r['excess_class'] == 'none'
    assert r['p_none'] > 0.3


# ------------------------------------------------ hosts off the main sequence
def _old_ln_lum(p, q):
    """The main-sequence luminosity term alone (the prior before the
    off-main-sequence mixture)."""
    q = np.asarray(q, float)
    d = p.delta_lum
    return np.minimum(0.0, -0.5 * ((d - np.log10(1.0 + q)) ** 2 - d ** 2)
                      / p.sigma_lum ** 2)


def _young(teff=2950.0, above=1.0, **kw):
    """A host `above` dex over the main sequence at `teff`."""
    llms, _ = fmod._ms_logl(teff)
    return Prior(ruwe=None, gmag=16.0, dist_pc=113.0,
                 lum_sys=10.0 ** (llms + above), teff_sys=teff,
                 teff_sys_err=50.0, **kw)


def test_main_sequence_host_is_not_gated_off():
    p = Prior(ruwe=None, **GJ876)
    assert abs(p.delta_lum) < 0.3
    assert p.p_offms < 2e-3
    assert p.summary()['p_off_main_sequence'] == p.p_offms


@pytest.mark.parametrize('teff2', [900.0, 2000.0, 2600.0, 3150.0, 3240.0])
def test_main_sequence_weights_unchanged(teff2):
    """On the main sequence the weight is the overluminosity term, as before
    the off-main-sequence mixture, to 0.05 in ln w down to ln w = -3 (a
    factor 20 of suppression)."""
    p = Prior(ruwe=None, **GJ876)
    q = np.geomspace(1e-4, 0.999, 200)
    new, old = p.ln_weight(teff2, q, 3250.0), _old_ln_lum(p, q)
    m = old > -3.0
    assert np.allclose(new[m], old[m], atol=0.05)
    assert np.all(new <= 1e-12)


def test_young_host_is_off_the_main_sequence():
    p = _young(above=1.0)
    assert p.delta_lum == pytest.approx(1.0, abs=1e-6)
    assert p.p_offms > 0.7 and _young(above=1.5).p_offms > 0.99
    # the overluminosity term alone admits every companion of such a host
    assert np.allclose(_old_ln_lum(p, np.geomspace(1e-3, 0.999, 30)), 0.0)


def test_coeval_term_penalises_an_equal_temperature_fainter_companion():
    """Same Teff and age means same luminosity: a companion at the primary's
    Teff with a third of its light is older than the primary."""
    p = _young(above=1.5)
    t1 = 2950.0
    lw_twin = p.ln_weight(2950.0, [0.999], t1)[0]
    lw_faint = p.ln_weight(2950.0, [0.3], t1)[0]
    lw_fainter = p.ln_weight(2950.0, [0.05], t1)[0]
    assert lw_twin > -0.01
    assert lw_fainter < lw_faint < lw_twin
    assert lw_fainter < -3.0
    # a cooler companion at (or above) the main-sequence light ratio is coeval
    l1, _ = fmod._ms_logl(t1)
    l2, _ = fmod._ms_logl(2700.0)
    qc = 10.0 ** (l2 - l1)
    assert p.ln_weight(2700.0, [qc], t1)[0] > -0.01
    assert p.ln_weight(2700.0, [min(3 * qc, 0.99)], t1)[0] > -0.01
    assert p.ln_weight(2700.0, [qc / 30.0], t1)[0] < -1.0
    # substellar companions have no main sequence: no claim
    assert abs(p.ln_weight(1500.0, [1e-3], t1)[0]) < 1e-12
    # never above one
    for t2 in (1500.0, 2400.0, 2700.0, 2950.0):
        assert np.all(p.ln_weight(t2, np.geomspace(1e-4, 0.999, 60), t1)
                      <= 1e-12)


def test_gate_is_continuous_in_overluminosity():
    lw = [_young(above=a).ln_weight(2950.0, [0.3], 2950.0)[0]
          for a in np.linspace(0.0, 1.5, 61)]
    assert np.max(np.abs(np.diff(lw))) < 0.5


@pytest.mark.parametrize('name', ['gj876', 'gliese1'])
def test_main_sequence_verdicts_unchanged_by_the_mixture(name):
    """GJ 876 and Gliese 1: the model selection with the off-main-sequence
    mixture equals the one with the gate closed (the prior as it was)."""
    args, kw, _ = _load(name)
    meta = json.load(open(os.path.join(DATA, f'irx_ms_{name}.json')))
    te = float(np.max(kw['teff_primary_err']))
    base = dict(ruwe=1.0, gmag=9.0, dist_pc=meta['dist_pc'],
                lum_sys=meta['lum_lsun'], teff_sys=meta['teff_primary'],
                teff_sys_err=te)
    new = Prior(**base)
    old = Prior(settings=dict(offms_gate_sigma=1e6), **base)
    assert old.p_offms == 0.0 and new.p_offms < 0.05
    for t2 in (2000.0, 2600.0, 3000.0):
        q = np.geomspace(1e-4, 0.999, 100)
        lo, ln = old.ln_weight(t2, q, base['teff_sys']), new.ln_weight(t2, q, base['teff_sys'])
        m = lo > -3.0
        assert np.allclose(ln[m], lo[m], atol=0.25)
    r_new = fmod._excess_model_selection(*args, comp_prior=new,
                                         primary_fill='model', **kw)
    r_old = fmod._excess_model_selection(*args, comp_prior=old,
                                         primary_fill='model', **kw)
    assert r_new['excess_class'] == r_old['excess_class'] == 'none'
    for k in ('p_none', 'p_disk', 'p_companion'):
        assert np.isclose(r_new[k], r_old[k], rtol=0.05, atol=1e-4), k


# ------------------------------------------------- one distance for R, L and d
class _Star:
    dist = 113.0


def test_irx_frame_moves_radius_and_luminosity_to_the_catalogue_distance():
    f = Fitter()
    f.star = _Star()
    f._norm = False
    bf = dict(rad=0.26, lum=0.0045, dist=56.5)
    d, r, re, lum, s = f._irx_frame(bf, dict(rad=(0.01, 0.02)))
    assert d == 113.0 and s == pytest.approx(2.0)
    assert r == pytest.approx(0.52) and lum == pytest.approx(0.018)
    assert re == pytest.approx((0.02, 0.04))
    # the angular radius R/d, which is what the SED measures, is unchanged
    assert r / d == pytest.approx(bf['rad'] / bf['dist'])
    # a fit at the catalogue distance is left alone
    assert f._irx_frame(dict(rad=0.26, lum=0.0045, dist=113.0), {})[1:] \
        == (0.26, None, 0.0045, 1.0)
    # no fitted distance (normalisation mode) or no catalogue distance
    f._norm = True
    assert f._irx_frame(bf, {})[4] == 1.0
    f._norm = False
    f.star = type('S', (), dict(dist=-1))()
    assert f._irx_frame(bf, {})[4] == 1.0
    assert f._irx_frame(dict(rad=0.26, lum=0.0045), {})[4] == 1.0


def _usco(consistent):
    """Model-selection inputs of the Upper Scorpius M dwarf Gaia DR3
    6244258239317113088 (primordial disk, RUWE 5.7; fitted at 54.6 pc against
    a catalogue distance of 113 pc)."""
    meta = json.load(open(os.path.join(DATA, 'irx_ms_usco.json')))
    arr = dict(np.load(os.path.join(DATA, 'irx_ms_usco.npz')))
    s = meta['dist_pc'] / meta['dist_fit'] if consistent else 1.0
    rad, lum = meta['rad_primary'] * s, meta['lum_lsun'] * s ** 2
    cool = [(g, load_grid(g)) for g in meta['cool']]
    args = [arr.pop('names'), arr.pop('waves_um'), arr.pop('excess'),
            arr.pop('excess_err'), cool, meta['z'], meta['av'],
            meta['dist_pc'], lum, meta['use_norm'], fitzpatrick99,
            meta['teff_primary']]
    kw = dict(arr)
    kw['rad_primary_err'] = kw['rad_primary_err'] * s
    kw.update(rad_primary=rad, logg_primary=meta['logg_primary'],
              wd_host=False, prim_ig=load_grid(meta['prim']))
    prior = Prior(ruwe=meta['ruwe'], gmag=meta['gmag'],
                  dist_pc=meta['dist_pc'], lum_sys=lum,
                  teff_sys=meta['teff_primary'],
                  teff_sys_err=float(np.max(kw['teff_primary_err'])))
    return args, kw, prior


def test_young_disk_host_is_a_disk_at_the_catalogue_distance():
    """With R and L at the catalogue distance the primordial disk is a disk,
    decisively, and its fractional luminosity is below the dust ceiling."""
    args, kw, prior = _usco(consistent=True)
    assert prior.p_offms > 0.8                    # 1.2 dex above the sequence
    r = fmod._excess_model_selection(*args, comp_prior=prior,
                                     primary_fill='model', **kw)
    assert r['excess_class'] == 'disk'
    assert r['lnbf_disk_companion'] > 20.0
    assert 0.02 < r['disk']['l_dust_over_lstar'] < 0.09


def test_fitted_radius_at_the_catalogue_distance_made_it_a_companion():
    """The defect: the fitted radius (at 54.6 pc) evaluated at 113 pc leaves
    the primary 4.3 times too faint, and a companion as bright as the primary
    supplies the missing photosphere."""
    args, kw, prior = _usco(consistent=False)
    r = fmod._excess_model_selection(*args, comp_prior=prior,
                                     primary_fill='model', **kw)
    assert r['excess_class'] == 'companion'
    assert r['lnbf_disk_companion'] < -100.0
    assert r['disk']['l_dust_over_lstar'] > 0.095   # pinned at the 0.1 ceiling


# ------------------------------------------- dust ceiling of a young host
def test_dust_ceiling_of_a_young_host():
    T = np.array([100.0, 500.0, 1500.0])
    assert np.allclose(fmod._disk_lmax(T), [1e-1, 1e-1, 1e-4])
    assert np.allclose(fmod._disk_lmax(T, young_host=True), [1.0, 1.0, 1e-1])
    assert np.allclose(fmod._disk_lmax(T, wd_host=True, young_host=True), 1e-1)
    mid = fmod._disk_lmax(np.array([866.0]), young_host=True)[0]
    assert 1e-1 < mid < 1.0


def test_primordial_disk_above_the_debris_ceiling_is_a_disk_for_a_young_host():
    """The Upper Scorpius star with its radius and luminosity as fitted (at
    54.6 pc) has L_dust/L_* = 0.22 against that luminosity, above the debris
    ceiling of 0.1. As a young host it is a disk at that luminosity; as a
    main-sequence host its dust is pinned at the ceiling."""
    args, kw, prior = _usco(consistent=False)
    young = fmod._excess_model_selection(*args, comp_prior=prior, primary_fill='model',
                                         young_host=True, **kw)
    ms = fmod._excess_model_selection(*args, comp_prior=prior, primary_fill='model', **kw)
    assert ms['disk']['l_dust_over_lstar'] > 0.095
    assert 0.15 < young['disk']['l_dust_over_lstar'] < 0.35
    assert young['lnbf_disk_companion'] > ms['lnbf_disk_companion'] + 20.0


@pytest.mark.parametrize('name', ['gj876', 'gliese1'])
def test_main_sequence_host_keeps_the_debris_ceiling(name):
    args, kw, _ = _load(name)
    a = fmod._excess_model_selection(*args, comp_prior=None, primary_fill='model', **kw)
    b = fmod._excess_model_selection(*args, comp_prior=None, primary_fill='model',
                                     young_host=False, compact_primary=False, **kw)
    for k in ('p_none', 'p_disk', 'p_companion'):
        assert a[k] == b[k]


# ------------------------------ compact primary: from the fitted radius
def test_radius_below_the_floor_at_the_catalogue_distance_is_not_compact():
    """A main-sequence fit railed at the 0.05 Rsun radius floor can have
    R < 0.05 Rsun at the catalogue distance. It keeps the main-sequence
    companion radius bound (max(R1, 0.12)), not the white-dwarf one."""
    args, kw, _ = _load('gj876')
    kw = dict(kw, rad_primary=0.04)
    ms = fmod._excess_model_selection(*args, comp_prior=None, primary_fill='model',
                                      compact_primary=False, **kw)
    wd = fmod._excess_model_selection(*args, comp_prior=None, primary_fill='model',
                                      compact_primary=True, **kw)
    auto = fmod._excess_model_selection(*args, comp_prior=None, primary_fill='model', **kw)
    assert ms['companion']['rad2'] <= 0.12 + 1e-9
    assert auto['p_companion'] == wd['p_companion']      # None -> decided by rad_primary
    assert max(s[1] for g in wd['companion']['per_grid'].values() for s in g['samples']) <= fmod._IRX_WD_R2_MAX + 1e-9
