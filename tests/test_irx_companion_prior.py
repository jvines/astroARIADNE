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
    assert np.allclose(p.ln_weight(3240.0, q, 3250.0), 0.0)


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
