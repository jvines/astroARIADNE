"""The default distance prior is the Bailer-Jones distance with its own
uncertainty: N(dist, dist_e) truncated at zero.

It used to be N(dist, 3 dist_e). The SED fixes R/d only, so the posterior of d
is its prior, and the widened prior let a fit sit far along the R/d degeneracy
from the catalogue distance: Gaia DR3 6244258239317113088 (113.0 +- 4.9 pc)
was fitted at 52-78 pc, and 126 of 1348 cal_run6 survey fits ended more than
10% from their catalogue distance.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter
from astroARIADNE.sed_library import _make_transform
from astroARIADNE.star import Star


def _fitter(dist, dist_e):
    star = Star('dist_prior_test', ra=180.0, dec=0.0, dist=dist, dist_e=dist_e,
                Av=0.05, offline=True, verbose=False,
                mag_dict={'2MASS_J': (10.0, 0.02), '2MASS_H': (9.7, 0.02),
                          '2MASS_Ks': (9.6, 0.02), 'GaiaDR2v2_G': (10.5, 0.01)})
    star.spectroscopic_params = None
    star.rave_params = None
    if not hasattr(star, 'Av_e'):
        star.Av_e = None
    f = Fitter.__new__(Fitter)
    f._star = star
    f.norm = False
    f.prior_sources = {}
    npars = 6 + int(star.used_filters.sum())
    f.coordinator = np.zeros(npars)
    f.fixed = np.zeros(npars)
    fitter_mod.order = np.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av'])
    return f


@pytest.mark.parametrize('dist,dist_e', [(113.0, 4.94), (50.0, 1.0), (4.67, 0.002)])
def test_distance_prior_has_the_catalogue_width(dist, dist_e):
    prior = _fitter(dist, dist_e)._default_priors()['dist']
    assert prior.dist.name == 'truncnorm'
    lo, med, hi = prior.ppf([0.158655, 0.5, 0.841345])
    assert med == pytest.approx(dist, rel=1e-6)
    assert hi - med == pytest.approx(dist_e, rel=1e-3)
    assert med - lo == pytest.approx(dist_e, rel=1e-3)
    # 4 sigma of the catalogue error is the far tail of the prior
    assert prior.cdf(dist - 4 * dist_e) < 1e-4


def test_distance_prior_is_truncated_at_zero():
    prior = _fitter(100.0, 60.0)._default_priors()['dist']
    assert prior.ppf(1e-9) >= 0.0
    assert prior.cdf(0.0) == 0.0
    assert prior.support()[0] == pytest.approx(0.0, abs=1e-9)


def test_fast_transform_matches_the_prior():
    prior = _fitter(113.0, 4.94)._default_priors()['dist']
    t = _make_transform(prior)
    u = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    assert np.allclose([t(x) for x in u], prior.ppf(u), rtol=1e-8)
