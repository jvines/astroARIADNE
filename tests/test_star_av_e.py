"""A user-supplied Av (and Av_e) must reach the Av prior (#89)."""
import numpy as np
import scipy.stats as st

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter
from astroARIADNE.star import Star

MAGS = {'2MASS_J': (9.763, 0.028), '2MASS_H': (9.135, 0.026),
        '2MASS_Ks': (8.899, 0.019), 'PS1_r': (12.9105, 0.0107)}


def _av_prior(**kw):
    # 500 pc: outside the d < 70 pc auto-pin, which hid the bug.
    star = Star('demo', 172.56, 7.588, offline=True, mag_dict=MAGS,
                dist=500., dist_e=10., verbose=False, **kw)
    f = Fitter.__new__(Fitter)
    f._star = star
    f.norm = False
    f.prior_sources = {}
    npars = 6 + int(star.used_filters.sum())
    f.coordinator = np.zeros(npars)
    f.fixed = np.zeros(npars)
    fitter_mod.order = np.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av'])
    return star, f._default_priors()['Av']


def test_av_with_error_gives_truncnorm():
    star, prior = _av_prior(Av=0.05, Av_e=0.01)
    assert star.Av_e == 0.01
    assert prior.dist.name == 'truncnorm'
    assert np.isclose(prior.kwds['loc'], 0.05)
    assert np.isclose(prior.kwds['scale'], 0.01)


def test_av_without_error_gives_uniform():
    star, prior = _av_prior(Av=0.05)
    assert star.Av_e is None
    assert prior.dist.name == 'uniform'
    assert np.isclose(prior.support()[1], 0.05)




def _sources(**kw):
    star = Star('demo', 172.56, 7.588, offline=True, mag_dict=MAGS,
                dist=500., dist_e=10., verbose=False, **kw)
    f = Fitter.__new__(Fitter)
    f._star = star
    f.norm = False
    f.prior_sources = {}
    npars = 6 + int(star.used_filters.sum())
    f.coordinator = np.zeros(npars)
    f.fixed = np.zeros(npars)
    fitter_mod.order = np.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av'])
    defaults = f._default_priors()
    return f, defaults


def test_prior_source_labels():
    f, _ = _sources(Av=0.05)
    assert f.prior_sources['Av'] == 'user_uniform'
    f, _ = _sources(Av=0.05, Av_e=0.01)
    assert f.prior_sources['Av'] == 'user_truncnorm'


def test_zero_av_e_pins_av():
    f, defaults = _sources(Av=0.05, Av_e=0.)
    assert defaults['Av'] is None
    assert f.coordinator[5] == 1 and f.fixed[5] == 0.05
    assert f.prior_sources['Av'] == 'user_fixed'


def test_negative_av_e_rejected():
    import pytest
    # InputError.__raise__ prints and exits, as elsewhere in Star.
    with pytest.raises(SystemExit):
        Star('demo', 172.56, 7.588, offline=True, mag_dict=MAGS,
             dist=500., dist_e=10., verbose=False, Av=0.05, Av_e=-0.01)
