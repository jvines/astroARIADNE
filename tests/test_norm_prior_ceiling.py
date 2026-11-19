"""The default norm = (R/D)^2 prior must not exclude nearby stars.

A fixed U(0, 1e-20) ceiling caps R/D at 1e-10, so a 0.41 R_sun M dwarf at
38 pc (K2-18, true norm 5.9e-20) was railed against the bound. With a
distance, the ceiling is now (100.05 R_sun / D_min)^2.
"""
import astropy.units as u
import numpy as np

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter
from test_default_priors_partial_spec import _build_offline_star

def _norm_prior(dist, dist_e):
    star = _build_offline_star()
    star.dist, star.dist_e = dist, dist_e
    f = Fitter.__new__(Fitter)
    f._star = star
    f.norm = True
    f.prior_sources = {}
    npars = 5 + int(star.used_filters.sum())
    f.coordinator = np.zeros(npars)
    f.fixed = np.zeros(npars)
    fitter_mod.order = np.array(['teff', 'logg', 'z', 'norm', 'Av'])
    return f._default_priors()['norm'], f.prior_sources['norm']

def _norm(rad_rsun, dist_pc):
    return (rad_rsun * u.solRad.to(u.pc) / dist_pc) ** 2

def test_nearby_star_inside_ceiling():
    prior, source = _norm_prior(38.10, 0.04)
    assert source == 'uniform_distance'
    up = prior.support()[1]
    assert np.isclose(up, _norm(100.05, 38.10 - 0.2))
    # K2-18 sits above the old 1e-20 wall and inside the new ceiling.
    assert 1e-20 < _norm(0.41, 38.10) < up
    assert _norm(100., 38.10) < up

def test_poor_parallax_floors_at_one_parsec():
    prior, _ = _norm_prior(10., 5.)
    assert np.isclose(prior.support()[1], _norm(100.05, 1.))

def test_no_distance_fixed_ceiling():
    prior, source = _norm_prior(-1, -1)
    assert source == 'uniform_default'
    assert np.isclose(prior.support()[1], 1e-16)

# ---- posterior railed against a prior bound -------------------------------
from astroARIADNE.fitter import _prior_bound_flags
import scipy.stats as st

_rng = np.random.default_rng(0)

def _railed():
    # The K2-18 symptom: norm piled against U(0, 1e-20), teff against 4200.
    return {'norm': 1e-20 - np.abs(_rng.normal(0, 3e-22, 4000)),
            'teff': 4200. - np.abs(_rng.normal(0, 15., 4000))}

def test_railed_posterior_flagged():
    priors = {'norm': st.uniform(0, 1e-20),
              'teff': st.uniform(400., 3800.)}
    flags = {(f['param'], f['edge']) for f in
             _prior_bound_flags(_railed(), priors)}
    assert flags == {('norm', 'upper'), ('teff', 'upper')}

def test_well_inside_prior_not_flagged():
    samples = {'norm': _rng.normal(5.9e-20, 3e-21, 4000),
               'teff': _rng.normal(3460., 50., 4000),
               'rad': _rng.normal(0.41, 0.02, 4000),
               'Av': np.abs(_rng.normal(0, 0.01, 4000))}
    priors = {'norm': st.uniform(0, 3.5e-15),
              'teff': st.uniform(400., 3800.),
              'rad': st.uniform(0.05, 100.),     # 0.41 is 0.4% of the span
              'Av': st.uniform(0, 0.5)}          # zero floor: not a wall
    assert _prior_bound_flags(samples, priors) == []

def test_prior_dominated_not_flagged():
    # Unconstrained logg filling its uniform prior.
    samples = {'logg': _rng.uniform(3.5, 6.0, 4000)}
    assert _prior_bound_flags(samples, {'logg': st.uniform(3.5, 2.5)}) == []

def test_show_priors_scientific_notation():
    from astroARIADNE.fitter import Fitter
    f = Fitter.__new__(Fitter)
    f.prior_sources = {}
    assert f._format_prior_notation('norm', st.uniform(0, 1e-20)) \
        == 'U(0.00, 1.00e-20)'
    assert f._format_prior_notation('teff', st.uniform(400., 3800.)) \
        == 'U(400.00, 4200.00)'
