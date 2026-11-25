"""Fit-health check on the mixing of the sampler, from the distance.

The likelihood depends on R/d only, so the posterior of d is its prior unless
the radius runs into a bound. A distance posterior that is displaced from the
prior or much narrower than it means the live points lost their spread: the
run did not mix, and its evidence and parameters are not converged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.stats as st

from astroARIADNE.fitter import _distance_mixing

RNG = np.random.default_rng(1)
PRIOR = st.truncnorm(a=-113.0 / 4.94, b=1e100, loc=113.0, scale=4.94)
RAD = st.uniform(loc=0.05, scale=100)


def _flag(dist, rad):
    return _distance_mixing(dict(dist=dist, rad=rad), dict(dist=PRIOR, rad=RAD))


def test_a_posterior_equal_to_the_prior_is_mixed():
    d = PRIOR.rvs(4000, random_state=RNG)
    out = _flag(d, 0.0046 * d)
    assert out['mixed'] is True
    assert abs(out['shift']) < 0.2 and 0.8 < out['width'] < 1.2


def test_a_collapsed_distance_is_flagged():
    d = RNG.normal(93.9, 1.9, 4000)                 # the Upper Sco fit at 25 steps
    out = _flag(d, 0.0042 * d)
    assert out['mixed'] is False
    assert out['shift'] < -3 and out['width'] < 0.5


def test_a_narrow_posterior_at_the_right_place_is_flagged():
    d = RNG.normal(113.0, 1.5, 4000)
    assert _flag(d, 0.0046 * d)['mixed'] is False


def test_radius_at_its_floor_is_not_a_mixing_failure():
    """R cannot go below 0.05 Rsun, so the data push d out: not the sampler."""
    d = RNG.normal(125.0, 2.0, 4000)
    out = _flag(d, np.full(4000, 0.0502) + RNG.uniform(0, 3e-4, 4000))
    assert out['mixed'] is None and out['reason'] == 'radius at a prior bound'


def test_without_a_distance_prior_there_is_no_verdict():
    assert _distance_mixing(dict(rad=np.ones(10)), dict())['mixed'] is None
    assert _distance_mixing(dict(dist=np.full(10, 50.0), rad=np.ones(10)),
                            dict(dist=PRIOR, rad=RAD))['mixed'] is None      # fixed distance
