"""Estimate logg using MIST isochrones."""

import os

from isochrones import SingleStarModel, get_ichrone
from isochrones.priors import GaussianPrior

from utils import credibility_interval


def estimate(bands, params):
    """Estimate logg using MIST isochrones."""
    mist = get_ichrone('mist', bands=bands)
    model = SingleStarModel(mist, **params)
    dist = 1 / (params['parallax'][0] * 0.001)
    dist_e = dist * params['parallax'][1] / params['parallax'][0]
    model._priors['distance'] = GaussianPrior(dist, 3 * dist_e)
    model.fit(resume=False, verbose=False, n_live_points=500)
    logg_samples = model.derived_samples['logg']
    med, lo, up = credibility_interval(logg_samples)
    med_e = 2 * max([med - lo, up - med])
    return med, med_e
