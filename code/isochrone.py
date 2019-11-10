"""Estimate logg using MIST isochrones."""

import os

import scipy as sp
from isochrones import SingleStarModel, get_ichrone
from isochrones.priors import GaussianPrior

import dynesty
from dynesty.utils import resample_equal
from utils import credibility_interval


def estimate(bands, params):
    """Estimate logg using MIST isochrones."""
    mist = get_ichrone('mist', bands=bands)
    model = SingleStarModel(mist, **params)
    dist = 1 / (params['parallax'][0] * 0.001)
    dist_e = dist * params['parallax'][1] / params['parallax'][0]
    model._priors['distance'] = GaussianPrior(dist, 3 * dist_e)
    sampler = dynesty.NestedSampler(
        loglike, prior_transform, model.n_params + len(bands),
        logl_args=([model, params, bands]),
        ptform_args=([model])
    )
    sampler.run_nested()
    results = sampler.results
    samples = resample_equal(
        results.samples, sp.exp(results.logwt - results.logz[-1])
    )
    # model.fit(resume=False, verbose=False, n_live_points=500)
    logg_samples = model.derived_samples['logg']
    med, lo, up = credibility_interval(logg_samples)
    med_e = 2 * max([med - lo, up - med])
    return med, med_e


# Written by Dan Foreman-mackey
# https://github.com/dfm/gaia-isochrones

# These functions wrap isochrones so that they can be used with dynesty:
def prior_transform(u, mod):
    cube = sp.copy(u)
    mod.mnest_prior(cube[: mod.n_params], None, None)
    cube[mod.n_params:] = -10 + 20 * cube[mod.n_params:]
    return cube


def loglike(theta, mod, params, jitter_vars):
    ind0 = mod.n_params
    lp0 = 0.0
    for i, k in enumerate(jitter_vars):
        err = sp.sqrt(params[k][1] ** 2 + sp.exp(theta[ind0 + i]))
        lp0 -= 2 * sp.log(err)  # This is to fix a bug in isochrones
        mod.kwargs[k] = (params[k][0], err)
    lp = lp0 + mod.lnpost(theta[: mod.n_params])
    if sp.isfinite(lp):
        return sp.clip(lp, -1e10, sp.inf)
    return -1e10
