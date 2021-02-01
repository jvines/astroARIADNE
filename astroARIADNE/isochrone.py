"""Estimate logg using MIST isochrones."""

import warnings
import pickle

import pandas as pd
import numpy as np
from isochrones import SingleStarModel, get_ichrone
from isochrones.mist import MIST_Isochrone
from isochrones.priors import FlatPrior, GaussianPrior
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
import dynesty
from dynesty.utils import resample_equal

from .error import DynestyError, InputError
from .utils import credibility_interval

warnings.filterwarnings(
    'ignore', category=NumbaDeprecationWarning, append=True)
warnings.filterwarnings(
    'ignore', category=NumbaPendingDeprecationWarning, append=True)


def get_isochrone(logage, feh):
    """Retrieve isochrone for given age and feh."""
    mist = MIST_Isochrone()
    iso = mist.isochrone(logage, feh)
    return iso


def estimate(bands, params, logg=True, out_folder='.'):
    """Estimate logg using MIST isochrones."""
    mist = get_ichrone('mist', bands=bands)
    model = SingleStarModel(mist, **params)
    if 'distance' in params.keys():
        dist, dist_e = params['distance']
    elif 'parallax' in params.keys():
        dist = 1 / (params['parallax'][0] * 0.001)
        dist_e = dist * params['parallax'][1] / params['parallax'][0]
    else:
        msg = 'No parallax or distance found.'
        msg += 'Aborting age and mass calculation.'
        InputError(msg).warn()
        return np.zeros(10), np.zeros(10)
    if 'feh' in params.keys():
        fe, fe_e = params['feh']
        if fe + fe_e >= 0.5:
            model._priors['feh'] = FlatPrior([-0.5, 0.5])
        else:
            model._priors['feh'] = GaussianPrior(fe, fe_e)
    if 'mass' in params.keys():
        m, m_e = params['mass']
        model._priors['mass'] = GaussianPrior(m, m_e)
    if 'AV' in params.keys():
        av, av_e = params['AV']
        model._priors['AV'] = GaussianPrior(av, av_e)
    model._priors['distance'] = GaussianPrior(dist, dist_e)
    sampler = dynesty.NestedSampler(
        loglike, prior_transform, model.n_params + len(bands),
        nlive=500, bound='multi', sample='rwalk',
        logl_args=([model, params, bands]),
        ptform_args=([model])
    )
    try:
        sampler.run_nested(dlogz=0.01)
    except ValueError as e:
        dump_out = f'{out_folder}/isochrone_DUMP.pkl'
        pickle.dump(sampler.results, open(dump_out, 'wb'))
        DynestyError(dump_out, 'isochrone', e).__raise__()
    results = sampler.results
    samples = resample_equal(
        results.samples, np.exp(results.logwt - results.logz[-1])
    )
    ###########################################################################
    # Written by Dan Foreman-mackey
    # https://github.com/dfm/gaia-isochrones
    df = model._samples = pd.DataFrame(
        dict(
            zip(
                list(model.param_names),
                samples.T,
            )
        )
    )
    model._derived_samples = model.ic(
        *[df[c].values for c in model.param_names])
    model._derived_samples["parallax"] = 1000.0 / df["distance"]
    model._derived_samples["distance"] = df["distance"]
    model._derived_samples["AV"] = df["AV"]
    ###########################################################################
    if logg:
        samples = model._derived_samples['logg']
        med, lo, up = credibility_interval(samples, 5)
        med_e = max([med - lo, up - med])
        return med, med_e
    else:
        age_samples = 10 ** (model._derived_samples['age'] - 9)
        mass_samples = model._derived_samples['mass']
        eep_samples = model._derived_samples['eep']
        return age_samples, mass_samples, eep_samples


# Written by Dan Foreman-mackey
# https://github.com/dfm/gaia-isochrones

# These functions wrap isochrones so that they can be used with dynesty:
def prior_transform(u, mod):
    cube = np.copy(u)
    mod.mnest_prior(cube[: mod.n_params], None, None)
    cube[mod.n_params:] = -10 + 20 * cube[mod.n_params:]
    return cube


def loglike(theta, mod, params, jitter_vars):
    ind0 = mod.n_params
    lp0 = 0.0
    for i, k in enumerate(jitter_vars):
        err = np.sqrt(params[k][1] ** 2 + np.exp(theta[ind0 + i]))
        lp0 -= 2 * np.log(err)  # This is to fix a bug in isochrones
        mod.kwargs[k] = (params[k][0], err)
    lp = lp0 + mod.lnpost(theta[: mod.n_params])
    if np.isfinite(lp):
        return np.clip(lp, -1e10, np.inf)
    return -1e10
