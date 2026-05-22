"""Estimate logg using MIST isochrones."""

import warnings
import pickle
from multiprocessing import Pool

import pandas as pd
import numpy as np
from isochrones import SingleStarModel, get_ichrone
from isochrones.mist import MIST_Isochrone
from isochrones.priors import FlatPrior, GaussianPrior
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
import dynesty
from dynesty.utils import resample_equal

from .error import (DynestyError, InputError)
from .utils import credibility_interval

# The isochrones model is large and not cheaply picklable, so to use a
# multiprocessing pool we pass it via module globals (inherited by the forked
# workers) instead of dynesty's logl_args/ptform_args. Each worker mutates its
# own forked copy of the model, so there is no cross-process shared state.
_ISO_MODEL = None
_ISO_PARAMS = None
_ISO_JITTER_VARS = None

warnings.filterwarnings(
    'ignore', category=NumbaDeprecationWarning, append=True)
warnings.filterwarnings(
    'ignore', category=NumbaPendingDeprecationWarning, append=True)


def get_isochrone(logage, feh):
    """Retrieve isochrone for given age and feh."""
    mist = MIST_Isochrone()
    iso = mist.isochrone(logage, feh)
    return iso


def estimate(bands, params, logg=True, out_folder='.', threads=1, dlogz=0.01):
    """Estimate logg (or age/mass/eep) using MIST isochrones.

    Parameters
    ----------
    threads : int
        Number of processes for the nested-sampling pool. >1 parallelises the
        isochrone fit (the dominant cost of a BMA run).
    dlogz : float
        Evidence tolerance for the isochrone nested sampling. Lower is more
        precise but slower.
    """
    global _ISO_MODEL, _ISO_PARAMS, _ISO_JITTER_VARS
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

    # Expose the model/params to the module-level loglike & prior_transform so
    # they can run in pool workers without pickling the model.
    _ISO_MODEL = model
    _ISO_PARAMS = params
    _ISO_JITTER_VARS = bands
    ndim = model.n_params + len(bands)

    def _build_sampler(pool=None, queue_size=None):
        return dynesty.NestedSampler(
            loglike, prior_transform, ndim,
            nlive=500, bound='multi', sample='rwalk',
            pool=pool, queue_size=queue_size
        )

    try:
        if threads > 1:
            with Pool(threads) as pool:
                sampler = _build_sampler(pool=pool, queue_size=threads)
                sampler.run_nested(dlogz=dlogz)
                results = sampler.results
        else:
            sampler = _build_sampler()
            sampler.run_nested(dlogz=dlogz)
            results = sampler.results
    except ValueError as e:
        dump_out = f'{out_folder}/isochrone_DUMP.pkl'
        pickle.dump(sampler.results, open(dump_out, 'wb'))
        DynestyError(dump_out, 'isochrone', e).__raise__()
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

# These functions wrap isochrones so that they can be used with dynesty.
# They read the model/params from module globals (set in ``estimate``) so they
# stay picklable-by-reference and work under a fork-based multiprocessing pool.
def prior_transform(u):
    mod = _ISO_MODEL
    cube = np.copy(u)
    mod.mnest_prior(cube[: mod.n_params], None, None)
    cube[mod.n_params:] = -10 + 20 * cube[mod.n_params:]
    return cube


def loglike(theta):
    mod = _ISO_MODEL
    params = _ISO_PARAMS
    jitter_vars = _ISO_JITTER_VARS
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
