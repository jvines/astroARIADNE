"""sed_library contain the model, prior and likelihood to be used."""

import scipy as sp

from phot_utils import *
from Star import *


def model_grid(theta, star, fixed_z=False):
    """Return the model grid in the selected filters.

    Parameters:
    -----------
    theta : array_like
        The parameters of the fit: teff, logg, z, radius, distance

    star : Star
        The Star object containing all relevant information regarding the star.

    Returns
    -------
    model : dict
        A dictionary whose keys are the filters and the values are the
        interpolated fluxes

    """
    if not fixed_z:
        teff, logg, z, rad, dist = theta
    else:
        teff, logg, rad, dist = theta
        z = -1
    model = dict()

    for f in star.filters:
        model[f] = star.get_interpolated_flux(
            teff, logg, z, f, fixed_z) * (rad / dist) ** 2
    return model


def log_prior(theta, prior_dict, fixed_z=False):
    if not fixed_z:
        teff, logg, z, rad, dist = theta
        lp_z = 0
    else:
        teff, logg, rad, dist = theta
    lp_teff, lp_logg, lp_rad, lp_dist = 0, 0, 0, 0, 0

    if not 2300 < teff < 12000:
        return -sp.inf
    if not 0 < logg < 6:
        return -sp.inf
    if not -4 < z < 1:
        return -sp.inf

    for k in prior_dict.keys():
        if k == 'teff':
            lp_teff += prior_dict[k].pdf(teff)
        elif k == 'logg':
            lp_logg += prior_dict[k].pdf(logg)
        elif k == 'z' and not fixed_z:
            lp_z += prior_dict[k].pdf(z)
        elif k == 'rad':
            lp_rad += prior_dict[k].pdf(rad)
        elif k == 'dist':
            lp_dist += prior_dict[k].pdf(dist)

    return lp_teff + lp_logg + lp_z + lp_rad + lp_dist


def log_likelihood(theta, filters, flux, flux_er, fixed_z=False):
    """flux is a dictionary where key = filter."""

    model_dict = model_grid(theta, filters, grids, fixed_z)
    residuals = []
    errs = []
    for f in filters:
        if f in model_dict.keys() and flux.keys():
            residuals.append(model_dict[f] - flux[f])
            errs.append(flux_er[f])

    residuals = sp.array(residuals)
    errs = sp.array(errs)

    lnl = (residuals ** 2).sum() / errs ** 2

    return -.5 * lnl


def log_probability(theta, params):
    star, prior_dict = params
    filters = star.filters
    flux = star.flux
    flux_er = star.flux_er
    fixed_z = star.fixed_z
    lp = log_prior(theta, prior_dict, fixed_z)
    if not sp.isfinite(lp):
        return -sp.inf
    return lp + log_likelihood(theta, filters, flux, flux_er, fixed_z)
