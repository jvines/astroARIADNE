"""sed_library contain the model, prior and likelihood to be used."""

import astropy.units as u
import scipy as sp
from extinction import apply, fitzpatrick99
from scipy.special import ndtr

from phot_utils import *
from Star import *

# GLOBAL VARIABLES

order = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']


def build_params(theta, coordinator):
    """Build the parameter vector that goes into the model."""
    params = sp.zeros(6)
    for i, k in enumerate(order):
        # params[i] = coordinator[k]
        params[i] = theta[i]
    return params


def get_interpolated_flux(temp, logg, z, star, interpolators):
    """Interpolate the grid of fluxes in a given teff, logg and z.

    Parameters
    ----------
    temp : float
        The effective temperature.

    logg : float
        The superficial gravity.

    z : float
        The metallicity.

    filt : str
        The desired filter.

    Returns
    -------
    flux : float
        The interpolated flux at temp, logg, z for filter filt.

    """
    values = (temp, logg, z)
    mask = star.filter_mask
    flux = sp.zeros(mask.shape[0])
    intps = interpolators[mask]
    for i, f in enumerate(intps):
        flux[i] = f(values)
    return flux


def model_grid(theta, star, interpolators):
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
    teff, logg, z, dist, rad, Av = theta
    model = dict()

    dist = (dist * u.pc).to(u.solRad).value
    Rv = 3.1  # For extinction.

    mask = star.filter_mask
    flux = get_interpolated_flux(teff, logg, z, star, interpolators)

    wav = star.wave[mask] * 1e4
    ext = fitzpatrick99(wav, Av, Rv)
    model = apply(ext, flux) * (rad / dist) ** 2
    return model


def get_residuals(theta, star, interpolators):
    """Calculate residuals of the model."""
    model = model_grid(theta, star, interpolators)
    # inflation = theta[-1]
    residuals = []
    errs = []
    mask = star.filter_mask
    residuals = star.flux[mask] - model
    errs = star.flux_er[mask]
    return residuals, errs


def log_prior(theta, prior_dict, coordinator):
    """Calculate prior."""
    teff, logg, z, dist, rad, Av = theta
    lp = 0

    if not 2300 <= teff <= 12000:
        return -sp.inf
    if not -4 < logg <= 6:
        return -sp.inf
    if not -4 <= z <= 1:
        return -sp.inf
    if not 1 < dist <= 1500:
        return -sp.inf
    if not 0 < rad <= 15:
        return -sp.inf
    if not 0 < Av < .032:
        return -sp.inf
    # if not 0 < sigma < 1:
    #     return -sp.inf

    lp += sp.log(prior_dict['teff'].pdf(teff))
    lp += sp.log(prior_dict['logg'].pdf(logg))
    lp += sp.log(prior_dict['z'].pdf(z))
    lp += sp.log(prior_dict['dist'].pdf(dist))
    lp += sp.log(prior_dict['rad'].pdf(rad))
    lp += sp.log(prior_dict['Av'].pdf(Av))

    return lp


def log_likelihood(theta, star, interpolators):
    """Calculate log likelihood of the model."""
    residuals, errs = get_residuals(theta, star, interpolators)

    c = sp.log(2 * sp.pi * errs ** 2)
    lnl = (c + (residuals ** 2 / errs ** 2)).sum()

    return -.5 * lnl


def log_probability(theta, star, prior_dict, coordinator, interpolators):
    """Calculate unnormalized posterior probability of the model."""
    params = build_params(theta, coordinator)
    lp = log_prior(params, prior_dict, coordinator)
    lnl = log_likelihood(params, star, interpolators)
    if not sp.isfinite(lnl) or not sp.isfinite(lp):
        return -sp.inf
    return lp + lnl


def prior_transform_dynesty(u, star, prior_dict):
    u2 = sp.array(u)

    if star.get_temp or star.temp:
        u2[0] = prior_dict['teff'].ppf(u[0])
    else:
        u2[0] = prior_dict['teff'](u[0])
    u2[1] = prior_dict['logg'](u[1])
    u2[2] = prior_dict['z'].ppf(u[2])
    u2[3] = prior_dict['dist'].ppf(u[3])
    u2[4] = prior_dict['rad'].ppf(u[4])
    u2[5] = prior_dict['Av'].ppf(u[5])
    return u2


def prior_transform_multinest(u, star, prior_dict):
    if star.get_temp or star.temp:
        u[0] = prior_dict['teff'].ppf(u[0])
    else:
        u[0] = prior_dict['teff'](u[0])
    u[1] = prior_dict['logg'](u[1])
    u[2] = prior_dict['z'].ppf(u[2])
    u[3] = prior_dict['dist'].ppf(u[3])
    u[4] = prior_dict['rad'].ppf(u[4])
    u[5] = prior_dict['Av'].ppf(u[5])
    pass
