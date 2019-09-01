"""sed_library contain the model, prior and likelihood to be used."""

import astropy.units as u
import scipy as sp
from extinction import apply, fitzpatrick99
from scipy.special import ndtr

from phot_utils import *
from Star import *

# GLOBAL VARIABLES

order = sp.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation'])


def build_params(theta, coordinator, fixed):
    """Build the parameter vector that goes into the model."""
    params = sp.zeros(7)
    i = 0
    for j, k in enumerate(order):
        params[j] = theta[i] if not coordinator[j] else fixed[j]
        if not coordinator[j]:
            i += 1
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
    teff, logg, z, dist, rad, Av, inflation = theta
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
    inflation = theta[-1]
    mask = star.filter_mask
    residuals = model - star.flux[mask]
    errs = star.flux_er[mask]
    errs = sp.sqrt(errs ** 2 * (1 + inflation ** 2))
    return residuals, errs


def log_prior(theta, prior_dict, coordinator):
    """Calculate prior."""
    teff, logg, z, dist, rad, Av, inflation = theta
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
    if not 0 < inflation < 5:
        return -sp.inf

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


def log_probability(theta, star, prior_dict, coordinator, interpolators,
                    fixed):
    """Calculate unnormalized posterior probability of the model."""
    params = build_params(theta, coordinator, fixed)
    lp = log_prior(params, prior_dict, coordinator)
    lnl = log_likelihood(params, star, interpolators)
    if not sp.isfinite(lnl) or not sp.isfinite(lp):
        return -sp.inf
    return lp + lnl


def prior_transform_dynesty(u, star, prior_dict, coordinator):
    """Transform the prior from the unit cube to the parameter space."""
    u2 = sp.array(u)

    i = 0
    for fixed, par in zip(coordinator, order):
        if fixed:
            continue
        if par == 'logg':
            try:
                u2[i] = prior_dict['logg'](u2[i])
            except TypeError:
                u2[i] = prior_dict['logg'].ppf(u2[i])
            i += 1
            continue
        if par == 'teff':
            u2[i] = prior_dict['teff'].ppf(
                u2[i]) if star.get_temp else prior_dict['teff'](u2[i])
            i += 1
            continue
        u2[i] = prior_dict[par].ppf(u2[i])
        i += 1
    return u2


def prior_transform_multinest(u, star, prior_dict, coordinator):
    """Transform the prior from the unit cube to the parameter space."""
    i = 0
    for fixed, par in zip(coordinator, order):
        if fixed:
            continue
        if par == 'logg':
            try:
                u[i] = prior_dict['logg'](u[i])
            except TypeError:
                u[i] = prior_dict['logg'].ppf(u[i])
            i += 1
            continue
        if par == 'teff':
            u[i] = prior_dict['teff'].ppf(
                u[i]) if star.get_temp else prior_dict['teff'](u[i])
            i += 1
            continue
        u[i] = prior_dict[par].ppf(u[i])
        i += 1
    pass
