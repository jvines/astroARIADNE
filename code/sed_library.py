"""sed_library contain the model, prior and likelihood to be used."""

import astropy.units as u
import scipy as sp
from extinction import apply, fitzpatrick99

from phot_utils import *
from Star import *

# GLOBAL VARIABLES

order = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']


def build_params(theta, coordinator):
    params = sp.zeros(6)

    for i, k in enumerate(order):
        if k in coordinator.keys():
            params[i] = coordinator[k]
        else:
            params[i] = theta[i]
    return params


def get_interpolated_flux(temp, logg, z, filt, interpolators):
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
    flux = interpolators[filt](values)
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
    for f in star.filters:
        wav = star.wave[f]  # wavelength in um.
        wav *= 1e4  # um to AA, the unit required for extinction.
        wav = sp.array([wav])
        ext = fitzpatrick99(wav, Av, Rv)  # Calculate extinction.
        model[f] = apply(ext, get_interpolated_flux(
            teff, logg, z, f, interpolators) * (rad / dist) ** 2)[0]
        # model[f] = star.get_interpolated_flux(
        #     teff, logg, z, f) * (rad / dist) ** 2
    return model


def log_prior(theta, prior_dict, coordinator):

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

    for k in prior_dict.keys():
        if k not in coordinator.keys():
            if k == 'teff':
                lp += sp.log(prior_dict[k].pdf(teff))
            elif k == 'logg':
                lp += sp.log(prior_dict[k].pdf(logg))
            elif k == 'z':
                lp += sp.log(prior_dict[k].pdf(z))
            elif k == 'rad':
                lp += sp.log(prior_dict[k].pdf(rad))
            elif k == 'dist':
                lp += sp.log(prior_dict[k].pdf(dist))
            elif k == 'extinction':
                lp += sp.log(prior_dict[k].pdf(Av))
            # elif k == 'inflation':
            #     lp += prior_dict[k].pdf(sigma)

    return lp


def log_likelihood(theta, star, interpolators):
    """flux is a dictionary where key = filter."""
    model_dict = model_grid(theta, star, interpolators)
    inflation = theta[-1]
    residuals = []
    errs = []
    for f in star.filters:
        residuals.append(star.flux[f] - model_dict[f])
        errs.append(star.flux_er[f])

    residuals = sp.array(residuals)
    errs = sp.array(errs)
    # errs = sp.sqrt(errs ** 2 + inflation ** 2)

    c = sp.log(2 * sp.pi * errs ** 2)

    lnl = (c + (residuals ** 2 / errs ** 2)).sum()

    return -.5 * lnl


def log_probability(theta, star, prior_dict, coordinator, interpolators):
    params = build_params(theta, coordinator)
    lp = log_prior(params, prior_dict, coordinator)
    lnl = log_likelihood(params, star, interpolators)
    if not sp.isfinite(lnl) or not sp.isfinite(lp):
        return -sp.inf
    return lp + lnl
