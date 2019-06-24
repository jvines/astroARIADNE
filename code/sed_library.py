"""sed_library contain the model, prior and likelihood to be used."""

import scipy as sp

from phot_utils import *
from Star import *


def extract_info(magnitudes, errors, filters):
    """Extract the flux information for a Star."""
    flux = []
    wave = []
    bandpass = []
    bands = []

    for mag, err, band in zip(magnitudes, errors, filters):
        # Get central wavelength
        leff = get_effective_wavelength(band)
        # Extract mag and error from dictionary
        mag = mag
        mag_err = err
        # get flux, flux error and bandpass
        flx, flx_err = mag_to_flux(mag, mag_err, band)
        bp_l, bp_u = get_bandpass(band)
        flux.append([flx * leff, flx_err * leff])
        wave.append(leff)
        bandpass.append([leff - bp_l, bp_u - leff])

        # print('Flux in band', end=' ')
        # print(band, end=': ')
        # print(flx, end=' ')
        # print(r'erg/cm2/s/um', end='; ')
        # print('Central wavelength:', end=' ')
        # print('{:2.3f}'.format(leff), end=' ')
        # print('Bandpass:', end=' ')
        # print('{:2.3f}'.format(bp_u - bp_l))

    wave = sp.array(wave_flux)
    flux = sp.array(flux)
    bandpass = sp.array(bandpass)

    return wave, flux, bandpass


def model_grid(theta, star):
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
    teff, logg, z, rad, dist = theta
    model = dict()

    for f in star.filters:
        model[f] = star.get_interpolated_flux(
            teff, logg, z, f) * (rad / dist) ** 2
    return model


def model_grid_fixed_z(theta, star):
    """Return the model grid in the selected filters.

    This model assumes a fixed metallicity

    Parameters:
    -----------
    theta : array_like
        The parameters of the fit: teff, logg, radius, distance

    star : Star
        The Star object containing all relevant information regarding the star.

    Returns
    -------
    model : dict
        A dictionary whose keys are the filters and the values are the
        interpolated fluxes

    """
    teff, logg, rad, dist = theta
    model = dict()

    for f in star.filters:
        model[f] = star.get_interpolated_flux(
            teff, logg, -1, f, fixed_z=True) * (rad / dist) ** 2

    return model


def log_prior(theta, priorf):
    teff, logg, z, rad, dist = theta
    lp_teff, lp_logg, lp_z, lp_rad, lp_dist = 0, 0, 0, 0, 0

    if not 2300 < teff < 12000:
        return -sp.inf
    if not 0 < logg < 6:
        return -sp.inf
    if not -4 < z < 1:
        return -sp.inf

    prior_dict = read_priors(priorf)

    for k in prior_dict.keys():
        if k == 'teff':
            lp_teff += prior_dict[k](teff)
        elif k == 'logg':
            lp_logg += prior_dict[k](logg)
        elif k == 'z':
            lp_z += prior_dict[k](z)
        elif k == 'rad':
            lp_rad += prior_dict[k](rad)
        elif k == 'dist':
            lp_dist += prior_dict[k](dist)

    return lp_teff + lp_logg + lp_z + lp_rad + lp_dist


def loglike(theta, filters, grids, flux, flux_er, fixed_Z=False):
    """flux is a dictionary where key = filter."""
    if not fixed_Z:
        model_dict = model_grid(theta, filters, grids)
    else:
        model_dict = model_grid_fixed_z(theta, filters, grids)

    residuals = []
    errs = []
    for f in filters:
        if f in model_dict.keys() and flux.keys():
            residuals.append(model_dict[f] - flux[f])
            errs.append(flux_er[f])

    residuals = sp.array(residuals)
    errs = sp.array(errs)

    lnl = (residuals**2).sum() / errs**2

    return -.5 * lnl
