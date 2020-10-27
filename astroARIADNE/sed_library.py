"""sed_library contain the model, prior and likelihood to be used."""

import pickle
from contextlib import closing

import astropy.units as u
import numpy as np
from extinction import apply

from .phot_utils import *
from .utils import get_noise_name


def build_params(theta, flux, flux_e, filts, coordinator, fixed, use_norm):
    """Build the parameter vector that goes into the model."""
    if use_norm:
        params = np.zeros(len(coordinator))
        order = np.array(['teff', 'logg', 'z', 'norm', 'av'])
    else:
        params = np.zeros(len(coordinator))
        order = np.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'av']
        )

    for filt, flx, flx_e in zip(filts, flux, flux_e):
        p_ = get_noise_name(filt) + '_noise'
        order = np.append(order, p_)
    i = 0
    for j, k in enumerate(order):
        params[j] = theta[i] if not coordinator[j] else fixed[j]
        if not coordinator[j]:
            i += 1
    return params


def get_interpolated_flux(temp, logg, z, filts, interpolators):
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
    values = (logg, temp, z)
    vals = interpolators(values, filts)
    flux = vals
    return flux


def model_grid(theta, filts, wave, interpolators, use_norm, av_law):
    """Return the model grid in the selected filters.

    Parameters:
    -----------
    theta : array_like
        The parameters of the fit: teff, logg, z, radius, distance

    star : Star
        The Star object containing all relevant information regarding the star.

    interpolators : dict
        A dictionary with the interpolated grid.

    use_norm : bool
        False for a full fit  (including radius and distance). True to fit
        for a normalization constant instead.

    Returns
    -------
    model : dict
        A dictionary whose keys are the filters and the values are the
        interpolated fluxes

    """
    model = dict()
    Rv = 3.1  # For extinction.

    if use_norm:
        teff, logg, z, norm, Av = theta[:5]
    else:
        teff, logg, z, dist, rad, Av = theta[:6]
        dist = (dist * u.pc).to(u.solRad).value

    flux = get_interpolated_flux(teff, logg, z, filts, interpolators)

    wav = wave * 1e4
    ext = av_law(wav, Av, Rv)
    if use_norm:
        model = apply(ext, flux) * norm
    else:
        model = apply(ext, flux) * (rad / dist) ** 2
    return model


def get_residuals(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                  av_law):
    """Calculate residuals of the model."""
    model = model_grid(theta, filts, wave, interpolators, use_norm, av_law)
    start = 5 if use_norm else 6
    inflation = theta[start:]
    residuals = model - flux
    errs = np.sqrt(flux_er ** 2 + inflation**2)
    return residuals, errs


def log_likelihood(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                   av_law):
    """Calculate log likelihood of the model."""
    res, ers = get_residuals(theta, flux, flux_er, wave,
                             filts, interpolators, use_norm, av_law)

    c = np.log(2 * np.pi * ers ** 2)
    lnl = (c + (res ** 2 / ers ** 2)).sum()

    if not np.isfinite(lnl):
        return -1e300

    return -.5 * lnl


def prior_transform_dynesty(u, flux, flux_er, filts, prior_dict, coordinator,
                            use_norm):
    """Transform the prior from the unit cube to the parameter space."""
    u2 = np.array(u)
    if use_norm:
        order = np.array(['teff', 'logg', 'z', 'norm', 'av'])
    else:
        order = np.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'av']
        )

    for filt, flx, flx_e in zip(filts, flux, flux_er):
        p_ = get_noise_name(filt) + '_noise'
        order = np.append(order, p_)

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
            try:
                u2[i] = prior_dict['teff'](u2[i])
            except TypeError:
                u2[i] = prior_dict['teff'].ppf(u2[i])
            i += 1
            continue
        u2[i] = prior_dict[par].ppf(u2[i])
        i += 1
    return u2


def prior_transform_multinest(u, flux, flux_er, filts, prior_dict, coordinator,
                              use_norm):
    """Transform the prior from the unit cube to the parameter space."""
    if use_norm:
        order = np.array(['teff', 'logg', 'z', 'norm', 'av'])
    else:
        order = np.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'av']
        )

    for filt, flx, flx_e in zip(filts, flux, flux_er):
        p_ = get_noise_name(filt) + '_noise'
        order = np.append(order, p_)

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
            try:
                u[i] = prior_dict['teff'](u[i])
            except TypeError:
                u[i] = prior_dict['teff'].ppf(u[i])
            i += 1
            continue
        u[i] = prior_dict[par].ppf(u[i])
        i += 1
    pass
