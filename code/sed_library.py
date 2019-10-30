"""sed_library contain the model, prior and likelihood to be used."""

import astropy.units as u
import scipy as sp
from extinction import apply
from scipy.special import ndtr

from phot_utils import *
from Star import *


def build_params(theta, coordinator, fixed, use_norm):
    """Build the parameter vector that goes into the model."""
    if use_norm:
        params = sp.zeros(6)
        order = sp.array(['teff', 'logg', 'z', 'norm', 'Av', 'inflation'])
    else:
        params = sp.zeros(7)
        order = sp.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation']
        )
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


def model_grid(theta, star, interpolators, use_norm, av_law):
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
    mask = star.filter_mask

    if use_norm:
        teff, logg, z, norm, Av, inflation = theta
    else:
        teff, logg, z, dist, rad, Av, inflation = theta
        dist = (dist * u.pc).to(u.solRad).value

    flux = get_interpolated_flux(teff, logg, z, star, interpolators)

    wav = star.wave[mask] * 1e4
    ext = av_law(wav, Av, Rv)
    if use_norm:
        model = apply(ext, flux) * norm
    else:
        model = apply(ext, flux) * (rad / dist) ** 2
    return model


def get_residuals(theta, star, interpolators, use_norm, av_law):
    """Calculate residuals of the model."""
    model = model_grid(theta, star, interpolators, use_norm, av_law)
    inflation = theta[-1]
    mask = star.filter_mask
    residuals = model - star.flux[mask]
    errs = star.flux_er[mask]
    errs = sp.sqrt(errs ** 2 * (1 + inflation ** 2))
    return residuals, errs


def log_prior(theta, star, prior_dict, coordinator, use_norm):
    """Calculate prior."""
    # DEPRECATED
    if use_norm:
        order = sp.array(['teff', 'logg', 'z', 'norm', 'Av', 'inflation'])
    else:
        order = sp.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation']
        )

    lp = 0
    i = 0
    for fixed, par in zip(coordinator, order):
        if fixed:
            continue
        if par == 'logg':
            if not 0 <= theta[i] <= 5:
                return -sp.inf
            try:
                lp += prior_dict['logg'].pdf(theta[i])
            except AttributeError:
                with closing(open('../Datafiles/logg_kde.pkl', 'rb')) as jar:
                    prior = pickle.load(jar)['logg']
                lp += prior(theta[i])
            i += 1
            continue
        if par == 'teff':
            if not 3500 <= theta[i] <= 12000:
                return -sp.inf
            lp += prior_dict['teff'].pdf(
                theta[i]) if star.get_temp else prior_dict['teff'](theta[i])
            i += 1
            continue
        if par == 'z' and not (-1 <= theta[i] <= 1):
            return -sp.inf
        if par == 'dist' and not (1 <= theta[i]):
            return -sp.inf
        if par == 'rad' and not (0 < theta[i]):
            return -sp.inf
        if par == 'norm' and not (theta[i] < 0):
            return -sp.inf
        if par == 'Av' and not (0 <= theta[i] <= star.Av):
            return -sp.inf
        if par == 'inflation' and not (0 <= theta[i] <= 5):
            return -sp.inf
        lp += prior_dict[par].pdf(theta[i])
        i += 1

    return lp


def log_likelihood(theta, star, interpolators, use_norm, av_law):
    """Calculate log likelihood of the model."""
    res, ers = get_residuals(theta, star, interpolators, use_norm, av_law)

    c = sp.log(2 * sp.pi * ers ** 2)
    lnl = (c + (res ** 2 / ers ** 2)).sum()

    if sp.isnan(lnl):
        return -sp.inf

    return -.5 * lnl


def log_probability(theta, star, prior_dict, coordinator, interpolators,
                    fixed):
    """Calculate unnormalized posterior probability of the model."""
    # DEPRECATED
    params = build_params(theta, coordinator, fixed)
    lp = log_prior(params, prior_dict, coordinator)
    lnl = log_likelihood(params, star, interpolators)
    if not sp.isfinite(lnl) or not sp.isfinite(lp):
        return -sp.inf
    return lp + lnl


def prior_transform_dynesty(u, star, prior_dict, coordinator, use_norm):
    """Transform the prior from the unit cube to the parameter space."""
    u2 = sp.array(u)
    if use_norm:
        order = sp.array(['teff', 'logg', 'z', 'norm', 'Av', 'inflation'])
    else:
        order = sp.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation']
        )

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


def prior_transform_multinest(u, star, prior_dict, coordinator, use_norm):
    """Transform the prior from the unit cube to the parameter space."""
    if use_norm:
        order = sp.array(['teff', 'logg', 'z', 'norm', 'Av', 'inflation'])
    else:
        order = sp.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation']
        )
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
