"""phot_utils module for SED fitting.

This module contains useful functions in order to obtain fluxes from different
broadband filters. It also has functions to convert to different units of flux

It uses the module pyphot to get the fluxes and bandpasses of different
broadband filters.
"""

import astropy.constants as const
import astropy.units as u
import pyphot
import numpy as np

_PYPHOT_LIB = pyphot.get_library()


def extract_info(magnitudes, errors, filters, systems=None):
    """Extract the flux information for a Star.

    ``systems`` maps band -> magnitude system for bands not in their default
    system (see ``mag_to_flux``)."""
    flux = dict()
    flux_er = dict()
    wave = dict()
    bandpass = dict()
    systems = systems or {}

    for mag, err, band in zip(magnitudes, errors, filters):
        # Get central wavelength
        leff = get_effective_wavelength(band)
        mag_err = err
        # get flux, flux error and bandpass
        flx, flx_err = mag_to_flux(mag, mag_err, band, systems.get(band))
        bp = get_bandpass(band)
        flux[band] = flx  # * leff
        flux_er[band] = flx_err  # * leff
        wave[band] = leff
        bandpass[band] = bp

    return wave, flux, flux_er, bandpass


def convert_jansky_to_ergs(j):
    """Convert flux from jansky to erg s-1 cm-2 Hz-1."""
    return j * 1e-23


def convert_jansky_to_ergs_lambda(j, l):
    """Convert flux from jansky to erg s-2 cm-2 lambda-1 in the units of l."""
    return j * const.c.to(u.micrometer / u.s).value / l ** 2


def convert_f_lambda_to_f_nu(f, l):
    """Convert flux from erg s-1 cm-2 lambda-1 to erg s-1 cm-2 Hz-1."""
    return f / const.c.to(u.micrometer / u.s).value * l ** 2


def convert_f_nu_to_f_lambda(f, l):
    """Convert flux from erf s-1 cm-2 Hz-1 to erg s-1 cm-2 lambda-1."""
    return f * const.c.to(u.micrometer / u.s).value / l ** 2


# Surveys whose catalogues are AB-native. Every other band defaults to Vega.
# HST photometry comes in VEGAMAG, ABMAG or STMAG depending on the source, so
# HST bands default to Vega and callers say otherwise per band (Star's
# ``mag_system``).
_AB_PREFIXES = ('PS1_', 'SDSS_', 'GALEX_', 'SkyMapper_')
_MAG_SYSTEMS = {'vega': 'Vega', 'ab': 'AB', 'st': 'ST'}


def normalize_mag_system(system):
    """'vega'/'AB'/'st' (any case) -> 'Vega'/'AB'/'ST'; None stays None."""
    if system is None:
        return None
    try:
        return _MAG_SYSTEMS[str(system).lower()]
    except KeyError:
        raise ValueError(f'unknown magnitude system {system!r}; use one of '
                         'Vega, AB, ST') from None


def default_mag_system(band):
    """The system a band's magnitudes are in unless told otherwise."""
    return 'AB' if any(p in band for p in _AB_PREFIXES) else 'Vega'


def zero_flux(band, system=None):
    """Flux of a zero-magnitude source in ``band``, erg s-1 cm-2 um-1.

    ``system`` is 'Vega', 'AB' or 'ST' (None: the band's default). AB is
    3631 Jy at the band's effective wavelength; ST is f_lambda =
    10**(-0.4 * 21.10) erg s-1 cm-2 AA-1; Vega is pyphot's Vega zero point.
    """
    system = normalize_mag_system(system) or default_mag_system(band)
    if system == 'AB':
        return convert_f_nu_to_f_lambda(3.631e-20,
                                        get_effective_wavelength(band))
    if system == 'ST':
        return 10 ** (-0.4 * 21.10) * 1e4
    return get_zero_flux(band)


def mag_to_flux(mag, mag_err, band, system=None):
    """Convert from magnitude to flux in erg s-1 cm-2 um-1.

    ``band`` must match a pyphot filter name. ``system`` is the magnitude
    system, 'Vega', 'AB' or 'ST'; None uses the band's default: AB for
    PanSTARRS, SDSS, GALEX and SkyMapper, Vega otherwise.
    """
    flux = 10 ** (-.4 * mag) * zero_flux(band, system)
    flux_err = abs(-.4 * flux * np.log(10) * mag_err)
    return flux, flux_err


def flux_to_mag(flux, flux_err, band, system=None):
    """Convert from flux (erg s-1 cm-2 um-1) to magnitude in ``system``
    (None: the band's default, as in ``mag_to_flux``)."""
    mag = -2.5 * np.log10(flux / zero_flux(band, system))
    mag_err = 2.5 * flux_err / (np.log(10) * flux)
    return mag, mag_err


def get_zero_flux(band):
    """Look for the filter information in the pyphot library of filters."""
    # Load photometry filter library
    filt = _PYPHOT_LIB[band]
    # Get Vega zero flux in erg / cm2 / s / um
    f0 = filt.Vega_zero_flux.to('erg/(um * cm ** 2 * s)').value
    return f0


def get_effective_wavelength(band):
    """Get central wavelength of a specific filter in um."""
    # Load photometry filter library
    filt = _PYPHOT_LIB[band]
    # Get central wavelength in um
    leff = filt.cl.to('um').value
    return leff


def get_bandpass(band):
    """Get the bandpass of a specific filter in um."""
    # Load photometry filter library
    filt = _PYPHOT_LIB[band]
    # Get lower and upper bandpass in um
    width = filt.width.to('um').value
    bp = width
    return bp / 2


def mag_to_flux_AB(mag, mag_err):
    """Calculate flux in erg s-1 cm-2 Hz-1."""
    flux = 10 ** (-.4 * (mag + 48.6))
    flux_err = abs(-.4 * flux * np.log(10) * mag_err)
    return flux, flux_err
