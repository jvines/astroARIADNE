"""phot_utils module for SED fitting.

This module contains useful functions in order to obtain fluxes from different
broadband filters. It also has functions to convert to different units of flux

It uses the module pyphot to get the fluxes and bandpasses of different
broadband filters.
"""


from __future__ import print_function, division

import scipy as sp
import astropy.units as u
import astropy.constants as const
import pyphot


def convert_jansky_to_ergs(j):
    """Convert flux from jansky to erg s-1 cm-2."""
    return j * 1e-23


def convert_jansky_to_ergs_lambda(j, l):
    """Convert flux from jansky to erg s-2 cm-2 lambda-1 in the units of l."""
    # TODO: fix the constant and use astropy constans instead
    return j * 2.99792458E-05 / l ** 2


def convert_f_lambda_to_f_nu(f, l):
    """Convert flux from erg s-1 cm-2 lambda-1 to erg s-1 cm-2 Hz-1."""
    # TODO: fix the constant and use astropy constants instead
    return sed / 2.99792458E+18 * l ** 2


def convert_f_nu_to_f_lambda(f, l):
    """Convert flux from erf s-1 cm-2 Hz-1 to erg s-1 cm-2 lambda-1."""
    return f * const.c.to(u.micrometer / u.s).value / l ** 2


def mag_to_flux(mag, mag_err, band):
    """Convert from magnitude to flux.

    mag_to_flux performs the conversion from magnitude to flux in
    erg s-1 cm-2 um-1.

    The band parameter is a string representing the filter used and it must
    match exactly the name in pyphots filter database

    If the filter is from PanSTARRS or SDSS, then the magnitude is in the AB
    system. Else it's in the Vega system.
    """
    if 'PS1_' in band or 'SDSS_' in band:
        # Get flux from AB mag
        flux, flux_err = mag_to_flux_AB(mag, mag_err)
        # Get effective wavelength for bandpass
        leff = get_effective_wavelength(band)
        # Convert from f_nu to f_lambda in erg / cm2 / s / um
        flux = convert_f_nu_to_f_lambda(flux, leff)
        flux_err = convert_f_nu_to_f_lambda(flux_err, leff)
    else:
        # Get flux in erg / cm2 / s / um
        f0 = get_band_info(band)
        flux = 10 ** (-.4 * mag) * f0
        flux_err = abs(-.4 * flux * sp.log(10) * mag_err)
    return flux, flux_err


def get_band_info(band):
    """Look for the filter information in the pyphot library of filters."""
    # TODO: rename?
    # Load photometry filter library
    filt = pyphot.get_library()[band]
    # Get Vega zero flux in erg / cm2 / s / um
    f0 = filt.Vega_zero_flux.to('erg/(um * cm ** 2 * s)').magnitude
    return f0


def get_effective_wavelength(band):
    """Get central wavelength of a specific filter in um."""
    # Load photometry filter library
    filt = pyphot.get_library()[band]
    # Get central wavelength in um
    leff = filt.cl.to('um').magnitude
    return leff


def get_bandpass(band):
    """Get the bandpass of a specific filter in um."""
    # Load photometry filter library
    filt = pyphot.get_library()[band]
    # Get lower and upper bandpass in um
    bp_low = filt.wavelength[0].to('um').magnitude
    bp_up = filt.wavelength[-1].to('um').magnitude
    return bp_low, bp_up


def mag_to_flux_AB(mag, mag_err):
    """Calculate flux in erg s-1 cm-2 Hz-1."""
    flux = 10 ** (-.4 * (mag + 48.57))
    flux_err = abs(-.4 * flux * sp.log(10) * mag_err)
    return flux, flux_err
