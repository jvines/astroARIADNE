"""phot_utils module for SED fitting.

This module contains useful functions in order to obtain fluxes from different
broadband filters. It also has functions to convert to different units of flux

It uses the module pyphot to get the fluxes and bandpasses of different
broadband filters.
"""

import os

import astropy.constants as const
import astropy.units as u
import pyphot
import numpy as np

from .config import custom_filters_path

_PYPHOT_LIB = pyphot.get_library()


def _load_custom_filters():
    """Build pyphot Filter objects for ARIADNE's custom (mid/far-IR) bands that
    pyphot's bundled library lacks -- Spitzer MIPS, AKARI/IRC, IRAS and Herschel
    SPIRE PLW. Curves and detector convention ship in Datafiles/custom_filters.h5
    (built by notebooks/build_custom_filters.py). Returns {} if the file is
    absent so the library degrades to the pyphot default instead of failing.
    """
    import h5py
    from pyphot.ezunits import unit
    out = {}
    if not os.path.isfile(custom_filters_path):
        return out
    with h5py.File(custom_filters_path, 'r') as h:
        for name in h:
            g = h[name]
            dt = g.attrs.get('dtype', 'photon')
            un = g.attrs.get('unit', 'Angstrom')
            dt = dt.decode() if isinstance(dt, bytes) else dt
            un = un.decode() if isinstance(un, bytes) else un
            out[name] = pyphot.Filter(
                np.asarray(g['wavelength'][:], float) * unit['AA'],
                np.asarray(g['transmission'][:], float),
                name=name, dtype=dt, unit=un)
    return out


_CUSTOM_LIB = _load_custom_filters()


def _get_filter(band):
    """Filter lookup preferring ARIADNE's custom library, then pyphot's."""
    if band in _CUSTOM_LIB:
        return _CUSTOM_LIB[band]
    return _PYPHOT_LIB[band]


def extract_info(magnitudes, errors, filters):
    """Extract the flux information for a Star."""
    flux = dict()
    flux_er = dict()
    wave = dict()
    bandpass = dict()

    for mag, err, band in zip(magnitudes, errors, filters):
        # Get central wavelength
        leff = get_effective_wavelength(band)
        mag_err = err
        # get flux, flux error and bandpass
        flx, flx_err = mag_to_flux(mag, mag_err, band)
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


def mag_to_flux(mag, mag_err, band):
    """Convert from magnitude to flux.

    mag_to_flux performs the conversion from magnitude to flux in
    erg s-1 cm-2 um-1.

    The band parameter is a string representing the filter used and it must
    match exactly the name in pyphots filter database

    If the filter is from PanSTARRS, SDSS, GALEX or SkyMapper, the magnitude
    is in the AB system. Else it's in the Vega system. (SkyMapper was wrongly
    routed through the Vega branch until 2026-06: its AB-native mags came out
    20-60% faint in flux -- the redder the band the worse -- which tilted
    every fit that used them.)
    """
    if 'PS1_' in band or 'SDSS_' in band or 'GALEX_' in band \
            or 'SkyMapper_' in band:
        # Get flux from AB mag
        flux, flux_err = mag_to_flux_AB(mag, mag_err)
        # Get effective wavelength for bandpass
        leff = get_effective_wavelength(band)
        # Convert from f_nu to f_lambda in erg / cm2 / s / um
        flux = convert_f_nu_to_f_lambda(flux, leff)
        flux_err = convert_f_nu_to_f_lambda(flux_err, leff)
    else:
        # Get flux in erg / cm2 / s / um
        f0 = get_zero_flux(band)
        flux = 10 ** (-.4 * mag) * f0
        flux_err = abs(-.4 * flux * np.log(10) * mag_err)
    return flux, flux_err


def flux_to_mag(flux, flux_err, band):
    """Convert from flux to magnitude.

    The flux is expected to be in the units of erg s-1 cm-2 um-1
    """
    leff = get_effective_wavelength(band)
    if 'PS1_' in band or 'SDSS_' in band or 'GALEX_' in band \
            or 'SkyMapper_' in band:
        f0 = convert_f_nu_to_f_lambda(3.631e-20, leff)
    else:
        f0 = get_zero_flux(band)
    mag = -2.5 * np.log10(flux / f0)
    mag_err = 2.5 * flux_err / (np.log(10) * flux)
    return mag, mag_err


def get_zero_flux(band):
    """Look for the filter information in the pyphot library of filters."""
    # Load photometry filter library
    filt = _get_filter(band)
    # Get Vega zero flux in erg / cm2 / s / um
    f0 = filt.Vega_zero_flux.to('erg/(um * cm ** 2 * s)').value
    return f0


def get_effective_wavelength(band):
    """Get central wavelength of a specific filter in um."""
    # Load photometry filter library
    filt = _get_filter(band)
    # Get central wavelength in um
    leff = filt.cl.to('um').value
    return leff


def get_bandpass(band):
    """Get the bandpass of a specific filter in um."""
    # Load photometry filter library
    filt = _get_filter(band)
    # Get lower and upper bandpass in um
    width = filt.width.to('um').value
    bp = width
    return bp / 2


def mag_to_flux_AB(mag, mag_err):
    """Calculate flux in erg s-1 cm-2 Hz-1."""
    flux = 10 ** (-.4 * (mag + 48.6))
    flux_err = abs(-.4 * flux * np.log(10) * mag_err)
    return flux, flux_err
