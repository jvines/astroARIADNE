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
                np.asarray(g['wavelength'][:], float) * u.AA,
                np.asarray(g['transmission'][:], float),
                name=name, dtype=dt, unit=un)
    return out


_CUSTOM_LIB = _load_custom_filters()


def _get_filter(band):
    """Filter lookup preferring ARIADNE's custom library, then pyphot's."""
    if band in _CUSTOM_LIB:
        return _CUSTOM_LIB[band]
    return _PYPHOT_LIB[band]


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


# Stromgren magnitudes are on the standard uvby system (catalogues give V, b-y,
# m1, c1 and the librarian builds u, v, b, y from them), in which Vega is NOT at
# zero. pyphot's Vega zero points put Vega at 0 in every band, so standard-system
# magnitudes must be shifted by Vega's own colours first; y is treated like
# Johnson V (Vega = 0), as for every other Vega-system band. Vega (HD 172167),
# Hauck & Mermilliod 1998 (II/215): b-y = 0.003, m1 = 0.157, c1 = 1.088, giving
# u-y = 1.411, v-y = 0.163, b-y = 0.003. Without this, every Stromgren u flux was
# 3.7x too faint (the robust photometry QC measured +1.32 mag in 28 roster
# stars; v +0.11, b -0.02, y +0.01).
_STROMGREN_VEGA = {'STROMGREN_u': 1.411, 'STROMGREN_v': 0.163,
                   'STROMGREN_b': 0.003, 'STROMGREN_y': 0.0}


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
    10**(-0.4 * 21.10) erg s-1 cm-2 AA-1; Vega is pyphot's Vega zero point,
    shifted for the standard Stromgren system (see _STROMGREN_VEGA).
    """
    system = normalize_mag_system(system) or default_mag_system(band)
    if system == 'AB':
        return convert_f_nu_to_f_lambda(3.631e-20,
                                        get_effective_wavelength(band))
    if system == 'ST':
        return 10 ** (-0.4 * 21.10) * 1e4
    return get_zero_flux(band) * 10 ** (0.4 * _STROMGREN_VEGA.get(band, 0.0))


def mag_to_flux(mag, mag_err, band, system=None):
    """Convert from magnitude to flux in erg s-1 cm-2 um-1.

    ``band`` must match a pyphot (or custom) filter name. ``system`` is the
    magnitude system, 'Vega', 'AB' or 'ST'; None uses the band's default:
    AB for PanSTARRS, SDSS, GALEX and SkyMapper, Vega otherwise. (SkyMapper
    was wrongly routed through Vega until 2026-06: its AB-native mags came
    out 20-60% faint in flux, the redder the band the worse.)
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
