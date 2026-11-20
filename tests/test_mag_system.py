"""Per-band magnitude systems (HST VEGAMAG / ABMAG / STMAG, #78)."""
import numpy as np
import pytest

from astroARIADNE.phot_utils import (convert_f_nu_to_f_lambda,
                                     default_mag_system, flux_to_mag,
                                     get_effective_wavelength, get_zero_flux,
                                     mag_to_flux, zero_flux)
from astroARIADNE.star import Star

HST = 'HST_WFC3_F275W'
MAGS = {'2MASS_J': (9.763, 0.028), '2MASS_H': (9.135, 0.026),
        '2MASS_Ks': (8.899, 0.019), HST: (14.2, 0.03)}


def _ab(band):
    return convert_f_nu_to_f_lambda(3.631e-20, get_effective_wavelength(band))


def test_defaults():
    assert default_mag_system(HST) == 'Vega'
    assert default_mag_system('SkyMapper_z') == 'AB'
    assert np.isclose(zero_flux(HST), get_zero_flux(HST), rtol=1e-12)


@pytest.mark.parametrize('system, f0', [
    ('AB', lambda b: _ab(b)),
    ('ab', lambda b: _ab(b)),
    ('ST', lambda b: 10 ** (-0.4 * 21.10) * 1e4),
    ('Vega', lambda b: get_zero_flux(b)),
])
def test_explicit_system(system, f0):
    flux, _ = mag_to_flux(0.0, 0.01, HST, system)
    assert np.isclose(flux, f0(HST), rtol=1e-12)
    mag, _ = flux_to_mag(flux * 10 ** (-0.4 * 12.3), 0., HST, system)
    assert np.isclose(mag, 12.3, atol=1e-10)


def test_unknown_system_rejected():
    with pytest.raises(ValueError):
        zero_flux(HST, 'johnson')


def _star(**kw):
    return Star('demo', 172.56, 7.588, offline=True, mag_dict=MAGS,
                dist=500., dist_e=10., Av=0.05, verbose=False, **kw)


def test_star_applies_override():
    i = list(_star().filter_names).index(HST)
    vega = _star().flux[i]
    ab = _star(mag_system={HST: 'AB'}).flux[i]
    assert np.isclose(vega, 10 ** (-0.4 * 14.2) * get_zero_flux(HST))
    assert np.isclose(ab, 10 ** (-0.4 * 14.2) * _ab(HST))


def test_add_mag_with_system():
    s = _star()
    s.add_mag(14.2, 0.03, HST, system='ST')
    i = list(s.filter_names).index(HST)
    assert s.mag_system == {HST: 'ST'}
    assert np.isclose(s.flux[i], 10 ** (-0.4 * (14.2 + 21.10)) * 1e4)


@pytest.mark.parametrize('bad', [{'NOT_A_BAND': 'AB'}, {HST: 'johnson'}])
def test_bad_mag_system_rejected(bad):
    # InputError.__raise__ prints and exits, as elsewhere in Star.
    with pytest.raises(SystemExit):
        _star(mag_system=bad)


