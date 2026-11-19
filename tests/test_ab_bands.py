"""AB-native surveys must be converted with the AB zero point."""
import numpy as np
import pytest

from astroARIADNE.phot_utils import (convert_f_nu_to_f_lambda, flux_to_mag,
                                     get_effective_wavelength, mag_to_flux)

AB_BANDS = ['SkyMapper_u', 'SkyMapper_v', 'SkyMapper_g', 'SkyMapper_r',
            'SkyMapper_i', 'SkyMapper_z', 'PS1_r', 'SDSS_g', 'GALEX_NUV']


@pytest.mark.parametrize('band', AB_BANDS)
def test_ab_zero_point(band):
    # m_AB = 0 is 3631 Jy (48.6 offset; the two agree to 6e-5).
    flux, _ = mag_to_flux(0.0, 0.01, band)
    expected = convert_f_nu_to_f_lambda(3.631e-20,
                                        get_effective_wavelength(band))
    assert np.isclose(flux, expected, rtol=1e-4, atol=0)


@pytest.mark.parametrize('band', AB_BANDS)
def test_ab_round_trip(band):
    flux, err = mag_to_flux(12.3, 0.02, band)
    mag, mag_err = flux_to_mag(flux, err, band)
    assert np.isclose(mag, 12.3, rtol=0, atol=1e-4) and np.isclose(mag_err, 0.02, rtol=1e-8)
