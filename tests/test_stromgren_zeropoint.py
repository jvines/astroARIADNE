"""Stromgren magnitudes are standard-system: Vega's own u is 1.41, not 0.

Regression: mag_to_flux treated them as Vega = 0, so every Stromgren u flux was
3.7x too faint (robust QC: +1.32 mag median offset across the roster).
"""
import numpy as np

from astroARIADNE.phot_utils import flux_to_mag, get_zero_flux, mag_to_flux


def test_vega_standard_u_maps_to_vega_flux():
    f, _ = mag_to_flux(1.411, 0.01, 'STROMGREN_u')     # Vega's standard-system u
    assert np.isclose(f, get_zero_flux('STROMGREN_u'), rtol=1e-6)


def test_round_trip_all_uvby():
    for b in ('STROMGREN_u', 'STROMGREN_v', 'STROMGREN_b', 'STROMGREN_y'):
        f, e = mag_to_flux(9.0, 0.02, b)
        m, me = flux_to_mag(f, e, b)
        assert np.isclose(m, 9.0) and np.isclose(me, 0.02)


def test_johnson_unchanged():
    f, _ = mag_to_flux(0.0, 0.01, 'GROUND_JOHNSON_V')
    assert np.isclose(f, get_zero_flux('GROUND_JOHNSON_V'))
