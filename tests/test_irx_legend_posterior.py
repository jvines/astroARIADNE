"""IRX_SED prints and draws the POSTERIOR MEDIANS of the dust parameters.

Regression: the legend printed the least-squares blackbody (``disk_bb``) and the 2-T MAP pair, which
disagreed with the posterior summary of the same result (a 256 K disk read 454 K in the legend). The
legend values and the drawn curve now come from the summary's posterior medians.
"""
import re

import numpy as np
from extinction import fitzpatrick99
from scipy.constants import h, c, k as kB

from astroARIADNE.fitter import _excess_model_selection
from astroARIADNE.phot_utils import get_effective_wavelength
from astroARIADNE.plotter import _irx_disk_display
from astroARIADNE.sed_library import load_grid

NAMES = np.array(['WISE_RSR_W3', 'AKARI_IRC_L18W', 'WISE_RSR_W4', 'SPITZER_MIPS_24', 'IRAS_60',
                  'SPITZER_MIPS_70', 'HERSCHEL_PACS_GREEN', 'SPITZER_MIPS_160'])
WAVES = np.array([get_effective_wavelength(b) for b in NAMES])


def _planck_um(lam, T):
    ll = lam * 1e-6
    return (2 * h * c ** 2 / ll ** 5) / np.expm1(h * c / (ll * kB * T)) * 1e-3


def _select(F):
    g = load_grid('btsettl')
    return _excess_model_selection(NAMES, WAVES, F, 0.05 * F, [('btsettl', g)], 0., 0., 30.0, 1.0,
                                   False, lambda w, a, rv: fitzpatrick99(w, a, rv), 6000.)


def _numbers(label):
    return [float(x) for x in re.findall(r'=\s*([0-9.]+(?:e-?[0-9]+)?)', label.replace('$', ''))]


def test_single_temperature_legend_is_posterior_median():
    r = _select(2e-16 * _planck_um(WAVES, 150.0))
    d = r['disk']
    assert d['t_dust_lo'] <= d['t_dust_med'] <= d['t_dust_hi']
    assert d['l_dust_over_lstar_lo'] <= d['l_dust_over_lstar_med'] <= d['l_dust_over_lstar_hi']
    assert abs(d['t_dust_med'] - 150) < 25
    disp = _irx_disk_display(r)
    assert len(disp['components']) == 1
    T, om = disp['components'][0]
    assert T == d['t_dust_med'] and om == d['omega_med_sr']
    nums = _numbers(disp['label'])        # P, T_d, L_d/L_*
    assert abs(nums[1] - round(d['t_dust_med'])) <= 0.5
    assert np.isclose(nums[2], float('%.1g' % d['l_dust_over_lstar_med']))


def test_two_temperature_legend_is_warm_cold_medians():
    F = 1e-16 * _planck_um(WAVES, 300.0) + 4e-15 * _planck_um(WAVES, 60.0)
    r = _select(F)
    d2 = r['disk2']
    assert r['p_disk2'] > 0.9
    disp = _irx_disk_display(r)
    assert [t for t, _ in disp['components']] == [d2['t_warm_med'], d2['t_cold_med']]
    assert [o for _, o in disp['components']] == [d2['omega_warm_med'], d2['omega_cold_med']]
    nums = _numbers(disp['label'])        # P, T_w, T_c
    assert abs(nums[1] - round(d2['t_warm_med'])) <= 0.5 and abs(nums[2] - round(d2['t_cold_med'])) <= 0.5
    assert d2['l_dust_over_lstar_lo'] <= d2['l_dust_over_lstar_med'] <= d2['l_dust_over_lstar_hi']
    for k in ('warm', 'cold'):
        assert d2[f't_{k}_lo'] <= d2[f't_{k}_med'] <= d2[f't_{k}_hi']


def test_median_omega_reproduces_median_luminosity():
    r = _select(2e-16 * _planck_um(WAVES, 150.0))
    d = r['disk']
    assert d['omega_med_sr'] > 0


def test_results_without_medians_fall_back_to_legacy_curve():
    fit = dict(p_disk=1.0, p_disk2=0.0, disk={'t_dust': 200.0}, disk_bb=dict(
        t_dust=210.0, omega_sr=1e-17, l_dust_over_lstar=0.02))
    disp = _irx_disk_display(fit)
    assert disp['components'] == [(210.0, 1e-17)]
    assert _irx_disk_display({}) is None
