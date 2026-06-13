"""Tests for the blackbody gate-Teff estimate (`_bb_gate_teff`).

When a star has no catalogue Teff, grid gating falls back to a blackbody fit
to the observed photometry, calibrated against BT-Settl synthetic photometry
in the star's own band set. The calibration matters: a raw blackbody fit runs
~500-600 K cool around the 4000-5000 K range (line/molecular blanketing), so
a true 4500 K dwarf would read as < 4000 K and be misgated into the
ultra-cool ensemble (BT triplet multi-counted, Kurucz/CK04 dropped).

Runs as a plain script (no pytest):
    python tests/test_bb_gate.py
"""
import os
import sys
import types
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
from astroARIADNE.fitter import _bb_gate_teff, _bb_fit_teff  # noqa: E402
from astroARIADNE.sed_library import get_interpolated_flux, load_grid  # noqa: E402

# A typical survey band set: Johnson/Tycho + Gaia + 2MASS + WISE.
BANDS = np.array([
    'TYCHO_B_MvB', 'TYCHO_V_MvB', 'GROUND_JOHNSON_B', 'GROUND_JOHNSON_V',
    'GaiaDR2v2_BP', 'GaiaDR2v2_G', 'GaiaDR2v2_RP',
    '2MASS_J', '2MASS_H', '2MASS_Ks', 'WISE_RSR_W1', 'WISE_RSR_W2',
])


def _synthetic_star(true_teff, grid):
    fl = np.asarray(get_interpolated_flux(true_teff, 4.5, 0.0, BANDS, grid),
                    dtype=float)
    waves = np.array([0.42, 0.53, 0.44, 0.55, 0.50, 0.62, 0.78,
                      1.24, 1.65, 2.16, 3.35, 4.60])
    ok = np.isfinite(fl) & (fl > 0)
    return types.SimpleNamespace(
        used_filters=np.ones(ok.sum()), wave=waves[ok], flux=fl[ok],
        flux_er=0.02 * fl[ok], filter_names=BANDS[ok])


def test_calibration_recovers_grid_teff():
    """Calibrated estimate lands within 150 K of the input across the gate
    boundaries (raw blackbody is off by up to ~600 K here)."""
    grid = load_grid('btsettl')
    for true_t in (3200., 3800., 4200., 4500., 5200., 6300.):
        s = _synthetic_star(true_t, grid)
        est = _bb_gate_teff(s)
        assert est is not None
        assert abs(est - true_t) < 150., (true_t, est)


def test_k_dwarf_not_misgated():
    """Regression: a true 4200 K dwarf must gate as > 4000 K. In this band
    set the raw blackbody fit lands at ~3820 K (cool blanketing bias) --
    below the 4000 K boundary, which would re-admit the BT clones and drop
    Kurucz/CK04."""
    grid = load_grid('btsettl')
    s = _synthetic_star(4200., grid)
    raw = _bb_fit_teff(s.wave, s.flux, s.flux_er)
    assert raw is not None and raw[0] < 4000.   # the misgate premise is real
    assert _bb_gate_teff(s) > 4000.             # the calibration fixes it


def test_hot_star_keeps_bosz():
    """Regression: above ~8000 K the raw blackbody runs HOT (+~2000 K at
    15000 K, Rayleigh-Jeans bias). Uncalibrated, a true 15000 K star gated
    > 16000 K, silently dropping BOSZ -- the correct grid for it."""
    grid = load_grid('btsettl')
    s = _synthetic_star(15000., grid)
    raw = _bb_fit_teff(s.wave, s.flux, s.flux_er)
    assert raw is not None and raw[0] > 16000.  # the hot bias premise
    est = _bb_gate_teff(s)
    assert 13000. < est < 16000., est           # bosz survives the gate


def test_u_band_does_not_flip_the_gate():
    """Regression: a u-band detection used to corrupt the calibration right
    at the 4000 K boundary (Balmer-jump bias is non-monotonic). u is now
    excluded from the gate fit entirely (_BB_GATE_MIN_UM)."""
    grid = load_grid('btsettl')
    bands_u = np.append(BANDS, 'SDSS_u')
    fl = np.asarray(get_interpolated_flux(4500., 4.5, 0.0, bands_u, grid),
                    dtype=float)
    waves = np.array([0.42, 0.53, 0.44, 0.55, 0.50, 0.62, 0.78,
                      1.24, 1.65, 2.16, 3.35, 4.60, 0.355])
    ok = np.isfinite(fl) & (fl > 0)
    s = types.SimpleNamespace(
        used_filters=np.ones(ok.sum()), wave=waves[ok], flux=fl[ok],
        flux_er=0.02 * fl[ok], filter_names=bands_u[ok])
    assert _bb_gate_teff(s) > 4000.


def test_too_few_bands_returns_none():
    s = types.SimpleNamespace(
        used_filters=np.ones(2), wave=np.array([0.55, 2.16]),
        flux=np.array([1e-10, 1e-11]), flux_er=np.array([1e-12, 1e-13]),
        filter_names=np.array(['GROUND_JOHNSON_V', '2MASS_Ks']))
    assert _bb_gate_teff(s) is None


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
    print('ALL PASS')
