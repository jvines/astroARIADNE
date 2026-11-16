"""Star.clip_outlier_magnitudes with a single Gaia band.

Regression: with G but no BP/RP the Gaia-anchored path could not fit its
blackbody (two parameters, one anchor), flagged every other band and recorded
no temperature, so the no-model SED plot had no blackbody overlay.
"""
import types

import numpy as np

from astroARIADNE.star import Star


def _fake_star(names, mags, errs):
    s = types.SimpleNamespace()
    s.filter_names = np.array(names)
    s.mags = np.array(mags, float)
    s.mag_errs = np.array(errs, float)
    s.filter_mask = np.arange(len(names))
    return s


def test_single_gaia_band_still_fits_blackbody():
    # photometry of Gaia DR3 4854879577968187392 (G only, no BP/RP)
    s = _fake_star(['GaiaDR2v2_G', '2MASS_J', '2MASS_H', '2MASS_Ks', 'WISE_RSR_W1', 'WISE_RSR_W2'],
                   [17.675, 12.965, 12.302, 12.024, 11.757, 11.504],
                   [0.004, 0.027, 0.028, 0.027, 0.022, 0.021])
    Star.clip_outlier_magnitudes(s, sigma=5.0, err_floor=0.05, warn_only=True)
    assert s._qc_bb_teff is not None and np.isfinite(s._qc_bb_teff)
    assert len(s._qc_bb_flagged_bands) < len(s.filter_names) - 1
