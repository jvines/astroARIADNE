"""Offline unit tests for the SPHEREx extraction internals (_spherex.py).

Runs as a plain script (no pytest):
    python tests/test_spherex_extract.py
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__

from astroARIADNE.librarian._spherex import (  # noqa: E402
    _annulus_median, _aperture_sum)


def test_annulus_median_ignores_nan():
    """A NaN pixel in the background annulus must not poison the median.

    Regression: np.median over a mask that kept NaN pixels returned NaN for
    any exposure with an unobserved region in the annulus, silently dropping
    it -- this killed the SPHEREX_2750/3050 super-bands (the 2MASS-WISE gap)
    for SCR 1845 despite enough good exposures per window.
    """
    img = np.full((41, 41), 7.0)
    img[20, 35] = np.nan                     # one NaN inside the annulus
    bkg = _annulus_median(img, 20.0, 20.0, 10.0, 16.0)
    assert np.isfinite(bkg), 'NaN annulus pixel poisoned the background'
    assert abs(bkg - 7.0) < 1e-12


def test_annulus_median_requires_coverage():
    """An annulus that is mostly unobserved still returns NaN (no
    background estimate from a handful of pixels)."""
    img = np.full((41, 41), np.nan)
    img[20, 32] = 7.0                        # 1 finite pixel only
    bkg = _annulus_median(img, 20.0, 20.0, 10.0, 16.0)
    assert not np.isfinite(bkg)


def test_aperture_sum_flat_field():
    """Subpixel aperture sum on a flat field equals value * pi r^2 to <1%."""
    img = np.full((41, 41), 2.0)
    s, area = _aperture_sum(img, 20.0, 20.0, 3.0)
    assert abs(s - 2.0 * np.pi * 9.0) / (2.0 * np.pi * 9.0) < 0.01
    assert abs(area - np.pi * 9.0) / (np.pi * 9.0) < 0.01


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
    print('ALL PASS')
