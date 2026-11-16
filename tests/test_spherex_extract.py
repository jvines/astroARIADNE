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


def _gaussian_epsf(monkeypatch_target, sigma_px=0.9, ovs=5, n=33):
    """Install a synthetic ePSF (one zone, Gaussian, unit-sum oversampled array)."""
    import astroARIADNE.librarian._spherex as S
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-0.5 * (((xx - c) / ovs) ** 2 + ((yy - c) / ovs) ** 2) / sigma_px ** 2)
    cube = (g / g.sum())[None]
    monkeypatch_target['orig'] = S._epsf
    S._epsf = lambda det: (np.array([1020.0]), np.array([1020.0]), cube, ovs)
    return S


def _restore(S, saved):
    S._epsf = saved['orig']


def test_epsf_native_samples_sum_to_one():
    saved = {}
    S = _gaussian_epsf(saved)
    try:
        u = np.arange(-4, 5)
        X, Y = np.meshgrid(u - 0.3, u + 0.2)
        assert abs(S._epsf_pix(1, 1020.0, 1020.0, X, Y).sum() - 1.0) < 0.01
    finally:
        _restore(S, saved)


def test_forced_psf_separates_a_blended_neighbour():
    """A neighbour 1.6 px from the target is inside the 3-px aperture: the
    aperture returns target + neighbour, the forced ePSF fit the target alone."""
    saved = {}
    S = _gaussian_epsf(saved)
    try:
        rng = np.random.default_rng(5)
        n = 39
        yy, xx = np.mgrid[0:n, 0:n]
        t, nb, F1, F2, bkg = (19.3, 18.7), (20.9, 19.0), 50.0, 80.0, 2.0
        img = bkg + F1 * S._epsf_pix(1, 1020., 1020., xx - t[0], yy - t[1]) \
            + F2 * S._epsf_pix(1, 1020., 1020., xx - nb[0], yy - nb[1])
        var = np.full((n, n), 0.01)
        img = img + rng.normal(0, 0.1, img.shape)
        flg = np.zeros((n, n), int)
        f_psf, e_psf = S._psf_flux(img, var, flg, t[0], t[1], 1020., 1020., 1, [nb])
        unit = S.OMEGA * 1e12
        assert abs(f_psf / unit - F1) / F1 < 0.03, f_psf / unit
        ap = S._measure_arrays(img, var, flg, t[0], t[1], 1.5)
        assert ap[1] / unit > 1.8 * F1        # the aperture took the neighbour too
    finally:
        _restore(S, saved)


def test_predicted_contamination():
    from astroARIADNE.librarian._spherex import predicted_contamination
    near = [dict(sep=5.0, ks=13.0)]            # 1 mag fainter, well inside the aperture
    far = [dict(sep=45.0, ks=10.0)]            # bright but far outside
    assert abs(predicted_contamination(near, 12.0) - 10 ** -0.4) < 0.01
    assert predicted_contamination(far, 12.0) < 1e-6
    assert predicted_contamination([dict(sep=5.0, ks=float('nan'))], 12.0) == 0.0


def test_fit_neighbours_keeps_only_impactful_and_caps():
    """Hundreds of faint catalogued sources must not all become free
    amplitudes: only those sending >1% of the target's flux into the fit
    region are kept, brightest first, at most NB_MAX."""
    from astroARIADNE.librarian._spherex import NB_MAX, fit_neighbours
    faint = [dict(sep=10.0, ks=20.0, g=np.nan) for _ in range(300)]
    bright = [dict(sep=8.0, ks=12.0 + 0.1 * i, g=np.nan) for i in range(30)]
    gaia_only = [dict(sep=8.0, ks=float('nan'), g=14.0)]      # G-2.5 = 11.5: kept
    kept = fit_neighbours(faint + bright + gaia_only, 12.0)
    assert len(kept) == NB_MAX
    assert kept[0]['g'] == 14.0                    # brightest contribution first
    assert all(n['ks'] < 20.0 or np.isfinite(n['g']) for n in kept)


def test_annulus_flux_counts_only_the_annulus():
    from astroARIADNE.librarian._spherex import annulus_flux
    nb = [dict(sep=80.0, ks=12.0), dict(sep=20.0, ks=8.0), dict(sep=150.0, ks=8.0),
          dict(sep=70.0, ks=float('nan'))]
    assert abs(annulus_flux(nb, 12.0) - 1.0) < 1e-9     # only the 80" source counts


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
    print('ALL PASS')
