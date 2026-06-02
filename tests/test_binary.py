"""Tests for unresolved-binary / SB2 mode.

Covers the gate (binary needs no-norm + a parallax + >=1 contrast), the
two-component forward model, the contrast likelihood, the teff1>=teff2 ordering,
and that single-star behaviour is byte-identical when binary is off.
"""
import sys
import types
import numpy as np

REPO = '/Users/jayvains/github/personal/astroARIADNE'
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
import astroARIADNE.sed_library as sl  # noqa: E402
from astroARIADNE.fitter import Fitter  # noqa: E402

import pandas as pd  # noqa: E402
from isochrones.interp import DFInterpolator  # noqa: E402
from astroARIADNE.config import gridsdir  # noqa: E402
from extinction import fitzpatrick99  # noqa: E402

GRID = DFInterpolator(pd.read_pickle(gridsdir + '/Phoenixv2_DF.pkl'))
FILTS = np.array(['2MASS_J', '2MASS_H', '2MASS_Ks', 'GaiaDR2v2_G'])
WAVE = np.array([1.235, 1.662, 2.159, 0.64])


def _avlaw(w, a, rv):
    return fitzpatrick99(w, a, rv)


# ----------------------------- gate -----------------------------------------
def _gate_fitter(binary=True, norm=False, bma=False,
                 contrast=(('GaiaDR2v2_G', 3.0, 0.1),), dist=50.0):
    f = Fitter.__new__(Fitter)
    f.__dict__['_guard_active'] = False
    f.binary = binary
    f._norm = norm
    f._bma = bma
    f.contrast = list(contrast) if contrast is not None else None
    f.star = types.SimpleNamespace(dist=dist)
    f.out = '.'
    return f


def _gate_raises(f, needle):
    try:
        f.initialize()
    except ValueError as e:
        assert needle in str(e), str(e)
        return
    except Exception:
        # any other failure means it got PAST the gate -> gate didn't fire
        raise AssertionError('expected ValueError from binary gate')
    raise AssertionError('binary gate did not raise')


def test_gate_requires_no_norm():
    _gate_raises(_gate_fitter(norm=True), 'norm=False')


def test_gate_requires_contrast():
    _gate_raises(_gate_fitter(contrast=None), 'flux-ratio')


def test_gate_requires_parallax():
    _gate_raises(_gate_fitter(dist=-1), 'parallax')


# --------------------------- forward model ----------------------------------
def test_binary_model_is_sum_of_components():
    t1, g1, z, d, r1, Av = 4800., 4.5, 0.0, 30., 0.8, 0.12
    t2, g2, r2 = 3300., 4.9, 0.3
    th = np.array([t1, g1, z, d, r1, Av, t2, g2, r2])
    m_bin = sl.model_grid(th, FILTS, WAVE, GRID, False, _avlaw, binary=True)
    m1 = sl.model_grid(np.array([t1, g1, z, d, r1, Av]), FILTS, WAVE, GRID,
                       False, _avlaw)
    m2 = sl.model_grid(np.array([t2, g2, z, d, r2, Av]), FILTS, WAVE, GRID,
                       False, _avlaw)
    assert np.allclose(m_bin, m1 + m2, rtol=1e-12)


def test_single_star_path_unchanged():
    th = np.array([5000., 4.4, 0.0, 25., 1.0, 0.05])
    a = sl.model_grid(th, FILTS, WAVE, GRID, False, _avlaw)            # default
    b = sl.model_grid(th, FILTS, WAVE, GRID, False, _avlaw, binary=False)
    assert np.array_equal(a, b)


def test_param_order_binary():
    o = list(sl._param_order(FILTS, False, binary=True)[:9])
    assert o == ['teff', 'logg', 'z', 'dist', 'rad', 'Av',
                 'teff2', 'logg2', 'rad2']


def test_contrast_loglike():
    t1, g1, z, d, r1, Av = 4800., 4.5, 0.0, 30., 0.8, 0.0
    t2, g2, r2 = 3300., 4.9, 0.3
    th = np.array([t1, g1, z, d, r1, Av, t2, g2, r2])
    cf = np.array(['GaiaDR2v2_G'])
    f1 = sl.get_interpolated_flux(t1, g1, z, cf, GRID)
    f2 = sl.get_interpolated_flux(t2, g2, z, cf, GRID)
    dmag = float(2.5 * np.abs(np.log10((r2 / r1) ** 2 * (f2 / f1)))[0])
    at_true = sl.contrast_loglike(th, GRID, (cf, np.array([dmag]),
                                             np.array([0.05])))
    off = sl.contrast_loglike(th, GRID, (cf, np.array([dmag + 0.05]),
                                         np.array([0.05])))
    assert abs(at_true) < 1e-6           # ~0 at the true contrast
    assert np.isclose(off, -0.5)         # 1-sigma off -> -0.5


def test_prior_transform_orders_teff():
    # build a free order/coordinator and confirm the swap enforces teff1>=teff2
    order = sl._param_order(FILTS, False, binary=True)
    coord = np.zeros(len(order))
    idx = sl._binary_swap_indices(order, coord)
    assert idx == (0, 6, 1, 7, 4, 8)
    # craft a unit vector that, after identity-ish transforms, has teff2>teff1,
    # then verify the swap in prior_transform_dynesty fixes the order. We use a
    # uniform prior dict so transforms are monotonic in u.
    import scipy.stats as st
    pd_ = {p: st.uniform(loc=0, scale=1) for p in order}
    pd_['teff'] = st.uniform(loc=3000, scale=4000)   # 3000-7000
    pd_['teff2'] = st.uniform(loc=3000, scale=4000)
    pd_['logg'] = st.uniform(loc=4, scale=2); pd_['logg2'] = st.uniform(loc=4, scale=2)
    pd_['rad'] = st.uniform(loc=0.1, scale=2); pd_['rad2'] = st.uniform(loc=0.1, scale=2)
    u = np.full(len(order), 0.5)
    u[0] = 0.1   # teff -> low
    u[6] = 0.9   # teff2 -> high  (out of order, must be swapped)
    out = sl.prior_transform_dynesty(u, None, None, FILTS, pd_, coord, False,
                                     binary=True)
    assert out[0] >= out[6]              # teff1 >= teff2 after the swap


def test_prep_contrast_validates_filter():
    f = Fitter()                          # loads the phoenix grid
    f.binary = True
    f.contrast = [('NOT_A_FILTER', 3.0, 0.1)]
    try:
        f._prep_contrast()
    except ValueError as e:
        assert 'not a column' in str(e)
        return
    raise AssertionError('expected ValueError for unknown contrast filter')


if __name__ == '__main__':
    for _n, _f in list(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
    print('ALL PASS')
