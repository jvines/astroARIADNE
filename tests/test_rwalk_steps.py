"""The number of rwalk steps per proposal scales with the dimension of the fit.

A fit has six stellar parameters plus one excess-noise term per band, 20 to 40
dimensions. With dynesty's flat walks = 25 the random walks are too short to
decorrelate: the live points lose their spread and the run ends early. On Gaia
DR3 6244258239317113088 (36 dimensions, six repeats) walks = 25 gave
ln Z = 605-646 for BT-Settl and a distance posterior of 93.9 +- 1.9 pc against
a prior of 113.0 +- 4.9 pc (the likelihood depends on R/d only, so the
posterior of d is its prior); walks = 150 gave ln Z = 651-658 and
110.2-113.1 pc with the prior's width. The default is max(25, 5 ndim), the
rule MOIRAI's daedalus backend uses.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from astroARIADNE.fitter import Fitter, default_walks


@pytest.mark.parametrize('ndim,expected', [(1, 25), (5, 25), (6, 30), (20, 100), (36, 180)])
def test_default_walks_scales_with_dimension(ndim, expected):
    assert default_walks(ndim) == expected


def test_fitter_default_is_adaptive():
    f = Fitter()
    assert f.walks is None                       # unset: decided from ndim
    f.ndim = 36
    assert f._rwalk_steps() == 180
    f.ndim = 8
    assert f._rwalk_steps() == 40


def test_explicit_walks_is_kept():
    f = Fitter()
    f.walks = 30
    f.ndim = 36
    assert f._rwalk_steps() == 30
    g = Fitter()
    g.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 1, False, 42]
    g.ndim = 36
    assert g.walks == 42 and g._rwalk_steps() == 42
    h = Fitter()
    h.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 1, False]
    assert h.walks is None
