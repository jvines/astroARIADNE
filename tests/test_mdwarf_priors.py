"""Tests for the M-dwarf-aware Teff/logg priors in Fitter._default_priors.

Cool stars (Gaia BP-RP > 1.8, or Teff < 4000 K) should not get the FGK population
Teff prior, and cool *dwarfs* (absolute G > 8, or L < 0.5 Lsun) should get a tight
main-sequence logg prior instead of the wide uniform / uninformative isochrone.
Detection is colour-primary because M dwarfs frequently lack Gaia Teff/lum. FGK
behaviour must be unchanged.
"""
import sys
import types
import numpy as np

REPO = '/Users/jayvains/github/personal/astroARIADNE'
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
import astroARIADNE.fitter as fm  # noqa: E402
from astroARIADNE.fitter import Fitter  # noqa: E402

F = Fitter._cool_dwarf_flags


def test_flags_colour_primary():
    # M dwarf: red + faint absolute mag
    assert F(None, None, 3.0, 12.0) == (True, True)
    # M giant: red but luminous (abs_g small) -> cool, not a dwarf
    assert F(None, None, 2.5, 0.0) == (True, False)
    # FGK dwarf: blue -> not cool
    assert F(None, None, 0.8, 4.5) == (False, False)
    # colour boundary is strict (>1.8)
    assert F(None, None, 1.8, 12.0)[0] is False
    assert F(None, None, 1.81, 12.0)[0] is True


def test_flags_temp_lum_fallback():
    # no colour -> fall back to Teff/lum estimates
    assert F(3200, 0.01) == (True, True)
    assert F(3500, 50.0) == (True, False)        # cool giant
    assert F(5700, 1.0) == (False, False)        # FGK
    assert F(None, None, None, None) == (False, False)
    assert F(0, 0, None, None) == (False, False)
    # the -1 / -999 "no Gaia data" sentinels must NOT read as cool
    assert F(-1, -1, None, None) == (False, False)
    assert F(-999, -999, None, None) == (False, False)
    # colour wins over a (possibly stale/absent) Teff when both present
    assert F(8000, 0.001, 3.0, 12.0) == (True, True)


def _mock_star(bp_rp, abs_g, dist=15.0, temp=None, lum=None):
    s = types.SimpleNamespace()
    s.temp, s.lum = temp, lum
    s.spectroscopic_params = None
    s.rave_params = None
    s.get_logg = False
    s.logg, s.logg_e = 5.0, 0.5
    s.dist, s.dist_e = dist, 0.1
    s.Av, s.Av_e = 0.1, None
    g = abs_g - 5.0 + 5.0 * np.log10(dist)
    rp, bp = g, g + bp_rp
    s.filter_names = np.array(
        ['GaiaDR2v2_BP', 'GaiaDR2v2_G', 'GaiaDR2v2_RP', '2MASS_J'])
    s.mags = np.array([bp, g, rp, 9.0])
    s.filter_mask = np.array([3])              # only 2MASS_J 'used' (noise loop)
    s.flux = np.array([0., 0., 0., 0.5])
    s.flux_er = np.array([0., 0., 0., 0.05])
    return s


def test_gaia_color_absmag_extraction():
    s = _mock_star(3.0, 12.0, dist=15.0)
    bp_rp, abs_g = Fitter._gaia_color_absmag(s)
    assert abs(bp_rp - 3.0) < 1e-9
    assert abs(abs_g - 12.0) < 1e-9
    # missing Gaia bands -> (None, None)
    s2 = types.SimpleNamespace(filter_names=np.array(['2MASS_J']),
                               mags=np.array([9.0]), dist=15.0)
    assert Fitter._gaia_color_absmag(s2) == (None, None)


def _priors_for(star):
    f = Fitter()
    f.star = star
    f._grid = 'phoenix'
    f._norm = False
    f.coordinator = np.zeros(8)
    f.fixed = np.zeros(8)
    fm.order = np.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av'])
    f._default_priors()
    return f.prior_sources


def test_mdwarf_priors_applied():
    ps = _priors_for(_mock_star(3.0, 12.0))          # M dwarf
    assert ps['teff'] == 'mdwarf_uniform', ps['teff']
    assert ps['logg'] == 'mdwarf_main_sequence', ps['logg']


def test_cool_giant_gets_cool_teff_but_not_ms_logg():
    ps = _priors_for(_mock_star(2.5, 0.0))           # red giant
    assert ps['teff'] == 'mdwarf_uniform'
    assert ps['logg'] != 'mdwarf_main_sequence'


def test_fgk_unchanged():
    ps = _priors_for(_mock_star(0.8, 4.5))           # FGK dwarf
    assert ps['teff'] == 'rave_population'
    assert ps['logg'] == 'uniform_default'


if __name__ == '__main__':
    for _n, _f in list(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
    print('ALL PASS')
