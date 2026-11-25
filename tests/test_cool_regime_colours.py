"""Cool-dwarf regime detection without a usable BP-RP.

The regime priors (cool-star Teff prior down to the coolest grid node, the
main-sequence logg prior) were switched on by BP-RP > 1.8 alone, with the Gaia
Teff as the only fallback. The faintest and reddest M and L dwarfs are the
stars that lack both: BP is too faint, or the photometry QC drops it, and
GSP-Phot has no solution. They then got the FGK population priors: 196 of the
1367 cal_run8 survey M dwarfs, whose fits ran hot (up to 4700 K), 18 of them
pinned at the 2300 K edge of that prior and 21 at the 0.05 Rsun radius floor.
The colour now falls back to G-RP, then G-Ks, each converted to the BP-RP of a
dwarf of that colour (Mamajek sequence), and with no colour at all to the
absolute G magnitude.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from astroARIADNE.fitter import Fitter

F = Fitter._cool_dwarf_flags


def _star(mags, dist=20.0, temp=0.0, lum=0.0):
    s = types.SimpleNamespace()
    s.temp, s.lum = temp, lum
    s.dist, s.dist_e = dist, 0.1
    s.filter_names = np.array(list(mags))
    s.mags = np.array([mags[k] for k in mags], float)
    return s


def test_bp_rp_is_used_when_present():
    s = _star({'GaiaDR2v2_BP': 15.0, 'GaiaDR2v2_G': 13.5, 'GaiaDR2v2_RP': 12.4, '2MASS_Ks': 9.5})
    bp_rp, abs_g = Fitter._gaia_color_absmag(s)
    assert bp_rp == pytest.approx(2.6)
    assert abs_g == pytest.approx(13.5 + 5 - 5 * np.log10(20.0))


def test_g_rp_stands_in_for_a_missing_bp():
    """BP removed (0 in the magnitude array, as Star.remove_mag leaves it)."""
    red = _star({'GaiaDR2v2_BP': 0.0, 'GaiaDR2v2_G': 19.8, 'GaiaDR2v2_RP': 18.2, '2MASS_Ks': 12.7})
    bp_rp, abs_g = Fitter._gaia_color_absmag(red)
    assert bp_rp is not None and bp_rp > 1.8          # G-RP = 1.6: late M / L
    assert F(0, 0, bp_rp, abs_g) == (True, True)
    k_dwarf = _star({'GaiaDR2v2_BP': 0.0, 'GaiaDR2v2_G': 10.0, 'GaiaDR2v2_RP': 9.3})
    bp_rp, abs_g = Fitter._gaia_color_absmag(k_dwarf)
    assert bp_rp is not None and bp_rp < 1.8          # G-RP = 0.7: mid K
    assert F(0, 0, bp_rp, abs_g)[0] is False


def test_g_rp_threshold_matches_the_bp_rp_one():
    """BP-RP = 1.8 on the dwarf sequence is G-RP = 0.91."""
    lo = _star({'GaiaDR2v2_G': 12.0, 'GaiaDR2v2_RP': 11.14})
    hi = _star({'GaiaDR2v2_G': 12.0, 'GaiaDR2v2_RP': 11.04})
    assert Fitter._gaia_color_absmag(lo)[0] < 1.8 < Fitter._gaia_color_absmag(hi)[0]


def test_g_ks_stands_in_when_rp_is_missing_too():
    red = _star({'GaiaDR2v2_G': 19.5, '2MASS_Ks': 12.6})          # G-Ks = 6.9
    bp_rp, abs_g = Fitter._gaia_color_absmag(red)
    assert bp_rp is not None and bp_rp > 1.8
    assert F(0, 0, bp_rp, abs_g) == (True, True)
    warm = _star({'GaiaDR2v2_G': 10.0, '2MASS_Ks': 8.0})         # G-Ks = 2.0: K dwarf
    assert Fitter._gaia_color_absmag(warm)[0] < 1.8


def test_absolute_magnitude_decides_without_any_colour():
    faint = _star({'GaiaDR2v2_G': 19.5}, dist=60.0)                # M_G = 15.6
    bp_rp, abs_g = Fitter._gaia_color_absmag(faint)
    assert bp_rp is None and abs_g == pytest.approx(19.5 + 5 - 5 * np.log10(60.0))
    assert F(0, 0, bp_rp, abs_g) == (True, True)
    bright = _star({'GaiaDR2v2_G': 9.0}, dist=60.0)                # M_G = 5.1
    assert F(0, 0, *Fitter._gaia_color_absmag(bright)) == (False, False)
    # a Gaia Teff still wins over the magnitude when there is no colour
    assert F(5200, 0.4, None, 9.5)[0] is False
    assert F(3300, 0.01, None, 4.0)[0] is True


def test_no_gaia_photometry_is_unknown():
    assert Fitter._gaia_color_absmag(_star({'2MASS_J': 9.0})) == (None, None)
    assert F(0, 0, None, None) == (False, False)
