"""Tests for the PHOENIX-NewEra grid (Husser et al. 2025).

Covers the built grid (schema, full-coverage band set, metal-poor reach) and
interpolation through the ARIADNE path. Runs as a plain script (no pytest):
    python tests/test_newera.py
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
from astroARIADNE.fitter import Fitter  # noqa: E402
from astroARIADNE.sed_library import get_interpolated_flux, load_grid  # noqa: E402
from astroARIADNE.config import filter_names, irx_bands  # noqa: E402

# Base (non-IR-excess) photosphere band set every grid must carry.
BASE_BANDS = set(filter_names) - set(irx_bands)


def test_grid_schema():
    ne = load_grid('newera')
    lg, tf, fe = ne.index_columns
    # NewEra's built grid is still base-only (its 0.09-930 um spectra reach the
    # full IR, but adding those bands needs a re-stream; until then, base set).
    assert set(ne.column_index) == BASE_BANDS
    assert tf.min() == 2300.0 and tf.max() == 12000.0
    assert fe.min() == -4.0                             # metal-poor reach
    # NewEra's cube is NaN-padded at missing corners; count the filled nodes
    assert np.isfinite(ne.grid).all(axis=-1).sum() > 5000


def test_interpolation_finite():
    interp = load_grid('newera')
    bands = np.array(['SDSS_g', '2MASS_J', 'GALEX_NUV', 'WISE_RSR_W1'])
    for node in [(5500.0, 4.5, 0.0), (3500.0, 5.0, -1.0), (8000.0, 4.0, -2.0)]:
        f = get_interpolated_flux(node[0], node[1], node[2], bands, interp)
        assert np.all(np.isfinite(np.asarray(f))), node


def test_grid_setter_loads_newera():
    f = Fitter()
    f.grid = 'newera'
    assert f._interpolator is not None


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
    print('ALL PASS')
