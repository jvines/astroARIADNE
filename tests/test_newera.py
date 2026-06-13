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

# Base (non-IR-excess) photosphere band set every grid must carry. The
# SPHEREx fit bands are excepted for NewEra: its cache-derived columns were
# too sparse (466 nodes) to fit with, so they are blanked until the raw
# reconvolution fills them -- the band-coverage guard then auto-drops newera
# from SPHEREx-fed fits instead of letting NaN pockets poison them.
BASE_BANDS = (set(filter_names) - set(irx_bands)
              - {f for f in filter_names
                 if f.startswith(('SPHEREX_', 'JPLUS_', 'IRS_'))})


def test_grid_schema():
    ne = load_grid('newera')
    lg, tf, fe = ne.index_columns
    # NewEra was re-streamed at the full 66-band set (2026-06, on yuki): its
    # 0.09-930 um spectra cover every band, IR-excess set included, making it
    # the deepest-IR modern grid in the ensemble.
    assert set(ne.column_index) == set(filter_names)
    assert BASE_BANDS <= set(ne.finite_bands)
    assert {'HERSCHEL_SPIRE_PLW', 'AKARI_FIS_N160'} <= set(ne.finite_bands)
    assert tf.min() == 2300.0 and tf.max() == 12000.0
    assert fe.min() == -4.0                             # metal-poor reach
    # NewEra's cube is NaN-padded at missing corners; count the filled nodes
    # over the 66 raw-convolved bands. The 12 SPHEREx columns are
    # cache-derived and exist only at the 466 plotting-cache nodes until the
    # raw reconvolution fills them (notebooks/augment_grids_spherex.py).
    non_spherex = [i for f, i in ne.column_index.items()
                   if not f.startswith(('SPHEREX_', 'JPLUS_', 'IRS_'))]
    assert np.isfinite(ne.grid[..., non_spherex]).all(axis=-1).sum() > 5000


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
