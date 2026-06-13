"""Cross-grid flux-scale audit.

Every model grid must store the same physical quantity: surface flux
F_lambda (erg s^-1 cm^-2 AA^-1). Upstream libraries disagree on conventions
-- Coelho 2014 ships astrophysical flux (off by pi), BOSZ 2024 ships
Eddington flux H_lambda (off by 4*pi) -- and a scale error this size poisons
the radius (sqrt(pi) ~ 1.77x) while leaving Teff untouched, so it survives
casual inspection. This test interpolates every grid at a node inside its
validity range and checks the median band-flux ratio against BT-Settl: a
convention bug shows up as a ratio of ~3.14 or ~12.57, far outside what
physics differences between model families produce (a few percent in the
median, tens of percent in single bands at worst).

Runs as a plain script (no pytest):
    python tests/test_grid_flux_scale.py
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
from astroARIADNE.sed_library import get_interpolated_flux, load_grid  # noqa: E402
from astroARIADNE.config import filter_names  # noqa: E402

BANDS = np.array(filter_names)

# (grid, comparison node (teff, logg, z)) -- node must sit inside both the
# grid's and BT-Settl's support. Koester is excluded: its log g range
# (6.5-9.5) has no overlap with BT-Settl's, so there is no shared node; its
# flux scale was validated against blackbody normalisation when it was built.
GRID_NODES = {
    'phoenix':   (5500., 4.5, 0.0),
    'btnextgen': (3200., 5.0, 0.0),
    'btcond':    (3200., 5.0, 0.0),
    'btdusty':   (3200., 5.0, 0.0),
    'kurucz':    (5500., 4.5, 0.0),
    'ck04':      (5500., 4.5, 0.0),
    'bosz':      (5500., 4.5, 0.0),
    'coelho':    (5500., 4.5, 0.0),
    'stagger':   (5500., 4.5, 0.0),
    'sphinx':    (3200., 5.0, 0.0),
    'newera':    (5500., 4.5, 0.0),
    'tlusty':    (20000., 4.0, 0.0),
    'atmo':      (1200., 5.0, 0.0),
    'sonora':    (1200., 5.0, 0.0),
}

# A pi error gives 3.14 (or 0.318), a 4*pi error 12.57 (or 0.0796). Median
# physics scatter between families is a few percent at these nodes; even
# SPHINX's strong optical deviations leave the all-band median well inside
# this window.
RATIO_LO, RATIO_HI = 0.6, 1.7


def _median_ratio(grid, node):
    ref = load_grid('btsettl')
    other = load_grid(grid)
    # Older/exotic grids carry fewer than the full 66 columns; compare only
    # the bands both grids actually have.
    shared = np.array([b for b in BANDS
                       if b in ref.column_index and b in other.column_index])
    f_ref = np.asarray(get_interpolated_flux(*node, shared, ref), dtype=float)
    f_oth = np.asarray(get_interpolated_flux(*node, shared, other), dtype=float)
    ok = np.isfinite(f_ref) & np.isfinite(f_oth) & (f_ref > 0) & (f_oth > 0)
    assert ok.sum() >= 10, f'{grid}: only {ok.sum()} shared finite bands'
    return float(np.median(f_oth[ok] / f_ref[ok])), int(ok.sum())


def test_flux_scale_vs_btsettl():
    failures = []
    for grid, node in sorted(GRID_NODES.items()):
        try:
            ratio, nb = _median_ratio(grid, node)
        except FileNotFoundError:
            print(f'skip {grid}: grid file not installed')
            continue
        tag = 'ok' if RATIO_LO < ratio < RATIO_HI else 'FAIL'
        print(f'{tag:>4} {grid:>10} @ {node}: median ratio {ratio:.3f} '
              f'({nb} bands)')
        if tag == 'FAIL':
            failures.append((grid, ratio))
    assert not failures, (
        f'flux-scale mismatch vs btsettl (pi=3.14, 4pi=12.57 are the usual '
        f'suspects): {failures}')


if __name__ == '__main__':
    test_flux_scale_vs_btsettl()
    print('ALL PASS')
