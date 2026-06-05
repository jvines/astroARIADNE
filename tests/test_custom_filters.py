"""Tests for ARIADNE's custom mid/far-IR filter library and the grid builder's
filter resolution + out-of-range NaN guard.

The custom library (Datafiles/custom_filters.h5, built by
notebooks/build_custom_filters.py) supplies the bands pyphot lacks -- Spitzer
MIPS 24/70/160, AKARI/IRC S9W & L18W, IRAS 12/25/60/100, Herschel SPIRE PLW --
in the energy (bolometer) convention SVO declares for them.

Runs as a plain script (no pytest):
    python tests/test_custom_filters.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'notebooks'))

import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
from astroARIADNE import phot_utils as pu  # noqa: E402

CUSTOM = ['SPITZER_MIPS_24', 'SPITZER_MIPS_70', 'SPITZER_MIPS_160',
          'AKARI_IRC_S9W', 'AKARI_IRC_L18W',
          'IRAS_12', 'IRAS_25', 'IRAS_60', 'IRAS_100',
          'HERSCHEL_SPIRE_PLW']


# ----------------------------- runtime side ---------------------------------
def test_custom_library_loads():
    assert set(pu._CUSTOM_LIB) == set(CUSTOM), set(pu._CUSTOM_LIB)


def test_custom_bands_are_energy_convention():
    # SVO tags all of these as energy counters (bolometer-calibrated systems).
    for b in CUSTOM:
        assert pu._CUSTOM_LIB[b].dtype == 'energy', b


def test_effective_wavelength_resolves_customs():
    # cl is Vega-independent, so it is finite even for SPIRE PLW (500 um), where
    # the Vega-based leff is NaN. Bands must come out monotonic in wavelength.
    cl = {b: pu.get_effective_wavelength(b) for b in CUSTOM}
    for b, w in cl.items():
        assert np.isfinite(w) and w > 0, (b, w)
    assert cl['SPITZER_MIPS_24'] < cl['SPITZER_MIPS_70'] < cl['SPITZER_MIPS_160']
    assert 450 < cl['HERSCHEL_SPIRE_PLW'] < 600          # ~511 um


def test_merged_lookup_keeps_pyphot_bands():
    # A pyphot-library band still resolves through the merged _get_filter path.
    w = pu.get_effective_wavelength('WISE_RSR_W4')
    assert 20 < w < 25, w


# --------------------------- grid-builder side ------------------------------
def _builder():
    import build_model_grids as B
    return B


def test_filter_sets():
    B = _builder()
    assert len(B.FILTERS) == 59                          # common set
    assert len(B._filters_for('tlusty')) == 62           # + SPIRE 250/350/500
    assert len(B._filters_for('bosz')) == 59             # no extras
    # every custom band is reachable by the builder's resolver
    fl = {n for n, *_ in B._build_filters(B._filters_for('tlusty'))}
    assert set(CUSTOM).issubset(fl)
    assert 'HERSCHEL_SPIRE_PLW' in fl


def test_nan_guard_clamps_to_reach():
    """A band whose 98%-transmission support is not spanned by the spectrum is
    NaN'd; bands fully within reach are finite and unit-consistent."""
    B = _builder()
    B._FILTS = B._build_filters(B._filters_for('tlusty'))

    def synth(top_um):
        w = np.linspace(0.1, top_um, 60000)
        return [(5000., 4.5, 0.0, w ** -4.0, w)]         # falling RJ-like tail

    # spectrum to 32 um (BOSZ-like): W4/MIPS24 finite, MIPS70/PACS/SPIRE NaN
    row = B._process_file('x', lambda p: synth(32.0))[0]
    assert np.isfinite(row['WISE_RSR_W4'])
    assert np.isfinite(row['SPITZER_MIPS_24'])
    assert np.isnan(row['SPITZER_MIPS_70'])
    assert np.isnan(row['HERSCHEL_PACS_RED'])
    assert np.isnan(row['HERSCHEL_SPIRE_PLW'])

    # spectrum to 700 um (TLUSTY far-IR): everything finite incl. SPIRE PLW
    row = B._process_file('x', lambda p: synth(700.0))[0]
    for b in CUSTOM:
        assert np.isfinite(row[b]), b
    # monotonic decreasing flux for a falling spectrum across the IR ladder
    ladder = ['WISE_RSR_W4', 'SPITZER_MIPS_24', 'SPITZER_MIPS_70',
              'SPITZER_MIPS_160', 'HERSCHEL_SPIRE_PLW']
    vals = [row[b] for b in ladder]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
    print('ALL PASS')
