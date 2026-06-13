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

# Custom bands that are also convolved into the model grids (mid/far-IR).
CUSTOM = ['SPITZER_MIPS_24', 'SPITZER_MIPS_70', 'SPITZER_MIPS_160',
          'AKARI_IRC_S9W', 'AKARI_IRC_L18W',
          'IRAS_12', 'IRAS_25', 'IRAS_60', 'IRAS_100',
          'HERSCHEL_SPIRE_PLW']
# AKARI FIS (65-160 um): runtime-only filters for ingesting the FIS Jy
# catalogue. They are NOT convolved into the grids (plot-only IR-excess bands),
# so they appear in the custom library but not in the grid-builder filter set.
CUSTOM_FIS = ['AKARI_FIS_N60', 'AKARI_FIS_WIDE_S',
              'AKARI_FIS_WIDE_L', 'AKARI_FIS_N160']
# SPHEREx super-bands: ACTIVE since 2026-06 (filter_names 66 -> 78; grid
# columns cache-derived via notebooks/augment_grids_spherex.py pending the
# raw reconvolution).
CUSTOM_SPHEREX = [f'SPHEREX_{c}' for c in
                  ('1050', '1130', '1400', '1850', '2350', '2750', '3050',
                   '3350', '3700', '4050', '4300', '4650', '4900')]
# J-PLUS narrow diagnostics: ACTIVE since 2026-06 (filter_names 78 -> 81;
# real OAJ/JPLUS curves, cache-derived grid columns pending raw rebuild).
# CASSIS/IRS super-bands: ACTIVE since 2026-06 (filter_names 82 -> 93; energy
# top-hats over the IRS low-res order layout, all >= 5 um -> irx_bands).
CUSTOM_IRS = [f'IRS_{c}' for c in
              ('5500', '6500', '8000', '10000', '12000', '13000',
               '15000', '18000', '23000', '30000', '35000')]
CUSTOM_JPLUS = ['JPLUS_J0515', 'JPLUS_J0660', 'JPLUS_J0861']
# Nothing currently staged-inert.
CUSTOM_STAGED = []


# ----------------------------- runtime side ---------------------------------
def test_custom_library_loads():
    assert set(pu._CUSTOM_LIB) == \
        set(CUSTOM) | set(CUSTOM_FIS) | set(CUSTOM_SPHEREX) \
        | set(CUSTOM_JPLUS) | set(CUSTOM_IRS) | set(CUSTOM_STAGED), \
        set(pu._CUSTOM_LIB)


def test_spherex_bands_active():
    # The 12 SPHEREx super-bands joined filter_names (the grids carry their
    # columns). ALL are fit bands: the fit/excess boundary is the ~5 um
    # wavelength rule, instrument-independent (2026-06 ruling) -- the
    # optional re-routing to the excess set is fit_irx(route_mid_ir=True).
    from astroARIADNE.config import filter_names, irx_bands, iso_mask
    assert set(CUSTOM_SPHEREX) <= set(filter_names)
    assert len(filter_names) == len(iso_mask), 'iso_mask misaligned'
    assert not (set(CUSTOM_SPHEREX) & set(irx_bands))


def test_staged_filters_are_inert():
    # The staged Phase-2 filters must NOT be in the active band set yet --
    # the grids lack their columns until a reconvolution.
    from astroARIADNE.config import filter_names
    assert not (set(CUSTOM_STAGED) & set(filter_names)), \
        'staged filters leaked into filter_names before grid reconvolution'


def test_custom_bands_are_energy_convention():
    # SVO tags all of these as energy counters (bolometer-calibrated systems).
    for b in CUSTOM + CUSTOM_FIS:
        assert pu._CUSTOM_LIB[b].dtype == 'energy', b


def test_effective_wavelength_resolves_customs():
    # cl is Vega-independent, so it is finite even for SPIRE PLW (500 um), where
    # the Vega-based leff is NaN. Bands must come out monotonic in wavelength.
    cl = {b: pu.get_effective_wavelength(b) for b in CUSTOM + CUSTOM_FIS}
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
    # The common set now matches config.filter_names: AKARI FIS + SPIRE were
    # promoted into _IR_EXTENSION (SPIRE used to be a TLUSTY-only extra).
    assert len(B.FILTERS) == 93                          # 66+13 SPHEREx+3 J-PLUS+11 IRS
    assert len(B._filters_for('tlusty')) == 93           # no extras anymore
    assert len(B._filters_for('bosz')) == 93
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


def test_skymapper_is_ab():
    # SkyMapper is AB-native; until 2026-06 it was wrongly converted with
    # Vega zero points (fluxes 20-60% faint, redder = worse).
    import numpy as np
    leff = pu.get_effective_wavelength('SkyMapper_z')
    f_ab = pu.convert_f_nu_to_f_lambda(3.631e-20 * 10 ** (-0.4 * 12.0), leff)
    f, _ = pu.mag_to_flux(12.0, 0.01, 'SkyMapper_z')
    assert abs(f / f_ab - 1) < 1e-3, (f, f_ab)
    # round trip
    m, _ = pu.flux_to_mag(f, f * 0.01, 'SkyMapper_z')
    assert abs(m - 12.0) < 1e-3   # zero-point defs differ by ~7e-5 mag


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
    print('ALL PASS')
