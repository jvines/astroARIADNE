"""Tests for the Gaia XP synthetic gap-filling (`Librarian._fetch_xp_synth`).

Offline: gaiaxpy is stubbed with a canned response, so these exercise the
selection logic (gap rule, Gaia-band coverage exclusion, quality cuts, the
Stromgren zero-point calibration) rather than the archive.

Runs as a plain script (no pytest):
    python tests/test_xp_fill.py
"""
import os
import sys
import types
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__


def _make_lib(mags):
    """A Librarian shell with just the state _fetch_xp_synth touches."""
    from astroARIADNE.librarian._api import Librarian
    lib = Librarian.__new__(Librarian)
    lib._gaia_id = 123
    lib._ignore = ()
    lib._magnitudes = dict(mags)
    return lib


class _FakeRow(dict):
    def get(self, k, default=None):
        return super().get(k, default)


def _stub_gaiaxpy(monkey_row):
    """Install a gaiaxpy stub returning one canned row."""
    class _PS:
        Stromgren_Std = 'Stromgren_Std'
        JKC_Std = 'JKC_Std'
        SDSS_Std = 'SDSS_Std'
        PanSTARRS1_Std = 'PanSTARRS1_Std'
        Sky_Mapper = 'Sky_Mapper'

    class _DF:
        def __init__(self, row):
            self._row = row
            self.iloc = self

        def __getitem__(self, i):
            return self._row

        def __len__(self):
            return 1

    mod = types.ModuleType('gaiaxpy')
    mod.PhotometricSystem = _PS
    mod.generate = lambda *a, **k: _DF(monkey_row)
    sys.modules['gaiaxpy'] = mod


# A bright G dwarf with Gaia + 2MASS only: every optical band is a gap.
BASE_MAGS = {
    'Gaia_G': (8.0, 0.003), 'Gaia_BP': (8.3, 0.004), 'Gaia_RP': (7.5, 0.004),
    '2MASS_J': (6.9, 0.02), '2MASS_H': (6.6, 0.02), '2MASS_Ks': (6.5, 0.02),
}

ROW = _FakeRow({
    'StromgrenStd_mag_v': 8.551, 'StromgrenStd_mag_error_v': 0.005,
    'StromgrenStd_mag_b': 8.016, 'StromgrenStd_mag_error_b': 0.005,
    'StromgrenStd_mag_y': 7.655, 'StromgrenStd_mag_error_y': 0.005,
    'JkcStd_mag_B': 8.18, 'JkcStd_mag_error_B': 0.005,
    'JkcStd_mag_V': 7.66, 'JkcStd_mag_error_V': 0.005,
    'JkcStd_mag_R': 7.34, 'JkcStd_mag_error_R': 0.005,
    'JkcStd_mag_I': 7.02, 'JkcStd_mag_error_I': 0.005,
    'SdssStd_mag_g': 7.84, 'SdssStd_mag_error_g': 0.005,
    'SdssStd_mag_r': 7.51, 'SdssStd_mag_error_r': 0.005,
    'SdssStd_mag_i': 7.41, 'SdssStd_mag_error_i': 0.005,
    'SdssStd_mag_z': 7.41, 'SdssStd_mag_error_z': 0.005,
    'Panstarrs1Std_mag_gp': 7.81, 'Panstarrs1Std_mag_error_gp': 0.005,
    'Panstarrs1Std_mag_rp': 7.52, 'Panstarrs1Std_mag_error_rp': 0.005,
    'Panstarrs1Std_mag_ip': 7.43, 'Panstarrs1Std_mag_error_ip': 0.005,
    'Panstarrs1Std_mag_zp': 7.43, 'Panstarrs1Std_mag_error_zp': 0.005,
    'Panstarrs1Std_mag_yp': 7.45, 'Panstarrs1Std_mag_error_yp': 0.005,
})


def test_fills_gaps_and_blocks_duplicates():
    _stub_gaiaxpy(ROW)
    lib = _make_lib(BASE_MAGS)
    lib._fetch_xp_synth()
    filled = lib._xp_filled
    assert filled, 'nothing filled on a wide-open optical gap'
    # Gaia ultra-broad bands must not have blocked the optical fills.
    assert 'STROMGREN_y' in filled or 'GROUND_JOHNSON_V' in filled
    # Mutual blocking: of the near-identical r-type bands (SDSS_r/PS1_r),
    # only one survives the 5% log-lambda rule.
    assert not ({'SDSS_r', 'PS1_r'} <= set(filled))
    assert not ({'SDSS_i', 'PS1_i'} <= set(filled))
    # Priority order: Stromgren (first in _XP_SYSTEMS) wins its wavelengths.
    assert 'STROMGREN_v' in filled and 'STROMGREN_b' in filled


def test_observed_band_blocks_fill():
    _stub_gaiaxpy(ROW)
    mags = dict(BASE_MAGS)
    mags['STROMGREN_b'] = (8.0, 0.01)        # observed -> nearby fills blocked
    lib = _make_lib(mags)
    lib._fetch_xp_synth()
    assert 'STROMGREN_b' not in lib._xp_filled
    # SDSS_g (0.477 um) sits within 5% log-lambda of Stromgren b (0.467) and
    # must be blocked by the OBSERVED band.
    assert 'SDSS_g' not in lib._xp_filled


def test_stromgren_calibration_applied():
    _stub_gaiaxpy(ROW)
    lib = _make_lib(BASE_MAGS)
    lib._fetch_xp_synth()
    mag_v, err_v = lib._magnitudes['STROMGREN_v']
    assert abs(mag_v - (8.551 + 0.066)) < 1e-9   # zero-point correction
    assert err_v >= 0.036                        # measured-scatter floor
    mag_y, err_y = lib._magnitudes['STROMGREN_y']
    assert abs(mag_y - 7.655) < 1e-9             # y uncorrected
    assert 0.013 <= err_y < 0.02


def test_faint_star_skipped():
    _stub_gaiaxpy(ROW)
    mags = dict(BASE_MAGS)
    mags['Gaia_G'] = (16.5, 0.005)               # past the hockey-stick cut
    lib = _make_lib(mags)
    lib._fetch_xp_synth()
    assert getattr(lib, '_xp_filled', []) == []


def test_noisy_band_rejected():
    row = _FakeRow(dict(ROW))
    row['StromgrenStd_mag_error_v'] = 0.2        # > _XP_MAX_ERR
    _stub_gaiaxpy(row)
    lib = _make_lib(BASE_MAGS)
    lib._fetch_xp_synth()
    assert 'STROMGREN_v' not in lib._xp_filled


def test_qc_drops_split_crossmatch():
    """An observed catalog band > 1 mag from its own XP synthetic is a
    wrong/split crossmatch (high-PM disease): dropped, then refilled by XP."""
    _stub_gaiaxpy(ROW)
    mags = dict(BASE_MAGS)
    mags['PS1_r'] = (12.1, 0.01)     # garbage: synthetic says 7.52
    lib = _make_lib(mags)
    lib._fetch_xp_synth()
    assert 'PS1_r' not in lib._magnitudes or \
        abs(lib._magnitudes['PS1_r'][0] - 7.52) < 0.2, \
        'garbage catalog band survived the XP QC'
    # something r-like got filled in its place (SDSS_r has fill priority)
    assert any(b in lib._xp_filled for b in ('SDSS_r', 'PS1_r'))


def test_fitter_spectra_optout():
    """Fitter.spectra = False drops XP-filled and SPHEREX_* bands only."""
    from astroARIADNE.fitter import Fitter
    f = Fitter()
    assert f.spectra is True                      # default: spectra in
    star = types.SimpleNamespace(
        filter_names=np.array(['GROUND_JOHNSON_V', 'STROMGREN_v',
                               'SPHEREX_3350', '2MASS_J']),
        used_filters=np.array([1., 1., 1., 1.]),
        filter_mask=np.array([0, 1, 2, 3]),
        xp_filled=['STROMGREN_v'])
    f.star = star
    f.spectra = False
    f._drop_spectra_bands()
    kept = f.star.filter_names[f.star.used_filters == 1]
    assert list(kept) == ['GROUND_JOHNSON_V', '2MASS_J']
    # the user's star object is untouched (shallow-copy contract)
    assert star.used_filters.sum() == 4


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
    print('ALL PASS')
