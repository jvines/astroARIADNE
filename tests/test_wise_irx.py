"""Tests that the librarian fetches AllWISE W3/W4 and routes them to IR-excess.

W3 (12 um) / W4 (22 um) are all-sky IR-excess bands. They are Vega mags like
W1/W2 (reuse the existing mag->flux path), QC'd via the same AllWISE ph_qual
string (qph; W3=idx 2, W4=idx 3), pass the name-based filter bridge (they are in
config.filter_names), and -- being in config.irx_bands -- are routed to the
star's irx set rather than the photosphere fit. Static (no network).

    python tests/test_wise_irx.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__
from astroARIADNE.librarian._api import _CATALOGS, _WISE_BAND_IDX  # noqa: E402
from astroARIADNE.librarian import _qc  # noqa: E402
from astroARIADNE.librarian._qc import _qc_wise_band, _qc_wise_blend  # noqa: E402
from astroARIADNE.librarian._filtermap import to_ariadne_filters  # noqa: E402
from astroARIADNE.config import irx_bands, filter_names  # noqa: E402
from astroARIADNE.phot_utils import get_zero_flux  # noqa: E402

import numpy as np  # noqa: E402


def test_wise_catalog_fetches_four_bands():
    bands = [b.pyphot for b in _CATALOGS['Wise'].bands]
    assert bands == ['WISE_RSR_W1', 'WISE_RSR_W2',
                     'WISE_RSR_W3', 'WISE_RSR_W4']
    # the AllWISE magnitude/error columns mirror the W1/W2 convention
    cols = {b.pyphot: (b.col, b.err_col) for b in _CATALOGS['Wise'].bands}
    assert cols['WISE_RSR_W3'] == ('W3mag', 'e_W3mag')
    assert cols['WISE_RSR_W4'] == ('W4mag', 'e_W4mag')


def test_band_index_map_covers_w3_w4():
    assert _WISE_BAND_IDX == {'WISE_RSR_W1': 0, 'WISE_RSR_W2': 1,
                              'WISE_RSR_W3': 2, 'WISE_RSR_W4': 3}


def test_qph_qc_passes_detections_rejects_upper_limits():
    # qph one char per band; A/B/C are detections, U is an upper limit.
    row = {'qph': 'ABBU', 'ex': 0}            # W1=A W2=B W3=B W4=U
    assert _qc_wise_band(row, _WISE_BAND_IDX['WISE_RSR_W3']) is True
    assert _qc_wise_band(row, _WISE_BAND_IDX['WISE_RSR_W4']) is False  # U
    # missing/short qph -> reject
    assert _qc_wise_band({'qph': 'AB'}, 3) is False


def test_ccf_contamination_and_saturation_rejected():
    W3 = _WISE_BAND_IDX['WISE_RSR_W3']
    # clean cc_flags passes
    assert _qc_wise_band({'qph': 'AAAA', 'ccf': '0000'}, W3) is True
    # UPPERCASE artefact (halo on W3) is always rejected
    assert _qc_wise_band({'qph': 'AAAA', 'ccf': '00H0'}, W3) is False
    # lowercase possible-contamination rejected iff strict (default on)
    assert _qc_wise_band({'qph': 'AAAA', 'ccf': '00h0'}, W3) is (
        not _qc._WISE_CCF_STRICT)
    # heavy saturation on W3 rejected, none kept
    assert _qc_wise_band({'qph': 'AAAA', 'ccf': '0000', 'sat3': 0.9}, W3) is False
    assert _qc_wise_band({'qph': 'AAAA', 'ccf': '0000', 'sat3': 0.0}, W3) is True
    # backward compatible: a row without ccf/sat is treated as clean
    assert _qc_wise_band({'qph': 'AAAA'}, W3) is True


def test_wise_blend_guard():
    assert _qc_wise_blend({'nb': 1}) is True   # single component
    assert _qc_wise_blend({'nb': 2}) is True   # one close neighbour -> kept
    assert _qc_wise_blend({'nb': 3}) is False  # crowded extraction
    assert _qc_wise_blend({'nb': 4}) is False
    assert _qc_wise_blend({}) is True          # missing -> keep


def test_w3_w4_pass_filter_bridge_and_route_to_irx():
    mags = {'WISE_RSR_W1': (7.0, 0.03), 'WISE_RSR_W3': (6.8, 0.05),
            'WISE_RSR_W4': (6.5, 0.06), 'NOT_A_BAND': (1.0, 1.0)}
    kept = to_ariadne_filters(mags)
    assert set(kept) == {'WISE_RSR_W1', 'WISE_RSR_W3', 'WISE_RSR_W4'}
    # W3/W4 are excess bands (not fit); W1 is a fit band
    assert 'WISE_RSR_W3' in irx_bands and 'WISE_RSR_W4' in irx_bands
    assert 'WISE_RSR_W1' not in irx_bands
    assert {'WISE_RSR_W3', 'WISE_RSR_W4'} <= set(filter_names)


def test_w3_w4_mag_to_flux_is_finite():
    # Vega zero flux exists for W3/W4 (pyphot), so the existing mag path works.
    for b in ['WISE_RSR_W3', 'WISE_RSR_W4']:
        assert np.isfinite(get_zero_flux(b)) and get_zero_flux(b) > 0


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
    print('ALL PASS')
