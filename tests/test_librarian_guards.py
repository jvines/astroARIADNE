"""Offline tests for the librarian crossmatch guards.

Runs as a plain script (no pytest):
    python tests/test_librarian_guards.py
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import astroARIADNE  # noqa: E402
assert 'site-packages' not in astroARIADNE.__file__, astroARIADNE.__file__

from astroARIADNE.librarian._api import Librarian  # noqa: E402


def _shell(pmra=None, pmdec=None):
    lib = Librarian.__new__(Librarian)
    lib._pmra = pmra
    lib._pmdec = pmdec
    lib._magnitudes = {}
    return lib


def test_pm_available():
    assert _shell(2583.19, 588.50)._pm_available() is True
    assert _shell(None, None)._pm_available() is False
    assert _shell(np.nan, 5.0)._pm_available() is False
    assert _shell('bogus', 5.0)._pm_available() is False


def test_match_sep_units():
    """_r separations honour the table unit (VizieR uses arcmin or arcsec
    depending on the query)."""
    from astropy.table import Table
    import astropy.units as u
    t = Table({'_r': [0.5]})
    t['_r'].unit = u.arcmin
    assert abs(Librarian._match_sep_arcsec(t, t[0]) - 30.0) < 1e-9
    t2 = Table({'_r': [7.5]})
    t2['_r'].unit = u.arcsec
    assert abs(Librarian._match_sep_arcsec(t2, t2[0]) - 7.5) < 1e-9
    t3 = Table({'_r': [3.0]})          # no unit -> assume arcsec
    assert abs(Librarian._match_sep_arcsec(t3, t3[0]) - 3.0) < 1e-9
    t4 = Table({'x': [1.0]})           # no _r column -> None
    assert Librarian._match_sep_arcsec(t4, t4[0]) is None


def test_far_ir_skipped_without_pm():
    """An epoch-distant far-IR catalog must NOT be queried when the proper
    motion is unknown (the J2016 position can be arcmin off; nearest-row
    matching would attach a field source -> fake IR excess)."""
    import astroARIADNE.librarian._api as api

    lib = _shell(None, None)
    called = []
    orig = api._with_timeout
    api._with_timeout = lambda *a, **k: called.append(a) or None
    try:
        cat_def, radius, max_sep = api._FAR_IR_CATALOGS['AKARI_IRC']
        lib._fetch_flux_catalog('AKARI_IRC', cat_def, radius, max_sep)
    finally:
        api._with_timeout = orig
    assert called == [], 'far-IR catalog was queried without proper motion'


def test_far_ir_tuples_have_max_sep():
    """Every far-IR catalog carries (def, radius, max_sep) with
    max_sep < radius (the cut must bite inside the cone)."""
    from astroARIADNE.librarian._api import _FAR_IR_CATALOGS
    for name, (cat_def, radius, max_sep) in _FAR_IR_CATALOGS.items():
        assert max_sep < radius, name


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
    print('ALL PASS')
