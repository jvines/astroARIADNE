"""Tests for the LAMOST and PASTEL spectroscopic-prior queries.

The offline tests mock VizieR. The live test hits the archive and runs only
with ARIADNE_LIVE_TESTS=1.
"""
import os
from unittest.mock import patch

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

import astroARIADNE.librarian._api as api
from astroARIADNE.librarian._api import Librarian


def _shell(ra=331.712489, dec=30.364975):
    lib = Librarian.__new__(Librarian)
    lib.ra, lib.dec = ra, dec
    lib._coord = SkyCoord(ra * u.deg, dec * u.deg)
    lib._search_radius = 3.0 * u.arcsec
    return lib


def _vizier_table(fields):
    """One-row table as astroquery parses a VizieR VOTable (FIELD names)."""
    from astroquery.vizier.core import VizierClass
    heads = "".join(f'<FIELD name="{n}" datatype="float"/>' for n in fields)
    cells = "".join(f"<TD>{v}</TD>" for v in fields.values())
    vot = ('<?xml version="1.0"?><VOTABLE version="1.4" '
           'xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE name="x">'
           f'<TABLE name="x">{heads}<DATA><TABLEDATA><TR>{cells}</TR>'
           '</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>')
    resp = type("R", (), {"content": vot.encode(), "text": vot})()
    return VizierClass()._parse_result(resp)[0]


class _TableList(dict):
    """The parts of astroquery's TableList used here: int and key indexing,
    in VizieR's table order."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return dict.__getitem__(self, key)


def test_query_lamost_reads_the_parameter_table():
    # A cone on all of V/162 answers with the general LRS table (dr11l, no
    # Teff/logg/[Fe/H]) first and the parameter table (dr11sl) after it;
    # parsing the first table dropped every LAMOST prior.
    general = Table({"_r": [0.1], "snrg": [46.3], "ObsID": [1]})
    params = _vizier_table({"snrg": 46.34, "Teff": 5534.95, "logg": 4.695,
                            "[Fe/H]": 0.088})
    params["_r"] = [0.1]
    asked = {}

    class _Fake:
        def query_region(self, coord, radius=None, catalog=None):
            asked["catalog"] = catalog
            if catalog == "V/162":
                return _TableList({"V/162/dr11l": general,
                                   "V/162/dr11sl": params})
            return _TableList({catalog: params})

    with patch.object(api, "Vizier", _Fake()):
        res = _shell()._query_lamost()
    assert asked["catalog"] == "V/162/dr11sl"
    assert res is not None
    assert res["teff"] == pytest.approx(5534.95)
    assert res["logg"] == pytest.approx(4.695)
    assert res["feh"] == pytest.approx(0.088)
    # dr11sl carries no error columns: the parser defaults apply.
    assert res["feh_err"] == pytest.approx(0.1)


def test_query_lamost_empty_answer_is_none():
    class _Fake:
        def query_region(self, coord, radius=None, catalog=None):
            return _TableList()

    with patch.object(api, "Vizier", _Fake()):
        assert _shell()._query_lamost() is None


def test_pastel_reads_vizier_metallicity_names():
    # astroquery names columns after the VOTable FIELD name ("[Fe/H]"), not
    # astropy's ID spelling ("__Fe_H_"); reading the latter dropped every
    # PASTEL and LAMOST [Fe/H].
    t = _vizier_table({"_r": 0.1, "Teff": 5987, "e_Teff": 50, "logg": 4.24,
                       "e_logg": 0.1, "[Fe/H]": -0.04, "e_[Fe/H]": 0.05})

    class _Fake:
        def query_region(self, coord, radius=None, catalog=None):
            return _TableList({catalog: t})

    with patch.object(api, "Vizier", _Fake()):
        out = _shell()._query_pastel()
    assert out is not None
    assert out["feh"] == pytest.approx(-0.04)
    assert out["feh_err"] == pytest.approx(0.05)


@pytest.mark.skipif(
    os.environ.get("ARIADNE_LIVE_TESTS", "0") in ("0", "", "false", "False"),
    reason="hits the live VizieR archive; set ARIADNE_LIVE_TESTS=1 to run")
def test_query_lamost_live():
    """Gaia DR3 1898291470849457024: Teff 5534.95, logg 4.695, [Fe/H] 0.088
    in LAMOST DR11. A cone on all of V/162 parsed the general table and
    never returned a prior."""
    res = _shell()._query_lamost()
    assert res is not None
    assert res["teff"] == pytest.approx(5534.95, abs=0.1)
    assert res["feh"] == pytest.approx(0.088, abs=1e-3)
