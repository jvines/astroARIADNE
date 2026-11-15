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


# -- APOGEE DR17 and GALAH DR3 ------------------------------------------------
def _query_shell(gaia_id=2851858288640):
    lib = _shell()
    lib._gaia_id = gaia_id
    lib._ids = {}
    return lib


class _FakeClient:
    def __init__(self, table):
        self.table = table
        self.seen = {}

    def query_constraints(self, catalog=None, **constraints):
        self.seen = {"catalog": catalog, **constraints}
        return _TableList({catalog: self.table})


def _patched(client, columns_out):
    def _make(columns):
        columns_out.extend(columns)
        return client
    return patch.object(api, "_vizier_columns", _make)


APOGEE_ROW = {"Teff": 5000, "e_Teff": 20, "logg": 3.1, "e_logg": 0.05,
              "[Fe/H]": -0.3, "e_[Fe/H]": 0.01, "f_[Fe/H]": 0, "SNR": 100}
GALAH_ROW = {"Teff": 5800, "e_Teff": 30, "logg": 4.4, "e_logg": 0.06,
             "[Fe/H]": 0.1, "e_[Fe/H]": 0.02, "Flagsp": 0, "q_[Fe/H]": 0}


def test_parse_apogee_reads_direct_feh_and_gates_on_its_flag():
    out = Librarian._parse_apogee(_vizier_table(APOGEE_ROW)[0])
    assert out == {"teff": 5000.0, "teff_err": 20.0, "logg": pytest.approx(3.1),
                   "logg_err": pytest.approx(0.05), "feh": pytest.approx(-0.3),
                   "feh_err": pytest.approx(0.01)}
    assert Librarian._parse_apogee(
        _vizier_table({**APOGEE_ROW, "f_[Fe/H]": 1})[0]) is None


def test_parse_galah_gates_on_flags_and_reads_bracket_names():
    out = Librarian._parse_galah(_vizier_table(GALAH_ROW)[0])
    assert out["teff"] == 5800.0 and out["feh"] == pytest.approx(0.1)
    assert out["feh_err"] == pytest.approx(0.02)
    for bad in ({"Flagsp": 1}, {"q_[Fe/H]": 1}):
        assert Librarian._parse_galah(
            _vizier_table({**GALAH_ROW, **bad})[0]) is None


def test_query_apogee_asks_catalog_by_gaia_id_and_takes_highest_snr():
    low = {**APOGEE_ROW, "SNR": 20, "Teff": 4000}
    t = _vizier_table(low)
    t.add_row(list(_vizier_table(APOGEE_ROW)[0]))   # SNR 100 row, Teff 5000
    client, cols = _FakeClient(t), []
    with _patched(client, cols):
        out = _query_shell()._query_apogee()
    assert client.seen == {"catalog": "III/286/catalog",
                           "GaiaEDR3": "2851858288640"}
    assert "f_[Fe/H]" in cols and "SNR" in cols
    assert out["teff"] == 5000.0


def test_query_galah_asks_stars_table_by_gaia_id():
    client, cols = _FakeClient(_vizier_table(GALAH_ROW)), []
    with _patched(client, cols):
        out = _query_shell(36734515982599808)._query_galah()
    assert client.seen == {"catalog": "J/MNRAS/506/150/stars",
                           "GaiaEDR3": "36734515982599808"}
    assert "Flagsp" in cols and "q_[Fe/H]" in cols
    assert out["feh"] == pytest.approx(0.1)


def test_query_apogee_galah_need_gaia_id():
    lib = _query_shell(None)
    assert lib._query_apogee() is None and lib._query_galah() is None


_LIVE = pytest.mark.skipif(
    os.environ.get("ARIADNE_LIVE_TESTS", "0") in ("0", "", "false", "False"),
    reason="hits the live VizieR archive; set ARIADNE_LIVE_TESTS=1 to run")


@_LIVE
def test_query_apogee_live():
    res = _query_shell(2851858288640)._query_apogee()
    assert res is not None
    assert 3000 < res["teff"] < 8000 and res["feh"] is not None


@_LIVE
def test_query_galah_live():
    res = _query_shell(36734515982599808)._query_galah()
    assert res is not None
    assert 3000 < res["teff"] < 8000 and res["feh"] is not None
