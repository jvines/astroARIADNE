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


# -- Hypatia as the first link of the chain -----------------------------------
def _chain_shell(feh_source="hypatia"):
    lib = _query_shell(1779546757669063552)
    lib._feh_source = feh_source
    lib._hypatia_statistic, lib._hypatia_uncertainty = "median", "std"
    lib._hypatia_solarnorm, lib._feh_floor = "asplund09", 0.10
    lib.calls = []
    for name in ("pastel", "apogee", "galah", "rave", "lamost"):
        setattr(lib, f"_query_{name}", _stub(lib, name))
    return lib


def _stub(lib, name):
    def _q():
        lib.calls.append(name)
        return {"teff": 5000.0, "teff_err": 100.0, "logg": 4.0,
                "logg_err": 0.2, "feh": -0.5, "feh_err": 0.1} \
            if name == "pastel" else None
    return _q


HYP_FEH = {"feh": 0.02, "feh_err": 0.05, "n_meas": 7, "source": "Hypatia",
           "solarnorm": "asplund09", "statistic": "median",
           "uncertainty": "std", "matched_name": "Gaia DR3 1779546757669063552"}
HYP_STAR = {"teff": 6075.0, "teff_err": 150.0, "logg": 4.37,
            "logg_err": 0.3, "matched_name": "Gaia DR3 1779546757669063552"}


def _hypatia(lib, feh, star):
    lib.hyp_calls = {"feh": 0, "star": 0}

    def _f():
        lib.hyp_calls["feh"] += 1
        return feh

    def _s():
        lib.hyp_calls["star"] += 1
        return star
    lib._query_hypatia, lib._query_hypatia_star = _f, _s


def test_hypatia_first_link_wins_and_skips_later_links():
    lib = _chain_shell()
    _hypatia(lib, HYP_FEH, HYP_STAR)
    lib._query_spectroscopic_priors()
    p = lib._spectroscopic_params
    assert lib.calls == []
    assert p["source"] == "Hypatia"
    assert (p["teff"], p["teff_err"], p["logg"], p["logg_err"]) == \
        (6075.0, 150.0, 4.37, 0.3)
    assert p["feh"] == 0.02 and p["feh_source"] == "Hypatia"
    assert p["feh_n_meas"] == 7 and p["feh_solarnorm"] == "asplund09"
    assert lib.hyp_calls == {"feh": 1, "star": 1}


def test_hypatia_without_teff_logg_continues_chain_and_overrides_feh():
    lib = _chain_shell()
    _hypatia(lib, HYP_FEH, None)
    lib._query_spectroscopic_priors()
    p = lib._spectroscopic_params
    assert lib.calls == ["pastel"]
    assert p["source"] == "PASTEL" and p["teff"] == 5000.0
    assert p["feh"] == 0.02 and p["feh_source"] == "Hypatia"
    assert lib.hyp_calls["feh"] == 1


def test_hypatia_not_found_leaves_chain_untouched():
    lib = _chain_shell()
    _hypatia(lib, None, None)
    lib._query_spectroscopic_priors()
    p = lib._spectroscopic_params
    assert p["source"] == "PASTEL" and p["feh"] == -0.5
    assert "feh_source" not in p
    assert lib.hyp_calls == {"feh": 1, "star": 0}


def test_survey_feh_source_never_queries_hypatia():
    lib = _chain_shell("survey")
    _hypatia(lib, HYP_FEH, HYP_STAR)
    lib._query_spectroscopic_priors()
    assert lib.hyp_calls == {"feh": 0, "star": 0}
    assert lib._spectroscopic_params["source"] == "PASTEL"
    assert lib._spectroscopic_params["feh"] == -0.5


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


def test_one_composition_request_per_star(monkeypatch):
    import requests
    seen = []

    def _get(url, params=None, timeout=None):
        seen.append(url)
        if url.endswith("/composition"):
            return _Resp([{"name": "x", "median_value": 0.02,
                           "all_values": [{"value": 0.0}, {"value": 0.04}]}])
        return _Resp([{"status": "found", "teff": 6075.0, "logg": 4.37}])

    monkeypatch.setattr(requests, "get", _get)
    lib = _chain_shell()
    lib._query_spectroscopic_priors()
    assert sum(u.endswith("/composition") for u in seen) == 1
    assert lib._spectroscopic_params["source"] == "Hypatia"
    assert lib.calls == []


def test_query_hypatia_star_url_names_and_not_found(monkeypatch):
    import requests
    from astroARIADNE.librarian._hypatia import query_hypatia_star
    seen = []

    def _get(url, params=None, timeout=None):
        seen.append((url, params["name"], timeout))
        if params["name"].startswith("Gaia"):
            return _Resp([{"status": "not-found", "teff": None, "logg": None}])
        return _Resp([{"status": "found", "teff": 6075.0, "logg": 4.37}])

    monkeypatch.setattr(requests, "get", _get)
    out = query_hypatia_star(1779546757669063552, ["2MASS J22031079+1859330"])
    assert [s[1] for s in seen] == ["Gaia DR3 1779546757669063552",
                                    "2MASS J22031079+1859330"]
    assert seen[0][0] == "https://hypatiacatalog.com/hypatia/api/v2/star/"
    assert out["teff"] == 6075.0 and out["teff_err"] == 150.0
    assert out["logg"] == 4.37 and out["logg_err"] == 0.3

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(
        [{"status": "not-found", "teff": None, "logg": None}]))
    assert query_hypatia_star(6031016407317733376) is None
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(
        [{"status": "found", "teff": 6000.0, "logg": None}]))
    assert query_hypatia_star(1) is None


@_LIVE
def test_hypatia_star_live_hd209458():
    from astroARIADNE.librarian._hypatia import query_hypatia_star
    out = query_hypatia_star(1779546757669063552)
    assert out is not None
    assert out["teff"] == pytest.approx(6075.0)
    assert out["logg"] == pytest.approx(4.37)
    assert query_hypatia_star(6031016407317733376) is None
