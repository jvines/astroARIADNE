"""Offline tests for the ESA Gaia archive fallback in the librarian.

All network is mocked: Vizier and Gaia are replaced in the module, so nothing
reaches CDS or the live ESA archive.
"""
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

import astroARIADNE.librarian._api as api
from astroARIADNE.librarian._api import Librarian

SRC = {
    "source_id": 12345, "ra": 10.0, "dec": -20.0, "pmra": 5.0, "pmdec": -3.0,
    "parallax": 10.0, "parallax_error": 0.1, "ruwe": 1.1,
    "phot_g_mean_mag": 8.5, "phot_g_mean_flux_over_error": 5000.0,
    "phot_bp_mean_mag": 8.9, "phot_bp_mean_flux_over_error": 2000.0,
    "phot_rp_mean_mag": 8.0, "phot_rp_mean_flux_over_error": 3000.0,
}
FLAME = {
    "radius_flame": 1.0, "radius_flame_lower": 0.9, "radius_flame_upper": 1.1,
    "lum_flame": 1.5, "lum_flame_lower": 1.2, "lum_flame_upper": 1.8,
    "mass_flame": 1.0, "mass_flame_lower": 0.9, "mass_flame_upper": 1.1,
    "age_flame": 4.6, "age_flame_lower": 3.0, "age_flame_upper": 6.0,
}
BJ = {"r_med_geo": 100.0, "r_lo_geo": 95.0, "r_hi_geo": 106.0}


def _fake_esa(adql):
    if "astrophysical_parameters" in adql:
        return dict(FLAME)
    if "gaiaedr3_distance" in adql:
        return dict(BJ)
    if "gaia_source" in adql:
        return dict(SRC)
    return None


def _shell():
    lib = Librarian.__new__(Librarian)
    lib._gaia_id = 12345
    lib._magnitudes = {}
    for a in ("parallax", "parallax_e", "teff", "teff_e", "radius_flame",
              "radius_flame_e", "luminosity", "luminosity_e", "mass", "mass_e",
              "age", "age_e", "distance", "distance_e"):
        setattr(lib, "_" + a, None)
    return lib


def _offline_gaia():
    gaia = MagicMock()
    gaia.launch_job.side_effect = Exception("no sync TAP")
    gaia.launch_job_async.side_effect = Exception("no async TAP")
    return gaia


def _run(vizier_result, esa=_fake_esa):
    """Run both queries with every VizieR query returning vizier_result."""
    viz = MagicMock()
    viz.query_constraints.return_value = vizier_result
    lib = _shell()
    with patch.object(api, "Vizier", viz), \
            patch.object(api, "Gaia", _offline_gaia()), \
            patch.object(api, "_esa_first_row", side_effect=esa) as esa_m:
        lib._query_gaia_params()
        lib._query_bailer_jones()
    return lib, [c.args[0] for c in esa_m.call_args_list]


def _job(table):
    job = MagicMock()
    job.get_results.return_value = table
    return job


def test_mag_err_adds_zero_point_like_vizier():
    e = api._gaia_mag_err(1e4, api._GAIA_ZP_ERR["G"])
    assert e == pytest.approx(np.hypot(2.5 / np.log(10) / 1e4, 0.0027553202))
    assert round(e, 4) == 0.0028


@pytest.mark.parametrize("foe", [None, 0.0, -1.0, np.nan])
def test_mag_err_invalid_flux_over_error(foe):
    assert api._gaia_mag_err(foe, 0.0028) is None


def test_translators_use_vizier_names():
    row = api._vizier_gaia_main(SRC)
    assert row["Plx"] == 10.0 and row["e_Plx"] == 0.1 and row["RUWE"] == 1.1
    assert row["pmRA"] == 5.0 and row["pmDE"] == -3.0
    assert row["RA_ICRS"] == 10.0 and row["DE_ICRS"] == -20.0
    assert row["Gmag"] == 8.5 and row["RPmag"] == 8.0
    assert row["e_BPmag"] == pytest.approx(
        np.hypot(2.5 / np.log(10) / 2000.0, 0.0027901700))
    fl = api._vizier_gaia_flame(FLAME)
    assert fl["Rad-Flame"] == 1.0 and fl["b_Rad-Flame"] == 0.9
    assert fl["B_Age-Flame"] == 6.0
    assert api._vizier_bailer_jones(BJ) == {
        "rgeo": 100.0, "b_rgeo": 95.0, "B_rgeo": 106.0}


def test_esa_first_row_lowercases_and_masks():
    t = Table(masked=True)
    t["Plx"] = [1.0]
    t["Other"] = [2.0]
    t["Other"].mask = [True]
    gaia = MagicMock()
    gaia.launch_job.return_value = _job(t)
    with patch.object(api, "Gaia", gaia):
        assert api._esa_first_row("q") == {"plx": 1.0, "other": None}
    gaia.launch_job.return_value = _job(t[:0])
    with patch.object(api, "Gaia", gaia):
        assert api._esa_first_row("q") == {}


def test_esa_first_row_retries_async_when_sync_queue_is_full():
    """ESA rejects sync jobs when its sync queue is full."""
    gaia = MagicMock()
    gaia.launch_job.side_effect = Exception(
        "503: maximum number of synchronous queued jobs (150) reached")
    gaia.launch_job_async.return_value = _job(Table({"ruwe": [1.23]}))
    with patch.object(api, "Gaia", gaia):
        assert api._esa_first_row("q") == {"ruwe": pytest.approx(1.23)}
    gaia.launch_job_async.assert_called_once_with("q")


def test_esa_first_row_none_when_sync_and_async_fail():
    with patch.object(api, "Gaia", _offline_gaia()):
        assert api._esa_first_row("q") is None


def test_vizier_failure_reads_everything_from_esa():
    lib, queries = _run(None)
    assert lib._parallax == pytest.approx(10.0 + api._PLX_ZEROPOINT)
    g, g_e = lib._magnitudes["Gaia_G"]
    assert g == 8.5
    assert g_e == pytest.approx(np.hypot(2.5 / np.log(10) / 5000.0, 0.0027553202))
    assert lib._radius_flame == 1.0
    assert lib._luminosity == 1.5
    assert lib._distance == 100.0 and lib._distance_e == 6.0
    assert any("gaiadr3.gaia_source" in q for q in queries)
    assert any("astrophysical_parameters" in q for q in queries)
    assert any("gaiaedr3_distance" in q for q in queries)


def test_vizier_empty_does_not_query_esa_for_flame_or_distance():
    lib, queries = _run([Table()])
    assert len(queries) == 1 and "gaiadr3.gaia_source" in queries[0]
    assert lib._radius_flame is None
    assert lib._distance is None
    assert lib._parallax == pytest.approx(10.0 + api._PLX_ZEROPOINT)


def test_vizier_row_does_not_query_esa():
    t = Table({"Plx": [9.0], "e_Plx": [0.1], "Gmag": [8.0], "e_Gmag": [0.01]})
    lib, queries = _run([t])
    assert queries == []
    assert lib._parallax == pytest.approx(9.0 + api._PLX_ZEROPOINT)


@pytest.mark.parametrize("vizier", [None, [Table()]])
@pytest.mark.parametrize("esa", [lambda adql: None, lambda adql: {}])
def test_no_gaia_row_anywhere_returns_quietly(vizier, esa):
    lib, _ = _run(vizier, esa=esa)
    assert lib._parallax is None
    assert lib._distance is None
    assert "Gaia_G" not in lib._magnitudes


def test_gaia_id_resolution_survives_a_full_sync_queue():
    lib = _shell()
    lib._gaia_id = None
    lib._coord = SkyCoord(ra=10.0, dec=-20.0, unit="deg")
    lib._search_radius = 5 * u.arcsec
    gaia = MagicMock()
    gaia.launch_job.side_effect = Exception("503: synchronous queue full")
    gaia.launch_job_async.return_value = _job(
        Table({"source_id": [777], "ang_sep": [0.0001]}))
    with patch.object(api, "Gaia", gaia):
        lib._resolve_gaia_id()
    assert lib._gaia_id == 777
