"""Offline tests for the ESA Gaia archive fallback in the librarian.

All network is mocked: Vizier and Gaia are replaced in the module, so nothing
reaches CDS or the live ESA archive.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import astroARIADNE.librarian._api as api  # noqa: E402
from astroARIADNE.librarian._api import GaiaQueryError, Librarian  # noqa: E402

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
    for a in ("parallax", "parallax_e", "ruwe", "pmra", "pmdec", "gaia_ra",
              "gaia_dec", "teff", "teff_e", "radius_flame", "radius_flame_e",
              "luminosity", "luminosity_e", "mass", "mass_e", "age", "age_e",
              "distance", "distance_e"):
        setattr(lib, "_" + a, None)
    return lib


def _run(vizier_result, esa=_fake_esa):
    """Run both queries with every VizieR query returning vizier_result."""
    viz = MagicMock()
    viz.query_constraints.return_value = vizier_result
    gaia = MagicMock()
    gaia.launch_job.side_effect = Exception("no sync TAP")
    gaia.launch_job_async.side_effect = Exception("no async TAP")
    lib = _shell()
    with patch.object(api, "Vizier", viz), patch.object(api, "Gaia", gaia), \
            patch.object(api, "_GAIA_LOGIN_ATTEMPTED", True), \
            patch.object(api, "_esa_first_row", side_effect=esa) as esa_m:
        try:
            lib._query_gaia_params()
            lib._query_bailer_jones()
        finally:
            lib._esa = esa_m
    return lib, esa_m


def _tables(esa_m):
    """ESA data queries, leaving out the RUWE lookup that always runs."""
    return [c.args[0] for c in esa_m.call_args_list
            if not c.args[0].startswith("SELECT ruwe")]


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
    with patch.object(api, "_maybe_login_gaia"), \
            patch.object(api, "_tap_query", return_value=t):
        assert api._esa_first_row("q") == {"plx": 1.0, "other": None}
    with patch.object(api, "_maybe_login_gaia"), \
            patch.object(api, "_tap_query", return_value=t[:0]):
        assert api._esa_first_row("q") == {}
    with patch.object(api, "_maybe_login_gaia"), \
            patch.object(api, "_tap_query", return_value=None):
        assert api._esa_first_row("q") is None


def test_vizier_failure_reads_everything_from_esa():
    lib, esa = _run(None)
    assert lib._parallax == pytest.approx(10.037)
    assert lib._ruwe == 1.1
    g, g_e = lib._magnitudes["Gaia_G"]
    assert g == 8.5
    assert g_e == pytest.approx(np.hypot(2.5 / np.log(10) / 5000.0, 0.0027553202))
    assert lib._pmra == 5.0 and lib._gaia_dec == -20.0
    assert lib._radius_flame == 1.0 and lib._radius_flame_e == pytest.approx(0.5)
    assert lib._luminosity == 1.5
    assert lib._distance == 100.0 and lib._distance_e == 6.0
    tabs = _tables(esa)
    assert any("gaiadr3.gaia_source" in q for q in tabs)
    assert any("astrophysical_parameters" in q for q in tabs)
    assert any("gaiaedr3_distance" in q for q in tabs)


def test_vizier_empty_does_not_query_esa_for_flame_or_distance():
    lib, esa = _run([Table()])
    tabs = _tables(esa)
    assert len(tabs) == 1 and "gaiadr3.gaia_source" in tabs[0]
    assert lib._radius_flame is None
    assert lib._distance is None
    assert lib._parallax == pytest.approx(10.037)


def test_mirror_empty_flame_and_distance_are_confirmed_with_esa():
    # CfA answers Source=<id> on I/355/paramp with an empty table even when it
    # holds the row; while CDS is down an empty answer is checked with ESA.
    from astroARIADNE.librarian import _cds
    _cds.breaker.trip(api._PRIMARY_VIZIER, "test")
    lib, esa = _run([Table()])
    tabs = _tables(esa)
    assert any("astrophysical_parameters" in q for q in tabs)
    assert any("gaiaedr3_distance" in q for q in tabs)
    assert lib._radius_flame == 1.0
    assert lib._distance == 100.0


def test_vizier_row_does_not_query_esa():
    t = Table({"Plx": [9.0], "e_Plx": [0.1], "Gmag": [8.0], "e_Gmag": [0.01]})
    lib, esa = _run([t])
    assert _tables(esa) == []
    assert lib._parallax == pytest.approx(9.037)


def test_esa_failure_after_vizier_failure_raises():
    with pytest.raises(GaiaQueryError, match="both VizieR and the ESA"):
        _run(None, esa=lambda adql: None)


def test_esa_failure_after_vizier_empty_does_not_raise():
    lib, _ = _run([Table()], esa=lambda adql: None)
    assert lib._parallax is None
    assert lib._distance is None
    assert "Gaia_G" not in lib._magnitudes


def test_esa_absent_source_returns_quietly():
    lib, _ = _run(None, esa=lambda adql: {})
    assert lib._parallax is None


def test_ruwe_survives_a_full_sync_queue():
    """ESA rejects sync jobs when its sync queue is full; RUWE retries async."""
    job = MagicMock()
    job.get_results.return_value = Table({"ruwe": [1.23]})
    gaia = MagicMock()
    gaia.launch_job.side_effect = Exception("503: maximum number of "
                                            "synchronous queued jobs")
    gaia.launch_job_async.return_value = job
    viz = MagicMock()
    viz.query_constraints.side_effect = lambda catalog=None, **kw: (
        [Table({"Plx": [9.0], "e_Plx": [0.1]})] if catalog == "I/355/gaiadr3"
        else [Table()])
    lib = _shell()
    with patch.object(api, "Vizier", viz), patch.object(api, "Gaia", gaia), \
            patch.object(api, "_GAIA_LOGIN_ATTEMPTED", True):
        lib._query_gaia_params()
    assert lib._ruwe == pytest.approx(1.23)
