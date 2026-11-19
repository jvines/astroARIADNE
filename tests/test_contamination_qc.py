"""Offline tests for the Gaia blend QC (issue #55): cross-match uniqueness and
unresolved-neighbour flux contamination. No network."""
import math

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

import astroARIADNE.librarian._api as api
from astroARIADNE.librarian import _contamination as cq
from astroARIADNE.librarian._api import Librarian

TARGET = 1000
FWHM = 6.1
SIGMA = FWHM / 2.3548


# ── formula ──────────────────────────────────────────────────────────────────

def test_equal_mag_neighbour_at_half_fwhm_contributes_half():
    # exp(-(FWHM/2)^2 / (2 sigma^2)) = exp(-ln 2) = 0.5
    c, terms = cq.contamination_fraction(10.0, [(1, FWHM / 2, 10.0)], FWHM)
    assert c == pytest.approx(0.5, rel=1e-3)
    assert terms[0][0] == 1


def test_flux_ratio_and_gaussian_weight():
    sep, dg = 4.0, 3.0
    c, _ = cq.contamination_fraction(10.0, [(1, sep, 10.0 + dg)], FWHM)
    expect = 10 ** (-0.4 * dg) * math.exp(-sep**2 / (2 * SIGMA**2))
    assert c == pytest.approx(expect, rel=1e-4)


def test_contributions_add_and_sort():
    c, terms = cq.contamination_fraction(
        10.0, [(1, 3.0, 14.0), (2, 3.0, 12.0)], FWHM)
    assert [t[0] for t in terms] == [2, 1]
    assert c == pytest.approx(sum(t[3] for t in terms))


def test_zero_separation_and_nan_are_ignored():
    c, terms = cq.contamination_fraction(
        10.0, [(1, 0.0, 10.0), (2, 2.0, float("nan"))], FWHM)
    assert c == 0.0 and terms == []


def test_far_neighbour_is_negligible_for_small_beam_not_large():
    nb = [(1, 8.0, 10.0)]
    c_2mass, _ = cq.contamination_fraction(10.0, nb, 2.5)
    c_w4, _ = cq.contamination_fraction(10.0, nb, 12.0)
    assert c_2mass < 1e-3 < 0.1 < c_w4


# ── thresholds ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("c,expect", [
    (0.0, "ok"), (0.01, "ok"), (0.0101, "flag"), (0.05, "flag"),
    (0.0501, "exclude"), (3.0, "exclude")])
def test_default_thresholds(c, expect):
    assert cq.classify(c) == expect


def test_custom_thresholds():
    assert cq.classify(0.02, flag=0.03, exclude=0.1) == "ok"
    assert cq.classify(0.2, flag=0.03, exclude=0.1) == "exclude"


def test_bad_threshold_order_rejected():
    lib = Librarian.__new__(Librarian)
    with pytest.raises(ValueError):
        lib._init_contamination(True, 0.1, 0.01)


# ── PSF table ────────────────────────────────────────────────────────────────

def test_psf_table_scope():
    assert cq.band_fwhm("WISE_RSR_W1") == 6.1
    assert cq.band_fwhm("WISE_RSR_W4") == 12.0
    assert cq.band_fwhm("2MASS_J") == 2.5
    for skipped in ("Gaia_G", "Gaia_BP", "Gaia_RP", "TESS", "SPHEREx_b1",
                    "TYCHO_V_MvB", "no_such_band"):
        assert cq.band_fwhm(skipped) is None


# ── neighbour extraction ─────────────────────────────────────────────────────

def _cone(rows, ra0=100.0, dec0=0.0):
    """rows: (source_id, dx_arcsec, dy_arcsec, G)."""
    return Table({
        "source_id": [r[0] for r in rows],
        "ra": [ra0 + r[1] / 3600.0 for r in rows],
        "dec": [dec0 + r[2] / 3600.0 for r in rows],
        "phot_g_mean_mag": [r[3] for r in rows],
    })


def test_target_excluded_by_source_id_and_seps_measured_from_it():
    tbl = _cone([(TARGET, 0, 0, 10.0), (1, 3, 4, 12.0), (2, 0, 10, 15.0)])
    g, nbrs = cq.neighbours_from_table(tbl, TARGET)
    assert g == 10.0
    assert {n[0] for n in nbrs} == {1, 2}
    assert dict((n[0], n[1]) for n in nbrs)[1] == pytest.approx(5.0, abs=1e-3)


def test_neighbour_without_g_dropped_and_missing_target():
    tbl = _cone([(TARGET, 0, 0, 10.0), (1, 3, 4, np.nan)])
    assert cq.neighbours_from_table(tbl, TARGET)[1] == []
    g, nbrs = cq.neighbours_from_table(_cone([(1, 3, 4, 12.0)]), TARGET)
    assert g is None


# ── per-band assessment ──────────────────────────────────────────────────────

def test_assess_bands_excludes_wise_keeps_2mass():
    # equal-brightness neighbour 4" away: negligible in 2MASS, huge in WISE
    nb = [(1, 4.0, 10.0)]
    rec = cq.assess_bands(["2MASS_J", "WISE_RSR_W1", "Gaia_G"], 10.0, nb)
    assert rec["2MASS_J"]["status"] == "ok"
    assert rec["WISE_RSR_W1"]["status"] == "exclude"
    assert "Gaia_G" not in rec
    assert rec["WISE_RSR_W1"]["neighbours"][0][0] == 1


def test_describe_names_the_neighbour():
    rec = cq.assess_bands(["WISE_RSR_W1"], 10.0, [(77, 4.0, 10.0)])
    text = cq.describe("WISE_RSR_W1", rec["WISE_RSR_W1"])
    assert "77" in text and "c = " in text and "6.1" in text


# ── cross-match uniqueness ───────────────────────────────────────────────────

@pytest.mark.parametrize("mates,nbrs,expect", [
    (0, 1, "ok"), (1, 1, "drop"), (3, 2, "drop"), (0, 2, "flag"),
    (None, None, "ok"), (np.ma.masked, np.ma.masked, "ok")])
def test_crossmatch_status(mates, nbrs, expect):
    assert cq.crossmatch_status(mates, nbrs) == expect


def _lib(**kw):
    lib = Librarian.__new__(Librarian)
    lib._init_contamination(kw.get("qc", True), 0.01, 0.05)
    lib._gaia_id = TARGET
    lib._coord = SkyCoord(ra=100.0, dec=0.0, unit="deg")
    lib._gaia_ra = lib._gaia_dec = None
    lib._magnitudes = {}
    return lib


def _xm(mates, nbrs):
    return Table({"original_ext_source_id": ["x"],
                  "number_of_neighbours": [nbrs],
                  "number_of_mates": [mates]})


def test_mates_drop_for_2mass_and_wise_only():
    lib = _lib()
    lib._note_crossmatch("2MASS", _xm(2, 1))
    lib._note_crossmatch("Wise", _xm(1, 1))
    lib._note_crossmatch("Pan-STARRS", _xm(1, 1))
    assert set(lib._mates_dropped) == {"2MASS", "Wise"}
    assert lib.xmatch_flags["Pan-STARRS"]["status"] == "drop"


def test_neighbours_gt1_is_flag_only():
    lib = _lib()
    lib._note_crossmatch("2MASS", _xm(0, 2))
    assert not lib._mates_dropped
    assert lib.xmatch_flags["2MASS"]["status"] == "flag"


def test_clean_match_and_table_without_mates_column():
    lib = _lib()
    lib._note_crossmatch("2MASS", _xm(0, 1))
    assert not lib._mates_dropped
    t = Table({"original_ext_source_id": ["t"], "number_of_neighbours": [3]})
    lib._note_crossmatch("TYCHO2", t)
    assert lib.xmatch_flags["TYCHO2"]["status"] == "flag"


def test_qc_off_records_nothing():
    lib = _lib(qc=False)
    lib._note_crossmatch("2MASS", _xm(5, 1))
    assert not lib._mates_dropped and lib.xmatch_flags == {}


def test_xmatch_select_columns():
    assert "number_of_mates" in cq.xmatch_select("2MASS")
    assert "number_of_mates" not in cq.xmatch_select("TYCHO2")


def _patch_esa(monkeypatch, tables):
    """Serve best-neighbour rows keyed by table name through whichever ESA
    helper this branch's librarian uses."""
    seen = []

    def fake(*args, **kwargs):
        adql = next(a for a in args if isinstance(a, str))
        seen.append(adql)
        for name, tbl in tables.items():
            if f"gaiadr3.{name}_best_neighbour" in adql:
                return tbl
        return Table({"original_ext_source_id": [], "number_of_neighbours": [],
                      "number_of_mates": []})

    for attr in ("_esa_table", "_tap_query"):
        if hasattr(api, attr):
            monkeypatch.setattr(api, attr, fake)
    return seen


def test_query_crossmatches_end_to_end_drops_blended_wise(monkeypatch):
    seen = _patch_esa(monkeypatch, {"tmass_psc_xsc": _xm(0, 1),
                                    "allwise": _xm(1, 1)})
    lib = _lib()
    lib._ignore = set()
    lib._ids = {}
    lib._query_crossmatches()
    assert set(lib._mates_dropped) == {"Wise"}
    assert lib._ids["Wise"] == "x"
    assert any("number_of_mates" in q for q in seen)


# ── Librarian-level contamination ────────────────────────────────────────────

def _bands():
    return {"2MASS_J": (9.0, 0.02), "WISE_RSR_W1": (8.5, 0.02),
            "Gaia_G": (10.0, 0.003)}


def test_apply_excludes_contaminated_band_and_records_c(monkeypatch):
    lib = _lib()
    lib._magnitudes = _bands()
    tbl = _cone([(TARGET, 0, 0, 10.0), (5, 4, 0, 10.5)])
    lib._run_adql = lambda adql: tbl
    lib._apply_contamination_qc()
    assert "WISE_RSR_W1" not in lib._magnitudes
    assert "2MASS_J" in lib._magnitudes and "Gaia_G" in lib._magnitudes
    c = lib.contamination
    assert c["WISE_RSR_W1"]["status"] == "exclude"
    assert c["2MASS_J"]["c"] < 0.01 and "Gaia_G" not in c
    assert any("EXCLUDED" in ln and "5" in ln
               for ln in lib._contamination_summary_lines())


def test_apply_flags_but_keeps_between_thresholds():
    lib = _lib()
    lib._magnitudes = {"WISE_RSR_W1": (8.5, 0.02)}
    # pick G so c lands between 0.01 and 0.05 at 6.1" separation
    g_nb = 10.0 + 2.5 * math.log10(1 / 0.03) + 2.5 * math.log10(
        math.exp(-(4.0**2) / (2 * SIGMA**2)))
    lib._run_adql = lambda adql: _cone([(TARGET, 0, 0, 10.0), (5, 4, 0, g_nb)])
    lib._apply_contamination_qc()
    assert lib.contamination["WISE_RSR_W1"]["status"] == "flag"
    assert "WISE_RSR_W1" in lib._magnitudes


def test_apply_custom_threshold_keeps_band():
    lib = _lib()
    lib._init_contamination(True, 0.5, 2.0)
    lib._magnitudes = {"WISE_RSR_W1": (8.5, 0.02)}
    lib._run_adql = lambda adql: _cone([(TARGET, 0, 0, 10.0), (5, 4, 0, 10.5)])
    lib._apply_contamination_qc()
    assert "WISE_RSR_W1" in lib._magnitudes


def test_apply_degrades_when_esa_unreachable(caplog):
    lib = _lib()
    lib._magnitudes = _bands()
    lib._run_adql = lambda adql: None
    with caplog.at_level("WARNING"):
        lib._apply_contamination_qc()
    assert set(lib._magnitudes) == set(_bands())
    assert lib.contamination == {}
    assert "NOT checked" in caplog.text


def test_apply_off_does_not_query():
    lib = _lib(qc=False)
    lib._magnitudes = _bands()

    def boom(adql):
        raise AssertionError("queried")
    lib._run_adql = boom
    lib._apply_contamination_qc()
    assert set(lib._magnitudes) == set(_bands())


def test_apply_target_without_g_keeps_everything(caplog):
    lib = _lib()
    lib._magnitudes = _bands()
    lib._run_adql = lambda adql: _cone([(TARGET, 0, 0, np.nan),
                                        (5, 4, 0, 9.0)])
    with caplog.at_level("WARNING"):
        lib._apply_contamination_qc()
    assert set(lib._magnitudes) == set(_bands())
    assert "NOT checked" in caplog.text


def test_cone_adql_is_centred_on_gaia_position():
    lib = _lib()
    lib._gaia_ra, lib._gaia_dec = 101.5, -2.25
    q = lib._gaia_cone_adql()
    assert "101.50000000" in q and "-2.25000000" in q and "CONTAINS" in q


# ── provenance: APASS-sourced SDSS bands use the APASS PSF ───────────────────

def test_band_fwhm_catalogue_key_with_band_only_fallback():
    assert cq.band_fwhm("SDSS_g") == 1.4
    assert cq.band_fwhm("SDSS_g", "SDSS") == 1.4
    assert cq.band_fwhm("SDSS_g", "APASS") == 5.0
    assert cq.band_fwhm("WISE_RSR_W1", "Wise") == 6.1


def test_add_mag_records_source_of_the_winning_write_only():
    lib = _lib()
    lib._add_mag("SDSS_g", 12.0, 0.02, src="APASS")
    lib._add_mag("SDSS_g", 12.5, 0.02)          # loses: first write wins
    lib._add_mag("SDSS_r", 12.0, 0.02)          # SDSS-sourced: no entry
    assert lib._band_src == {"SDSS_g": "APASS"}
    assert lib._magnitudes["SDSS_g"][0] == 12.0


def test_apass_sourced_sdss_g_checked_with_apass_psf():
    # neighbour 4" away, 1 mag fainter: negligible in a 1.4" PSF, large in 5"
    cone = _cone([(TARGET, 0, 0, 10.0), (5, 4, 0, 11.0)])
    out = {}
    for src in (None, "APASS"):
        lib = _lib()
        lib._add_mag("SDSS_g", 12.0, 0.02, src=src)
        lib._run_adql = lambda adql: cone
        lib._apply_contamination_qc()
        out[src] = lib.contamination["SDSS_g"]
    assert out[None]["status"] == "ok" and out[None]["fwhm"] == 1.4
    assert out["APASS"]["fwhm"] == 5.0
    assert out["APASS"]["status"] == "exclude"
