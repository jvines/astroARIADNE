"""Offline tests for the proper-motion-widened cone fallback in the librarian.

A fast mover observed early or late in a survey sits far from its mean-epoch
position, so the 3" cone misses it even after PM propagation. With the Gaia
crossmatch ID known, the cone widens by |PM| x the survey half-span and takes
only the ID match. All VizieR access is mocked.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import astroARIADNE.librarian._api as api  # noqa: E402
from astroARIADNE.librarian._api import Librarian  # noqa: E402

TMASS_ID = "02025226+0542205"
PS1_ID = 114840307240755015


def _shell(ids, pmra=2323.64, pmdec=-707.82):
    """Librarian for Gaia DR3 2519329585637068160 (2.43"/yr), no network."""
    lib = Librarian.__new__(Librarian)
    lib.ra, lib.dec = 30.727661, 5.702675
    lib._coord = SkyCoord(lib.ra * u.deg, lib.dec * u.deg)
    lib._gaia_id = 2519329585637068160
    lib._gaia_ra, lib._gaia_dec = lib.ra, lib.dec
    lib._gaia_ref_epoch = 2016.0
    lib._pmra, lib._pmdec = pmra, pmdec
    lib._parallax = 34.85
    lib._search_radius = 3.0 * u.arcsec
    lib._ids = dict(ids)
    lib._ignore = []
    lib._magnitudes = {}
    return lib


def _tmass_table(rows):
    t = Table(rows=rows, names=("_r", "2MASS", "Jmag", "e_Jmag", "Hmag",
                                "e_Hmag", "Kmag", "e_Kmag", "Qflg", "Cflg"),
              dtype=(float, str, float, float, float, float, float, float,
                     str, str))
    t["_r"].unit = u.arcsec
    return t


def _ps1_table(rows):
    names = ("_r", "objID", "Qual") + tuple(
        x for b in "grizy" for x in (f"{b}mag", f"e_{b}mag"))
    t = Table(rows=rows, names=names,
              dtype=(float, "i8", int) + (float,) * 10)
    t["_r"].unit = u.arcsec
    return t


def _vizier(table_by_cat):
    viz = MagicMock()
    viz.query_region.side_effect = (
        lambda coord, radius, catalog: {catalog: table_by_cat[catalog].copy()})
    return viz


GOOD_2MASS = (3.4, TMASS_ID, 9.474, 0.023, 8.900, 0.026, 8.684, 0.020,
              "AAA", "000")
NEIGHBOUR_2MASS = (1.0, "02025200+0542100", 15.1, 0.05, 14.6, 0.06, 14.4,
                   0.07, "AAA", "000")


def test_widened_cone_recovers_the_id_match():
    lib = _shell({"2MASS": TMASS_ID})
    viz = _vizier({"II/246/out": _tmass_table([GOOD_2MASS])})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("2MASS", api._CATALOGS["2MASS"])
    radius = viz.query_region.call_args.kwargs["radius"]
    # 3" + 2.43"/yr x 2.0 yr
    assert radius.to(u.arcsec).value == pytest.approx(3.0 + 2.429 * 2.0,
                                                      abs=0.01)
    assert lib._magnitudes["2MASS_J"] == (pytest.approx(9.474), 0.023)
    assert set(lib._magnitudes) == {"2MASS_J", "2MASS_H", "2MASS_Ks"}


def test_id_match_beats_a_nearer_neighbour():
    lib = _shell({"2MASS": TMASS_ID})
    viz = _vizier({"II/246/out": _tmass_table([NEIGHBOUR_2MASS, GOOD_2MASS])})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("2MASS", api._CATALOGS["2MASS"])
    assert lib._magnitudes["2MASS_J"][0] == pytest.approx(9.474)


def test_widened_cone_never_takes_a_far_non_id_source():
    """ID absent from the cone: a source only the widening reached is
    rejected, never adopted as the nearest match."""
    lib = _shell({"2MASS": TMASS_ID})
    far = (5.0, "02025300+0542000") + GOOD_2MASS[2:]
    viz = _vizier({"II/246/out": _tmass_table([far])})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("2MASS", api._CATALOGS["2MASS"])
    assert lib._magnitudes == {}


def test_no_id_keeps_the_base_radius_and_nearest_match():
    lib = _shell({})
    near = (1.0, "02025226+0542205") + GOOD_2MASS[2:]
    viz = _vizier({"II/246/out": _tmass_table([near])})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("2MASS", api._CATALOGS["2MASS"])
    radius = viz.query_region.call_args.kwargs["radius"]
    assert radius.to(u.arcsec).value == pytest.approx(3.0)
    assert "2MASS_J" in lib._magnitudes


def test_widening_is_capped():
    lib = _shell({"2MASS": TMASS_ID}, pmra=20000.0, pmdec=0.0)
    lib._ids["SDSS"] = 1
    viz = _vizier({"V/147/sdss12": Table(names=("_r", "objID"),
                                         dtype=(float, "i8"))})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("SDSS", api._CATALOGS["SDSS"], require_id=True)
    radius = viz.query_region.call_args.kwargs["radius"]
    assert radius.to(u.arcsec).value == pytest.approx(60.0)


def test_optical_catalog_needs_its_id():
    lib = _shell({})
    viz = _vizier({})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("Pan-STARRS", api._CATALOGS["Pan-STARRS"],
                           require_id=True)
    viz.query_region.assert_not_called()
    assert lib._magnitudes == {}


def test_ps1_recovered_by_id_and_neighbour_ignored():
    lib = _shell({"Pan-STARRS": PS1_ID})
    good = (8.0, PS1_ID, 52) + (12.9, 0.01, 11.8, 0.01, 11.1, 0.01, 10.7,
                                0.01, 10.5, 0.01)
    other = (0.5, 114840307240755999, 52) + (18.0, 0.02) * 5
    viz = _vizier({"II/349/ps1": _ps1_table([other, good])})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("Pan-STARRS", api._CATALOGS["Pan-STARRS"],
                           require_id=True)
    assert lib._magnitudes["PS1_g"][0] == pytest.approx(12.9)
    assert len(lib._magnitudes) == 5


def test_ps1_without_id_match_adds_nothing():
    lib = _shell({"Pan-STARRS": PS1_ID})
    other = (0.5, 114840307240755999, 52) + (18.0, 0.02) * 5
    viz = _vizier({"II/349/ps1": _ps1_table([other])})
    with patch.object(api, "Vizier", viz):
        lib._fetch_by_cone("Pan-STARRS", api._CATALOGS["Pan-STARRS"],
                           require_id=True)
    assert lib._magnitudes == {}


def _run_fetch_photometry(lib, viz):
    """Run only the bulk + PM-cone stage of _fetch_photometry."""
    with patch.object(api, "Vizier", viz):
        lib._fetch_photometry()


def test_fetch_photometry_runs_pm_cone_for_missed_catalogs():
    lib = _shell({"2MASS": TMASS_ID, "Pan-STARRS": PS1_ID})
    lib._ignore = ["Wise", "SDSS", "TYCHO2", "GLIMPSE", "CatWISE", "far_ir",
                   "Spitzer", "TESS", "APASS", "SkyMapper", "GALEX",
                   "MERMILLIOD", "STROMGREN", "XP", "DESI", "BOSS",
                   "SPHEREX", "CASSIS", "IUE"]
    good_ps1 = (8.0, PS1_ID, 52) + (12.9, 0.01) * 5
    tables = {"II/246/out": _tmass_table([GOOD_2MASS]),
              "II/349/ps1": _ps1_table([good_ps1])}
    calls = []

    def query_region(coord, radius, catalog):
        calls.append((catalog, radius.to(u.arcsec).value))
        if isinstance(catalog, list):  # bulk cone at J2016: misses both
            return {}
        return {catalog: tables[catalog].copy()}

    viz = MagicMock()
    viz.query_region.side_effect = query_region
    _run_fetch_photometry(lib, viz)
    assert {"2MASS_J", "2MASS_H", "2MASS_Ks"} <= set(lib._magnitudes)
    assert {"PS1_g", "PS1_y"} <= set(lib._magnitudes)
    cone = {c: r for c, r in calls if not isinstance(c, list)}
    assert cone["II/246/out"] == pytest.approx(3.0 + 2.429 * 2.0, abs=0.01)
    assert cone["II/349/ps1"] == pytest.approx(3.0 + 2.429 * 2.5, abs=0.01)


def test_fetch_photometry_skips_pm_cone_when_bulk_matched():
    """Bulk cone found the 2MASS row but QC rejected every band: the cone
    fallback must not re-query (it would find the same row)."""
    lib = _shell({"2MASS": TMASS_ID})
    lib._ignore = ["Wise", "Pan-STARRS", "SDSS", "TYCHO2", "GLIMPSE",
                   "CatWISE", "far_ir", "Spitzer", "TESS", "APASS",
                   "SkyMapper", "GALEX", "MERMILLIOD", "STROMGREN", "XP",
                   "DESI", "BOSS", "SPHEREX", "CASSIS", "IUE"]
    flagged = (0.1, TMASS_ID, 11.6, 0.05, 11.1, 0.05, 10.6, 0.05, "AAU",
               "cc0")
    calls = []

    def query_region(coord, radius, catalog):
        calls.append(catalog)
        return {"II/246/out": _tmass_table([flagged])}

    viz = MagicMock()
    viz.query_region.side_effect = query_region
    _run_fetch_photometry(lib, viz)
    assert lib._magnitudes == {}
    assert len(calls) == 1
