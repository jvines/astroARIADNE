"""Offline tests for Gaia source resolution and the ESA archive breaker.

Vizier and Gaia are mocked in the module; nothing reaches CDS or ESA.
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import astroARIADNE.librarian._api as api  # noqa: E402
from astroARIADNE.librarian import _cds  # noqa: E402
from astroARIADNE.librarian._api import GaiaQueryError, Librarian  # noqa: E402

RA, DEC = 120.0, 30.0
RADIUS = 15 * u.arcsec


def _cone(with_r=True):
    """Unsorted cone: row 0 is the farthest, the nearest is source 222."""
    t = Table()
    t["Source"] = [111, 222, 333]
    t["RA_ICRS"] = [RA + 4e-3, RA + 1e-4, RA + 2e-3]
    t["DE_ICRS"] = [DEC, DEC, DEC]
    if with_r:
        t["_r"] = [14.0, 0.3, 7.0]
    return t


def _vizier(result):
    viz = MagicMock()
    viz.query_region.return_value = result
    return viz


def _failing_vizier():
    viz = MagicMock()
    viz.query_region.side_effect = ConnectionError("CDS down")
    return viz


def _shell():
    lib = Librarian.__new__(Librarian)
    lib._gaia_id = None
    lib._coord = SkyCoord(RA * u.deg, DEC * u.deg)
    lib._search_radius = RADIUS
    return lib


@pytest.mark.parametrize("with_r", [True, False])
def test_vizier_cone_returns_nearest_when_esa_times_out(with_r):
    gaia = MagicMock()
    gaia.launch_job.side_effect = TimeoutError("ESA down")
    with patch.object(api, "Vizier", _vizier([_cone(with_r)])), \
            patch.object(api, "Gaia", gaia), \
            patch.object(api, "_esa_first_row") as esa:
        assert api._resolve_gaia_source(RA, DEC, RADIUS) == 222
    esa.assert_not_called()


def test_vizier_failure_falls_back_to_esa():
    with patch.object(api, "Vizier", _failing_vizier()), \
            patch.object(api, "_esa_first_row",
                         return_value={"source_id": 777, "ang_sep": 1e-5}) as esa:
        assert api._resolve_gaia_source(RA, DEC, RADIUS) == 777
    assert "ORDER BY ang_sep" in esa.call_args.args[0]


def test_empty_vizier_cone_is_no_source_without_esa():
    with patch.object(api, "Vizier", _vizier([Table(names=["Source", "_r"])])), \
            patch.object(api, "_esa_first_row") as esa:
        assert api._resolve_gaia_source(RA, DEC, RADIUS) is None
    esa.assert_not_called()


def test_empty_mirror_cone_is_confirmed_with_esa():
    _cds.breaker.trip(api._PRIMARY_VIZIER, "test")
    with patch.object(api, "Vizier", _vizier([Table(names=["Source", "_r"])])), \
            patch.object(api, "_esa_first_row", return_value={"source_id": 5}):
        assert api._resolve_gaia_source(RA, DEC, RADIUS) == 5


def test_esa_answering_no_row_is_no_source():
    with patch.object(api, "Vizier", _failing_vizier()), \
            patch.object(api, "_esa_first_row", return_value={}):
        assert api._resolve_gaia_source(RA, DEC, RADIUS) is None


def test_both_services_down_raises_not_empty_result():
    lib = _shell()
    with patch.object(api, "Vizier", _failing_vizier()), \
            patch.object(api, "_esa_first_row", return_value=None):
        with pytest.raises(GaiaQueryError):
            lib._resolve_gaia_id()
    assert lib._gaia_id is None


def test_librarian_resolver_uses_the_shared_helper():
    lib = _shell()
    with patch.object(api, "_resolve_gaia_source", return_value=42) as helper:
        lib._resolve_gaia_id()
    helper.assert_called_once_with(RA, DEC, RADIUS)
    assert lib._gaia_id == 42


def test_from_gaia_goes_through_the_breaker_aware_esa_helper():
    from astroARIADNE.star import Star
    with patch.object(api, "_esa_first_row",
                      return_value={"ra": RA, "dec": DEC}) as esa, \
            patch.object(Star, "__init__", return_value=None) as init:
        Star.from_gaia(42)
    assert "source_id = 42" in esa.call_args.args[0]
    assert init.call_args.args == ("Gaia DR3 42", RA, DEC)
    assert init.call_args.kwargs["g_id"] == 42


def test_from_gaia_raises_when_esa_is_down():
    from astroARIADNE.star import Star
    with patch.object(api, "_esa_first_row", return_value=None):
        with pytest.raises(SystemExit):
            Star.from_gaia(42)


# ── ESA circuit breaker ──────────────────────────────────────────


def test_open_breaker_skips_esa_without_calling_launch_job():
    gaia = MagicMock()
    _cds.breaker.trip(api._ESA_HOST, "test")
    t0 = time.monotonic()
    with patch.object(api, "Gaia", gaia), patch.object(api, "_maybe_login_gaia"):
        assert api._esa_first_row("SELECT 1") is None
    assert time.monotonic() - t0 < 1.0
    gaia.launch_job.assert_not_called()
    gaia.launch_job_async.assert_not_called()


@pytest.mark.parametrize("exc", [ConnectionError("refused"), TimeoutError("t")])
def test_unreachable_esa_trips_breaker_and_skips_async(exc):
    gaia = MagicMock()
    gaia.launch_job.side_effect = exc
    with patch.object(api, "Gaia", gaia), patch.object(api, "_maybe_login_gaia"):
        assert api._esa_first_row("SELECT 1") is None
    gaia.launch_job_async.assert_not_called()
    assert _cds.breaker.is_down(api._ESA_HOST)


def test_sync_timeout_trips_breaker_and_skips_async():
    gaia = MagicMock()
    gaia.launch_job.side_effect = lambda q: time.sleep(0.5)
    with patch.object(api, "Gaia", gaia):
        assert api._tap_query(gaia, "q", timeout=0.05,
                              host=api._ESA_HOST) is None
    gaia.launch_job_async.assert_not_called()
    assert _cds.breaker.is_down(api._ESA_HOST)


def test_server_error_still_retries_async_without_tripping():
    gaia = MagicMock()
    gaia.launch_job.side_effect = ValueError("sync queue full")
    gaia.launch_job_async.return_value.get_results.return_value = Table(
        {"x": [1]})
    with patch.object(api, "Gaia", gaia), patch.object(api, "_maybe_login_gaia"):
        assert api._esa_first_row("SELECT 1") == {"x": 1}
    gaia.launch_job_async.assert_called_once()
    assert not _cds.breaker.is_down(api._ESA_HOST)


def test_success_resets_the_breaker(monkeypatch):
    gaia = MagicMock()
    gaia.launch_job.return_value.get_results.return_value = Table({"x": [1]})
    # Window elapsed: the entry has expired, but success must still clear it.
    monkeypatch.setattr(_cds, "DOWN_SECONDS", 0.0)
    _cds.breaker.trip(api._ESA_HOST, "test")
    assert not _cds.breaker.is_down(api._ESA_HOST)
    with patch.object(api, "Gaia", gaia), patch.object(api, "_maybe_login_gaia"):
        assert api._esa_first_row("SELECT 1") == {"x": 1}
    assert api._ESA_HOST not in _cds.breaker._until


def test_crossmatch_goes_straight_to_fallback_while_esa_is_down():
    lib = _shell()
    lib._gaia_id = 42
    lib._ids = {}
    lib._ignore = ()
    _cds.breaker.trip(api._ESA_HOST, "test")
    gaia = MagicMock()
    with patch.object(api, "Gaia", gaia), \
            patch.object(lib, "_xmatch_fallback") as fb:
        lib._query_crossmatches()
    gaia.launch_job.assert_not_called()
    gaia.launch_job_async.assert_not_called()
    fb.assert_called_once()
    assert len(fb.call_args.args[0]) > 1
