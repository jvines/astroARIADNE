"""Offline tests: VizieR-first Gaia source resolution and the ESA breaker."""
import time
from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

import astroARIADNE.librarian._api as api
from astroARIADNE.librarian import _cds
from astroARIADNE.librarian._api import GaiaQueryError, Librarian

COORD = SkyCoord(ra=10.0, dec=-20.0, unit="deg")
RADIUS = 15 * u.arcsec


def _cone(table):
    return MagicMock(**{"query_region.return_value": table})


def _unsorted_cone():
    # Row 0 is the farthest source; the nearest is source 222.
    return [Table({"Source": [111, 222, 333], "_r": [9.0, 0.4, 5.0]})]


def _esa_answering(source_id):
    gaia = MagicMock()
    tbl = Table({"source_id": [source_id], "ang_sep": [0.0001]})
    gaia.launch_job.return_value.get_results.return_value = tbl
    gaia.launch_job_async.return_value.get_results.return_value = tbl
    return gaia


def _esa_timing_out():
    gaia = MagicMock()
    gaia.launch_job.side_effect = TimeoutError("timed out")
    gaia.launch_job_async.side_effect = TimeoutError("timed out")
    return gaia


def _lib(gaia_id=None):
    lib = Librarian.__new__(Librarian)
    lib._gaia_id = gaia_id
    lib._coord = COORD
    lib._search_radius = RADIUS
    return lib


def test_vizier_unsorted_cone_gives_nearest_with_esa_down():
    with patch.object(api, "Vizier", _cone(_unsorted_cone())), \
            patch.object(api, "Gaia", _esa_timing_out()) as gaia:
        assert api._resolve_gaia_source(COORD, RADIUS) == 222
    gaia.launch_job.assert_not_called()


def test_nearest_without_r_column_uses_positions():
    t = Table({"Source": [111, 222], "RA_ICRS": [10.01, 10.0],
               "DE_ICRS": [-20.0, -20.0]})
    with patch.object(api, "Vizier", _cone([t])):
        assert api._resolve_gaia_source(COORD, RADIUS) == 222


def test_vizier_failure_falls_back_to_esa():
    with patch.object(api, "Vizier", _cone(None)), \
            patch.object(api, "Gaia", _esa_answering(777)):
        assert api._resolve_gaia_source(COORD, RADIUS) == 777


def test_vizier_empty_with_cds_up_means_no_source():
    gaia = _esa_answering(777)
    with patch.object(api, "Vizier", _cone([Table()])), \
            patch.object(api, "Gaia", gaia):
        assert api._resolve_gaia_source(COORD, RADIUS) is None
    gaia.launch_job.assert_not_called()


def test_vizier_empty_from_mirror_is_confirmed_with_esa():
    _cds.breaker.trip(api._PRIMARY_VIZIER, "test")
    with patch.object(api, "Vizier", _cone([Table()])), \
            patch.object(api, "Gaia", _esa_answering(777)):
        assert api._resolve_gaia_source(COORD, RADIUS) == 777


def test_esa_empty_answer_is_no_source_not_an_error():
    gaia = MagicMock()
    gaia.launch_job.return_value.get_results.return_value = Table(
        {"source_id": [1]})[:0]
    with patch.object(api, "Vizier", _cone(None)), \
            patch.object(api, "Gaia", gaia):
        assert api._resolve_gaia_source(COORD, RADIUS) is None


def test_both_failing_raises_gaia_query_error():
    lib = _lib()
    with patch.object(api, "Vizier", _cone(None)), \
            patch.object(api, "Gaia", _esa_timing_out()):
        with pytest.raises(GaiaQueryError):
            lib._resolve_gaia_id()
    assert issubclass(GaiaQueryError, RuntimeError)
    assert lib._gaia_id is None


def test_librarian_construction_raises_when_gaia_services_are_down():
    with patch.object(api, "Vizier", _cone(None)), \
            patch.object(api, "Gaia", _esa_timing_out()):
        with pytest.raises(GaiaQueryError):
            Librarian(10.0, -20.0, verbose=False)


def test_resolve_gaia_id_uses_the_shared_helper():
    lib = _lib()
    with patch.object(api, "_resolve_gaia_source", return_value=42) as h:
        lib._resolve_gaia_id()
    h.assert_called_once_with(COORD, RADIUS)
    assert lib._gaia_id == 42
    # a supplied id is never re-resolved
    lib = _lib(gaia_id=7)
    with patch.object(api, "_resolve_gaia_source") as h:
        lib._resolve_gaia_id()
    h.assert_not_called()


def test_esa_timeout_trips_breaker_and_skips_async_retry():
    gaia = _esa_timing_out()
    with patch.object(api, "Gaia", gaia):
        assert api._esa_first_row("q") is None
    gaia.launch_job_async.assert_not_called()
    assert _cds.breaker.is_down(_cds.ESA_HOST)


def test_esa_http_refusal_does_not_trip_breaker():
    import requests
    gaia = MagicMock()
    gaia.launch_job.side_effect = requests.HTTPError("503 sync queue full")
    gaia.launch_job_async.return_value.get_results.return_value = Table(
        {"x": [1]})
    with patch.object(api, "Gaia", gaia):
        assert api._esa_first_row("q") == {"x": 1}
    assert not _cds.breaker.is_down(_cds.ESA_HOST)


def test_open_breaker_skips_esa_immediately():
    _cds.breaker.trip(_cds.ESA_HOST, "test")
    gaia = _esa_answering(1)
    t0 = time.monotonic()
    with patch.object(api, "Gaia", gaia):
        assert api._esa_first_row("q") is None
        assert api._esa_gaia_main(1) is None
        assert api._esa_gaia_flame(1) is None
        assert api._esa_bailer_jones(1) is None
    assert time.monotonic() - t0 < 1.0
    gaia.launch_job.assert_not_called()
    gaia.launch_job_async.assert_not_called()


def test_esa_success_resets_breaker_after_window(monkeypatch):
    monkeypatch.setattr(_cds, "DOWN_SECONDS", 0.05)
    _cds.breaker.trip(_cds.ESA_HOST, "test")
    time.sleep(0.08)  # window over: the host is tried again
    with patch.object(api, "Gaia", _esa_answering(5)):
        assert api._esa_first_row("q") == {"source_id": 5, "ang_sep": 0.0001}
    assert not _cds.breaker.is_down(_cds.ESA_HOST)
    assert _cds.ESA_HOST not in _cds.breaker._until


def test_crossmatch_skips_tap_while_breaker_open():
    _cds.breaker.trip(_cds.ESA_HOST, "test")
    lib = _lib(gaia_id=1)
    lib._ids = {}
    lib._ignore = ()
    gaia = _esa_answering(1)
    with patch.object(api, "Gaia", gaia), \
            patch.object(Librarian, "_xmatch_fallback") as fb:
        lib._query_crossmatches()
    gaia.launch_job_async.assert_not_called()
    fb.assert_called_once()
    assert fb.call_args[0][0]  # every catalogue went to the fallback
