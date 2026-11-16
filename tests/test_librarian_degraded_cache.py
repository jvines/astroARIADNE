"""A lookup that saw failed queries is not written to the per-source cache.

2026-11-13: with the ESA archive answering 429, lookups lost their Gaia
cross-matches; the degraded result was cached under the current package hash
and replayed on every rerun as if it were the archives' real answer.
"""
import pytest

import astroARIADNE.librarian._api as api
from astroARIADNE.librarian._api import Librarian

STEPS = ('_resolve_gaia_id', '_query_gaia_params', '_query_crossmatches',
         '_query_bailer_jones', '_fetch_photometry', '_impute_zero_errors',
         '_query_spectroscopic_priors', '_query_extinction')


@pytest.fixture
def offline(monkeypatch, tmp_path):
    saved = []
    for name in STEPS:
        monkeypatch.setattr(Librarian, name, lambda self: None)
    monkeypatch.setattr(Librarian, '_result_cache_path',
                        lambda self: str(tmp_path / 'star.pkl'))
    monkeypatch.setattr(Librarian, '_save_cache', lambda self: saved.append(1))
    return saved


def _failing(self):
    api._with_timeout(lambda: (_ for _ in ()).throw(RuntimeError('Error 429')))


def test_clean_lookup_is_cached(offline):
    Librarian(10.0, -20.0, gaia_id=1, verbose=False)
    assert offline == [1]


def test_lookup_with_a_failed_query_is_not_cached(offline, monkeypatch):
    monkeypatch.setattr(Librarian, '_query_crossmatches', _failing)
    Librarian(10.0, -20.0, gaia_id=1, verbose=False)
    assert offline == []


def test_failures_before_the_lookup_do_not_count(offline):
    api._with_timeout(lambda: (_ for _ in ()).throw(RuntimeError('earlier star')))
    Librarian(10.0, -20.0, gaia_id=1, verbose=False)
    assert offline == [1]


# ── ESA 429 backoff in _tap_query ────────────────────────────────────

class _Job:
    def __init__(self, rows):
        self.rows = rows

    def get_results(self):
        return self.rows


class _Service:
    """launch_job answers 429 for the first ``n429`` calls, then rows."""

    def __init__(self, n429):
        self.n429, self.sync_calls, self.async_calls = n429, 0, 0

    def launch_job(self, query):
        self.sync_calls += 1
        if self.sync_calls <= self.n429:
            raise RuntimeError("Error 429:\nUser (anonymous) has reached the "
                               "maximum number of jobs allowed per user in the queue (50).")
        return _Job(["row"])

    def launch_job_async(self, query):
        self.async_calls += 1
        return _Job(["async-row"])


def test_tap_429_waits_and_retries_sync(monkeypatch):
    slept = []
    monkeypatch.setattr(api.time, "sleep", slept.append)
    svc = _Service(n429=2)
    assert api._tap_query(svc, "SELECT 1") == ["row"]
    assert slept == list(api._TAP_429_BACKOFF[:2])
    assert svc.async_calls == 0


def test_tap_429_that_never_clears_gives_up_without_async(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    svc = _Service(n429=99)
    before = api._QUERY_FAILURES
    assert api._tap_query(svc, "SELECT 1") is None
    assert svc.sync_calls == len(api._TAP_429_BACKOFF) + 1
    assert svc.async_calls == 0
    assert api._QUERY_FAILURES == before + 1


def test_other_sync_errors_still_fall_back_to_async(monkeypatch):
    class _Broken(_Service):
        def launch_job(self, query):
            raise RuntimeError("Error 500: null")
    svc = _Broken(0)
    assert api._tap_query(svc, "SELECT 1") == ["async-row"]
