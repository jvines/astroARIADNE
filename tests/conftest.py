import pytest
import requests


@pytest.fixture(autouse=True)
def _no_live_cds_fallback(monkeypatch):
    """Keep the CDS failover offline in tests.

    A test that mocks a CDS query as failing would otherwise reach the live
    VizieR mirror through the local xMatch. Tests of the fallback itself patch
    ``_cds._cone`` or ``_cds.local_xmatch`` explicitly.
    """
    from astroARIADNE.librarian import _cds

    _cds.breaker.clear()

    def _offline(*args, **kwargs):
        raise requests.ConnectionError("local xMatch disabled in tests")

    monkeypatch.setattr(_cds, "local_xmatch", _offline)
    # No TCP probe of the real xMatch host; tests decide reachability.
    monkeypatch.setattr(_cds, "_reachable", lambda *a, **k: True)
    yield
    _cds.breaker.clear()
