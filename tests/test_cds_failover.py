"""Offline tests for the CDS failover (VizieR mirrors, local xMatch, SIMBAD).

Nothing here reaches the network: the HTTP layer under MirroredVizier, the
VizieR cone queries under the local xMatch, and the SIMBAD clients are fakes.
"""
import os
import sys

import astropy.units as u
import numpy as np
import pytest
import requests
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table
from regions import CircleSkyRegion

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import astroARIADNE.librarian._api as api  # noqa: E402
from astroARIADNE.librarian import _cds  # noqa: E402
from astroARIADNE.librarian._api import Librarian  # noqa: E402

# conftest disables the local xMatch for every test; keep the real one here.
REAL_LOCAL_XMATCH = _cds.local_xmatch

CDS = "vizier.cds.unistra.fr"
CFA = "vizier.cfa.harvard.edu"


class _Resp:
    def __init__(self, status=200, host=None):
        self.status_code = status
        self.host = host


@pytest.fixture
def http(monkeypatch):
    """Fake the HTTP layer under MirroredVizier. Configure ``http.plan``
    with host -> status code or exception; every call is recorded."""

    class _HTTP:
        plan = {}
        calls = []

        def __call__(self, inner_self, method, url, **kwargs):
            host = url.split("/")[2]
            self.calls.append((host, kwargs))
            outcome = self.plan.get(host, 200)
            if isinstance(outcome, Exception):
                raise outcome
            return _Resp(outcome, host)

    fake = _HTTP()
    fake.plan, fake.calls = {}, []
    monkeypatch.setattr(_cds.VizierClass, "_request",
                        lambda s, method, url, **kw: fake(s, method, url, **kw))
    return fake


def _viz():
    return _cds.MirroredVizier(row_limit=-1, columns=["all"], timeout=60)


def _get(v):
    return v._request(method="POST", url=f"https://{CDS}/viz-bin/votable",
                      data="-source=II/246/out", timeout=60, cache=False)


# ── VizieR mirror failover ───────────────────────────────────────────

def test_vizier_moves_to_mirror_when_cds_answers_404(http):
    http.plan = {CDS: 404}
    resp = _get(_viz())
    assert resp.host == CFA
    assert [h for h, _ in http.calls] == [CDS, CFA]
    assert _cds.breaker.is_down(CDS)


def test_vizier_moves_to_mirror_on_connection_error(http):
    http.plan = {CDS: requests.ConnectionError("refused")}
    assert _get(_viz()).host == CFA


def test_vizier_moves_to_mirror_on_server_error(http):
    http.plan = {CDS: 503}
    assert _get(_viz()).host == CFA


def test_a_good_answer_is_not_retried_elsewhere(http):
    # 200 covers the empty-result case: VizieR answers 200 with no rows.
    resp = _get(_viz())
    assert resp.host == CDS
    assert [h for h, _ in http.calls] == [CDS]


def test_a_bad_request_is_not_treated_as_an_outage(http):
    http.plan = {CDS: 400}
    assert _get(_viz()).status_code == 400
    assert [h for h, _ in http.calls] == [CDS]
    assert not _cds.breaker.is_down(CDS)


def test_dead_primary_is_skipped_while_tripped(http):
    http.plan = {CDS: 404}
    v = _viz()
    _get(v)
    http.calls.clear()
    assert _get(v).host == CFA
    assert [h for h, _ in http.calls] == [CFA]


def test_primary_comes_back_after_success(http):
    _cds.breaker.trip(CDS, "test")
    http.plan = {CFA: 404}
    assert _get(_viz()).host == CDS
    assert not _cds.breaker.is_down(CDS)


def test_requests_use_a_short_connect_timeout(http):
    _get(_viz())
    assert http.calls[0][1]["timeout"] == (10.0, 60.0)


def test_all_hosts_down_returns_the_error_response(http):
    http.plan = {CDS: 404, CFA: 503}
    assert _get(_viz()).status_code == 503


def test_all_hosts_unreachable_raises(http):
    http.plan = {CDS: requests.ConnectionError("a"),
                 CFA: requests.ConnectTimeout("b")}
    with pytest.raises(requests.ConnectTimeout):
        _get(_viz())


def test_cached_error_response_is_deleted(http, tmp_path):
    http.plan = {CDS: 404}
    v = _viz()
    v._cache_location = tmp_path
    url = f"https://{CDS}/viz-bin/votable"
    q = _cds.AstroQuery("POST", url, data="-source=II/246/out")
    stale = q.request_file(tmp_path)
    stale.write_bytes(b"pickled 404")
    v._request(method="POST", url=url, data="-source=II/246/out",
               timeout=60, cache=True)
    assert not stale.exists()


def test_mirrored_vizier_keeps_the_vizier_cache_directory():
    assert _viz().cache_location.name == "Vizier"


def test_calling_the_instance_keeps_the_failover():
    assert isinstance(_viz()(columns=["Source"]), _cds.MirroredVizier)


# ── local xMatch ─────────────────────────────────────────────────────

C = SkyCoord(10.0, -20.0, unit="deg")
AREA = CircleSkyRegion(C, 5 * u.arcsec)


def _offset(arcsec_east):
    return C.spherical_offsets_by(arcsec_east * u.arcsec, 0 * u.arcsec)


def _gaia(*offsets, propagated=None):
    pos = [_offset(o) for o in offsets]
    t = Table()
    t["RA_ICRS"] = [p.ra.deg for p in pos]
    t["DE_ICRS"] = [p.dec.deg for p in pos]
    t["Source"] = np.arange(1, len(pos) + 1, dtype=np.int64)
    t["Gmag"] = 8.0 + np.arange(len(pos))
    shifted = [_offset(o + (propagated or 0)) for o in offsets]
    t["_RAJ2000"] = [p.ra.deg for p in shifted]
    t["_DEJ2000"] = [p.dec.deg for p in shifted]
    return t


def _tmass(*offsets, sexagesimal=False):
    pos = [_offset(o) for o in offsets]
    t = Table()
    if sexagesimal:
        t["RAJ2000"] = [p.ra.to_string(u.hour, sep=" ") for p in pos]
        t["DEJ2000"] = [p.dec.to_string(sep=" ") for p in pos]
    t["_RAJ2000"] = [p.ra.deg for p in pos]
    t["_DEJ2000"] = [p.dec.deg for p in pos]
    t["_2MASS"] = [f"J{i}" for i in range(len(pos))]
    t["Kmag"] = 7.0 + np.arange(len(pos))
    return t


def _cones(monkeypatch, t1, t2):
    tables = {"I/355/gaiadr3": t1, "II/246/out": t2}
    seen = []

    def fake(catalog, center, radius, timeout):
        seen.append((catalog, radius))
        return tables[catalog]

    monkeypatch.setattr(_cds, "_cone", fake)
    return seen


def _local(max_distance=3 * u.arcsec):
    return REAL_LOCAL_XMATCH("vizier:I/355/gaiadr3", "vizier:II/246/out",
                             max_distance, area=AREA)


def test_local_xmatch_pairs_rows_within_max_distance(monkeypatch):
    seen = _cones(monkeypatch, _gaia(0.0, 4.0), _tmass(0.5, 20.0))
    xm = _local()
    assert len(xm) == 1
    assert xm["Source"][0] == 1 and xm["_2MASS"][0] == "J0"
    assert xm["angDist"][0] == pytest.approx(0.5, abs=1e-3)
    assert xm.colnames[0] == "angDist"
    assert seen == [("I/355/gaiadr3", 5 * u.arcsec), ("II/246/out", 5 * u.arcsec)]


def test_local_xmatch_returns_every_pair_like_xmatch(monkeypatch):
    _cones(monkeypatch, _gaia(0.0, 1.0), _tmass(0.5))
    xm = _local()
    assert sorted(xm["Source"]) == [1, 2]


def test_local_xmatch_renames_clashing_second_catalogue_columns(monkeypatch):
    _cones(monkeypatch, _gaia(0.0), _tmass(0.2))
    xm = _local()
    assert "_RAJ2000" in xm.colnames and "_RAJ2000_2" in xm.colnames
    assert xm["_RAJ2000"][0] == pytest.approx(_gaia(0.0)["_RAJ2000"][0])


def test_local_xmatch_matches_gaia_at_its_epoch_2016_position(monkeypatch):
    # VizieR's computed _RAJ2000 for Gaia is propagated to epoch 2000; xMatch
    # pairs on RA_ICRS, so a 10 arcsec propagation must not break the match.
    _cones(monkeypatch, _gaia(0.0, propagated=10.0), _tmass(0.3))
    assert len(_local()) == 1


def test_local_xmatch_uses_computed_degrees_when_native_is_sexagesimal(monkeypatch):
    _cones(monkeypatch, _gaia(0.0), _tmass(0.3, sexagesimal=True))
    assert len(_local()) == 1


def test_local_xmatch_skips_rows_without_a_position(monkeypatch):
    t2 = _tmass(0.3, 0.6)
    t2["_RAJ2000"] = MaskedColumn(t2["_RAJ2000"], mask=[True, False])
    _cones(monkeypatch, _gaia(0.0), t2)
    xm = _local()
    assert list(xm["_2MASS"]) == ["J1"]


def test_local_xmatch_empty_when_nothing_is_close(monkeypatch):
    _cones(monkeypatch, _gaia(0.0), _tmass(4.0))
    xm = _local()
    assert len(xm) == 0 and "angDist" in xm.colnames


def test_local_xmatch_empty_when_a_cone_finds_nothing(monkeypatch):
    _cones(monkeypatch, _gaia(0.0), None)
    assert len(_local()) == 0


def test_local_xmatch_needs_an_area():
    with pytest.raises(ValueError):
        REAL_LOCAL_XMATCH("vizier:I/355/gaiadr3", "vizier:II/246/out",
                          3 * u.arcsec, area=None)


def test_local_xmatch_only_handles_vizier_tables():
    with pytest.raises(ValueError):
        REAL_LOCAL_XMATCH("simbad", "vizier:II/246/out", 3 * u.arcsec, area=AREA)


def test_local_xmatch_failure_raises(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("mirror down too")

    monkeypatch.setattr(_cds, "_cone", down)
    with pytest.raises(requests.ConnectionError):
        _local()


# ── xmatch(): real first, local when CDS xMatch is down ─────────────

class _Client:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, 0

    def query(self, **kwargs):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _record_local(monkeypatch):
    calls = []

    def fake(cat1, cat2, max_distance, *, area, timeout=60):
        calls.append((cat1, cat2))
        return Table({"angDist": [0.1]})

    monkeypatch.setattr(_cds, "local_xmatch", fake)
    return calls


def test_xmatch_uses_cds_when_it_answers(monkeypatch):
    local = _record_local(monkeypatch)
    real = Table({"angDist": [0.2]})
    out = _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA,
                      client=_Client(real))
    assert out is real and local == []


def test_xmatch_falls_back_when_cds_is_down(monkeypatch):
    local = _record_local(monkeypatch)
    client = _Client(requests.ConnectTimeout("hang"))
    out = _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA,
                      client=client)
    assert out["angDist"][0] == 0.1 and local == [("vizier:a", "vizier:b")]
    # The next call skips the dead service entirely.
    _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA, client=client)
    assert client.calls == 1


def test_xmatch_does_not_hide_programming_errors(monkeypatch):
    _record_local(monkeypatch)
    with pytest.raises(TypeError):
        _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA,
                    client=_Client(TypeError("bad call")))


# ── SIMBAD ───────────────────────────────────────────────────────────

class _Simbad:
    instances = []

    def __init__(self, outcome=None, server="simbad.cds.unistra.fr"):
        self.server, self.outcome = server, outcome
        self.ROW_LIMIT = -1
        self.columns_in_output = []
        _Simbad.instances.append(self)

    def query_objectids(self, name):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return f"{self.server}:{name}"


def test_simbad_moves_to_the_harvard_mirror(monkeypatch):
    monkeypatch.setattr(_cds, "SimbadClass", lambda: _Simbad("ok", server=None))
    primary = _Simbad(requests.HTTPError("404"))
    assert _cds.simbad("query_objectids", "HD 1", client=primary) == \
        "simbad.harvard.edu:HD 1"


def test_simbad_bad_query_is_not_retried(monkeypatch):
    made = []
    monkeypatch.setattr(_cds, "SimbadClass", lambda: made.append(1))
    with pytest.raises(ValueError):
        _cds.simbad("query_objectids", "HD 1", client=_Simbad(ValueError("x")))
    assert made == []


# ── librarian wiring ─────────────────────────────────────────────────

def test_librarian_vizier_client_fails_over():
    assert isinstance(api.Vizier, _cds.MirroredVizier)


def test_xmatch_connect_timeout_is_short():
    assert api.XMatch.TIMEOUT[0] == 10.0


def test_mermilliod_survives_a_cds_xmatch_outage(monkeypatch):
    monkeypatch.setattr(api.XMatch, "query",
                        lambda **k: (_ for _ in ()).throw(
                            requests.ConnectTimeout("cds down")))
    row = Table({"angDist": [1.0], "Source": np.array([12345], dtype=np.int64),
                 "Vmag": [7.65], "e_Vmag": [0.01], "B-V": [0.59],
                 "e_B-V": [0.01], "U-B": [0.10], "e_U-B": [0.02]})
    monkeypatch.setattr(_cds, "local_xmatch", lambda *a, **k: row)
    lib = Librarian.__new__(Librarian)
    lib._gaia_id = 12345
    lib._coord = C
    lib._magnitudes = {}
    lib._fetch_mermilliod()
    assert lib._magnitudes["GROUND_JOHNSON_V"][0] == pytest.approx(7.65)
    assert lib._magnitudes["GROUND_JOHNSON_B"][0] == pytest.approx(8.24)
    assert lib._magnitudes["GROUND_JOHNSON_U"][0] == pytest.approx(8.34)
