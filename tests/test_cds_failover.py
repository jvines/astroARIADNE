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
REAL_REACHABLE = _cds._reachable

CDS = "vizier.cds.unistra.fr"
CFA = "vizier.cfa.harvard.edu"


DB_ERROR = "DB"  # plan value: a 200 carrying VizieR's database-down INFO
MALFORMED = "BAD"  # plan value: a 200 whose VOTable is not well-formed XML
BAD_VOTABLE = ('<?xml version="1.0"?><VOTABLE version="1.4"><RESOURCE><TABLE>'
               '<DATA><TABLEDATA><TR><TD>1\x00</TD></TR>')


class _Resp:
    def __init__(self, status=200, host=None, text=""):
        self.status_code = status
        self.host = host
        self.text = text


@pytest.fixture
def http(monkeypatch):
    """Fake the HTTP layer under MirroredVizier. Configure ``http.plan``
    with host -> status code, DB_ERROR, exception, or a list of those (one
    per call); every call is recorded."""

    class _HTTP:
        plan = {}
        calls = []

        def __call__(self, inner_self, method, url, **kwargs):
            host = url.split("/")[2]
            self.calls.append((host, kwargs))
            outcome = self.plan.get(host, 200)
            if isinstance(outcome, list):
                outcome = outcome.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome == MALFORMED:
                return _Resp(200, host, BAD_VOTABLE)
            if outcome == DB_ERROR:
                return _Resp(200, host, '<INFO ID="Error" name="Error" '
                             'value="The database is not currently reachable."/>')
            return _Resp(outcome, host)

    fake = _HTTP()
    fake.plan, fake.calls = {}, []
    monkeypatch.setattr(_cds.time, "sleep", lambda s: None)
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


# ── mirror data integrity ────────────────────────────────────────────

STAR_BODY = ("-source=II/246/out\n-out.max=50\n-out=*\n-out.add=_r\n"
             "-c=330.79502759+18.88432490\n-c.rs=5.0\n-out.form=mini")


def test_mirror_requests_ask_for_all_columns(http):
    # CfA returns shifted, corrupt values for -out=* and explicit lists.
    http.plan = {CDS: 404}
    _viz()._request(method="POST", url=f"https://{CDS}/viz-bin/votable",
                    data=STAR_BODY, timeout=60, cache=False)
    (_, to_cds), (_, to_cfa) = http.calls
    assert to_cds["data"] == STAR_BODY
    lines = to_cfa["data"].split("\n")
    assert "-out.all" in lines and not any(ln.startswith("-out=") for ln in lines)
    assert "-out.add=_r" in lines and "-out.max=50" in lines


def test_all_columns_rewrite():
    assert _cds._all_columns("-source=X\n-out=Source,Plx\n-sort=_r") == \
        "-source=X\n-out.all\n-sort=_r"
    assert _cds._all_columns("-source=X\n-out.all\n-out.add=_r") == \
        "-source=X\n-out.all\n-out.add=_r"
    assert _cds._all_columns("-source=X\nSource=1") == "-source=X\nSource=1"
    assert _cds._all_columns(None) is None


def test_database_error_sent_as_200_is_retried_on_the_mirror(http):
    http.plan = {CDS: 404, CFA: [DB_ERROR, DB_ERROR, 200]}
    resp = _get(_viz())
    assert resp.host == CFA and resp.text == ""
    assert [h for h, _ in http.calls] == [CDS, CFA, CFA, CFA]
    assert not _cds.breaker.is_down(CFA)


def test_database_error_on_every_try_raises_instead_of_reading_empty(http):
    # An empty answer would read as "not in the catalogue"; raising lets the
    # caller take its own fallback (e.g. the ESA archive for Gaia).
    http.plan = {CDS: 404, CFA: DB_ERROR}
    with pytest.raises(requests.HTTPError, match="database not reachable"):
        _get(_viz())
    assert len(http.calls) == 1 + _cds.MIRROR_TRIES


def test_database_error_on_the_primary_moves_to_the_mirror(http):
    http.plan = {CDS: DB_ERROR}
    assert _get(_viz()).host == CFA


def test_a_real_empty_answer_is_not_a_database_error():
    assert not _cds._db_error(_Resp(200, CFA, "<VOTABLE><RESOURCE/></VOTABLE>"))
    assert not _cds._db_error(_Resp(200, CFA, '<INFO name="Error" value="bad column"/>'))


def test_malformed_votable_is_detected():
    assert _cds._malformed(_Resp(200, CFA, BAD_VOTABLE))
    assert not _cds._malformed(_Resp(200, CFA, "<VOTABLE><RESOURCE/></VOTABLE>"))
    assert not _cds._malformed(_Resp(200, CFA, "not xml at all"))   # e.g. TSV
    assert not _cds._malformed(_Resp(200, CFA, ""))


def test_malformed_votable_is_retried_on_the_mirror(http):
    http.plan = {CDS: 404, CFA: [MALFORMED, MALFORMED, 200]}
    resp = _get(_viz())
    assert resp.host == CFA and resp.text == ""
    assert [h for h, _ in http.calls] == [CDS, CFA, CFA, CFA]


def test_malformed_votable_on_every_try_raises(http):
    http.plan = {CDS: 404, CFA: MALFORMED}
    with pytest.raises(requests.HTTPError, match="malformed"):
        _get(_viz())
    assert len(http.calls) == 1 + _cds.MIRROR_TRIES


def test_cached_malformed_response_is_deleted(http, tmp_path):
    v = _viz()
    v.cache_location = str(tmp_path)
    forgotten = []
    v._forget = lambda method, url, kw: forgotten.append(url)
    http.plan = {CDS: MALFORMED}          # the primary gets one try
    assert _get(v).host == CFA
    assert forgotten == [f"https://{CDS}/viz-bin/votable"]


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


def test_local_xmatch_clash_keeps_the_counterpart_column_name(monkeypatch):
    # Gaia DR3 carries its own TYC2; the Tycho-2 ID must win the bare name.
    t1, t2 = _gaia(0.0), _tmass(0.2)
    t1["TYC2"] = MaskedColumn([0], mask=[True])
    t2["TYC2"] = [1821]
    _cones(monkeypatch, t1, t2)
    xm = _local()
    assert xm["TYC2"][0] == 1821 and "TYC2_1" in xm.colnames
    assert xm["_RAJ2000"][0] == pytest.approx(t2["_RAJ2000"][0])
    assert xm["_RAJ2000_1"][0] == pytest.approx(t1["_RAJ2000"][0])


def test_local_xmatch_clash_keeps_the_gaia_source_id(monkeypatch):
    t1, t2 = _gaia(0.0), _tmass(0.2)
    t2["Source"] = ["other"]
    _cones(monkeypatch, t1, t2)
    xm = _local()
    assert xm["Source"][0] == 1 and xm["Source_2"][0] == "other"


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


def test_unreachable_xmatch_goes_local_without_calling_it(monkeypatch):
    # astroquery's table-name check ignores XMatch.TIMEOUT, so a dead host
    # must be caught by the TCP probe before the client is ever called.
    monkeypatch.setattr(_cds, "_reachable", lambda *a, **k: False)
    local = _record_local(monkeypatch)
    client = _Client(Table({"angDist": [0.2]}))
    _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA, client=client)
    assert client.calls == 0 and len(local) == 1
    assert _cds.breaker.is_down(_cds.XMATCH_HOST)


def test_reachable_is_false_for_a_refused_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens there now
    assert REAL_REACHABLE(f"http://127.0.0.1:{port}/x", timeout=2) is False


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


# ── blank rows from the mirror's VOTables ────────────────────────────

def _tablelist(**tables):
    from astroquery.utils import commons
    return commons.TableList(tables)


def test_blank_mirror_row_is_dropped():
    t = Table(masked=True)
    t["Source"] = MaskedColumn([0, 4888833115509354496], mask=[True, False])
    t["pmRA"] = MaskedColumn([0.0, 28.72], mask=[True, False])
    t["Plx"] = MaskedColumn([0.0, np.nan], mask=[True, True])   # a real NULL
    t["2MASS"] = ["", ""]
    out = _cds._drop_blank_rows(_tablelist(**{"I/355/gaiadr3": t}))
    g = out["I/355/gaiadr3"]
    assert len(g) == 1 and g["Source"][0] == 4888833115509354496
    assert np.ma.is_masked(g["Plx"][0])                     # NULLs survive


def test_tables_without_blank_rows_or_numbers_are_untouched():
    real = Table({"RAJ2000": [1.0, 2.0], "Name": ["a", "b"]})
    text = Table({"Name": ["", ""]})
    out = _cds._drop_blank_rows(_tablelist(a=real, b=text))
    assert len(out["a"]) == 2 and len(out["b"]) == 2
    assert _cds._drop_blank_rows("unparsed raw text") == "unparsed raw text"


def test_blank_row_with_int_zero_is_dropped():
    # I/352 junk row: VOTable ints without a null value read an empty cell as 0.
    t = Table(masked=True)
    t["Source"] = MaskedColumn([0, 6668917117751909632], mask=[False, False])
    t["RA_ICRS"] = MaskedColumn([0.0, 304.118], mask=[True, False])
    t["rgeo"] = MaskedColumn([0.0, 147.54], mask=[True, False])
    t["Flag"] = MaskedColumn([0, 10033], mask=[True, False])
    out = _cds._drop_blank_rows(_tablelist(**{"I/352/gedr3dis": t}))
    g = out["I/352/gedr3dis"]
    assert len(g) == 1 and g["rgeo"][0] == 147.54


def test_a_real_row_with_a_zero_id_but_finite_floats_is_kept():
    t = Table({"Seq": [0, 1], "RAJ2000": [10.0, 11.0], "Vmag": [np.nan, 9.1]})
    out = _cds._drop_blank_rows(_tablelist(a=t))
    assert len(out["a"]) == 2
