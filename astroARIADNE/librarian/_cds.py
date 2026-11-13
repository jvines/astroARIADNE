"""CDS services with failover: VizieR mirrors, a local xMatch, SIMBAD mirrors.

CDS (Strasbourg) serves VizieR, xMatch and SIMBAD from one site, so an outage
there takes out every catalogue at once. VizieR is mirrored in full elsewhere
(CfA by default) and SIMBAD at Harvard. xMatch has no mirror; it is reproduced
locally from two VizieR cone queries, which then go through the mirrors.

A host that fails is skipped for ``CDS_DOWN_SECONDS`` (default 600) so a dead
primary costs one connect timeout per process, not one per query.

Environment:
    CDS_VIZIER_MIRRORS   comma-separated VizieR hosts tried after the primary
                         (default: vizier.cfa.harvard.edu)
    CDS_SIMBAD_MIRRORS   comma-separated SIMBAD hosts (default: simbad.harvard.edu)
    CDS_DOWN_SECONDS     how long a failed host is skipped (default: 600)
"""

import logging
import os
import socket
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Column, Table, hstack
from astroquery.query import AstroQuery
from astroquery.simbad import Simbad, SimbadClass
from astroquery.vizier import VizierClass
from astroquery.xmatch import XMatch

logger = logging.getLogger(__name__)


def _hosts(env, default):
    raw = os.environ.get(env, default)
    return tuple(h.strip() for h in raw.split(",") if h.strip())


VIZIER_MIRRORS = _hosts("CDS_VIZIER_MIRRORS", "vizier.cfa.harvard.edu")
SIMBAD_MIRRORS = _hosts("CDS_SIMBAD_MIRRORS", "simbad.harvard.edu")
DOWN_SECONDS = float(os.environ.get("CDS_DOWN_SECONDS", "600"))
CONNECT_TIMEOUT = 10.0
XMATCH_HOST = urlsplit(XMatch.URL).hostname

# HTTP statuses that mean the service is broken rather than the query bad.
# CDS answers 404 on every VizieR path while it is down.
_DOWN_STATUS = {404, 408, 429}


def _is_down_status(code):
    return code in _DOWN_STATUS or code >= 500


def connect_read(timeout):
    """(connect, read) timeout: fail fast on a dead host, wait on a slow one."""
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
        return (min(CONNECT_TIMEOUT, float(timeout)), float(timeout))
    return timeout


class _Breaker:
    """Remembers which hosts failed recently, so they are tried last."""

    def __init__(self):
        self._until = {}
        self._lock = threading.Lock()

    def is_down(self, host):
        with self._lock:
            return time.monotonic() < self._until.get(host, 0.0)

    def trip(self, host, reason):
        with self._lock:
            was_up = time.monotonic() >= self._until.get(host, 0.0)
            self._until[host] = time.monotonic() + DOWN_SECONDS
        if was_up:
            logger.warning("%s unavailable (%s); skipping it for %.0f s",
                           host, reason, DOWN_SECONDS)

    def reset(self, host):
        with self._lock:
            self._until.pop(host, None)

    def clear(self):
        with self._lock:
            self._until.clear()


breaker = _Breaker()


def _order(primary, mirrors):
    """Primary first, then mirrors; hosts that failed recently go last."""
    hosts = [primary] + [m for m in mirrors if m != primary]
    return [h for h in hosts if not breaker.is_down(h)] + \
           [h for h in hosts if breaker.is_down(h)]


# A VizieR host whose database is down still answers HTTP 200, with an error
# INFO and no rows; astroquery parses that into an empty result, which would
# read as "not in the catalogue". Observed on the CfA mirror (2026-11-11), up
# to 4 of 6 identical queries failing this way. Under load (2026-11-13) it
# also answers "ERROR : temporary unavailable" / "(pipe) Request has been
# interrupted"; 184 of those were cached in one batch and replayed as empty.
# Permanent errors ("Table ... does not exist") are deliberately not markers:
# retrying them is useless and would trip the breaker on a healthy host.
_DB_ERROR_MARKERS = ("not currently reachable", "STDB_ERR", "connect error",
                     "temporary unavailable", "Request has been interrupted")
MIRROR_TRIES = int(os.environ.get("CDS_MIRROR_TRIES", "5"))


def _db_error(resp):
    """True for a 200 response that is a VizieR database error."""
    text = getattr(resp, "text", "")
    if not isinstance(text, str) or 'name="Error"' not in text:
        return False
    return any(m in text for m in _DB_ERROR_MARKERS)


def _empty_votable(resp):
    """True for a VOTable answer with no data rows.

    From the CfA mirror such an answer is not proof of absence: on 2026-11-13
    it answered a 3" cone on II/246/out around HIP 66924 with no table (no
    error, the same warnings as a real empty) and the same query minutes
    later with the 2MASS row, three times out of three. astroquery cached the
    false empty, so every rerun of the star lost 2MASS.
    """
    text = getattr(resp, "text", "")
    return isinstance(text, str) and "<VOTABLE" in text and "<TR>" not in text


def _malformed(resp):
    """True for a 200 VOTable body that is not well-formed XML.

    The CfA mirror intermittently serves corrupted VOTables (2026-11-13:
    "not well-formed (invalid token)" at a random line). astroquery fails to
    parse them, the query reads as failed, and the bad body stays in
    astroquery's request cache, replayed on every rerun.
    """
    content = getattr(resp, "content", None)
    if not isinstance(content, (bytes, bytearray)):
        text = getattr(resp, "text", "")
        content = text.encode("utf-8", "replace") if isinstance(text, str) else b""
    if b"<VOTABLE" not in content[:4096]:
        return False
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(content)
    except ET.ParseError:
        return True
    return False


def _empty_cells(col):
    """Per row: True where every element of the cell is masked, NaN or 0.

    Ints: a VOTable int without a null value reads an empty cell as 0.
    Floats: the II/246/out junk row (2026-11-13) carried RAJ2000 = 0.0 with
    every other number masked, so 0.0 counts as empty too. A real row always
    has some finite non-zero number (a coordinate at least).
    """
    data = np.ma.getdata(col)
    empty = np.ma.getmaskarray(col).copy()
    if data.dtype.kind in "fc":
        empty |= ~np.isfinite(data)
    empty |= data == 0
    return empty.reshape(len(col), -1).all(axis=1)


def _drop_blank_rows(result):
    """Remove rows with no value in any numeric column from a parsed answer.

    The CfA mirror (2026-11-13) intermittently returned VOTables with an extra
    first row: every float cell masked, int cells 0 (no null value declared),
    strings empty or a URL fragment (``https://vizier.idia.ac.za/...``).
    astroquery caches those answers, so they replay for a week. A single-row
    lookup read the junk row as THE answer: Gaia DR3 by Source lost PM,
    parallax and G; I/352 lost the Bailer-Jones distance; the ESA fallbacks
    never ran because a row was present. A real catalogue row always carries a
    finite float (coordinates at least), so tables without float columns are
    left alone.
    """
    if isinstance(result, (str, bytes)) or not hasattr(result, "keys"):
        return result
    for key in list(result.keys()):
        t = result[key]
        if not len(t) or not any(t[c].dtype.kind in "fc" for c in t.colnames):
            continue
        blank = np.ones(len(t), bool)
        for c in t.colnames:
            if t[c].dtype.kind in "iufc":
                blank &= _empty_cells(t[c])
        if blank.any():
            logger.info("VizieR %s: dropped %d blank row(s)", key, int(blank.sum()))
            t.remove_rows(np.nonzero(blank)[0])
    return result


def _all_columns(data):
    """The request body with any ``-out=<list>`` replaced by ``-out.all``.

    The CfA mirror returns corrupt rows for ``-out=*`` and explicit column
    lists (values shifted between columns, e.g. 2MASS J/H/Ks = 0.06/999999/
    0.011 for HD 209458, true 6.591/6.366/6.308) and correct ones for
    ``-out.all``. Every requested column is in ``-out.all``; computed columns
    (``-out.add``) and sorting are kept.
    """
    if not isinstance(data, str):
        return data
    lines = data.split("\n")
    kept = [ln for ln in lines if not ln.startswith("-out=")]
    if len(kept) == len(lines) or "-out.all" in kept:
        return "\n".join(kept)
    return "\n".join(kept[:1] + ["-out.all"] + kept[1:])


class MirroredVizier(VizierClass):
    """astroquery Vizier that moves to a mirror when a host is down.

    Failover happens per request, so it is thread-safe and covers every query
    method (query_region, query_constraints, query_object, ...). An empty
    result is a valid answer and is never retried elsewhere; a database error
    sent as an empty 200 is not, and is retried. Requests to a mirror always
    ask for ``-out.all`` (see ``_all_columns``). If every host fails the query
    raises, so callers take their own fallback instead of reading "no data".
    Blank rows are dropped from every result (see ``_drop_blank_rows``). An
    empty answer from a mirror is never cached and is asked once more (see
    ``_empty_votable``).
    """

    mirrors = VIZIER_MIRRORS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Share astroquery's "Vizier" cache directory instead of starting an
        # empty "MirroredVizier" one.
        self.name = "Vizier"

    def _parse_result(self, response, **kwargs):
        return _drop_blank_rows(super()._parse_result(response, **kwargs))

    def _request(self, method, url, *args, **kwargs):
        parts = urlsplit(url)
        primary = parts.hostname
        if args or primary is None:
            return super()._request(method, url, *args, **kwargs)
        kwargs["timeout"] = connect_read(kwargs.get("timeout"))
        hosts = _order(primary, self.mirrors)
        last_exc, last_resp, db_failed = None, None, None
        for host in hosts:
            target = urlunsplit(parts._replace(netloc=host))
            send = dict(kwargs)
            if host in self.mirrors:
                send["data"] = _all_columns(send.get("data"))
            tries = MIRROR_TRIES if host in self.mirrors else 1
            empty_retried = False
            for attempt in range(tries):
                try:
                    resp = super()._request(method, target, **send)
                except requests.RequestException as exc:
                    breaker.trip(host, type(exc).__name__)
                    last_exc = exc
                    break
                status = getattr(resp, "status_code", 200)
                if _is_down_status(status):
                    breaker.trip(host, f"HTTP {status}")
                    self._forget(method, target, send)
                    last_resp = resp
                    break
                if _db_error(resp) or _malformed(resp):
                    self._forget(method, target, send)
                    db_failed = resp
                    if attempt + 1 < tries:
                        time.sleep(0.5 * (attempt + 1))
                    continue
                if host in self.mirrors and _empty_votable(resp):
                    # A mirror's empty answer can be false, with nothing in it
                    # to tell it from a real one (see _empty_votable). Never
                    # keep it in the cache, where it would be replayed for
                    # every later query, and ask once more, live.
                    self._forget(method, target, send)
                    if not empty_retried and attempt + 1 < tries:
                        empty_retried = True
                        continue
                breaker.reset(host)
                if host != primary:
                    logger.info("VizieR query served by mirror %s", host)
                return resp
            else:
                breaker.trip(host, "database not reachable")
        if db_failed is not None:
            raise requests.HTTPError(
                "VizieR database not reachable (or its answer malformed) on any "
                "host", response=db_failed)
        if last_resp is not None:
            return last_resp
        raise last_exc

    def _forget(self, method, url, kwargs):
        """Drop a cached error response so it is not replayed for a week."""
        if not kwargs.get("cache"):
            return
        try:
            q = AstroQuery(method, url, params=kwargs.get("params"),
                           data=kwargs.get("data"),
                           headers=kwargs.get("headers"),
                           files=kwargs.get("files"), json=kwargs.get("json"))
            path = q.request_file(self.cache_location)
            if os.path.exists(path):
                os.remove(path)
        except Exception:  # cache housekeeping must never break a query
            pass


# ── xMatch ───────────────────────────────────────────────────────────

# Position columns used to pair rows, first numeric pair wins. Gaia's RA_ICRS
# is its epoch-2016 position, which is what CDS xMatch matches on; VizieR's
# computed _RAJ2000 for Gaia is propagated to epoch 2000, so it comes second.
# For every other catalogue _RAJ2000 is the catalogued position, in degrees
# even when the native columns are sexagesimal strings.
_POSITION_COLUMNS = (
    ("RA_ICRS", "DE_ICRS"),
    ("_RAJ2000", "_DEJ2000"),
    ("RAJ2000", "DEJ2000"),
    ("RA(ICRS)", "DE(ICRS)"),
    ("RAdeg", "DEdeg"),
)

# cat1 columns that keep their name on a clash: the Gaia DR3 ID.
_CAT1_KEEPS = ("Source",)


def _vizier_id(cat):
    cat = str(cat)
    if not cat.startswith("vizier:"):
        raise ValueError(f"local xMatch only handles VizieR tables, got {cat!r}")
    return cat[len("vizier:"):]


def _cone(catalog, center, radius, timeout):
    """Every column of ``catalog`` within ``radius``; None when nothing is there."""
    v = MirroredVizier(row_limit=-1, columns=["all", "_RAJ2000", "_DEJ2000"],
                       timeout=timeout)
    res = v.query_region(center, radius=radius, catalog=catalog)
    if res is None or len(res) == 0:
        return None
    key = catalog if catalog in res.keys() else res.keys()[0]
    tbl = res[key]
    if "_r" in tbl.colnames:
        tbl.remove_column("_r")
    return tbl


TIC_VIZIER = "IV/39/tic82"


def tic_from_vizier(center, radius, gaia_id, timeout=60):
    """A star's TIC-8 row from VizieR (IV/39/tic82), keyed like MAST's TIC.

    For when MAST's catalogue service fails: on 2026-11-13 one TIC query took
    401 s and the next failed after 90 s, while VizieR (through the mirror
    failover) served the same TIC-8 in 2 s. Matches on the Gaia ID as the
    MAST path does. A Gaia source can carry several TIC rows (DUPLICATE,
    SPLIT, ARTIFACT in ``Disp``); the row without a disposition wins. VizieR
    names MAST's ``ID`` ``TIC`` and its ``objType`` ``S/G``. Returns None when
    the star is not in the answer; raises when VizieR fails.
    """
    tbl = _cone(TIC_VIZIER, center, radius, timeout)
    if tbl is None or "GAIA" not in tbl.colnames:
        return None

    def val(row, col):
        if col not in tbl.colnames or np.ma.is_masked(row[col]):
            return None
        return row[col]

    gid = str(int(gaia_id))
    rows = [r for r in tbl
            if val(r, "GAIA") is not None and str(int(r["GAIA"])) == gid]
    if not rows:
        return None
    rows.sort(key=lambda r: bool(str(val(r, "Disp") or "").strip()))
    r = rows[0]
    s_g = val(r, "S/G")
    return {"ID": int(r["TIC"]), "GAIA": gid, "Tmag": val(r, "Tmag"),
            "e_Tmag": val(r, "e_Tmag"), "KIC": val(r, "KIC"),
            "objType": None if s_g is None else str(s_g).strip()}


def _coords(tbl):
    for ra, de in _POSITION_COLUMNS:
        if ra in tbl.colnames and de in tbl.colnames \
                and tbl[ra].dtype.kind in "fi" and tbl[de].dtype.kind in "fi":
            a = np.ma.filled(np.ma.asarray(tbl[ra], dtype=float), np.nan)
            d = np.ma.filled(np.ma.asarray(tbl[de], dtype=float), np.nan)
            good = np.isfinite(a) & np.isfinite(d)
            return SkyCoord(a[good], d[good], unit="deg"), np.flatnonzero(good)
    raise KeyError(f"no numeric position columns among {tbl.colnames}")


def local_xmatch(cat1, cat2, max_distance, *, area, timeout=60):
    """CDS xMatch reproduced from two VizieR cone queries.

    Returns every (cat1, cat2) pair closer than ``max_distance`` inside the
    circular ``area``, with cat1's columns, then cat2's, and ``angDist`` in
    arcsec -- the layout CDS xMatch returns. On a name clash the matched
    catalogue (cat2) keeps the name and cat1's column becomes ``<name>_1``:
    callers read the counterpart's IDs and magnitudes by their VizieR names
    (Gaia DR3 carries its own ``TYC2``, for one). The exception is cat1's
    ``Source`` (the Gaia DR3 ID every caller filters on), which keeps its name.
    An empty table means no counterpart; a failed query raises.
    """
    if area is None:
        raise ValueError("local xMatch needs a circular area")
    t1 = _cone(_vizier_id(cat1), area.center, area.radius, timeout)
    t2 = _cone(_vizier_id(cat2), area.center, area.radius, timeout)
    empty = Table([Column([], name="angDist", dtype=float)])
    if t1 is None or t2 is None or len(t1) == 0 or len(t2) == 0:
        return empty
    c1, rows1 = _coords(t1)
    c2, rows2 = _coords(t2)
    if len(c1) == 0 or len(c2) == 0:
        return empty
    i1, i2, sep, _ = search_around_sky(c1, c2, seplimit=max_distance)
    if len(i1) == 0:
        return empty
    left = t1[rows1[i1]]
    right = t2[rows2[i2]]
    for name in list(left.colnames):
        if name not in right.colnames:
            continue
        if name in _CAT1_KEEPS:
            right.rename_column(name, f"{name}_2")
        else:
            left.rename_column(name, f"{name}_1")
    out = hstack([left, right], join_type="exact", metadata_conflicts="silent")
    out.add_column(Column(sep.to_value(u.arcsec), name="angDist"), index=0)
    return out


def _reachable(url, timeout=CONNECT_TIMEOUT):
    """Can a TCP connection to ``url``'s host be opened within ``timeout``?

    astroquery's XMatch first validates the table names with a request that
    ignores ``XMatch.TIMEOUT``, so a dead host costs the OS connect timeout
    (75 s on macOS) -- longer than the callers' per-query budget, which then
    cancels the call before the local fallback can run.
    """
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def xmatch(cat1, cat2, max_distance, *, area=None, client=None, timeout=60,
           **kwargs):
    """CDS xMatch, or the local reproduction when xMatch is down.

    ``client`` is the astroquery XMatch object to try first (pass the caller's
    module-level ``XMatch`` so tests that patch it keep working).
    """
    client = XMatch if client is None else client
    if not breaker.is_down(XMATCH_HOST) and not _reachable(XMatch.URL):
        breaker.trip(XMATCH_HOST, "no TCP connection")
    if not breaker.is_down(XMATCH_HOST):
        try:
            res = client.query(cat1=cat1, cat2=cat2, max_distance=max_distance,
                               area=area, **kwargs)
            breaker.reset(XMATCH_HOST)
            return res
        except requests.RequestException as exc:
            breaker.trip(XMATCH_HOST, type(exc).__name__)
    logger.info("xMatch %s x %s reproduced from VizieR cone queries", cat1, cat2)
    return local_xmatch(cat1, cat2, max_distance, area=area, timeout=timeout)


# ── SIMBAD ───────────────────────────────────────────────────────────

# Errors that mean "bad query", not "service down"; these are never retried.
_SIMBAD_QUERY_ERRORS = (ValueError, TypeError, KeyError, AttributeError)


def simbad(method, *args, client=None, **kwargs):
    """Call ``Simbad.<method>``, moving to a SIMBAD mirror if the host is down."""
    client = Simbad if client is None else client
    primary = getattr(client, "server", None)
    last = None
    for host in _order(primary, SIMBAD_MIRRORS):
        if host == primary:
            c = client
        else:
            c = SimbadClass()
            c.server = host
            c.ROW_LIMIT = getattr(client, "ROW_LIMIT", c.ROW_LIMIT)
            try:
                c.columns_in_output = list(client.columns_in_output)
            except Exception:
                pass
        try:
            res = getattr(c, method)(*args, **kwargs)
        except _SIMBAD_QUERY_ERRORS:
            raise
        except Exception as exc:
            breaker.trip(host, type(exc).__name__)
            last = exc
            continue
        breaker.reset(host)
        return res
    raise last
