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


class MirroredVizier(VizierClass):
    """astroquery Vizier that moves to a mirror when a host is down.

    Failover happens per request, so it is thread-safe and covers every query
    method (query_region, query_constraints, query_object, ...). An empty
    result is a valid answer and is never retried elsewhere.
    """

    mirrors = VIZIER_MIRRORS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Share astroquery's "Vizier" cache directory instead of starting an
        # empty "MirroredVizier" one.
        self.name = "Vizier"

    def _request(self, method, url, *args, **kwargs):
        parts = urlsplit(url)
        primary = parts.hostname
        if args or primary is None:
            return super()._request(method, url, *args, **kwargs)
        kwargs["timeout"] = connect_read(kwargs.get("timeout"))
        hosts = _order(primary, self.mirrors)
        last_exc, last_resp = None, None
        for i, host in enumerate(hosts):
            target = urlunsplit(parts._replace(netloc=host))
            try:
                resp = super()._request(method, target, **kwargs)
            except requests.RequestException as exc:
                breaker.trip(host, type(exc).__name__)
                last_exc = exc
                continue
            status = getattr(resp, "status_code", 200)
            if _is_down_status(status):
                breaker.trip(host, f"HTTP {status}")
                self._forget(method, target, kwargs)
                last_resp = resp
                continue
            breaker.reset(host)
            if i:
                logger.info("VizieR query served by mirror %s", host)
            return resp
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
    circular ``area``, with cat1's columns, then cat2's (renamed ``<name>_2``
    on a clash), and ``angDist`` in arcsec -- the layout CDS xMatch returns.
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
    for name in list(right.colnames):
        if name in left.colnames:
            right.rename_column(name, f"{name}_2")
    out = hstack([left, right], join_type="exact", metadata_conflicts="silent")
    out.add_column(Column(sep.to_value(u.arcsec), name="angDist"), index=0)
    return out


def xmatch(cat1, cat2, max_distance, *, area=None, client=None, timeout=60,
           **kwargs):
    """CDS xMatch, or the local reproduction when xMatch is down.

    ``client`` is the astroquery XMatch object to try first (pass the caller's
    module-level ``XMatch`` so tests that patch it keep working).
    """
    client = XMatch if client is None else client
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
