"""SPHEREx QR2 per-star spectro-photometry for the librarian.

Extracts a 0.75-5 um spectrum from the public QR2 weekly spectral images
(one LVF wavelength per overlapping exposure; ~150-230 clean exposures per
star) and bins it into ARIADNE's SPHEREX_* super-bands. Validated against
2MASS/WISE on GJ 1214 and TRAPPIST-1 (band ratios 0.94-1.07; see
notebooks/spherex_extract.py for the standalone validation tool).

Method per exposure: 3-px aperture + 10-16 px background annulus on the IBE
cutout (subpixel-weighted, numpy-only -- the SPHEREx PSF holds 99.6% of its
energy inside 3 px, zone spread 0.2%, so no aperture correction applies);
MJy/sr -> uJy via the 6.2" pixel solid angle; wavelength from the
science-grade per-detector CWAVE array at the star's RAW detector pixel (NOT
the L2 WCS-WAVE table, which the mission designates visualization-only -- see
_cwave). Exposures with any bad flag inside the aperture are dropped (bits
1 = OVERFLOW per the official SDT guidance and 21 = benign frame-wide marker
are ignored), then a 4-sigma MAD clip per wavelength bin.

Binned results are cached per source id under
~/.astroARIADNE/spherex/<CACHE_VERSION>/ so a star is only ever extracted once.
"""
import logging
import os
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

OMEGA = (6.2 * 4.84813681e-6) ** 2          # sr per 6.2" SPHEREx pixel
AP_R, ANN_IN, ANN_OUT = 3.0, 10.0, 16.0
IGNORE_BITS = (1, 21)
C_UM = 2.99792458e14
# Bumped whenever the extraction changes NUMERICALLY, so two solutions can never
# be mixed inside one corpus. v1 (unversioned, ~/.astroARIADNE/spherex/*.csv) was
# written with the celestial SIP leaking into the wavelength lookup (see
# _alt_wcs): its lambda column is biased ~0.005 um blueward. v2 fixed that but
# still used the visualization-grade WCS-WAVE table; v3 takes lambda from the
# science-grade per-detector CWAVE product (_cwave) -- bump again if
# CAL_WCS_VERSION changes, it sets the wavelengths. Old versions are left in
# place -- ignored, not deleted -- so their results stay reproducible.
CACHE_VERSION = 'v3'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'spherex',
                         CACHE_VERSION)
# Withheld until the raw grid reconvolution covers them at warm nodes (the
# cache-derived columns only reach 4.63 um for most grids).
WITHHELD = ('SPHEREX_4650', 'SPHEREX_4900')
# Usable point-source window (Vega Ks): saturation flags begin ~9 Vega,
# all-sky 5-sigma depth ~16.5 (mission paper, 2 um).
KS_BRIGHT, KS_FAINT = 8.5, 16.5
MIN_EXP_PER_BAND = 3
ABS_CAL_FLOOR = 0.02                        # fractional


def _superband_windows():
    """SPHEREX_* -> (lo_um, hi_um) from the filter library (single source of
    truth: custom_filters.h5 via phot_utils)."""
    from ..phot_utils import _CUSTOM_LIB
    out = {}
    for name, filt in _CUSTOM_LIB.items():
        if not name.startswith('SPHEREX_') or name in WITHHELD:
            continue
        wl = filt.wavelength.to('um').value
        tr = np.asarray(filt.transmit, float)
        on = wl[tr > 0.5]
        out[name] = (float(on.min()), float(on.max()))
    return out


def _aperture_sum(img, x, y, r, oversamp=5):
    """Subpixel-weighted circular aperture sum (numpy only). Accurate to
    <<1% at r=3 px with 5x oversampling; photutils-free by design (the
    ariadne env pins numpy)."""
    ny, nx = img.shape
    x0, x1 = int(max(0, np.floor(x - r - 1))), int(min(nx, np.ceil(x + r + 2)))
    y0, y1 = int(max(0, np.floor(y - r - 1))), int(min(ny, np.ceil(y + r + 2)))
    sub = (np.arange(oversamp) + 0.5) / oversamp - 0.5
    xs = np.arange(x0, x1)[:, None] + sub[None, :]      # (npx, os)
    ys = np.arange(y0, y1)[:, None] + sub[None, :]
    dx2 = (xs - x) ** 2                                  # (npx, os)
    dy2 = (ys - y) ** 2
    inside = (dy2[:, None, :, None] + dx2[None, :, None, :]) <= r * r
    frac = inside.mean(axis=(2, 3))                      # (npy, npx)
    return float(np.sum(img[y0:y1, x0:x1] * frac)), frac.sum()


def _annulus_median(img, x, y, r_in, r_out):
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - x, yy - y)
    # finite-only: a single NaN pixel (detector edge / unobserved region)
    # otherwise poisons np.median and silently drops the whole exposure --
    # this killed the SPHEREX_2750/3050 super-bands (the 2MASS-WISE gap)
    # for SCR 1845 despite 4 good exposures per window.
    m = (rr >= r_in) & (rr <= r_out) & np.isfinite(img)
    if m.sum() < 10:
        return np.nan
    return float(np.median(img[m]))


# SPHEREx L2 headers carry SIP coefficients for the CELESTIAL solution (CTYPE
# 'RA---TAN-SIP') alongside two lettered solutions we need: 'A' ("0-based active
# pixel coordinates" -- cutout pixel -> raw detector pixel) and 'W' (the
# visualization-grade WAVE-TAB lookup). astropy's SIP reader is not
# alternate-key aware -- it reads the unkeyed A_/B_ coefficients whatever `key`
# is asked for -- so building EITHER off the same header picks the celestial
# distortion up and (a) logs a long "Inconsistent SIP distortion" record per
# exposure (astropy.wcs, via log.info: a log record, not a warning, so no
# warnings filter silences it; ~150-230 exposures per star) and (b) applies the
# polynomial to a pixel frame it has no business touching. Both solutions are
# defined on RAW detector pixels -- where the photon physically lands -- so the
# distortion must not be applied: on real QR2 cutouts the leak displaced the
# wavelength by a median -0.005 um (max 0.014), ~11% of a SPHEREx resolution
# element, systematically blueward.
# This is IRSA's own prescription -- "the wavelength per pixel is a property of
# the detector-filter combination and is independent of optical distortion in
# the telescope [...] so we turn the SIP distortion off for this WCS"
# (irsa-tutorials spherex-intro / spherex-cutouts, which do
# `spectral_wcs.sip = None`); the open astropy issue is #13105. We drop the
# keywords from a header copy rather than nulling `.sip` afterwards: the INFO is
# logged inside WCS.__init__, so post-hoc nulling fixes the numbers but still
# prints, and silencing it would mean muting astropy's logger.
_SIP_KW = re.compile(r'^(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)$')


def _alt_wcs(hdr, hdul, key):
    """An alternate-key WCS off a SPHEREx header, with the celestial SIP
    stripped. Both lettered solutions we use need this -- 'W' (WAVE-TAB) and
    'A' ("0-based active pixel coordinates")."""
    from astropy.wcs import WCS
    clean = hdr.copy()
    # Dedup the keywords first: `del header[kw]` drops EVERY card with that name
    # and then raises KeyError, so a duplicated SIP card would blow up on the
    # second pass -- and _fetch_one's bare except would swallow it, costing the
    # star every exposure with no log line. (astropy's own _read_sip_kw dedups
    # through a set for exactly this reason.)
    for k in {m.group() for m in map(_SIP_KW.match, list(clean)) if m}:
        del clean[k]
    return WCS(clean, fobj=hdul, key=key)


def _raw_pixel(hdr, hdul, x, y):
    """Cutout pixel -> 0-based RAW detector pixel, via the header's own 'A'
    solution (WCSNAMEA = "0-based active pixel coordinates"). IRSA re-references
    it to the cutout origin, so this is the mission's declared mapping onto the
    2040^2 detector grid the calibration arrays are indexed on."""
    return (float(v) for v in
            _alt_wcs(hdr, hdul, 'A').wcs_pix2world(np.array([[x, y]]), 0)[0])


# --------------------------------------------------------------------------- #
# Science-grade wavelength: the per-detector CWAVE calibration array
# --------------------------------------------------------------------------- #
# The WCS-WAVE table carried by each L2 file is explicitly NOT for science:
# "The fidelity of the WCS-WAVE lookup table is intended for visualization
# purposes. For science analysis, use the CWAVE and CBAND extensions in the
# Spectral WCS calibration product" (IRSA SPHEREx data-products page; ExpSupp
# v1.6 4.3.1). So we index CWAVE at the star's raw detector pixel instead.
# One 33 MB product per detector, six detectors, fetched once and reused.
CAL_WCS_VERSION = 'cal-wcs-v4-2025-254'
CAL_DIR = os.path.join(os.path.expanduser('~'), '.astroARIADNE', 'spherex',
                       'cal', CAL_WCS_VERSION)
_CAL_NAME = 'spectral_wcs_D{det}_spx_' + CAL_WCS_VERSION + '.fits'
_CAL_URL = ('https://irsa.ipac.caltech.edu/ibe/data/spherex/qr2/spectral_wcs/'
            + CAL_WCS_VERSION + '/{det}/' + _CAL_NAME)
_CAL_TIMEOUT = 300      # s, 33 MB


class CalibrationUnavailable(RuntimeError):
    """The detector's CWAVE product could not be fetched.

    Raised rather than swallowed: falling back to WAVE-TAB would silently mix
    two wavelength solutions in one corpus, and returning "no exposures" would
    cache an empty result and mark the star as having no SPHEREx data forever.
    """


_CAL_LOCKS = {}
_CAL_LOCKS_GUARD = None


def _cal_lock(det):
    """One lock per detector, so N workers hitting a cold cache download the
    33 MB product once rather than N times."""
    global _CAL_LOCKS_GUARD
    import threading
    if _CAL_LOCKS_GUARD is None:
        _CAL_LOCKS_GUARD = threading.Lock()
    with _CAL_LOCKS_GUARD:
        return _CAL_LOCKS.setdefault(det, threading.Lock())


@lru_cache(maxsize=None)
def _cwave(det):
    """Per-pixel central wavelength (um) for one detector, 2040x2040.

    Cached on disk under CAL_DIR and in memory per process. The download is
    serialised per detector and staged through a per-writer temp file, then
    renamed: concurrent first touches cost one download, a reader never sees a
    partial product, and a second writer cannot pull the temp file out from
    under a first (which it could when the temp name was keyed on the pid
    alone, since threads share it).
    """
    from astropy.io import fits
    import threading
    det = int(det)
    path = os.path.join(CAL_DIR, _CAL_NAME.format(det=det))
    if not os.path.exists(path):
        with _cal_lock(det):
            if not os.path.exists(path):             # another worker may have won
                os.makedirs(CAL_DIR, exist_ok=True)
                tmp = '%s.%d.%d.part' % (path, os.getpid(), threading.get_ident())
                try:
                    with urllib.request.urlopen(_CAL_URL.format(det=det),
                                                timeout=_CAL_TIMEOUT) as r, \
                            open(tmp, 'wb') as fh:
                        shutil.copyfileobj(r, fh)
                    os.replace(tmp, path)
                except Exception as e:               # noqa: BLE001
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise CalibrationUnavailable(
                        'SPHEREx spectral-WCS product for detector %d (%s): %s: %s'
                        % (det, CAL_WCS_VERSION, type(e).__name__, e)) from e
    try:
        with fits.open(path) as h:
            return np.asarray(h['CWAVE'].data, np.float32)
    except Exception as e:                               # noqa: BLE001
        raise CalibrationUnavailable(
            'unreadable CWAVE product %s: %s: %s'
            % (path, type(e).__name__, e)) from e


def _bilinear(a, x, y):
    """a[y, x] at fractional 0-based indices, or NaN off the grid.

    Half a pixel of overhang is clamped rather than rejected: the first and last
    pixel CENTRES sit at 0 and n-1, so a source legitimately on an edge row can
    land a hair outside (a star on detector row 0 comes back as -2e-12), and
    NaN there would silently drop the exposure."""
    ny, nx = a.shape
    if not (-0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5):
        return float('nan')
    x = min(max(x, 0.0), nx - 1.0)
    y = min(max(y, 0.0), ny - 1.0)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    fx, fy = x - x0, y - y0
    # CWAVE ships float32; weight in float64 so the result does not depend on
    # the array's storage dtype.
    c00, c01 = float(a[y0, x0]), float(a[y0, x1])
    c10, c11 = float(a[y1, x0]), float(a[y1, x1])
    return (c00 * (1 - fx) * (1 - fy) + c01 * fx * (1 - fy) +
            c10 * (1 - fx) * fy + c11 * fx * fy)


def _measure_arrays(img, var, flg, x, y, lam):
    """The photometry itself, on pixels already in hand.

    Split out of `_measure_cutout` so the local-file path and the ranged
    remote path (`_measure_remote`) run the SAME science code -- the two
    are asserted bit-identical in the tests. `x`/`y` are the star's position
    within the given window; `lam` its CWAVE wavelength.
    """
    ny, nx = img.shape
    if not (ANN_OUT <= x <= nx - ANN_OUT - 1 and
            ANN_OUT <= y <= ny - ANN_OUT - 1):
        return None
    ignore = sum(1 << b for b in IGNORE_BITS)
    for b in range(32):
        if b not in IGNORE_BITS and (flg >> b & 1).mean() > 0.99:
            ignore |= 1 << b                 # auto-detected universal marker
    yy, xx = np.mgrid[0:ny, 0:nx]
    near = np.hypot(xx - x, yy - y) <= AP_R + 0.5
    if ((flg & ~ignore)[near] != 0).any():
        return None
    bkg = _annulus_median(img, x, y, ANN_IN, ANN_OUT)
    if not np.isfinite(bkg):
        return None
    s, area = _aperture_sum(img, x, y, AP_R)
    v, _ = _aperture_sum(var, x, y, AP_R)
    f = (s - bkg * area) * OMEGA * 1e12      # MJy/sr -> uJy
    e = np.sqrt(max(v, 0.0)) * OMEGA * 1e12
    if not (np.isfinite(f) and f > 0):
        return None
    if not np.isfinite(lam):
        # A star off the calibrated grid yields NaN rather than an exception. A
        # NaN lambda used to be cached and then matched no super-band and no MAD
        # bin -- a silent drop. Reject it so the exposure is accounted for.
        return None
    return lam, f, e


def _star_in_window(hdr, hdul, ra, dec):
    """`(x, y, xa, ya, det)`: the star's position in the window this header
    describes, and on the raw detector grid."""
    from astropy.wcs import WCS
    x, y = (float(v) for v in WCS(hdr).world_to_pixel_values(ra, dec))
    xa, ya = _raw_pixel(hdr, hdul, x, y)
    return x, y, xa, ya, int(hdr["DETECTOR"])


def _measure_cutout(path, ra, dec):
    """One exposure -> (wavelength_um, flux_uJy, err_uJy) or None.

    The local-file path: reads a whole cutout off disk. Retained as the
    reference implementation the ranged path is validated against, and as the
    fallback when `USE_RANGED_READS` is off."""
    from astropy.io import fits
    with fits.open(path) as h:
        img = np.asarray(h["IMAGE"].data, float)
        var = np.asarray(h["VARIANCE"].data, float)
        flg = np.asarray(h["FLAGS"].data, int)
        hdr = h["IMAGE"].header
        x, y, xa, ya, det = _star_in_window(hdr, h, ra, dec)
    return _measure_arrays(img, var, flg, x, y, _bilinear(_cwave(det), xa, ya))


# Network timeouts: a single hung IRSA request must never block the star (and
# hence a whole bulk chunk -- the pool waits for ALL jobs to finish).
_CUTOUT_TIMEOUT = 60    # s, per cutout download (urlopen connect + each read)
_SIA_TIMEOUT = 120      # s, IRSA SIA query


# ── transport: ranged reads of the static L2 file ───────────────────────────
# The IRSA cutout service hands back a 5.09 MB MEF of which 4.93 MB is a PSF
# cube we never open, and it is the fragile part of the system: measured 503
# rates of 0% at 2 concurrent requests but 37.5% at 16, against 0% for the
# static file at the same concurrency. So we read the static L2 file directly
# and pull only the ~41 detector rows the aperture and annulus touch: 1.0 MB and
# 2 round trips per exposure instead of 5.09 MB, measured 4.1x faster and
# bit-identical (see tests).
#
# Two hosts serve the identical bytes, and the SIA access_url maps to the S3 key
# by a pure prefix swap. They differ in ONE way that matters: IRSA honours
# multi-range requests (206 multipart/byteranges), S3 silently IGNORES them and
# returns the WHOLE 71.6 MB object -- 14x worse than the cutout it replaces. So
# the multi-range shape is per-host, never assumed.
_L2_PREFIX = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
_L2_HOSTS = (
    ("irsa.ipac.caltech.edu", "/ibe/data/spherex/", True),
    ("nasa-irsa-spherex.s3.us-east-1.amazonaws.com", "/", False),
)
_BLOCK = 2880           # FITS block
_WIN = 39               # window side, = the 240 arcsec cutout at 6.15"/px
_FITS_DTYPE = {-64: ">f8", -32: ">f4", 64: ">i8", 32: ">i4", 16: ">i2", 8: "u1"}
_RANGE_TIMEOUT = 60
_RANGE_RETRIES = 3      # a swallowed 503 is a silently lost exposure


class _Reader:
    """Keep-alive ranged reader over one L2 host, with retry and host failover.

    NOT thread-safe (http.client connections are not): the pool gives each
    worker thread its own. Reuse matters -- a fresh connection costs a measured
    ~0.30 s of TCP+TLS, which at 2 requests per exposure would dominate.
    """

    def __init__(self, host_index=0):
        self._i = host_index
        self._conn = None

    @property
    def multirange(self):
        return _L2_HOSTS[self._i][2]

    def _connect(self):
        import http.client
        host = _L2_HOSTS[self._i][0]
        self._conn = http.client.HTTPSConnection(host, timeout=_RANGE_TIMEOUT)

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:                            # noqa: BLE001
                pass
            self._conn = None

    def get(self, key, ranges):
        """`ranges` = [(start, end), ...] inclusive. Returns [bytes, ...].

        Falls back to sequential single-range GETs on a host that does not do
        multipart, and to the next host after exhausting retries here.
        """
        import time
        last = None
        for host_try in range(len(_L2_HOSTS)):
            path = _L2_HOSTS[self._i][1] + key
            for attempt in range(_RANGE_RETRIES):
                try:
                    if self.multirange and len(ranges) > 1:
                        return self._multi(path, ranges)
                    return [self._one(path, a, b) for a, b in ranges]
                except Exception as e:                   # noqa: BLE001
                    last = e
                    self.close()
                    if attempt + 1 < _RANGE_RETRIES:
                        time.sleep(0.4 * (attempt + 1))
            self._i = (self._i + 1) % len(_L2_HOSTS)     # try the other host
            self.close()
        raise last

    def _request(self, path, hdr):
        if self._conn is None:
            self._connect()
        self._conn.request("GET", path, headers=hdr)
        r = self._conn.getresponse()
        body = r.read()
        if r.status not in (200, 206):
            raise OSError(f"HTTP {r.status} for {path}")
        return r, body

    def _one(self, path, a, b):
        r, body = self._request(path, {"Range": f"bytes={a}-{b}"})
        if r.status == 200 and len(body) > (b - a + 1) * 2:
            # the host ignored Range and sent the whole object
            raise OSError("Range ignored by host")
        return body

    def _multi(self, path, ranges):
        spec = ",".join(f"{a}-{b}" for a, b in ranges)
        r, body = self._request(path, {"Range": f"bytes={spec}"})
        ctype = r.getheader("Content-Type", "")
        if "multipart" not in ctype:
            raise OSError("multi-range not honoured")
        parts = _split_multipart(body, ctype)
        # Key each part by the offset in its OWN Content-Range: servers are not
        # required to return the parts in the order requested.
        return [parts[a] for a, _ in ranges]


def _split_multipart(body, ctype):
    """`{start_offset: payload}` from a multipart/byteranges body."""
    bound = ("--" + ctype.split("boundary=")[1].strip()).encode()
    out = {}
    for part in body.split(bound):
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep or b"Content-Range" not in head:
            continue
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-range"):
                start = int(line.split(b"bytes")[1].strip().split(b"-")[0])
                out[start] = data[:-2] if data.endswith(b"\r\n") else data
    return out


def _data_span(h):
    """Bytes of data (padded to whole FITS blocks) following a header."""
    if h.get("NAXIS", 0) == 0:
        return 0
    n = abs(h["BITPIX"]) // 8
    for i in range(1, h["NAXIS"] + 1):
        n *= h[f"NAXIS{i}"]
    n += h.get("PCOUNT", 0)
    return ((n + _BLOCK - 1) // _BLOCK) * _BLOCK


def _header_at(reader, key, off, nblk=12):
    """Parse the FITS header starting at `off`; returns (Header, data_offset)."""
    from astropy.io import fits
    raw = b""
    while True:
        want = nblk * _BLOCK
        raw += reader.get(key, [(off + len(raw), off + len(raw) + want - 1)])[0]
        for i in range(0, len(raw), 80):
            if raw[i:i + 8] == b"END     ":
                n = ((i + 80 + _BLOCK - 1) // _BLOCK) * _BLOCK
                return fits.Header.fromstring(raw[:n].decode("ascii")), off + n
        if len(raw) > 400 * _BLOCK:
            raise OSError(f"no END card in {key}")


# The HDU order is PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE -- note
# FLAGS BEFORE VARIANCE. Offsets are stable per (processing version, detector)
# but NOT across them: an l2b-v24 IMAGE header runs one 2880-byte block longer
# than an l2b-v20 one, and it varies detector to detector within a version. So
# the layout is cached per variant and re-validated on every use against the
# IMAGE header we have to read anyway; a mismatch just re-walks.
_LAYOUTS = {}
_LAYOUT_LOCK = None


def _variant(key):
    parts = key.split("/")
    return tuple(parts[3:5]) if len(parts) > 4 else (key,)


def _hdu_offsets(reader, key):
    """`{EXTNAME: data_offset}` for IMAGE / FLAGS / VARIANCE, plus the IMAGE
    header. One round trip when the variant's layout is already known."""
    global _LAYOUT_LOCK
    if _LAYOUT_LOCK is None:
        import threading
        _LAYOUT_LOCK = threading.Lock()
    v = _variant(key)
    with _LAYOUT_LOCK:
        known = _LAYOUTS.get(v)
    if known is not None:
        ih, iend = _header_at(reader, key, known["image_hdr"])
        if iend == known["IMAGE"][0]:             # layout still valid
            return ih, known
    prim, doff = _header_at(reader, key, 0)
    image_hdr = doff + _data_span(prim)
    ih, ioff = _header_at(reader, key, image_hdr)
    # (data offset, BITPIX): the pixel type is read from the file, never assumed
    # -- QR2 ships float32/int32 today, and a silently mis-cast slab would be
    # numerically wrong rather than an error.
    offs = {"image_hdr": image_hdr, "IMAGE": (ioff, ih["BITPIX"])}
    o = ioff + _data_span(ih)
    for _ in range(4):
        hh, dd = _header_at(reader, key, o)
        offs[hh.get("EXTNAME", "?")] = (dd, hh["BITPIX"])
        o = dd + _data_span(hh)
        if {"FLAGS", "VARIANCE"} <= set(offs):
            break
    with _LAYOUT_LOCK:
        _LAYOUTS[v] = offs
    return ih, offs


# The 11 super-bands cover 1.795 um of the 0.75-5 um range, so most exposures
# fall between windows and are discarded by superband_mags after being paid for.
# Skipping them changes results only through the MAD clip, whose 0.15 um bins can
# straddle a window edge: measured over 509 real cached stars, 1.86% of band
# slots move at all and 0.60% by more than 5 mmag -- inside the 2% (~22 mmag)
# absolute-calibration floor. Set ARIADNE_SPHEREX_ALL_BANDS=1 to keep them.
SKIP_OUT_OF_BAND = not os.environ.get("ARIADNE_SPHEREX_ALL_BANDS")


@lru_cache(maxsize=1)
def _superband_edges():
    return tuple(sorted(_superband_windows().values()))


def _in_any_superband(lam):
    if not np.isfinite(lam):
        return False
    return any(lo <= lam <= hi for lo, hi in _superband_edges())


def l2_key(access_url):
    """SIA `access_url` -> the path both L2 hosts serve it under."""
    return access_url[len(_L2_PREFIX):] if access_url.startswith(_L2_PREFIX) \
        else access_url


def _measure_remote(reader, key, ra, dec):
    """One exposure by ranged reads -- bit-identical to `_measure_cutout`.

    Two round trips: the IMAGE header (needed for this exposure's pointing), then
    one bundled request for the three row slabs the aperture and annulus touch.
    """
    ih, offs = _hdu_offsets(reader, key)
    nx, ny = ih["NAXIS1"], ih["NAXIS2"]
    x, y, xa, ya, det = _star_in_window(ih, None, ra, dec)
    # Crop to the SAME window the cutout service would have returned. It is not
    # cosmetic: _measure_arrays derives the universal-flag marker from the
    # fraction of flagged pixels in its window, so handing it a full-width slab
    # would change `ignore` and hence the reject decisions.
    x0, y0 = int(round(xa)) - _WIN // 2, int(round(ya)) - _WIN // 2
    if not (0 <= x0 and x0 + _WIN <= nx and 0 <= y0 and y0 + _WIN <= ny):
        return None                     # the edge guard, before fetching pixels
    lam = _bilinear(_cwave(det), xa, ya)
    if SKIP_OUT_OF_BAND and not _in_any_superband(lam):
        # 65% of exposures land between the super-band windows: they are
        # measured, cached, and then silently discarded by superband_mags. The
        # header we just read gives the EXACT wavelength, so we can drop them
        # before paying for the megabyte of pixels -- no footprint-based
        # prediction, no guard band, no chance of skipping a usable exposure.
        return None
    planes = ("IMAGE", "FLAGS", "VARIANCE")
    spans, dtypes = [], []
    for name in planes:
        off, bitpix = offs[name]
        itemsize = abs(bitpix) // 8
        row = nx * itemsize
        spans.append((off + y0 * row, off + (y0 + _WIN) * row - 1))
        dtypes.append(_FITS_DTYPE[bitpix])
    bufs = reader.get(key, spans)

    def arr(buf, dt):
        a = np.frombuffer(buf, dtype=dt).reshape(_WIN, nx)
        return np.asarray(a[:, x0:x0 + _WIN], float)

    img, flg, var = (arr(b, d) for b, d in zip(bufs, dtypes))
    return _measure_arrays(img, var, flg.astype(int), xa - x0, ya - y0, lam)


# Ranged reads are the default; set ARIADNE_SPHEREX_CUTOUTS=1 to fall back to the
# IRSA cutout service (the two are asserted bit-identical, so this is an escape
# hatch for a bad day at the archive, not a science switch).
USE_RANGED_READS = not os.environ.get("ARIADNE_SPHEREX_CUTOUTS")


class _ReaderPool:
    """A fixed set of keep-alive readers, leased one per in-flight exposure.

    Explicitly pooled rather than thread-local: `extract_spectrum` builds a
    fresh ThreadPoolExecutor for every star, so thread-local connections would
    be abandoned (and their sockets left to the garbage collector) 17,839 times
    over a catalog run.
    """

    def __init__(self, n):
        import queue
        self._q = queue.LifoQueue()
        self._all = [_Reader() for _ in range(max(1, n))]
        for r in self._all:
            self._q.put(r)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for r in self._all:
            r.close()

    def lease(self):
        return _Lease(self._q)


class _Lease:
    def __init__(self, q):
        self._q = q

    def __enter__(self):
        self._r = self._q.get()
        return self._r

    def __exit__(self, *exc):
        self._q.put(self._r)
        return False


def _fetch_one(args):
    url, ra, dec, tmpdir, i = args[:5]
    pool = args[5] if len(args) > 5 else None
    if USE_RANGED_READS:
        try:
            if pool is None:
                with _ReaderPool(1) as p, p.lease() as rd:
                    return _measure_remote(rd, l2_key(url), ra, dec)
            with pool.lease() as rd:
                return _measure_remote(rd, l2_key(url), ra, dec)
        except CalibrationUnavailable:
            raise
        except Exception as e:                           # noqa: BLE001
            # Both hosts and every retry failed. Say so: a swallowed failure is
            # indistinguishable from a legitimate reject and quietly shrinks the
            # star's exposure count.
            logger.warning("SPHEREx: ranged read failed for %s (%s: %s)",
                           url.rsplit("/", 1)[-1], type(e).__name__, e)
            return None
    path = os.path.join(tmpdir, f"spx_{i:04d}.fits")
    try:
        req = f"{url}?center={ra},{dec}d&size=240arcsec"
        for attempt in range(_RANGE_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=_CUTOUT_TIMEOUT) as resp, \
                        open(path, "wb") as fh:
                    fh.write(resp.read())
                break
            except Exception:                            # noqa: BLE001
                # The cutout service returns 503 under concurrency (0% at 2
                # concurrent, 37.5% at 16). Retrying is the difference between a
                # slow exposure and a silently missing one.
                if attempt + 1 == _RANGE_RETRIES:
                    raise
                import time
                time.sleep(0.4 * (attempt + 1))
        return _measure_cutout(path, ra, dec)
    except CalibrationUnavailable:
        raise            # never degrade to "this star has no SPHEREx data"
    except Exception as e:                               # noqa: BLE001
        logger.warning("SPHEREx: cutout failed for %s (%s: %s)",
                       url.rsplit("/", 1)[-1], type(e).__name__, e)
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def extract_spectrum(ra, dec, source_id, threads=4):
    """Per-exposure (lam, flux_uJy, err_uJy) arrays for a PM-propagated
    position; cached per source_id."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{source_id}.csv')
    if os.path.exists(cache):
        arr = np.loadtxt(cache, delimiter=',', ndmin=2)
        if arr.size:
            return arr[:, 0], arr[:, 1], arr[:, 2]
        return np.array([]), np.array([]), np.array([])

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.irsa import Irsa
    Irsa.TIMEOUT = _SIA_TIMEOUT      # bound the SIA query (no indefinite hang)
    res = Irsa.query_sia(
        pos=(SkyCoord(ra=ra * u.deg, dec=dec * u.deg), 10 * u.arcsec),
        collection='spherex_qr2')
    urls = [str(u_) for u_ in res['access_url']]
    logger.info('SPHEREx: %d overlapping QR2 planes', len(urls))
    import contextlib
    import tempfile
    rows = []
    # One keep-alive reader per worker, closed with the pool.
    pool_cm = _ReaderPool(threads) if USE_RANGED_READS else contextlib.nullcontext()
    with tempfile.TemporaryDirectory(prefix='spherex_') as tmpdir, pool_cm as pool:
        jobs = [(u_, ra, dec, tmpdir, i, pool) for i, u_ in enumerate(urls)]
        with ThreadPoolExecutor(threads) as ex:
            for r in ex.map(_fetch_one, jobs):
                if r is not None:
                    rows.append(r)
    if not rows:
        np.savetxt(cache, np.empty((0, 3)), delimiter=',')
        return np.array([]), np.array([]), np.array([])
    arr = np.array(rows)
    # 4-sigma MAD clip per 0.15 um bin (junk exposures the flags miss)
    lam, f, e = arr.T
    keep = np.ones(len(lam), bool)
    for lo in np.arange(0.7, 5.1, 0.15):
        m = (lam >= lo) & (lam < lo + 0.15)
        if m.sum() >= 4:
            med = np.median(f[m])
            sig = 1.4826 * np.median(np.abs(f[m] - med))
            keep[m] &= np.abs(f[m] - med) < 4 * max(sig, 1e-3 * med)
    arr = arr[keep]
    np.savetxt(cache, arr, delimiter=',')
    return arr[:, 0], arr[:, 1], arr[:, 2]


def superband_mags(lam, f, e):
    """Bin per-exposure fluxes into SPHEREX_* Vega mags.

    Returns {band: (mag, mag_err, n_exposures)}. Median flux per window;
    error = MAD-scatter/sqrt(n) with a 2% absolute-calibration floor.
    """
    from ..phot_utils import flux_to_mag
    out = {}
    for name, (lo, hi) in sorted(_superband_windows().items()):
        m = (lam >= lo) & (lam <= hi)
        n = int(m.sum())
        if n < MIN_EXP_PER_BAND:
            continue
        fmu = float(np.median(f[m]))
        if not (np.isfinite(fmu) and fmu > 0):
            continue
        scatter = 1.4826 * float(np.median(np.abs(f[m] - fmu)))
        fer = max(scatter / np.sqrt(n), ABS_CAL_FLOOR * fmu)
        lam_c = 0.5 * (lo + hi)
        flam = fmu * 1e-29 * C_UM / lam_c ** 2          # erg/cm2/s/um
        flam_e = fer * 1e-29 * C_UM / lam_c ** 2
        mag, mag_e = flux_to_mag(flam, flam_e, name)
        if np.isfinite(mag) and np.isfinite(mag_e):
            out[name] = (float(mag), float(mag_e), n)
    return out
