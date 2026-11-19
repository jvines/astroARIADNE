"""Blend / contamination QC from Gaia DR3 (issue #55).

Two independent checks, both pure functions here so they are unit-testable
without any network access; ``_api.Librarian`` supplies the archive data.

1. **Cross-match uniqueness** (Contardo & Hogg 2024, Sec. 2). The Gaia DR3
   best-neighbour tables carry ``number_of_mates`` (how many *other* Gaia
   sources have the same external object as their best neighbour) and
   ``number_of_neighbours`` (how many external candidates lie within the
   position errors). ``number_of_mates > 0`` means the 2MASS/AllWISE object is
   an unresolved blend of several Gaia sources and its photometry is not the
   target's: the catalogue is dropped. ``number_of_neighbours > 1`` means the
   match itself is ambiguous: kept but flagged.

2. **Unresolved-neighbour flux contamination** (Contardo & Hogg 2024,
   Sec. 3.1, "Crowding"). Gaia resolves neighbours down to ~1", so a Gaia
   source inside a catalogue's PSF leaks flux into that catalogue's
   measurement. For every retrieved band we estimate the fractional
   contaminating flux

       c = sum_nb 10**(-0.4 (G_nb - G_target)) * exp(-sep**2 / (2 sigma**2)),
       sigma = FWHM_band / 2.3548,

   using Gaia G as the flux proxy (no colour term: the neighbour's SED is
   unknown, so a G -> band conversion would be fragile modelling; for red
   bands this under-estimates the contamination of a cool neighbour and
   over-estimates that of a hot one). The target's own source_id is excluded.
   ``c > CONTAMINATION_FLAG`` flags the band, ``c > CONTAMINATION_EXCLUDE``
   removes it from the fit (c = 0.05 is a ~0.05 mag bias).

   This complements, and does not replace, the catalogue-native flags the
   librarian already applies (2MASS ``Cflg``, AllWISE ``ccf``/``nb``).
"""
from __future__ import annotations

import math

import numpy as np

# ── Defaults (module-level so a user can change them in one place) ──────────

CONTAMINATION_FLAG = 0.01      # c above this: keep the band, record + warn
CONTAMINATION_EXCLUDE = 0.05   # c above this: drop the band from the fit
CONE_RADIUS_ARCSEC = 20.0      # Gaia cone: 3.9 sigma of the widest beam (W4)

# Catalogues whose bands are dropped when number_of_mates > 0.
MATES_DROP_CATALOGS = ("2MASS", "Wise")

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))   # 1/2.3548

# ── Per-band PSF FWHM (arcsec) ───────────────────────────────────────────────
# Keyed by pyphot band name. Any band NOT listed is skipped by the check:
#   * Gaia (G/BP/RP): neighbours beyond ~1" are resolved, closer ones are the
#     astrometric/RUWE problem, not something a flux ratio can fix.
#   * TESS (21" pixels; the TIC already carries a contamination ratio),
#     Tycho-2 / Stromgren / Mermilliod (photoelectric, aperture >> 10"),
#     IRAS/AKARI/Herschel/CASSIS/SPHEREx/XP/IUE (spectro-photometry or
#     arcminute beams: a point-neighbour model does not apply).
# Values are typical survey PSF FWHM, not per-source measurements.
PSF_FWHM_ARCSEC: dict[str, float] = {
    # 2MASS: ~2.5-3" seeing-limited, Cutri+03
    "2MASS_J": 2.5, "2MASS_H": 2.5, "2MASS_Ks": 2.5,
    # WISE: 6.1 / 6.4 / 6.5 / 12.0 " in W1-W4 (Wright+10)
    "WISE_RSR_W1": 6.1, "WISE_RSR_W2": 6.4,
    "WISE_RSR_W3": 6.5, "WISE_RSR_W4": 12.0,
    # Spitzer/IRAC 3.6, 4.5 um: ~1.7" (Fazio+04)
    "SPITZER_IRAC_36": 1.7, "SPITZER_IRAC_45": 1.7,
    # GALEX FUV/NUV: 4.2" / 5.3" (Morrissey+07); FUV rounded up to 4.5
    "GALEX_FUV": 4.5, "GALEX_NUV": 5.3,
    # Pan-STARRS1 median seeing grizy ~1.1-1.3" (Magnier+16)
    "PS1_g": 1.3, "PS1_r": 1.2, "PS1_i": 1.1, "PS1_z": 1.1, "PS1_y": 1.0,
    # SDSS median seeing ~1.4" (r band)
    "SDSS_u": 1.5, "SDSS_g": 1.4, "SDSS_r": 1.4, "SDSS_i": 1.4, "SDSS_z": 1.4,
    # SkyMapper: median seeing ~2-2.5"
    "SkyMapper_u": 2.5, "SkyMapper_v": 2.5, "SkyMapper_g": 2.2,
    "SkyMapper_r": 2.2, "SkyMapper_i": 2.0, "SkyMapper_z": 2.0,
    # APASS: 2.6" pixels, PSF ~5"
    "GROUND_JOHNSON_V": 5.0, "GROUND_JOHNSON_B": 5.0,
}


# (catalogue, band) overrides of the band-only table above.
PSF_FWHM_BY_CATALOGUE: dict[tuple[str, str], float] = {
    ("APASS", "SDSS_g"): 5.0, ("APASS", "SDSS_r"): 5.0,
    ("APASS", "SDSS_i"): 5.0,
}


def band_fwhm(pyphot_name: str, catalogue: str | None = None) -> float | None:
    """PSF FWHM (arcsec) for a band, or None when the band is not checked.

    The lookup is keyed on (catalogue, band) first, then on the band alone.
    SDSS_g/r/i double as APASS bands in the librarian (first write wins), so
    an APASS-sourced SDSS_g is checked with the APASS PSF.
    """
    if catalogue is not None:
        v = PSF_FWHM_BY_CATALOGUE.get((catalogue, pyphot_name))
        if v is not None:
            return v
    return PSF_FWHM_ARCSEC.get(pyphot_name)


# ── Check 2: flux contamination ──────────────────────────────────────────────

def neighbours_from_table(tbl, target_id, *, ra0=None, dec0=None):
    """Split a Gaia cone table into ``(g_target, neighbours)``.

    ``tbl`` needs ``source_id, ra, dec, phot_g_mean_mag`` (any case).
    ``neighbours`` is a list of ``(source_id, sep_arcsec, G)`` excluding the
    target and any source without a G magnitude or at zero separation
    (a duplicate of the target would otherwise contaminate it with c = 1).
    Separations are measured from the target's own row; ``ra0, dec0`` are the
    fallback when the target is not in the table, in which case ``g_target``
    is None and the caller must skip the check.
    """
    cols = {c.lower(): c for c in tbl.colnames}
    sid = np.asarray(tbl[cols["source_id"]]).astype("int64")
    ra = np.asarray(tbl[cols["ra"]], dtype=float)
    dec = np.asarray(tbl[cols["dec"]], dtype=float)
    g = np.ma.filled(np.ma.asarray(tbl[cols["phot_g_mean_mag"]], dtype=float),
                     np.nan)
    is_t = sid == int(target_id)
    if is_t.any():
        i = int(np.argmax(is_t))
        ra0, dec0, g_t = float(ra[i]), float(dec[i]), float(g[i])
        g_t = g_t if np.isfinite(g_t) else None
    else:
        g_t = None
        if ra0 is None or dec0 is None:
            return None, []
    # small-angle offset is fine at <= 20"
    dx = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    dy = (dec - dec0) * 3600.0
    sep = np.hypot(dx, dy)
    out = [(int(sid[k]), float(sep[k]), float(g[k]))
           for k in range(len(sid))
           if not is_t[k] and sep[k] > 0.0 and np.isfinite(g[k])]
    return g_t, out


def contamination_fraction(g_target, neighbours, fwhm):
    """Fractional contaminating flux ``c`` and the per-neighbour terms.

    Returns ``(c, terms)`` with ``terms`` a list of
    ``(source_id, sep_arcsec, G, contribution)`` sorted by contribution.
    """
    sigma = float(fwhm) * _FWHM_TO_SIGMA
    terms = []
    for sid, sep, g in neighbours:
        if sep <= 0.0 or not np.isfinite(g):
            continue
        contrib = (10.0 ** (-0.4 * (g - g_target))
                   * math.exp(-sep ** 2 / (2.0 * sigma ** 2)))
        terms.append((sid, float(sep), float(g), float(contrib)))
    terms.sort(key=lambda t: -t[3])
    return float(sum(t[3] for t in terms)), terms


def classify(c, flag=CONTAMINATION_FLAG, exclude=CONTAMINATION_EXCLUDE):
    """'exclude' if c > exclude, 'flag' if c > flag, else 'ok'."""
    if c > exclude:
        return "exclude"
    if c > flag:
        return "flag"
    return "ok"


def assess_bands(bands, g_target, neighbours, *, sources=None,
                 flag=CONTAMINATION_FLAG, exclude=CONTAMINATION_EXCLUDE):
    """Contamination record for every band that has a PSF entry.

    Returns ``{band: {"c", "status", "fwhm", "neighbours"}}`` where
    ``neighbours`` holds the (up to 3) dominant contributors as
    ``(source_id, sep_arcsec, G, contribution)``. Bands without a PSF entry
    are absent from the result.
    """
    out = {}
    if g_target is None:
        return out
    for band in bands:
        fwhm = band_fwhm(band, (sources or {}).get(band))
        if fwhm is None:
            continue
        c, terms = contamination_fraction(g_target, neighbours, fwhm)
        out[band] = {"c": c, "status": classify(c, flag, exclude),
                     "fwhm": fwhm, "neighbours": terms[:3]}
    return out


def describe(band, rec):
    """One actionable line for a flagged / excluded band."""
    nb = "; ".join(f"Gaia {s} at {sep:.1f}\" G={g:.2f} (c={ct:.3f})"
                   for s, sep, g, ct in rec["neighbours"] if ct > 1e-4)
    return (f"{band}: contaminating flux c = {rec['c']:.3f} within a "
            f"{rec['fwhm']:g}\" PSF from {nb or 'unresolved neighbours'}")


# ── Check 1: cross-match uniqueness ──────────────────────────────────────────

def crossmatch_status(number_of_mates, number_of_neighbours):
    """'drop' (blend of several Gaia sources), 'flag' (ambiguous match), 'ok'.

    ``None`` / masked counts are treated as unknown, i.e. 'ok'.
    """
    def _n(v):
        try:
            if v is None or np.ma.is_masked(v):
                return 0
            return int(v)
        except (TypeError, ValueError):
            return 0

    if _n(number_of_mates) > 0:
        return "drop"
    if _n(number_of_neighbours) > 1:
        return "flag"
    return "ok"


# ── Librarian mixin ──────────────────────────────────────────────────────────

import logging

_log = logging.getLogger(__name__)

# Gaia best-neighbour tables that carry ``number_of_mates``
# (tycho2tdsc_merge_best_neighbour has only ``number_of_neighbours``).
_NO_MATES_COLUMN = ("TYCHO2",)
_LABEL = {"exclude": "EXCLUDED", "flag": "FLAGGED"}


def xmatch_select(cat_name):
    """SELECT list for a best-neighbour query, with the uniqueness columns."""
    cols = "xmatch.original_ext_source_id, xmatch.number_of_neighbours"
    if cat_name not in _NO_MATES_COLUMN:
        cols += ", xmatch.number_of_mates"
    return cols


class ContaminationMixin:
    """Cross-match uniqueness and PSF-contamination QC for the Librarian.

    The host class provides ``_run_adql(adql) -> Table | None`` (None = the
    ESA archive failed or is marked down), ``_gaia_id``, ``_coord``,
    ``_gaia_ra/_gaia_dec`` (Gaia-epoch position or None), ``_magnitudes`` and
    ``_contamination_qc / _contamination_flag / _contamination_exclude``.
    """

    # Class-level defaults so a Librarian built without __init__ (tests, cache
    # restores) still works; the mutable ones are only ever replaced, never
    # mutated in place.
    _contamination_qc = True
    _contamination_flag = CONTAMINATION_FLAG
    _contamination_exclude = CONTAMINATION_EXCLUDE
    _contamination: dict = {}
    _xmatch_flags: dict = {}
    _mates_dropped: frozenset = frozenset()
    _band_src: dict = {}   # band -> catalogue, where the PSF depends on it
    _gaia_ra = None
    _gaia_dec = None

    def _init_contamination(self, qc, flag, exclude):
        if not 0 <= flag <= exclude:
            raise ValueError("need 0 <= contamination_flag <= "
                             f"contamination_exclude, got {flag}, {exclude}")
        self._contamination_qc = bool(qc)
        self._contamination_flag = float(flag)
        self._contamination_exclude = float(exclude)
        self._contamination: dict = {}
        self._xmatch_flags: dict = {}
        self._mates_dropped = frozenset()

    # -- check 1 ---------------------------------------------------------

    def _note_crossmatch(self, cat_name, result):
        """Record uniqueness info from a best-neighbour row set and decide.

        ``result`` is the (non-empty) astropy Table of the best-neighbour
        query. A catalogue in ``MATES_DROP_CATALOGS`` with
        ``number_of_mates > 0`` is added to ``self._mates_dropped``.
        """
        if not self._contamination_qc or len(result) == 0:
            return
        names = result.colnames
        mates = result["number_of_mates"][0] if "number_of_mates" in names else None
        nbrs = (result["number_of_neighbours"][0]
                if "number_of_neighbours" in names else None)
        status = crossmatch_status(mates, nbrs)
        if mates is None and nbrs is None:
            return
        info = {"number_of_mates": None if mates is None else int(mates),
                "number_of_neighbours": None if nbrs is None else int(nbrs),
                "status": status}
        self._xmatch_flags = {**self._xmatch_flags, cat_name: info}
        if status == "drop" and cat_name in MATES_DROP_CATALOGS:
            self._mates_dropped = frozenset(self._mates_dropped | {cat_name})
            _log.warning(
                "%s: the %s counterpart of Gaia DR3 %s is shared with %d other "
                "Gaia source(s) (number_of_mates > 0): it is an unresolved "
                "blend, so its bands are dropped. Pass contamination_qc=False "
                "to keep them.", cat_name, cat_name, self._gaia_id, int(mates))
        elif status == "flag":
            _log.warning(
                "%s: %d catalogue objects lie within the Gaia position errors "
                "of Gaia DR3 %s (number_of_neighbours > 1); the match may be "
                "wrong. Bands kept, check them in star.xmatch_flags.",
                cat_name, int(nbrs), self._gaia_id)
        elif status == "drop":
            _log.info("%s: number_of_mates=%d for Gaia DR3 %s (blend); "
                         "recorded, not dropped for this catalogue.",
                         cat_name, int(mates), self._gaia_id)

    # -- check 2 ---------------------------------------------------------

    def _gaia_cone_adql(self):
        ra = self._gaia_ra if self._gaia_ra is not None else self._coord.ra.deg
        dec = (self._gaia_dec if self._gaia_dec is not None
               else self._coord.dec.deg)
        return ("SELECT TOP 1000 source_id, ra, dec, phot_g_mean_mag "
                "FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
                "POINT('ICRS', ra, dec), CIRCLE('ICRS', "
                f"{float(ra):.8f}, {float(dec):.8f}, "
                f"{CONE_RADIUS_ARCSEC / 3600.0:.8f}))")

    def _apply_contamination_qc(self):
        """Estimate c per retrieved band; flag or remove the contaminated ones.

        Fills ``self._contamination``; removes bands with c above the exclude
        threshold from ``self._magnitudes``. Degrades to a warning (all bands
        kept, nothing recorded) when the ESA archive cannot be reached.
        """
        if not self._contamination_qc or self._gaia_id is None:
            return
        checkable = [b for b in self._magnitudes
                     if band_fwhm(b, self._band_src.get(b)) is not None]
        if not checkable:
            return
        tbl = self._run_adql(self._gaia_cone_adql())
        if tbl is None:
            _log.warning(
                "Gaia neighbour cone unavailable (ESA archive unreachable): "
                "flux contamination of %d band(s) NOT checked; all kept.",
                len(checkable))
            return
        g_t, nbrs = neighbours_from_table(
            tbl, self._gaia_id, ra0=self._gaia_ra, dec0=self._gaia_dec)
        if g_t is None:
            _log.warning(
                "Gaia DR3 %s has no G magnitude or is absent from the %.0f\" "
                "cone: flux contamination NOT checked; all bands kept.",
                self._gaia_id, CONE_RADIUS_ARCSEC)
            return
        recs = assess_bands(checkable, g_t, nbrs, sources=self._band_src,
                            flag=self._contamination_flag,
                            exclude=self._contamination_exclude)
        self._contamination = recs
        for band, rec in recs.items():
            if rec["status"] == "exclude":
                self._magnitudes.pop(band, None)
                _log.warning("EXCLUDED %s (c > %g). Set contamination_qc=False "
                             "or raise contamination_exclude to keep it.",
                             describe(band, rec), self._contamination_exclude)
            elif rec["status"] == "flag":
                _log.warning("FLAGGED %s (c > %g), kept.",
                             describe(band, rec), self._contamination_flag)

    def _contamination_summary_lines(self):
        """Lines for the retrieval summary: dropped catalogues, excluded and
        flagged bands, each with its reason."""
        lines = []
        for cat in sorted(self._mates_dropped):
            info = self._xmatch_flags.get(cat, {})
            lines.append(f"DROPPED {cat}: shared with "
                         f"{info.get('number_of_mates')} other Gaia source(s)")
        for band, rec in self._contamination.items():
            if rec["status"] != "ok":
                lines.append(f"{_LABEL[rec['status']]} {describe(band, rec)}")
        return lines
