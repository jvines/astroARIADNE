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

3. **SPHEREx aperture contamination.** The SPHEREx super-bands are our own
   3-px (18.45") aperture photometry, so the Gaussian-beam model of check 2
   is replaced by the exact in-aperture fraction of each neighbour's ePSF at
   its separation, from a 40" Gaia cone, with Gaia RP as the flux proxy where
   available. Same thresholds. Bands measured by the crowded-field forced
   ePSF fit (``_spherex``, neighbours modelled) are recorded as 'deblended'
   and kept. Details at "Check 3" below.
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
#     IRAS/AKARI/Herschel/CASSIS/XP/IUE (spectro-photometry or arcminute
#     beams: a point-neighbour model does not apply).
#   * SPHEREx: our own aperture photometry, checked with the exact aperture
#     fraction instead (check 3, after the SPHEREx fetch).
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
    if "aperture" in rec:                      # SPHEREx (check 3)
        nb = "; ".join(f"Gaia {t[0]} at {t[1]:.1f}\" {t[4]}={t[2]:.2f} "
                       f"(c={t[3]:.3f})"
                       for t in rec["neighbours"] if t[3] > 1e-4)
        return (f"{band}: contaminating flux c = {rec['c']:.3f} inside the "
                f"{rec['aperture']:.1f}\" SPHEREx extraction aperture from "
                f"{nb or 'unresolved neighbours'}")
    nb = "; ".join(f"Gaia {s} at {sep:.1f}\" G={g:.2f} (c={ct:.3f})"
                   for s, sep, g, ct in rec["neighbours"] if ct > 1e-4)
    return (f"{band}: contaminating flux c = {rec['c']:.3f} within a "
            f"{rec['fwhm']:g}\" PSF from {nb or 'unresolved neighbours'}")


# ── Check 3: SPHEREx aperture contamination ──────────────────────────────────
# SPHEREx super-bands are our own 3-px aperture photometry (``_spherex``: r =
# 3 px = 18.45", 10-16 px median background annulus, no aperture correction).
# The aperture is known exactly, so a Gaia neighbour's leak into it is not a
# Gaussian-beam guess but the fraction of its PSF that falls inside that circle
# at its separation, and nothing in the extraction subtracts it.
#
#   c = sum_nb (F_nb / F_target) * f_ap(sep_nb) / f_ap(0)
#
# f_ap(d) is tabulated below per super-band from the mission's R7 ePSF products
# (cal-epsf-2026-191, detectors 1-6): every ePSF zone whose CWAVE lies inside the
# super-band window is averaged (PSF flux is linear, so averaging the stamps
# averages f_ap), the stamp is sampled at native pixel centres exactly as the
# extraction's subpixel-weighted aperture sees it (5x5 subpixel aperture
# weights), and the result is averaged over 16 subpixel phases of the target
# and 12 position angles of the neighbour. f_ap(0) ~ 0.99 (the target's own
# enclosed energy); f_ap(3.25 px) ~ 0.30; f_ap = 0 beyond 6.5 px (the ePSF
# stamp reaches 3.2 px from its centre), so SPHEREX_CONE_RADIUS_ARCSEC = 40"
# (6.5 px) holds every neighbour that can reach the aperture. The profile is
# nearly achromatic (band-to-band spread in f_ap at most 0.04, near 2.25 px).
#
# The background annulus (10-16 px) is a median over ~490 px: a point
# neighbour adds a handful of bright pixels to its tail and moves the median
# negligibly, and no neighbour inside the 40" cone reaches it (6.5 + 3.2 <
# 10 px), so it is not modelled.
#
# Flux proxy: Gaia RP when the target and the neighbour both have it (RP,
# 0.64-1.05 um, is the Gaia band nearest SPHEREx, and costs nothing extra in
# the cone query), else Gaia G. No colour term beyond that: a neighbour redder
# than the target is still under-estimated at 2-5 um.

SPHEREX_CONE_RADIUS_ARCSEC = 40.0   # aperture (18.45") + ePSF reach (3.2 px)
SPHEREX_PIX_ARCSEC = 6.15           # _spherex.PIX_ARCSEC
SPHEREX_AP_R_PX = 3.0               # _spherex.AP_R
SPHEREX_SEP_PX = tuple(0.25 * i for i in range(27))   # 0 .. 6.5 px

# f_ap(d) on SPHEREX_SEP_PX, per super-band (see the derivation above).
SPHEREX_APERTURE_FRACTION: dict[str, tuple[float, ...]] = {
    "SPHEREX_1050": (
        0.9918, 0.9915, 0.9906, 0.9890, 0.9863, 0.9815, 0.9734, 0.9565,
        0.9219, 0.8585, 0.7557, 0.6169, 0.4566, 0.3035, 0.1801, 0.0956,
        0.0475, 0.0231, 0.0121, 0.0069, 0.0043, 0.0029, 0.0019, 0.0012,
        0.0007, 0.0004, 0.0002,
    ),
    "SPHEREX_1130": (
        0.9911, 0.9908, 0.9900, 0.9884, 0.9860, 0.9818, 0.9748, 0.9594,
        0.9266, 0.8643, 0.7612, 0.6206, 0.4574, 0.3016, 0.1765, 0.0918,
        0.0445, 0.0213, 0.0113, 0.0066, 0.0043, 0.0030, 0.0021, 0.0013,
        0.0008, 0.0004, 0.0002,
    ),
    "SPHEREX_1400": (
        0.9900, 0.9896, 0.9887, 0.9870, 0.9845, 0.9801, 0.9730, 0.9574,
        0.9245, 0.8621, 0.7593, 0.6192, 0.4570, 0.3019, 0.1773, 0.0927,
        0.0454, 0.0221, 0.0120, 0.0071, 0.0047, 0.0033, 0.0023, 0.0014,
        0.0009, 0.0005, 0.0002,
    ),
    "SPHEREX_1850": (
        0.9870, 0.9865, 0.9855, 0.9836, 0.9809, 0.9761, 0.9691, 0.9537,
        0.9220, 0.8609, 0.7594, 0.6193, 0.4564, 0.3006, 0.1761, 0.0923,
        0.0460, 0.0232, 0.0133, 0.0083, 0.0056, 0.0041, 0.0029, 0.0018,
        0.0012, 0.0006, 0.0003,
    ),
    "SPHEREX_2350": (
        0.9850, 0.9845, 0.9833, 0.9810, 0.9777, 0.9719, 0.9634, 0.9459,
        0.9120, 0.8497, 0.7490, 0.6122, 0.4543, 0.3033, 0.1817, 0.0986,
        0.0514, 0.0271, 0.0158, 0.0098, 0.0066, 0.0048, 0.0033, 0.0021,
        0.0013, 0.0007, 0.0003,
    ),
    "SPHEREX_2750": (
        0.9917, 0.9913, 0.9898, 0.9873, 0.9829, 0.9755, 0.9631, 0.9403,
        0.8995, 0.8322, 0.7310, 0.6001, 0.4524, 0.3111, 0.1948, 0.1117,
        0.0607, 0.0320, 0.0174, 0.0097, 0.0057, 0.0036, 0.0022, 0.0012,
        0.0007, 0.0003, 0.0001,
    ),
    "SPHEREX_3050": (
        0.9918, 0.9913, 0.9898, 0.9870, 0.9822, 0.9742, 0.9608, 0.9370,
        0.8953, 0.8274, 0.7267, 0.5973, 0.4517, 0.3124, 0.1973, 0.1146,
        0.0632, 0.0339, 0.0186, 0.0104, 0.0061, 0.0037, 0.0022, 0.0012,
        0.0007, 0.0003, 0.0001,
    ),
    "SPHEREX_3350": (
        0.9920, 0.9915, 0.9897, 0.9867, 0.9816, 0.9730, 0.9591, 0.9347,
        0.8924, 0.8244, 0.7240, 0.5956, 0.4511, 0.3131, 0.1989, 0.1165,
        0.0649, 0.0352, 0.0195, 0.0110, 0.0064, 0.0038, 0.0023, 0.0012,
        0.0007, 0.0003, 0.0001,
    ),
    "SPHEREX_3700": (
        0.9920, 0.9914, 0.9896, 0.9864, 0.9811, 0.9724, 0.9583, 0.9338,
        0.8915, 0.8235, 0.7232, 0.5950, 0.4510, 0.3133, 0.1992, 0.1169,
        0.0654, 0.0357, 0.0199, 0.0113, 0.0066, 0.0040, 0.0023, 0.0013,
        0.0007, 0.0003, 0.0001,
    ),
    "SPHEREX_4050": (
        0.9909, 0.9904, 0.9887, 0.9859, 0.9809, 0.9730, 0.9599, 0.9366,
        0.8954, 0.8280, 0.7272, 0.5976, 0.4515, 0.3120, 0.1967, 0.1140,
        0.0628, 0.0338, 0.0189, 0.0108, 0.0065, 0.0040, 0.0024, 0.0014,
        0.0008, 0.0003, 0.0001,
    ),
    "SPHEREX_4300": (
        0.9921, 0.9916, 0.9898, 0.9870, 0.9819, 0.9737, 0.9602, 0.9366,
        0.8947, 0.8269, 0.7260, 0.5968, 0.4515, 0.3126, 0.1977, 0.1150,
        0.0635, 0.0342, 0.0189, 0.0107, 0.0063, 0.0038, 0.0022, 0.0012,
        0.0006, 0.0003, 0.0001,
    ),
}
# Super-bands without their own row (e.g. withheld ones) use the band mean.
_SPHEREX_FRACTION_MEAN = tuple(
    float(np.mean(col)) for col in zip(*SPHEREX_APERTURE_FRACTION.values()))


def is_spherex_band(band: str) -> bool:
    return band.upper().startswith("SPHEREX_")


def spherex_aperture_fraction(band: str, sep_arcsec: float) -> float:
    """Fraction of a point source's flux inside the 3-px SPHEREx extraction
    aperture when it lies ``sep_arcsec`` from the aperture centre."""
    prof = SPHEREX_APERTURE_FRACTION.get(band, _SPHEREX_FRACTION_MEAN)
    d = float(sep_arcsec) / SPHEREX_PIX_ARCSEC
    if not np.isfinite(d) or d >= SPHEREX_SEP_PX[-1]:
        return 0.0
    return float(np.interp(d, SPHEREX_SEP_PX, prof))


def spherex_neighbours_from_table(tbl, target_id, *, ra0=None, dec0=None,
                                  max_sep=SPHEREX_CONE_RADIUS_ARCSEC):
    """Split a Gaia cone into ``(target, neighbours)`` for the SPHEREx check.

    ``target`` is ``(G, RP)`` of the target row (NaN where missing) or None
    when the target is absent from the cone; ``neighbours`` is a list of
    ``(source_id, sep_arcsec, G, RP)`` within ``max_sep``. ``phot_rp_mean_mag``
    is optional (all-NaN when the column is missing)."""
    cols = {c.lower(): c for c in tbl.colnames}

    def _mag(name):
        if name not in cols:
            return np.full(len(tbl), np.nan)
        return np.ma.filled(np.ma.asarray(tbl[cols[name]], dtype=float),
                            np.nan)

    sid = np.asarray(tbl[cols["source_id"]]).astype("int64")
    ra = np.asarray(tbl[cols["ra"]], dtype=float)
    dec = np.asarray(tbl[cols["dec"]], dtype=float)
    g, rp = _mag("phot_g_mean_mag"), _mag("phot_rp_mean_mag")
    is_t = sid == int(target_id)
    if not is_t.any():
        return None, []
    i = int(np.argmax(is_t))
    ra0, dec0 = float(ra[i]), float(dec[i])
    dx = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    dy = (dec - dec0) * 3600.0
    sep = np.hypot(dx, dy)
    out = [(int(sid[k]), float(sep[k]), float(g[k]), float(rp[k]))
           for k in range(len(sid))
           if not is_t[k] and 0.0 < sep[k] <= max_sep
           and (np.isfinite(g[k]) or np.isfinite(rp[k]))]
    return (float(g[i]), float(rp[i])), out


def _flux_ratio(target, nb_g, nb_rp):
    """(ratio, proxy) of a neighbour to the target: RP when both have it, else
    G; (None, None) when neither pair is finite."""
    t_g, t_rp = target
    if np.isfinite(t_rp) and np.isfinite(nb_rp):
        return 10.0 ** (-0.4 * (nb_rp - t_rp)), "RP"
    if np.isfinite(t_g) and np.isfinite(nb_g):
        return 10.0 ** (-0.4 * (nb_g - t_g)), "G"
    return None, None


def spherex_contamination(band, target, neighbours):
    """``(c, terms)`` for one SPHEREx super-band: the neighbours' flux inside
    the extraction aperture over the target's own.

    ``target`` = ``(G, RP)``; ``neighbours`` = ``[(source_id, sep, G, RP)]``.
    ``terms`` = ``[(source_id, sep, proxy_mag, contribution, proxy)]`` sorted
    by contribution."""
    f0 = spherex_aperture_fraction(band, 0.0)
    terms = []
    for sid, sep, g, rp in neighbours:
        ratio, proxy = _flux_ratio(target, g, rp)
        if ratio is None or sep <= 0.0:
            continue
        frac = spherex_aperture_fraction(band, sep)
        if frac <= 0.0:
            continue
        mag = rp if proxy == "RP" else g
        terms.append((sid, float(sep), float(mag), float(ratio * frac / f0),
                      proxy))
    terms.sort(key=lambda t: -t[3])
    return float(sum(t[3] for t in terms)), terms


def assess_spherex_bands(bands, target, neighbours, *,
                         flag=CONTAMINATION_FLAG,
                         exclude=CONTAMINATION_EXCLUDE):
    """Contamination record for each SPHEREx super-band in ``bands``:
    ``{band: {"c", "status", "aperture", "neighbours"}}`` with ``aperture``
    the extraction radius in arcsec and ``neighbours`` the (up to 3) dominant
    ``(source_id, sep, proxy_mag, contribution, proxy)``."""
    out = {}
    if target is None:
        return out
    ap = SPHEREX_AP_R_PX * SPHEREX_PIX_ARCSEC
    for band in bands:
        if not is_spherex_band(band):
            continue
        c, terms = spherex_contamination(band, target, neighbours)
        out[band] = {"c": c, "status": classify(c, flag, exclude),
                     "aperture": ap, "neighbours": terms[:3]}
    return out


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
_LABEL = {"exclude": "EXCLUDED", "flag": "FLAGGED",
          "deblended": "DEBLENDED (kept)"}


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

    def _gaia_cone_adql(self, radius=CONE_RADIUS_ARCSEC,
                        columns="source_id, ra, dec, phot_g_mean_mag",
                        top=1000):
        ra = self._gaia_ra if self._gaia_ra is not None else self._coord.ra.deg
        dec = (self._gaia_dec if self._gaia_dec is not None
               else self._coord.dec.deg)
        return (f"SELECT TOP {int(top)} {columns} "
                "FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
                "POINT('ICRS', ra, dec), CIRCLE('ICRS', "
                f"{float(ra):.8f}, {float(dec):.8f}, "
                f"{radius / 3600.0:.8f}))")

    # -- check 3 ---------------------------------------------------------

    def _spherex_cone_adql(self):
        return self._gaia_cone_adql(
            radius=SPHEREX_CONE_RADIUS_ARCSEC,
            columns="source_id, ra, dec, phot_g_mean_mag, phot_rp_mean_mag",
            top=5000)

    def _apply_spherex_contamination_qc(self, bands, method="aperture"):
        """Estimate c for the SPHEREx super-bands just added; flag or remove
        the contaminated ones. Returns the set of removed bands.

        ``method`` is the extraction's (``_spherex`` meta): with ``'psf'`` the
        crowded-field forced ePSF fit already modelled the catalogued
        neighbours, so a band over the thresholds is recorded as 'deblended'
        and kept. Degrades to a warning (all kept) when ESA is unreachable."""
        if not self._contamination_qc or self._gaia_id is None:
            return set()
        bands = [b for b in bands
                 if is_spherex_band(b) and b in self._magnitudes]
        if not bands:
            return set()
        tbl = self._run_adql(self._spherex_cone_adql())
        if tbl is None:
            _log.warning(
                "Gaia neighbour cone unavailable (ESA archive unreachable): "
                "SPHEREx aperture contamination of %d band(s) NOT checked; "
                "all kept.", len(bands))
            return set()
        target, nbrs = spherex_neighbours_from_table(tbl, self._gaia_id)
        if target is None or not np.isfinite(target).any():
            _log.warning(
                "Gaia DR3 %s has no G/RP magnitude or is absent from the "
                "%.0f\" cone: SPHEREx contamination NOT checked; all kept.",
                self._gaia_id, SPHEREX_CONE_RADIUS_ARCSEC)
            return set()
        recs = assess_spherex_bands(bands, target, nbrs,
                                    flag=self._contamination_flag,
                                    exclude=self._contamination_exclude)
        removed = set()
        for band, rec in recs.items():
            if method == "psf" and rec["status"] != "ok":
                rec["status"] = "deblended"
                _log.info("SPHEREx %s; measured by forced ePSF photometry "
                          "with the neighbours fitted, kept.",
                          describe(band, rec))
            elif rec["status"] == "exclude":
                self._magnitudes.pop(band, None)
                removed.add(band)
                _log.warning("EXCLUDED %s (c > %g). Set contamination_qc="
                             "False or raise contamination_exclude to keep "
                             "it.", describe(band, rec),
                             self._contamination_exclude)
            elif rec["status"] == "flag":
                _log.warning("FLAGGED %s (c > %g), kept.",
                             describe(band, rec), self._contamination_flag)
        self._contamination = {**self._contamination, **recs}
        return removed

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
