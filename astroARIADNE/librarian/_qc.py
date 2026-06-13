"""Per-catalogue quality-control predicates.

Each function returns True if the row passes that catalogue's standard
quality cuts (photometric flags, contamination/confusion, point-source
likelihood, artefact masks). Pulled out of the original monolithic
librarian.py so the QC logic is editable and testable in isolation.
"""

import numpy as np
import numpy.ma as ma


def _col(row, col):
    """Extract value from an astropy Row. Returns None if missing or masked."""
    try:
        v = row[col]
    except (KeyError, IndexError):
        return None
    if ma.is_masked(v):
        return None
    return v


def _qc_mag(mag, err, max_err=1.0):
    """Universal magnitude quality check."""
    if mag is None or ma.is_masked(mag):
        return False
    try:
        if np.isnan(float(mag)):
            return False
    except (TypeError, ValueError):
        return False
    if err is not None and not ma.is_masked(err):
        try:
            if float(err) > max_err:
                return False
        except (TypeError, ValueError):
            pass
    return True


def _qc_2mass_band(row, band_idx):
    """QC for a single 2MASS band (J=0, H=1, Ks=2).

    Checks Qflg (photometric quality) and Cflg (contamination/confusion).
    """
    qflg = str(_col(row, "Qflg") or "UUU")
    cflg = str(_col(row, "Cflg") or "999")
    if band_idx >= len(qflg) or band_idx >= len(cflg):
        return False
    return qflg[band_idx] in "ABCD" and cflg[band_idx] == "0"


def _qc_wise_band(row, band_idx):
    """QC for a single WISE band (W1=0, W2=1)."""
    qph = str(_col(row, "qph") or "UU")
    if band_idx >= len(qph):
        return False
    return qph[band_idx] in "ABC"


def _qc_wise_extended(row):
    """WISE extended-source flag (AllWISE ext_flg / ``ex``). Reject only
    genuinely extended morphologies (ext_flg >= 2). ext_flg == 1 is point-source
    morphology that merely has a 2MASS-XSC positional association (bright nearby
    stars routinely pick one up); its point-source photometry is reliable, so it
    is kept. A missing flag is kept."""
    ex = _col(row, "ex")
    if ex is None:
        return True
    return int(ex) <= 1


# ── Far-IR (Jy) catalog quality ──────────────────────────────────

def _akari_qual(qcol):
    """AKARI IRC/FIS per-band flux quality: keep only q==3 (high quality,
    confirmed, flux reliable). q<=2 is unconfirmed / unreliable."""
    def _pred(row):
        try:
            return int(_col(row, qcol)) == 3
        except (TypeError, ValueError):
            return False
    return _pred


def _iras_qual(qcol):
    """IRAS per-band flux quality: keep q>=2 (3=high, 2=moderate); reject q==1,
    which flags an UPPER LIMIT, not a detection. The per-band ``q`` flag is the
    catalogue's own quality call and is sufficient -- the separate ``Confuse``
    flag is NOT a flux-reliability cut (beta Pic, the archetypal IRAS debris
    disk, is q=3 yet Confuse=4), so cutting on it wrongly drops real sources."""
    def _pred(row):
        try:
            return int(_col(row, qcol)) >= 2
        except (TypeError, ValueError):
            return False
    return _pred


def _qc_hppsc(row):
    """Herschel/PACS point source: not at a map edge, not blended, SNR>=3."""
    try:
        snr_ok = float(_col(row, "snr")) >= 3.0
    except (TypeError, ValueError):
        snr_ok = False
    edge = _col(row, "Edge")
    blend = _col(row, "Blend")
    blend_ok = blend is None or str(blend).strip() in ("0", "0.0", "")
    return str(edge).strip().lower() in ("f", "false") and blend_ok and snr_ok


def _qc_hspsc(row):
    """Herschel/SPIRE point source: SNR>=3 only. Deliberately does NOT reject
    fExt='T' -- debris disks are resolved/confused in the large SPIRE beam, so
    the extended flag marks exactly the IR-excess sources we want."""
    try:
        return float(_col(row, "SNR")) >= 3.0
    except (TypeError, ValueError):
        return False


def _qc_sdss(row):
    """SDSS: star (class=6), good photometry (Q=2 or 3)."""
    cls = _col(row, "class")
    q = _col(row, "Q")
    if cls is None or q is None:
        return False
    return int(cls) == 6 and int(q) in (2, 3)


def _qc_ps1(row):
    """Pan-STARRS quality bitflag check."""
    qual = _col(row, "Qual")
    if qual is None:
        return False
    q = int(qual)
    is_star = not (q & 1 and q & 2)
    is_good = (q & 4 or q & 16) and not (q & 128)
    return is_star and bool(is_good)


def _qc_galex_band(row, pyphot_name):
    """GALEX per-band artifact/extraction flag check."""
    if pyphot_name == "GALEX_FUV":
        for col_name in ("Fexf", "Fafl"):
            v = _col(row, col_name)
            if v is not None and int(v) > 0:
                return False
    elif pyphot_name == "GALEX_NUV":
        for col_name in ("Nexf", "Nafl"):
            v = _col(row, col_name)
            if v is not None and int(v) > 0:
                return False
    return True


def _qc_skymapper(row):
    """SkyMapper: flags must be 0."""
    flags = _col(row, "flags")
    if flags is None:
        return False
    return int(flags) == 0
