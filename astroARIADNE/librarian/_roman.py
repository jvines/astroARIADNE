"""Roman Space Telescope WFI ingestion for the librarian -- STAGED.

Roman (launch 2026) delivers, for ARIADNE's cool/substellar regime, exactly the
two data products this module will consume:

  * WFI imaging photometry -- eight broadbands 0.62-2.13 um (ROMAN_F062..F213),
    deep (~26 mag) and resolved (0.11"), so it deblends the close pairs that
    2MASS/WISE/SPHEREx merge. Mapped one-to-one onto the ROMAN_F* fit bands.
  * P127 prism spectrophotometry -- 0.75-1.80 um at R~100, ~3 mag more sensitive
    than the grism, optimised for faint red sources with broad molecular
    features. Median-binned into the ROMANP_* super-bands (gap-placed between
    the imaging centres so the prism adds spectral shape, not a re-measurement
    of the broadband flux), the same spectrum -> super-band recipe as _spherex.

STATUS: the filter curves (custom_filters.h5) and grid columns
(augment_grids_roman.py) are in place, and ROMAN_*/ROMANP_* are in
config.filter_names -- so a Roman fit works the day photometry lands. What is
NOT here is the catalogue access: the WFI/prism science archives (STScI/MAST,
IRSA) publish ~1-2 yr after launch, and their data model / query API / absolute
calibration are not yet fixed. `fetch_wfi_photometry` and `fetch_prism_spectrum`
therefore raise NotImplementedError until those products exist. The DOWNSTREAM
half -- the prism -> super-band binning -- is implemented and testable now, so
only the fetch remains to wire in.

Naming follows _spherex: bands come from the single source of truth
(custom_filters.h5 via phot_utils), so this module never hard-codes windows.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

C_UM = 2.99792458e14
# WFI imaging is AB-native (like SDSS/SkyMapper); the prism super-bands inherit
# the top-hat zero point from custom_filters.h5. Absolute-calibration floor is a
# placeholder pending the Roman photometric error budget.
ABS_CAL_FLOOR = 0.02                         # fractional
MIN_BINS_PER_SUPERBAND = 3


def _prism_superband_windows():
    """ROMANP_* -> (lo_um, hi_um) from the filter library (single source of
    truth: custom_filters.h5 via phot_utils), matching _spherex's pattern."""
    from ..phot_utils import _CUSTOM_LIB
    out = {}
    for name, filt in _CUSTOM_LIB.items():
        if not name.startswith('ROMANP_'):
            continue
        wl = filt.wavelength.to('um').value
        tr = np.asarray(filt.transmit, float)
        on = wl[tr > 0.5]
        out[name] = (float(on.min()), float(on.max()))
    return out


def prism_superband_mags(lam, f, e):
    """Bin a P127 prism spectrum into ROMANP_* mags.

    Inputs: per-sample wavelength [um], F_lambda-consistent flux and error in
    the prism's native units (uJy, as F_nu -- converted to F_lambda here, as in
    _spherex). Returns {band: (mag, mag_err, n_samples)}; median flux per window,
    error = MAD-scatter/sqrt(n) with the ABS_CAL_FLOOR floor. Implemented now so
    only the upstream fetch remains once Roman prism spectra are public.
    """
    from ..phot_utils import flux_to_mag
    lam = np.asarray(lam, float)
    f = np.asarray(f, float)
    out = {}
    for name, (lo, hi) in sorted(_prism_superband_windows().items()):
        m = (lam >= lo) & (lam <= hi)
        n = int(m.sum())
        if n < MIN_BINS_PER_SUPERBAND:
            continue
        fmu = float(np.median(f[m]))
        if not (np.isfinite(fmu) and fmu > 0):
            continue
        scatter = 1.4826 * float(np.median(np.abs(f[m] - fmu)))
        fer = max(scatter / np.sqrt(n), ABS_CAL_FLOOR * fmu)
        lam_c = 0.5 * (lo + hi)
        flam = fmu * 1e-29 * C_UM / lam_c ** 2           # uJy -> erg/cm2/s/um
        flam_e = fer * 1e-29 * C_UM / lam_c ** 2
        mag, mag_e = flux_to_mag(flam, flam_e, name)
        if np.isfinite(mag) and np.isfinite(mag_e):
            out[name] = (float(mag), float(mag_e), n)
    return out


def fetch_wfi_photometry(ra, dec, source_id, epoch=None):
    """Query the Roman WFI source catalogue at a PM-propagated position and
    return {ROMAN_F*: (mag, mag_err)}.

    Not implemented: the WFI catalogue is not public yet. When it is, this
    resolves the source at the survey epoch (Roman astrometry is Gaia-tied, so
    the same PM back/forward-propagation _spherex uses applies), pulls the eight
    AB broadband magnitudes, and returns them keyed by ROMAN_F062..F213. WFI's
    0.11" resolution is the deblending win: return the RESOLVED per-component
    row, not a merged one.
    """
    raise NotImplementedError(
        'Roman WFI catalogue ingestion is staged: filter curves + grid columns '
        'are in place, but the WFI science archive is not public yet (expected '
        '~1-2 yr post-launch). Wire the catalogue query here when it ships.')


def fetch_prism_spectrum(ra, dec, source_id, epoch=None):
    """Retrieve a P127 prism spectrum at a PM-propagated position; return
    (lam_um, flux, err) for prism_superband_mags.

    Not implemented: the prism spectral products are not public yet. When they
    are, this pulls the extracted 1D spectrum (slitless, so contamination-
    decontaminated) and hands it to prism_superband_mags for the ROMANP_* bands.
    """
    raise NotImplementedError(
        'Roman P127 prism ingestion is staged: the ROMANP_* super-bands + grid '
        'columns exist and prism_superband_mags() is ready, but the prism '
        'spectral archive is not public yet. Wire the retrieval here when it '
        'ships.')
