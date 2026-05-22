"""Librarian — automated photometry and astrometry retrieval.

Queries Gaia DR3 best_neighbour tables for catalog crossmatching, with
VizieR XMatch fallback when Gaia TAP is unavailable. Uses pyphot filter
names internally (ARIADNE-compatible).

The package is split into:
- ``_qc``: per-catalogue quality-control predicates and ``_col``.
- ``_api``: the ``Librarian`` class plus the catalogue registry, network
  transport helpers, and the spectroscopic-prior fetchers.

Public surface stays ``from astroARIADNE.librarian import Librarian``;
internal symbols are also re-exported so the existing test suite can
import them without churn.
"""

__all__ = ["Librarian"]

from ._api import (
    CatalogDef,
    Catalogs,
    Gaia,
    Librarian,
    MagSpec,
    Vizier,
    XMatch,
    _CATALOGS,
    _CONE_ONLY_FALLBACK,
    _STANDARD_CATALOGS,
    _XMATCH_CONFIG,
    _tap_query,
    _with_timeout,
)
from ._qc import (
    _col,
    _qc_2mass_band,
    _qc_galex_band,
    _qc_mag,
    _qc_ps1,
    _qc_sdss,
    _qc_skymapper,
    _qc_wise_band,
    _qc_wise_extended,
)
