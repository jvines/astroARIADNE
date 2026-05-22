"""Adapt the new Librarian interface to ARIADNE's ``Star`` data contract.

The new ``Librarian`` (``astroARIADNE.librarian._api``) exposes photometry and
stellar parameters through properties returning ``(value, error)`` tuples (or
``None``) and a ``magnitudes`` dict keyed by pyphot names. ARIADNE's ``Star``,
however, was written against the OLD ``astroARIADNE.librarian.Librarian``, which
exposed:

* parallel float64 arrays ``used_filters``, ``mags``, ``mag_errs`` of length
  ``len(config.filter_names)``, aligned to ``config.filter_names``;
* scalar attributes ``plx, plx_e, dist, dist_e, rad, rad_e, temp, temp_e,
  lum, lum_e`` (consumed via ``extract_from_lib``);
* ``g_id, tic, kic, rave_params, spectroscopic_params``.

This module bridges the two so ``Star`` can be rewritten with minimal logic.

Conventions replicated verbatim from the old code
--------------------------------------------------
``used_filters`` sentinels (from old ``Librarian._add_mags``):
    * ``1`` -> band present with a valid (``> 0``) error,
    * ``2`` -> band present but error is ``0`` / masked (no-error / upper limit),
    * ``0`` -> unused (array default).
  The old ``__init__`` later collapses ``2 -> 1`` (``used_filters[used >= 1] =
  1``); that final collapse is intentionally left to ``Star`` (Task 5), so the
  adapter preserves the richer ``1``/``2`` distinction.

Arrays are ``float64`` (old code built them with ``np.zeros(...)``).

Missing-scalar sentinels (PER-FIELD, matching the old extractors exactly):
    * ``parallax`` (plx, plx_e) missing -> ``-1, -1``
    * ``distance`` (dist, dist_e) missing -> ``-1, -1``
    * ``radius`` (rad, rad_e) missing -> ``0, 0``
    * ``teff`` (temp, temp_e) missing -> ``0, 0``
    * ``luminosity`` (lum, lum_e) missing -> ``0, 0``

  These are NOT uniform. The old ``_get_parallax``/``_get_distance`` returned
  ``-1`` on a miss, while ``_get_radius``/``_get_teff``/``_get_lum`` returned
  ``0``. This asymmetry is load-bearing: downstream code guards the FLAME family
  with ``!= 0`` rather than ``is not None``, e.g.

    * ``star.py:437``  ``if self.temp is not None and self.temp_e != 0:``
    * ``star.py:439``  ``if self.lum is not None and self.lum != 0:``
    * ``star.py:442``  ``if self.get_rad and self.rad is not None and self.rad != 0:``
    * ``fitter.py:1712``  ``if self.star.lum != 0 and self.star.lum_e != 0:``

  A uniform ``-1`` sentinel would make these branches WRONGLY fire on a missing
  field (feeding ``Teff = (-1, -1)`` to the isochrone, computing
  ``np.log10(-1) = NaN`` for luminosity, etc.). So a missing FLAME field MUST
  be ``0`` and a missing parallax/distance MUST be ``-1``.

Return type
-----------
``adapt_librarian`` returns an ``AdaptedStar`` dataclass. Task 5 consumes its
attributes directly; the names mirror the old ``Librarian`` attribute names so
``Star`` assignments stay one-to-one.
"""
from __future__ import annotations

__all__ = ["AdaptedStar", "adapt_librarian"]

from dataclasses import dataclass

import numpy as np

from .. import config
from ._filtermap import to_ariadne_filters

# Per-field missing sentinels (value and error). NOT uniform: parallax/distance
# miss to -1, while the FLAME family (radius/teff/luminosity) misses to 0 so the
# downstream ``!= 0`` guards in star.py / fitter.py don't wrongly fire.
_MISSING_PLX = -1
_MISSING_FLAME = 0


@dataclass
class AdaptedStar:
    """ARIADNE ``Star`` contract built from a new-style ``Librarian``.

    Attributes mirror the old ``Librarian`` attribute names consumed by
    ``Star``/``extract_from_lib``.
    """

    used_filters: np.ndarray
    mags: np.ndarray
    mag_errs: np.ndarray
    plx: float
    plx_e: float
    dist: float
    dist_e: float
    rad: float
    rad_e: float
    temp: float
    temp_e: float
    lum: float
    lum_e: float
    g_id: int | None
    tic: int | None
    kic: int | None
    rave_params: dict | None
    spectroscopic_params: dict | None


def _scalar(pair, missing):
    """Map a ``(value, error)`` tuple or ``None`` to a ``(value, error)`` pair.

    ``None`` -> ``(missing, missing)``; the caller supplies the per-field
    sentinel (``-1`` for parallax/distance, ``0`` for the FLAME family).
    """
    if pair is None:
        return missing, missing
    return pair[0], pair[1]


def adapt_librarian(lib) -> AdaptedStar:
    """Convert a new-style ``Librarian`` into ARIADNE's ``Star`` contract.

    Parameters
    ----------
    lib : object
        Anything exposing the new Librarian interface: ``magnitudes`` dict and
        properties ``parallax``, ``distance``, ``radius``, ``teff``,
        ``luminosity`` (each ``(value, error)`` or ``None``), plus ``gaia_id``,
        ``tic_id``, ``_kic_id``, ``rave_params``, ``spectroscopic_params``.

    Returns
    -------
    AdaptedStar
        Filled arrays + scalars + pass-through identifiers/params.
    """
    n = config.filter_names.shape[0]
    used_filters = np.zeros(n)
    mags = np.zeros(n)
    mag_errs = np.zeros(n)

    # Bridge pyphot names (Gaia DR3 -> DR2) and drop bands ARIADNE can't model.
    ariadne_mags = to_ariadne_filters(lib.magnitudes)

    # Build a name -> index lookup once (config.filter_names is small/static).
    name_to_idx = {name: i for i, name in enumerate(config.filter_names)}
    for name, (mag, err) in ariadne_mags.items():
        idx = name_to_idx.get(name)
        if idx is None:  # defensive: to_ariadne_filters already guarantees this
            continue
        # Replicate old _add_mags: err == 0 (or masked) -> sentinel 2, else 1.
        if err == 0 or np.ma.is_masked(err):
            used_filters[idx] = 2
        else:
            used_filters[idx] = 1
        mags[idx] = mag
        mag_errs[idx] = err

    plx, plx_e = _scalar(lib.parallax, _MISSING_PLX)
    dist, dist_e = _scalar(lib.distance, _MISSING_PLX)
    rad, rad_e = _scalar(lib.radius, _MISSING_FLAME)
    temp, temp_e = _scalar(lib.teff, _MISSING_FLAME)
    lum, lum_e = _scalar(lib.luminosity, _MISSING_FLAME)

    return AdaptedStar(
        used_filters=used_filters,
        mags=mags,
        mag_errs=mag_errs,
        plx=plx, plx_e=plx_e,
        dist=dist, dist_e=dist_e,
        rad=rad, rad_e=rad_e,
        temp=temp, temp_e=temp_e,
        lum=lum, lum_e=lum_e,
        g_id=lib.gaia_id,
        tic=lib.tic_id,
        kic=lib._kic_id,
        rave_params=lib.rave_params,
        spectroscopic_params=lib.spectroscopic_params,
    )
