"""Hypatia Catalog [Fe/H] query.

The Hypatia Catalog (Hinkel et al. 2014, https://www.hypatiacatalog.com) is a
compilation of high-resolution-spectroscopy abundance measurements from many
independent literature sources, all renormalised to a single solar abundance
scale. Querying Hypatia for [Fe/H] gives two advantages over PASTEL:

  * the returned uncertainty is the *spread across independent studies*, which
    is realistic — unlike PASTEL's inverse-variance weighted-mean σ, which is
    unrealistically tight and biased toward whichever study quoted the smallest
    statistical error;
  * every star can be placed on the same solar normalisation.

API notes (v2.2.0, verified 2026-05):
  * endpoint:  GET https://hypatiacatalog.com/hypatia/api/v2/composition
  * no authentication, no rate limiting
  * query by any SIMBAD name; "Gaia DR3 <id>" and "2MASS J<id>" both resolve
  * the response's ``std`` field is BROKEN — it returns log10 of the std of the
    *linear* abundances (negative numbers). We never use it; we recompute the
    dex std ourselves from ``all_values``.

Reliable response fields used here:
  * ``median_value`` / ``mean`` — central [Fe/H] in dex on the chosen solarnorm
  * ``plusminus``               — (max−min)/2 spread in dex (2+ studies)
  * ``all_values``              — list of {value, catalog} per-study measurements
"""
from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

API_URL = "https://hypatiacatalog.com/hypatia/api/v2/composition"

# Single-measurement systematic floor (dex). A lone study cannot capture
# method-to-method systematics; the project's 70-star cross-comparison put the
# inter-method scatter at ~0.065 dex, so 0.10 dex is a conservative floor.
DEFAULT_FLOOR = 0.10
DEFAULT_SOLARNORM = "asplund09"


def _name_candidates(gaia_id: int | None,
                     names: Iterable[str] = ()) -> list[str]:
    """Build an ordered list of Hypatia query names, most-reliable first."""
    cand: list[str] = []
    if gaia_id is not None:
        cand.append(f"Gaia DR3 {int(gaia_id)}")
    for n in names:
        if n:
            cand.append(str(n))
    # de-dup, preserve order
    seen: set[str] = set()
    return [c for c in cand if not (c in seen or seen.add(c))]


def query_hypatia_feh(
    gaia_id: int | None = None,
    names: Iterable[str] = (),
    *,
    statistic: str = "median",
    uncertainty: str = "std",
    solarnorm: str = DEFAULT_SOLARNORM,
    floor: float = DEFAULT_FLOOR,
    timeout: float = 30.0,
) -> dict | None:
    """Query Hypatia for a star's [Fe/H]. Returns None on miss/error.

    Parameters
    ----------
    gaia_id : int | None
        Gaia DR3 source_id. Queried first as ``"Gaia DR3 <id>"`` — the most
        reliable resolver since every benchmark star has one.
    names : iterable of str
        Fallback SIMBAD names tried in order if the Gaia ID misses
        (e.g. ``"HD 10700"``, ``"2MASS J01440402-1556141"``).
    statistic : {"median", "mean"}
        Central-value estimator. ``"median"`` (default) is robust to outlier
        studies; ``"mean"`` matches the colleague's stated v1 preference.
    uncertainty : {"std", "spread"}
        ``"std"`` (default) recomputes the inter-study standard deviation in dex
        from the individual measurements (ddof=1). ``"spread"`` uses Hypatia's
        ``plusminus`` = (max−min)/2.
    solarnorm : str
        Solar normalisation ID (default ``"asplund09"``). See the catalog's
        ``/solarnorm/`` endpoint for the full list.
    floor : float
        Minimum reported σ in dex (default 0.10). Applied after the chosen
        uncertainty so single-study stars get a defensible error bar.
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    dict | None
        ``{"feh", "feh_err", "n_meas", "source", "solarnorm",
           "statistic", "uncertainty", "matched_name"}`` or ``None``.
    """
    import requests
    import numpy as np

    if statistic not in ("median", "mean"):
        raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")
    if uncertainty not in ("std", "spread"):
        raise ValueError(f"uncertainty must be 'std' or 'spread', got {uncertainty!r}")

    for qname in _name_candidates(gaia_id, names):
        try:
            resp = requests.get(
                API_URL,
                params={"name": [qname], "element": ["Fe"], "solarnorm": [solarnorm]},
                timeout=timeout,
            )
        except Exception as e:
            logger.warning("Hypatia request failed for %s: %s", qname, e)
            continue
        if resp.status_code != 200:
            logger.warning("Hypatia HTTP %s for %s", resp.status_code, qname)
            continue
        try:
            payload = resp.json()
        except Exception:
            continue
        rec = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(rec, dict) or rec.get("name") in (None, "not-found"):
            continue

        central = rec.get("median_value") if statistic == "median" else rec.get("mean")
        if central is None:
            continue

        all_vals = [v["value"] for v in (rec.get("all_values") or [])
                    if v.get("value") is not None]
        n_meas = len(all_vals)

        if uncertainty == "std":
            sigma = float(np.std(all_vals, ddof=1)) if n_meas >= 2 else None
        else:  # spread
            pm = rec.get("plusminus")
            sigma = float(pm) if pm is not None else None

        feh_err = max(sigma, floor) if sigma is not None else floor

        return {
            "feh": float(central),
            "feh_err": float(feh_err),
            "n_meas": n_meas,
            "source": "Hypatia",
            "solarnorm": solarnorm,
            "statistic": statistic,
            "uncertainty": uncertainty,
            "matched_name": rec.get("name"),
        }

    return None
