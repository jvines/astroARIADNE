"""Bridge between librarian pyphot filter names and ARIADNE config names.

The librarian (``astroARIADNE.librarian._api``) returns photometry in a
``magnitudes`` dict keyed by pyphot filter names. Most of those names already
match the names ARIADNE's model grids expect (``astroARIADNE.config.filter_names``)
exactly. The one known divergence is Gaia: the librarian emits Gaia DR3 names
(``Gaia_G/BP/RP``) while ARIADNE grids are built on Gaia DR2 passbands
(``GaiaDR2v2_G/BP/RP``). We rename DR3 -> DR2 so the existing grids keep working,
accepting a small passband systematic.

The mapping is derived directly from the librarian's catalog registry so it can
never silently drift from the names the librarian actually emits.
"""
from .. import config
from ._api import _CATALOGS

# Gaia DR3 (librarian) -> Gaia DR2 (ARIADNE grids). Accepts a small passband
# systematic to reuse the existing DR2-based model grids.
_GAIA_DR3_TO_DR2 = {
    "Gaia_G": "GaiaDR2v2_G",
    "Gaia_BP": "GaiaDR2v2_BP",
    "Gaia_RP": "GaiaDR2v2_RP",
}

# Names emitted by the librarian that are not advertised through a registry
# band's ``pyphot`` attribute (constructed inline in _api.py).
_EXTRA_EMITTED = ("GROUND_JOHNSON_V", "GROUND_JOHNSON_B", "GROUND_JOHNSON_U")


def _build_map():
    """Build the librarian->ARIADNE name map from the real emitted names."""
    emitted = set(_EXTRA_EMITTED)
    for cat in _CATALOGS.values():
        for band in cat.bands:
            emitted.add(band.pyphot)

    mapping = {}
    for name in emitted:
        # Explicit rename (Gaia DR3 -> DR2).
        if name in _GAIA_DR3_TO_DR2:
            mapping[name] = _GAIA_DR3_TO_DR2[name]
        else:
            # Shared name: identity map.
            mapping[name] = name
    return mapping


# Librarian pyphot name -> ARIADNE config.filter_names name.
LACHESIS_TO_ARIADNE = _build_map()


def to_ariadne_filters(magnitudes: dict) -> dict:
    """Rename librarian photometry keys to ARIADNE filter names.

    Parameters
    ----------
    magnitudes : dict
        ``{pyphot_name: (mag, err)}`` as emitted by the librarian.

    Returns
    -------
    dict
        ``{ariadne_name: (mag, err)}``. Any key whose mapped name is not a
        member of ``astroARIADNE.config.filter_names`` is dropped (not raised),
        as is any key with no entry in :data:`LACHESIS_TO_ARIADNE`. The
        ``(mag, err)`` tuple values are preserved verbatim.
    """
    valid = set(config.filter_names)
    out = {}
    for key, value in magnitudes.items():
        mapped = LACHESIS_TO_ARIADNE.get(key)
        if mapped is None or mapped not in valid:
            continue
        out[mapped] = value
    return out
