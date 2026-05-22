"""Bridge between librarian pyphot filter names and ARIADNE config names.

The librarian (``astroARIADNE.librarian._api``) returns photometry in a
``magnitudes`` dict keyed by pyphot filter names. Almost all of those names are
identical to the names ARIADNE's model grids expect
(``astroARIADNE.config.filter_names``). The one known divergence is Gaia: the
librarian emits Gaia DR3 names (``Gaia_G/BP/RP``) while ARIADNE grids are built
on Gaia DR2 passbands (``GaiaDR2v2_G/BP/RP``). We rename DR3 -> DR2 so the
existing grids keep working, accepting a small passband systematic.

The bridge does not enumerate the names the librarian emits. Instead it applies
the Gaia rename and then keeps any key that is a member of
``config.filter_names`` by identity, dropping the rest. This way every name that
both the librarian and ARIADNE share -- including the Stromgren bands and any
band ARIADNE adds in the future -- passes through automatically, with no
hand-maintained list that could drift out of sync.
"""
from .. import config

# Gaia DR3 (librarian) -> Gaia DR2 (ARIADNE grids). Accepts a small passband
# systematic to reuse the existing DR2-based model grids. This is the only
# genuine name divergence between the librarian and ARIADNE.
GAIA_RENAMES = {
    "Gaia_G": "GaiaDR2v2_G",
    "Gaia_BP": "GaiaDR2v2_BP",
    "Gaia_RP": "GaiaDR2v2_RP",
}

# Backwards-compatible alias: the Gaia rename dict is the only explicit mapping.
LACHESIS_TO_ARIADNE = GAIA_RENAMES


def to_ariadne_filters(magnitudes: dict) -> dict:
    """Rename librarian photometry keys to ARIADNE filter names.

    Parameters
    ----------
    magnitudes : dict
        ``{pyphot_name: (mag, err)}`` as emitted by the librarian.

    Returns
    -------
    dict
        ``{ariadne_name: (mag, err)}``. For each input key the Gaia DR3 -> DR2
        rename is applied if applicable, then the (renamed) key is kept iff it
        is a member of ``astroARIADNE.config.filter_names``; otherwise it is
        dropped (not raised). The ``(mag, err)`` tuple values are preserved
        verbatim.
    """
    valid = set(config.filter_names)
    out = {}
    for key, value in magnitudes.items():
        mapped = GAIA_RENAMES.get(key, key)
        if mapped not in valid:
            continue
        out[mapped] = value
    return out
