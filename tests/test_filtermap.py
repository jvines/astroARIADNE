"""Tests for the librarian -> ARIADNE filter-name bridge."""
from astroARIADNE import config
from astroARIADNE.librarian._filtermap import (
    GAIA_RENAMES,
    LACHESIS_TO_ARIADNE,
    to_ariadne_filters,
)


def test_gaia_renamed_to_dr2():
    """Librarian Gaia DR3 keys map to ARIADNE DR2 grid names."""
    assert GAIA_RENAMES["Gaia_G"] == "GaiaDR2v2_G"
    assert GAIA_RENAMES["Gaia_BP"] == "GaiaDR2v2_BP"
    assert GAIA_RENAMES["Gaia_RP"] == "GaiaDR2v2_RP"
    # Backwards-compatible alias.
    assert LACHESIS_TO_ARIADNE is GAIA_RENAMES


def test_gaia_rename_targets_are_valid():
    """Every Gaia rename target is a valid ARIADNE filter."""
    valid = set(config.filter_names)
    for out_name in GAIA_RENAMES.values():
        assert out_name in valid, out_name


def test_shared_names_unchanged():
    """Names already shared with ARIADNE pass through by identity."""
    for name in ("2MASS_J", "GROUND_JOHNSON_V", "WISE_RSR_W1",
                 "SDSS_g", "PS1_g", "STROMGREN_y", "STROMGREN_b"):
        out = to_ariadne_filters({name: (10.0, 0.01)})
        assert out == {name: (10.0, 0.01)}


def test_stromgren_band_kept():
    """Stromgren bands are shared with ARIADNE and must not be dropped."""
    assert to_ariadne_filters({"STROMGREN_y": (8.0, 0.01)}) == {
        "STROMGREN_y": (8.0, 0.01)
    }
    for band in ("STROMGREN_u", "STROMGREN_v", "STROMGREN_b", "STROMGREN_y"):
        assert to_ariadne_filters({band: (8.0, 0.01)}) == {band: (8.0, 0.01)}


def test_to_ariadne_filters_renames_and_preserves_values():
    """Keys are renamed; (mag, err) tuples preserved verbatim."""
    mags = {
        "Gaia_G": (12.3, 0.01),
        "2MASS_J": (10.1, 0.02),
        "SDSS_g": (13.0, 0.03),
    }
    out = to_ariadne_filters(mags)
    assert out["GaiaDR2v2_G"] == (12.3, 0.01)
    assert out["2MASS_J"] == (10.1, 0.02)
    assert out["SDSS_g"] == (13.0, 0.03)
    assert "Gaia_G" not in out


def test_to_ariadne_filters_drops_unknown():
    """Keys with no ARIADNE equivalent are dropped, not raised."""
    mags = {
        "Gaia_BP": (11.0, 0.01),
        "SOME_UNKNOWN_FILTER": (9.0, 0.05),
    }
    out = to_ariadne_filters(mags)
    assert out == {"GaiaDR2v2_BP": (11.0, 0.01)}


def test_to_ariadne_filters_output_keys_all_valid():
    """Output of the bridge only ever contains valid ARIADNE filters."""
    valid = set(config.filter_names)
    mags = {name: (10.0, 0.01) for name in valid}
    mags.update({name: (10.0, 0.01) for name in GAIA_RENAMES})
    mags["TOTALLY_BOGUS"] = (9.0, 9.0)
    out = to_ariadne_filters(mags)
    assert all(k in valid for k in out)
    assert "TOTALLY_BOGUS" not in out
