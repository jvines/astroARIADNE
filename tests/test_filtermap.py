"""Tests for the librarian -> ARIADNE filter-name bridge."""
from astroARIADNE import config
from astroARIADNE.librarian._filtermap import (
    LACHESIS_TO_ARIADNE,
    to_ariadne_filters,
)


def test_gaia_renamed_to_dr2():
    """Librarian Gaia DR3 keys map to ARIADNE DR2 grid names."""
    assert LACHESIS_TO_ARIADNE["Gaia_G"] == "GaiaDR2v2_G"
    assert LACHESIS_TO_ARIADNE["Gaia_BP"] == "GaiaDR2v2_BP"
    assert LACHESIS_TO_ARIADNE["Gaia_RP"] == "GaiaDR2v2_RP"


def test_shared_names_unchanged():
    """Names already shared with ARIADNE are identity-mapped."""
    for name in ("2MASS_J", "GROUND_JOHNSON_V", "WISE_RSR_W1",
                 "SDSS_g", "PS1_g"):
        assert LACHESIS_TO_ARIADNE[name] == name


def test_all_outputs_in_config_filter_names():
    """Every mapped output name is a valid ARIADNE filter."""
    valid = set(config.filter_names)
    for out_name in LACHESIS_TO_ARIADNE.values():
        assert out_name in valid, out_name


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
    mags = {name: (10.0, 0.01) for name in LACHESIS_TO_ARIADNE}
    out = to_ariadne_filters(mags)
    assert all(k in valid for k in out)
