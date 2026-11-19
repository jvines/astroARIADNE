"""Offline tests for the SPHEREx aperture-contamination QC (issue #55, check
3): Gaia neighbours leaking into our own 3-px SPHEREx extraction aperture.
No network."""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroARIADNE.librarian import _contamination as cq
from astroARIADNE.librarian import _spherex
from astroARIADNE.librarian._api import Librarian

PIX = cq.SPHEREX_PIX_ARCSEC
BANDS = tuple(cq.SPHEREX_APERTURE_FRACTION)

# TIC-275647095 = Gaia DR3 5377927940883353216 and its ESA cone (2026-11-19):
# (source_id, ra, dec, G, RP). All 11 SPHEREx super-bands sat +13-23% above
# the BMA model while 2MASS / WISE / Gaia agreed to < 2%.
TIC_TARGET = 5377927940883353216
TIC_CONE = [
    (5377927940883353216, 179.33751107060513, -45.54654025686794, 12.742165, 12.174939),
    (5377927940885029504, 179.33845598145814, -45.543798713891334, 17.924135, 17.392103),
    (5377928662439768960, 179.3333683058011, -45.55059402701535, 20.353851, 19.639479),
    (5377927945181990016, 179.34423249175546, -45.54946531737626, 14.306417, 13.548859),
    (5377927940885034368, 179.34150633979155, -45.54065910228472, 19.935215, 18.886093),
    (5377929422649153024, 179.34203623056794, -45.54012014322396, 20.687012, 19.91521),
    (5377928666733519488, 179.32848634274723, -45.550071265594745, 18.954666, 17.571993),
    (5377928696799276544, 179.33024707660934, -45.541315548036174, 20.520891, 19.888199),
    (5377928666733519616, 179.3279944221432, -45.54992281328619, 19.622066, np.nan),
    (5377928666733518336, 179.32929897039762, -45.55211513044424, 19.945457, 19.37808),
    (5377928666733518592, 179.3279316228714, -45.55119909145996, 17.331188, 16.688978),
    (5377928696799277312, 179.32647134811506, -45.54162791767593, 17.818737, 17.296349),
    (5377927945180403584, 179.35197002853886, -45.54936078895071, 20.607218, 19.658915),
]


def _tic_table():
    return Table(rows=TIC_CONE, names=("source_id", "ra", "dec",
                                       "phot_g_mean_mag", "phot_rp_mean_mag"))


def _cone(rows, ra0=100.0, dec0=0.0):
    """rows: (source_id, dx_arcsec, dy_arcsec, G, RP)."""
    return Table({
        "source_id": [r[0] for r in rows],
        "ra": [ra0 + r[1] / 3600.0 for r in rows],
        "dec": [dec0 + r[2] / 3600.0 for r in rows],
        "phot_g_mean_mag": [r[3] for r in rows],
        "phot_rp_mean_mag": [r[4] for r in rows],
    })


# ── aperture fraction (from the mission ePSF) ────────────────────────────────

def test_table_covers_every_fitted_superband():
    windows = {b for b in _spherex._superband_windows()}
    assert windows <= set(BANDS)
    for prof in cq.SPHEREX_APERTURE_FRACTION.values():
        assert len(prof) == len(cq.SPHEREX_SEP_PX)


def test_geometry_matches_the_extraction():
    assert cq.SPHEREX_AP_R_PX == _spherex.AP_R
    assert cq.SPHEREX_PIX_ARCSEC == _spherex.PIX_ARCSEC


@pytest.mark.parametrize("band", BANDS)
def test_profile_is_physical(band):
    prof = np.array(cq.SPHEREX_APERTURE_FRACTION[band])
    # the target's own enclosed energy: the docstring's 99.6% within 3 px,
    # less the subpixel-phase average of the stamp-truncated ePSF
    assert 0.98 < prof[0] <= 1.0
    assert np.all(np.diff(prof) <= 0)          # monotone in separation
    # a source on the aperture edge puts about half its flux inside
    edge = cq.spherex_aperture_fraction(band, 3.0 * PIX)
    assert 0.40 < edge < 0.50
    assert prof[-1] < 1e-3


def test_fraction_is_zero_beyond_the_epsf_reach_and_cone_covers_it():
    assert cq.spherex_aperture_fraction("SPHEREX_1050", 6.5 * PIX) == 0.0
    assert cq.spherex_aperture_fraction("SPHEREX_1050", 100.0) == 0.0
    assert cq.SPHEREX_CONE_RADIUS_ARCSEC >= cq.SPHEREX_SEP_PX[-1] * PIX


def test_fraction_is_nearly_achromatic():
    rows = np.array(list(cq.SPHEREX_APERTURE_FRACTION.values()))
    assert np.ptp(rows, axis=0).max() < 0.05


def test_unknown_superband_uses_the_mean_profile():
    f = cq.spherex_aperture_fraction("SPHEREX_4900", 2.0 * PIX)
    rows = np.array(list(cq.SPHEREX_APERTURE_FRACTION.values()))
    assert f == pytest.approx(rows[:, 8].mean(), rel=1e-9)   # 2.0 px = index 8


def test_fraction_at_the_tic_neighbour_separation():
    # 19.95" = 3.24 px: just outside the 3-px edge, ~30% of its flux inside
    for band in BANDS:
        assert cq.spherex_aperture_fraction(band, 19.95) == pytest.approx(
            0.30, abs=0.02)


# ── contamination, flux proxy ────────────────────────────────────────────────

def test_rp_is_the_proxy_when_both_have_it():
    nb = [(1, 2.0 * PIX, 15.0, 12.0)]           # G ratio 0.063, RP ratio 0.4
    c, terms = cq.spherex_contamination("SPHEREX_1850", (12.0, 11.0), nb)
    f = (cq.spherex_aperture_fraction("SPHEREX_1850", 2.0 * PIX)
         / cq.spherex_aperture_fraction("SPHEREX_1850", 0.0))
    assert terms[0][4] == "RP"
    assert c == pytest.approx(10 ** (-0.4 * 1.0) * f, rel=1e-9)


def test_g_fallback_without_rp():
    nb = [(1, 2.0 * PIX, 13.0, np.nan)]
    c, terms = cq.spherex_contamination("SPHEREX_1850", (12.0, 11.0), nb)
    assert terms[0][4] == "G"
    f = (cq.spherex_aperture_fraction("SPHEREX_1850", 2.0 * PIX)
         / cq.spherex_aperture_fraction("SPHEREX_1850", 0.0))
    assert c == pytest.approx(10 ** (-0.4) * f, rel=1e-9)


def test_neighbour_without_any_magnitude_or_out_of_reach_is_ignored():
    nb = [(1, 2.0, np.nan, np.nan), (2, 45.0, 5.0, 5.0)]
    c, terms = cq.spherex_contamination("SPHEREX_1850", (12.0, 11.0), nb)
    assert c == 0.0 and terms == []


def test_neighbours_from_table_measures_from_target_and_cuts_at_cone():
    tbl = _cone([(7, 0, 0, 12.0, 11.5), (1, 3, 4, 14.0, 13.0),
                 (2, 0, 41, 9.0, 8.0), (3, 6, 8, np.nan, 15.0)])
    target, nbrs = cq.spherex_neighbours_from_table(tbl, 7)
    assert target == (12.0, 11.5)
    seps = {n[0]: n[1] for n in nbrs}
    assert set(seps) == {1, 3}
    assert seps[1] == pytest.approx(5.0, abs=1e-3)
    assert cq.spherex_neighbours_from_table(tbl, 999) == (None, [])


def test_neighbours_from_table_without_rp_column():
    tbl = _cone([(7, 0, 0, 12.0, 0), (1, 3, 4, 14.0, 0)])
    del tbl["phot_rp_mean_mag"]
    target, nbrs = cq.spherex_neighbours_from_table(tbl, 7)
    assert target[0] == 12.0 and np.isnan(target[1])
    assert np.isnan(nbrs[0][3])


# ── TIC-275647095 ────────────────────────────────────────────────────────────

def test_tic_275647095_every_superband_excluded():
    target, nbrs = cq.spherex_neighbours_from_table(_tic_table(), TIC_TARGET)
    recs = cq.assess_spherex_bands(BANDS, target, nbrs)
    assert set(recs) == set(BANDS)
    for band, rec in recs.items():
        assert rec["status"] == "exclude", band
        assert 0.09 < rec["c"] < 0.11, (band, rec["c"])
        top = rec["neighbours"][0]
        assert top[0] == 5377927945181990016 and top[4] == "RP"
        assert top[1] == pytest.approx(19.95, abs=0.02)
    # the G proxy of #55 alone would also exclude (0.24 x 0.30)
    g_only = [(s, sep, g, np.nan) for s, sep, g, _ in nbrs]
    for rec in cq.assess_spherex_bands(BANDS, target, g_only).values():
        assert rec["status"] == "exclude"


def test_spherex_bands_are_not_in_the_beam_check():
    for band in BANDS:
        assert cq.band_fwhm(band) is None
    assert cq.assess_spherex_bands(["2MASS_J"], (10.0, 9.5), []) == {}


def test_describe_spherex():
    target, nbrs = cq.spherex_neighbours_from_table(_tic_table(), TIC_TARGET)
    rec = cq.assess_spherex_bands(["SPHEREX_1050"], target, nbrs)
    text = cq.describe("SPHEREX_1050", rec["SPHEREX_1050"])
    assert "5377927945181990016" in text and "18.5\" SPHEREx" in text
    assert "RP=13.55" in text


# ── Librarian-level ──────────────────────────────────────────────────────────

def _lib(qc=True):
    lib = Librarian.__new__(Librarian)
    lib._init_contamination(qc, 0.01, 0.05)
    lib._gaia_id = TIC_TARGET
    lib._coord = SkyCoord(ra=179.33751107061, dec=-45.54654025687, unit="deg")
    lib._gaia_ra = lib._gaia_dec = None
    lib._magnitudes = {"2MASS_Ks": (11.1, 0.02)}
    for b in BANDS:
        lib._magnitudes[b] = (11.0, 0.03)
    return lib


def test_apply_excludes_and_records_c():
    lib = _lib()
    seen = []
    lib._run_adql = lambda adql: seen.append(adql) or _tic_table()
    removed = lib._apply_spherex_contamination_qc(list(BANDS))
    assert removed == set(BANDS)
    assert set(lib._magnitudes) == {"2MASS_Ks"}
    assert all(lib.contamination[b]["status"] == "exclude" for b in BANDS)
    assert "phot_rp_mean_mag" in seen[0]
    assert f"{40.0 / 3600.0:.8f}" in seen[0]
    lines = lib._contamination_summary_lines()
    assert sum("EXCLUDED SPHEREX_" in ln for ln in lines) == len(BANDS)


def test_apply_keeps_records_from_the_catalogue_check():
    lib = _lib()
    lib._contamination = {"WISE_RSR_W1": {"c": 0.0, "status": "ok",
                                          "fwhm": 6.1, "neighbours": []}}
    lib._run_adql = lambda adql: _tic_table()
    lib._apply_spherex_contamination_qc(["SPHEREX_1050"])
    assert set(lib.contamination) == {"WISE_RSR_W1", "SPHEREX_1050"}


def test_apply_psf_method_marks_deblended_and_keeps():
    lib = _lib()
    lib._run_adql = lambda adql: _tic_table()
    removed = lib._apply_spherex_contamination_qc(list(BANDS), method="psf")
    assert removed == set()
    assert all(b in lib._magnitudes for b in BANDS)
    assert all(lib.contamination[b]["status"] == "deblended" for b in BANDS)
    assert any(ln.startswith("DEBLENDED")
               for ln in lib._contamination_summary_lines())


def test_apply_clean_field_keeps_everything():
    lib = _lib()
    lib._run_adql = lambda adql: _tic_table()[:2]   # target + the 10" G=17.9
    removed = lib._apply_spherex_contamination_qc(list(BANDS))
    assert removed == set()
    assert all(lib.contamination[b]["status"] == "ok" for b in BANDS)


def test_apply_degrades_when_esa_unreachable(caplog):
    lib = _lib()
    lib._run_adql = lambda adql: None
    with caplog.at_level("WARNING"):
        assert lib._apply_spherex_contamination_qc(list(BANDS)) == set()
    assert all(b in lib._magnitudes for b in BANDS)
    assert lib.contamination == {} and "NOT checked" in caplog.text


def test_apply_off_or_nothing_to_check_does_not_query():
    def boom(adql):
        raise AssertionError("queried")
    lib = _lib(qc=False)
    lib._run_adql = boom
    assert lib._apply_spherex_contamination_qc(list(BANDS)) == set()
    lib = _lib()
    lib._run_adql = boom
    assert lib._apply_spherex_contamination_qc(["2MASS_Ks"]) == set()


def test_fetch_spherex_drops_excluded_bands_from_the_added_list(monkeypatch):
    lib = _lib()
    lib._magnitudes = {"2MASS_Ks": (11.1, 0.02)}
    lib._spherex_added = []
    lib._propagated_coord = lambda ep: lib._coord
    lib._run_adql = lambda adql: _tic_table()

    def fake_extract(*a, meta=None, **k):
        meta.update(method="aperture")
        return np.array([1.0]), np.array([1.0]), np.array([0.1])
    monkeypatch.setattr(_spherex, "extract_spectrum", fake_extract)
    monkeypatch.setattr(_spherex, "superband_mags", lambda *a: {
        "SPHEREX_1050": (11.0, 0.03, 12), "SPHEREX_2350": (10.9, 0.03, 9)})
    lib._fetch_spherex()
    assert lib._spherex_added == []
    assert set(lib._magnitudes) == {"2MASS_Ks"}
    assert lib.contamination["SPHEREX_2350"]["status"] == "exclude"
