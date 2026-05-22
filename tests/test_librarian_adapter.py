"""Tests for the new-librarian -> ARIADNE Star contract adapter.

The adapter (``astroARIADNE.librarian._adapter.adapt_librarian``) takes an
object exposing the *new* Librarian interface (``astroARIADNE.librarian._api``)
and produces the array + scalar contract that ARIADNE's ``Star`` consumes.

Contract replicated from the OLD ``astroARIADNE.librarian.Librarian``:

``used_filters`` (float64, length ``len(config.filter_names)``):
    * 1 -> band present with a valid (>0) error
    * 2 -> band present but error is 0 / masked (upper-limit, no error)
    * 0 -> unused (default)
  This matches the old ``_add_mags``. (The old ``__init__`` later collapses
  2 -> 1; that collapse is the Star's responsibility, not the adapter's.)

``mags`` / ``mag_errs`` (float64, same length): value at the band's index.

Missing scalar (property is ``None``) -> ``-1`` sentinel for value AND error,
matching ``extract_from_lib(None) == [-1] * 10``.
"""
import numpy as np

from astroARIADNE import config
from astroARIADNE.librarian._adapter import adapt_librarian


class FakeLibrarian:
    """Stub exposing the new Librarian interface for adapter testing."""

    def __init__(self, **kw):
        # New-librarian magnitudes are keyed by pyphot names (Gaia = DR3).
        self.magnitudes = kw.get("magnitudes", {})
        self._parallax = kw.get("parallax", None)
        self._distance = kw.get("distance", None)
        self._radius = kw.get("radius", None)
        self._teff = kw.get("teff", None)
        self._luminosity = kw.get("luminosity", None)
        self._mass = kw.get("mass", None)
        self._age = kw.get("age", None)
        self._gaia_id = kw.get("gaia_id", None)
        self._tic_id = kw.get("tic_id", None)
        self._kic_id = kw.get("_kic_id", None)
        self._spec = kw.get("spectroscopic_params", None)
        self._rave = kw.get("rave_params", None)
        self.Av = kw.get("Av", None)

    @property
    def parallax(self):
        return self._parallax

    @property
    def distance(self):
        return self._distance

    @property
    def radius(self):
        return self._radius

    @property
    def teff(self):
        return self._teff

    @property
    def luminosity(self):
        return self._luminosity

    @property
    def mass(self):
        return self._mass

    @property
    def age(self):
        return self._age

    @property
    def gaia_id(self):
        return self._gaia_id

    @property
    def tic_id(self):
        return self._tic_id

    @property
    def spectroscopic_params(self):
        return self._spec

    @property
    def rave_params(self):
        return self._rave


def _make_full():
    """A fairly complete fake librarian with Gaia + Stromgren + others."""
    return FakeLibrarian(
        magnitudes={
            "Gaia_G": (12.3, 0.01),     # DR3 -> must land at GaiaDR2v2_G
            "Gaia_BP": (12.6, 0.02),
            "STROMGREN_y": (12.1, 0.03),
            "2MASS_J": (10.5, 0.04),
        },
        parallax=(8.5, 0.05),
        distance=(117.0, 1.2),
        radius=(1.05, 0.10),
        teff=(5700.0, 90.0),
        luminosity=(1.3, 0.15),
        mass=(1.02, 0.08),
        age=(4.5, 1.0),
        gaia_id=123456789,
        tic_id=987654,
        _kic_id=555,
        spectroscopic_params={"teff": 5700.0, "feh": 0.0, "source": "PASTEL"},
        rave_params=None,
        Av=0.1,
    )


def test_array_shapes_and_dtype():
    out = adapt_librarian(_make_full())
    n = len(config.filter_names)
    for arr in (out.used_filters, out.mags, out.mag_errs):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (n,)
        assert arr.dtype == np.float64


def test_gaia_lands_at_dr2_index():
    out = adapt_librarian(_make_full())
    gi = np.where(config.filter_names == "GaiaDR2v2_G")[0][0]
    assert out.used_filters[gi] == 1
    assert out.mags[gi] == 12.3
    assert out.mag_errs[gi] == 0.01
    # The DR3 name must NOT appear (it isn't an ARIADNE filter).
    assert "Gaia_G" not in set(config.filter_names)


def test_stromgren_and_other_bands_set():
    out = adapt_librarian(_make_full())
    for name, (mag, err) in [
        ("GaiaDR2v2_BP", (12.6, 0.02)),
        ("STROMGREN_y", (12.1, 0.03)),
        ("2MASS_J", (10.5, 0.04)),
    ]:
        idx = np.where(config.filter_names == name)[0][0]
        assert out.used_filters[idx] == 1
        assert out.mags[idx] == mag
        assert out.mag_errs[idx] == err


def test_unused_filters_are_zero():
    out = adapt_librarian(_make_full())
    used_names = {"GaiaDR2v2_G", "GaiaDR2v2_BP", "STROMGREN_y", "2MASS_J"}
    for i, name in enumerate(config.filter_names):
        if name not in used_names:
            assert out.used_filters[i] == 0
            assert out.mags[i] == 0
            assert out.mag_errs[i] == 0


def test_used_filters_count():
    out = adapt_librarian(_make_full())
    assert int((out.used_filters >= 1).sum()) == 4


def test_zero_error_band_marked_two():
    """Band with err == 0 -> used_filters sentinel 2 (matches old _add_mags)."""
    fake = FakeLibrarian(magnitudes={"2MASS_J": (10.5, 0.0)})
    out = adapt_librarian(fake)
    idx = np.where(config.filter_names == "2MASS_J")[0][0]
    assert out.used_filters[idx] == 2
    assert out.mags[idx] == 10.5
    assert out.mag_errs[idx] == 0.0


def test_scalars_passthrough():
    out = adapt_librarian(_make_full())
    assert out.plx == 8.5
    assert out.plx_e == 0.05
    assert out.dist == 117.0
    assert out.dist_e == 1.2
    assert out.rad == 1.05
    assert out.rad_e == 0.10
    assert out.temp == 5700.0
    assert out.temp_e == 90.0
    assert out.lum == 1.3
    assert out.lum_e == 0.15


def test_missing_scalars_are_minus_one():
    fake = FakeLibrarian(
        magnitudes={"2MASS_J": (10.5, 0.04)},
        parallax=None, distance=None, radius=None, teff=None, luminosity=None,
    )
    out = adapt_librarian(fake)
    for v in (out.plx, out.plx_e, out.dist, out.dist_e, out.rad, out.rad_e,
              out.temp, out.temp_e, out.lum, out.lum_e):
        assert v == -1


def test_id_and_param_passthrough():
    fake = _make_full()
    out = adapt_librarian(fake)
    assert out.g_id == 123456789
    assert out.tic == 987654
    assert out.kic == 555
    assert out.rave_params is None
    assert out.spectroscopic_params == {
        "teff": 5700.0, "feh": 0.0, "source": "PASTEL"
    }
