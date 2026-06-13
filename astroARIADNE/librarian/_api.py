"""Librarian — automated photometry and astrometry retrieval.

Queries Gaia DR3 best_neighbour tables for catalog crossmatching,
with VizieR XMatch fallback when Gaia TAP is unavailable.
Uses pyphot filter names internally (ARIADNE-compatible).

Ported from astroARIADNE with cleaner declarative structure.
"""

__all__ = ["Librarian"]

import logging
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.ma as ma
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier as _VizierClass
from astroquery.xmatch import XMatch
from regions import CircleSkyRegion

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, append=True)

# Local Vizier instance — never mutate the module-level singleton, which is
# shared across the whole Python process and breaks any other consumer of
# astroquery in the same interpreter.
Vizier = _VizierClass(row_limit=-1, columns=["all"], timeout=60)
Gaia.TIMEOUT = 60
XMatch.TIMEOUT = 60
Catalogs.TIMEOUT = 60

# Shared executor used by _with_timeout — avoids per-call pool spawn/teardown.
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="librarian")

# Module-level dustmap caches. Constructing SFDQuery loads ~600 MB of FITS;
# repeating per-Star creation makes catalogue-scale runs unusable.
_DUSTMAP_CACHE: dict = {}

# Schlafly+11 SFD V-band coefficient: A_V = 2.742 * E(B-V).
_AV_PER_EBV_SFD = 2.742


# ── Data structures ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MagSpec:
    """One photometric band within a catalog."""

    col: str
    err_col: str
    pyphot: str
    # None -> the column holds a Vega/AB magnitude (default). 'Jy'/'mJy'/'uJy'
    # -> a flux density, converted via Librarian._add_flux (far-IR catalogs).
    flux_unit: str | None = None


@dataclass(frozen=True, slots=True)
class FluxSpec:
    """One flux-density band (Jy/mJy/uJy) within a far-IR catalog.

    Converted to a magnitude on intake via Librarian._add_flux. Carries the
    far-IR quirks: a per-band quality predicate, the IRAS percent-error
    convention, and an optional VizieR sub-table (Herschel splits one catalog
    into one table per band).
    """

    col: str
    err_col: str
    pyphot: str
    flux_unit: str = "Jy"
    err_is_percent: bool = False          # IRAS: e_Fnu_* is % relative, not Jy
    qual: Callable | None = None          # (row) -> bool, per-band quality
    table: str | None = None              # VizieR sub-table (multi-table cats)


@dataclass(frozen=True, slots=True)
class CatalogDef:
    """Declarative catalog definition."""

    vizier_id: str | None
    bands: tuple[MagSpec, ...]
    xmatch_table: str | None = None
    xmatch_id_cols: tuple[str, ...] | None = None
    qc: Callable | None = None


# ── Helpers ──────────────────────────────────────────────────────

_QUERY_TIMEOUT = 60  # seconds per network query


def _with_timeout(func, *args, timeout=_QUERY_TIMEOUT, **kwargs):
    """Run any callable with a hard timeout. Returns None on timeout or error."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    future = _TIMEOUT_POOL.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        logger.warning("Query timed out after %ds", timeout)
        return None
    except Exception as e:
        logger.warning("Query failed: %s", e)
        return None


def _tap_query(service, query, timeout=_QUERY_TIMEOUT):
    """Run a TAP async query with a hard timeout. Returns astropy Table or None."""
    def _run():
        job = service.launch_job_async(query)
        return job.get_results()
    return _with_timeout(_run, timeout=timeout)


from ._qc import (
    _akari_qual,
    _col,
    _iras_qual,
    _qc_2mass_band,
    _qc_galex_band,
    _qc_hppsc,
    _qc_hspsc,
    _qc_mag,
    _qc_ps1,
    _qc_sdss,
    _qc_skymapper,
    _qc_wise_band,
    _qc_wise_extended,
)


# ── Band index maps ─────────────────────────────────────────────

_2MASS_BAND_IDX = {"2MASS_J": 0, "2MASS_H": 1, "2MASS_Ks": 2}
# W3/W4 index into the same AllWISE ph_qual string (qph) as W1/W2. They feed
# IR-excess (config.irx_bands), not the photosphere fit, so they are routed to
# the star's irx set downstream rather than constraining Teff/radius.
_WISE_BAND_IDX = {"WISE_RSR_W1": 0, "WISE_RSR_W2": 1,
                  "WISE_RSR_W3": 2, "WISE_RSR_W4": 3}

# Mean observation epochs (decimal year) for the positionally-fetched IR
# catalogs. Used to proper-motion-propagate the Gaia position before a cone-
# search fallback, so high-PM stars (whose catalog-epoch positions sit many
# arcsec from J2016) are still recovered. AllWISE: mission mean (MJD ~55400 =
# 2010.56). 2MASS: survey mean (~1999.3).
_CATALOG_EPOCH = {"2MASS": 1999.3, "Wise": 2010.56,
                  "AKARI_IRC": 2006.5, "AKARI_FIS": 2006.5, "IRAS": 1983.5,
                  "HPPSC": 2011.0, "HSPSC": 2011.0}


# ── Catalog registry ────────────────────────────────────────────

_CATALOGS = {
    "Gaia": CatalogDef(
        vizier_id="I/355/gaiadr3",
        bands=(
            MagSpec("Gmag", "e_Gmag", "Gaia_G"),
            MagSpec("BPmag", "e_BPmag", "Gaia_BP"),
            MagSpec("RPmag", "e_RPmag", "Gaia_RP"),
        ),
    ),
    "2MASS": CatalogDef(
        vizier_id="II/246/out",
        bands=(
            MagSpec("Jmag", "e_Jmag", "2MASS_J"),
            MagSpec("Hmag", "e_Hmag", "2MASS_H"),
            MagSpec("Kmag", "e_Kmag", "2MASS_Ks"),
        ),
        xmatch_table="tmass_psc_xsc",
        xmatch_id_cols=("2MASS", "_2MASS", "_2M"),
    ),
    "Wise": CatalogDef(
        vizier_id="II/328/allwise",
        bands=(
            MagSpec("W1mag", "e_W1mag", "WISE_RSR_W1"),
            MagSpec("W2mag", "e_W2mag", "WISE_RSR_W2"),
            # W3 (12 um) / W4 (22 um): all-sky IR-excess bands. Vega mags like
            # W1/W2 (reuse the existing mag->flux path); QC via the same qph
            # string (W3=idx 2, W4=idx 3). Routed to the star's irx set.
            MagSpec("W3mag", "e_W3mag", "WISE_RSR_W3"),
            MagSpec("W4mag", "e_W4mag", "WISE_RSR_W4"),
        ),
        xmatch_table="allwise",
        xmatch_id_cols=("AllWISE",),
    ),
    "Pan-STARRS": CatalogDef(
        vizier_id="II/349/ps1",
        bands=(
            MagSpec("gmag", "e_gmag", "PS1_g"),
            MagSpec("rmag", "e_rmag", "PS1_r"),
            MagSpec("imag", "e_imag", "PS1_i"),
            MagSpec("zmag", "e_zmag", "PS1_z"),
            MagSpec("ymag", "e_ymag", "PS1_y"),
        ),
        xmatch_table="panstarrs1",
        xmatch_id_cols=("objID",),
        qc=_qc_ps1,
    ),
    "SDSS": CatalogDef(
        vizier_id="V/147/sdss12",
        bands=(
            MagSpec("umag", "e_umag", "SDSS_u"),
            MagSpec("gmag", "e_gmag", "SDSS_g"),
            MagSpec("rmag", "e_rmag", "SDSS_r"),
            MagSpec("imag", "e_imag", "SDSS_i"),
            MagSpec("zmag", "e_zmag", "SDSS_z"),
        ),
        xmatch_table="sdssdr13",
        xmatch_id_cols=("objID",),
        qc=_qc_sdss,
    ),
    "TYCHO2": CatalogDef(
        vizier_id="I/259/tyc2",
        bands=(
            MagSpec("BTmag", "e_BTmag", "TYCHO_B_MvB"),
            MagSpec("VTmag", "e_VTmag", "TYCHO_V_MvB"),
        ),
        xmatch_table="tycho2tdsc_merge",
    ),
    "APASS": CatalogDef(
        vizier_id=None,
        bands=(
            MagSpec("vmag", "e_vmag", "GROUND_JOHNSON_V"),
            MagSpec("bmag", "e_bmag", "GROUND_JOHNSON_B"),
            MagSpec("g_mag", "e_g_mag", "SDSS_g"),
            MagSpec("r_mag", "e_r_mag", "SDSS_r"),
            MagSpec("i_mag", "e_i_mag", "SDSS_i"),
        ),
        xmatch_table="apassdr9",
    ),
    "SkyMapper": CatalogDef(
        vizier_id=None,
        bands=(
            MagSpec("u_psf", "e_u_psf", "SkyMapper_u"),
            MagSpec("v_psf", "e_v_psf", "SkyMapper_v"),
            MagSpec("g_psf", "e_g_psf", "SkyMapper_g"),
            MagSpec("r_psf", "e_r_psf", "SkyMapper_r"),
            MagSpec("i_psf", "e_i_psf", "SkyMapper_i"),
            MagSpec("z_psf", "e_z_psf", "SkyMapper_z"),
        ),
        xmatch_table="skymapperdr2",
        xmatch_id_cols=("ObjectId", "object_id"),
        qc=_qc_skymapper,
    ),
    "TESS": CatalogDef(
        vizier_id=None,
        bands=(MagSpec("Tmag", "e_Tmag", "TESS"),),
    ),
    "GALEX": CatalogDef(
        vizier_id="II/312/ais",
        bands=(
            MagSpec("FUV", "e_FUV", "GALEX_FUV"),
            MagSpec("NUV", "e_NUV", "GALEX_NUV"),
        ),
    ),
    "GLIMPSE": CatalogDef(
        vizier_id="II/293/glimpse",
        bands=(
            MagSpec("_3.6mag", "e_3.6mag", "SPITZER_IRAC_36"),
            MagSpec("_4.5mag", "e_4.5mag", "SPITZER_IRAC_45"),
        ),
        xmatch_id_cols=("2MASS", "_2MASS", "_2M", "Designation"),
    ),
}

# Standard catalogs: VizieR cone search + crossmatched ID match
_STANDARD_CATALOGS = ("2MASS", "Wise", "Pan-STARRS", "SDSS", "TYCHO2", "GLIMPSE")

# VizieR XMatch fallback config: catalog_name -> (vizier_cat, id_cols)
_XMATCH_CONFIG = {
    "2MASS": ("vizier:II/246/out", ("2MASS", "_2MASS", "_2M")),
    "Wise": ("vizier:II/328/allwise", ("AllWISE",)),
    "Pan-STARRS": ("vizier:II/349/ps1", ("objID",)),
    "SDSS": ("vizier:V/147/sdss12", ("objID",)),
    "TYCHO2": ("vizier:I/259/tyc2", None),
    "APASS": ("vizier:II/336/apass9", None),
}

# Catalogs only available via VizieR cone search (not XMatch)
_CONE_ONLY_FALLBACK = {
    "RAVE": ("III/283/madera", ("ObsID",)),
    "SkyMapper": ("II/379", ("ObjectId", "object_id")),
}

# Far-IR Jy/mJy-native catalogs. Fetched positionally (no Gaia best_neighbour)
# via _fetch_flux_catalog, PM-propagated to each catalog's epoch (see
# _CATALOG_EPOCH -- IRAS is 1983.5, so the offset is large for high-PM stars).
# All bands are IR-excess bands (config.irx_bands): evaluated post-fit, never
# constraining Teff/radius. Value = (CatalogDef, cone radius arcsec, max
# accepted match separation arcsec). The radius tracks the beam: tiny for
# AKARI IRC, ~1.5' for IRAS' big far-IR beam. max_sep tracks the catalog's
# positional accuracy (+margin): a nearest-row match farther out is a field
# source, not the star -- accepting it injects a fake IR excess (a
# 48x-photosphere PACS 100 um "detection" of GJ 1214 motivated the cut).
# AKARI FIS bands ARE wired (config + grids); Spitzer SEIP has its own
# IRSA-TAP fetcher (_fetch_spitzer).
_FAR_IR_CATALOGS = {
    "AKARI_IRC": (CatalogDef(
        vizier_id="II/297/irc",
        bands=(
            FluxSpec("S09", "e_S09", "AKARI_IRC_S9W", "Jy",
                     qual=_akari_qual("q_S09")),
            FluxSpec("S18", "e_S18", "AKARI_IRC_L18W", "Jy",
                     qual=_akari_qual("q_S18")),
        )), 6.0, 3.0),
    "AKARI_FIS": (CatalogDef(
        vizier_id="II/298/fis",
        bands=(
            FluxSpec("S65", "e_S65", "AKARI_FIS_N60", "Jy",
                     qual=_akari_qual("q_S65")),
            FluxSpec("S90", "e_S90", "AKARI_FIS_WIDE_S", "Jy",
                     qual=_akari_qual("q_S90")),
            FluxSpec("S140", "e_S140", "AKARI_FIS_WIDE_L", "Jy",
                     qual=_akari_qual("q_S140")),
            FluxSpec("S160", "e_S160", "AKARI_FIS_N160", "Jy",
                     qual=_akari_qual("q_S160")),
        )), 40.0, 12.0),
    "IRAS": (CatalogDef(
        vizier_id="II/125/main",
        bands=(
            FluxSpec("Fnu_12", "e_Fnu_12", "IRAS_12", "Jy",
                     err_is_percent=True, qual=_iras_qual("q_Fnu_12")),
            FluxSpec("Fnu_25", "e_Fnu_25", "IRAS_25", "Jy",
                     err_is_percent=True, qual=_iras_qual("q_Fnu_25")),
            FluxSpec("Fnu_60", "e_Fnu_60", "IRAS_60", "Jy",
                     err_is_percent=True, qual=_iras_qual("q_Fnu_60")),
            FluxSpec("Fnu_100", "e_Fnu_100", "IRAS_100", "Jy",
                     err_is_percent=True, qual=_iras_qual("q_Fnu_100")),
        )), 90.0, 45.0),
    "HPPSC": (CatalogDef(
        vizier_id="VIII/106",
        bands=(
            FluxSpec("Flux", "snrnoise", "HERSCHEL_PACS_BLUE", "mJy",
                     qual=_qc_hppsc, table="VIII/106/hppsc070"),
            FluxSpec("Flux", "snrnoise", "HERSCHEL_PACS_GREEN", "mJy",
                     qual=_qc_hppsc, table="VIII/106/hppsc100"),
            FluxSpec("Flux", "snrnoise", "HERSCHEL_PACS_RED", "mJy",
                     qual=_qc_hppsc, table="VIII/106/hppsc160"),
        )), 12.0, 5.0),
    "HSPSC": (CatalogDef(
        vizier_id="VIII/112",
        bands=(
            FluxSpec("F250", "s_F250", "HERSCHEL_SPIRE_PSW", "mJy",
                     qual=_qc_hspsc, table="VIII/112/spsc250"),
            FluxSpec("F350", "s_F350", "HERSCHEL_SPIRE_PMW", "mJy",
                     qual=_qc_hspsc, table="VIII/112/spsc350"),
            FluxSpec("F500", "s_F500", "HERSCHEL_SPIRE_PLW", "mJy",
                     qual=_qc_hspsc, table="VIII/112/spsc500"),
        )), 25.0, 10.0),
}

# ── Constants ────────────────────────────────────────────────────

_PLX_ZEROPOINT = 0.037  # Lindegren+21 parallax zero-point (mas)
_PLX_SYS_ERR = 0.02  # systematic floor (mas), added in quadrature


# ── Librarian ────────────────────────────────────────────────────


class Librarian:
    """Automated photometry and astrometry retrieval for isochrone fitting.

    Uses Gaia DR3 best_neighbour tables for catalog crossmatching,
    with VizieR XMatch fallback when Gaia TAP is unavailable.

    Parameters
    ----------
    ra, dec : float
        Right ascension and declination in degrees.
    gaia_id : int, optional
        Gaia DR3 source_id (skips cone search if provided).
    radius : float
        Search radius in arcseconds (default 3).
    ignore : sequence of str
        Catalog names to skip (e.g. ["SDSS", "APASS"]).
    verbose : bool
        Print retrieval summary.
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        *,
        gaia_id: int | None = None,
        radius: float = 3.0,
        ignore: Sequence[str] = (),
        verbose: bool = True,
        feh_source: str = "hypatia",
        hypatia_statistic: str = "median",
        hypatia_uncertainty: str = "std",
        hypatia_solarnorm: str = "asplund09",
        feh_floor: float = 0.10,
    ):
        self.ra = ra
        self.dec = dec
        self._search_radius = radius * u.arcsec
        self._coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        self._ignore = set(ignore)
        # [Fe/H] prior configuration.
        #   feh_source: "hypatia" → Hypatia for [Fe/H] (survey fallback on miss),
        #               surveys for Teff/logg; "survey" → surveys only.
        self._feh_source = feh_source
        self._hypatia_statistic = hypatia_statistic
        self._hypatia_uncertainty = hypatia_uncertainty
        self._hypatia_solarnorm = hypatia_solarnorm
        self._feh_floor = feh_floor

        self._magnitudes: dict[str, tuple[float, float]] = {}
        self._gaia_id: int | None = gaia_id
        self._ids: dict[str, str | int | None] = {}
        self._parallax: float | None = None
        self._parallax_e: float | None = None
        # Gaia proper motion + reference epoch, for PM-propagating the search
        # position to a catalog's epoch (high-PM crossmatch recovery).
        self._pmra: float | None = None
        self._pmdec: float | None = None
        self._gaia_ra: float | None = None
        self._gaia_dec: float | None = None
        self._gaia_ref_epoch: float = 2016.0
        self._teff: float | None = None
        self._teff_e: float | None = None
        self._radius_flame: float | None = None
        self._radius_flame_e: float | None = None
        self._luminosity: float | None = None
        self._luminosity_e: float | None = None
        self._mass: float | None = None
        self._mass_e: float | None = None
        self._age: float | None = None
        self._age_e: float | None = None
        self._distance: float | None = None
        self._distance_e: float | None = None
        self._spectroscopic_params: dict | None = None
        self._tic_id: int | None = None
        self._kic_id: int | None = None
        self.Av: float | None = None

        # Per-source result cache: re-running the same star skips every
        # network call. The key folds in the inputs AND a hash of the
        # librarian package source, so any librarian code change invalidates
        # the cache automatically.
        self._cache_path = self._result_cache_path()
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                self._restore_cache()
                from termcolor import colored
                print(colored(
                    '\t\t*** Librarian: per-source cache hit -- no catalogs '
                    'queried (delete ~/.astroARIADNE/librarian to force a '
                    'refetch). ***', 'cyan'))
                if verbose:
                    self._print_summary()
                return
            except Exception as e:
                logger.warning('librarian cache restore failed (%s); '
                               'refetching', e)

        self._resolve_gaia_id()
        self._query_gaia_params()
        self._query_crossmatches()
        self._query_bailer_jones()
        self._fetch_photometry()
        self._impute_zero_errors()
        self._query_spectroscopic_priors()
        self._query_extinction()
        if self._cache_path:
            try:
                self._save_cache()
            except Exception as e:
                logger.warning('librarian cache save failed: %s', e)

        if verbose:
            self._print_summary()

    # ── per-source result cache ──────────────────────────────────

    # Everything the query sequence mutates; symmetric save/restore.
    _CACHE_FIELDS = (
        '_magnitudes', '_gaia_id', '_ids', '_parallax', '_parallax_e',
        '_pmra', '_pmdec', '_gaia_ra', '_gaia_dec', '_gaia_ref_epoch',
        '_teff', '_teff_e', '_radius_flame', '_radius_flame_e',
        '_luminosity', '_luminosity_e', '_mass', '_mass_e', '_age', '_age_e',
        '_distance', '_distance_e', '_spectroscopic_params', '_tic_id',
        '_kic_id', 'Av', '_xp_filled', '_spherex_added', '_cassis_added',
    )

    @staticmethod
    def _package_hash():
        import hashlib
        d = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.md5()
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.py'):
                with open(os.path.join(d, fn), 'rb') as fh:
                    h.update(fh.read())
        # The cached result also depends on the band registry and filter
        # library (e.g. SPHEREx super-band windows): a new band silently
        # missing from every cached star is worse than a refetch.
        pkg = os.path.dirname(d)
        for extra in (os.path.join(pkg, 'config.py'),
                      os.path.join(pkg, 'Datafiles', 'custom_filters.h5')):
            try:
                with open(extra, 'rb') as fh:
                    h.update(fh.read())
            except OSError:
                pass
        return h.hexdigest()[:12]

    def _result_cache_path(self):
        import hashlib
        key = repr((round(self.ra, 6), round(self.dec, 6), self._gaia_id,
                    tuple(sorted(self._ignore)), self._feh_source,
                    self._hypatia_statistic, self._hypatia_uncertainty,
                    self._hypatia_solarnorm, self._feh_floor,
                    float(self._search_radius.value),
                    self._package_hash()))
        name = hashlib.md5(key.encode()).hexdigest()[:16]
        d = os.path.join(os.path.expanduser('~'), '.astroARIADNE',
                         'librarian')
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return None
        return os.path.join(d, f'{name}.pkl')

    def _save_cache(self):
        import pickle
        state = {f: getattr(self, f, None) for f in self._CACHE_FIELDS}
        tmp = self._cache_path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(state, fh)
        os.replace(tmp, self._cache_path)

    def _restore_cache(self):
        import pickle
        with open(self._cache_path, 'rb') as fh:
            state = pickle.load(fh)
        for f in self._CACHE_FIELDS:
            if f in state:
                setattr(self, f, state[f])

    # ── Gaia ID resolution ───────────────────────────────────────

    def _resolve_gaia_id(self):
        """Find Gaia DR3 source_id via a Gaia TAP cone query if not provided.

        Uses an ADQL cone search against gaiadr3.gaia_source ordered by angular
        separation (nearest match). This is more robust than the VizieR cone
        search and avoids astroquery's ``Gaia.cone_search`` whose ``radius``
        argument became keyword-only in recent versions.
        """
        if self._gaia_id is not None:
            return
        ra = self._coord.ra.deg
        dec = self._coord.dec.deg
        radius_deg = self._search_radius.to(u.deg).value
        adql = (
            "SELECT source_id, "
            f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) "
            "AS ang_sep FROM gaiadr3.gaia_source "
            "WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) "
            "ORDER BY ang_sep ASC"
        )
        try:
            res = _with_timeout(lambda: Gaia.launch_job(adql).get_results())
            if res is None or len(res) == 0:
                logger.warning("No Gaia DR3 source within search radius")
                return
            self._gaia_id = int(res["source_id"][0])
            logger.info("Resolved Gaia DR3 ID: %d", self._gaia_id)
        except Exception as e:
            logger.warning("Gaia ID resolution failed: %s", e)

    # ── Gaia stellar parameters ──────────────────────────────────

    def _query_gaia_params(self):
        """Query Gaia DR3 for parallax, Teff, photometry, and FLAME params."""
        if self._gaia_id is None:
            return

        main_cats = _with_timeout(
            Vizier.query_constraints,
            catalog="I/355/gaiadr3", Source=str(self._gaia_id),
        )
        if not main_cats or len(main_cats[0]) == 0:
            logger.warning("Gaia DR3 source %d not found", self._gaia_id)
            return
        main = main_cats[0][0]

        # Proper motion + Gaia position, for PM-propagated cone fallbacks.
        self._pmra = _col(main, "pmRA")     # mas/yr, already *cos(dec)
        self._pmdec = _col(main, "pmDE")
        self._gaia_ra = _col(main, "RA_ICRS")
        self._gaia_dec = _col(main, "DE_ICRS")

        # FLAME astrophysical parameters table
        ap_cats = _with_timeout(
            Vizier.query_constraints,
            catalog="I/355/paramp", Source=str(self._gaia_id),
        )
        ap = ap_cats[0][0] if ap_cats and len(ap_cats[0]) > 0 else None

        # Parallax (Lindegren+21 correction)
        plx = _col(main, "Plx")
        plx_e = _col(main, "e_Plx")
        if plx is not None:
            self._parallax = float(plx) + _PLX_ZEROPOINT
            if plx_e is not None:
                self._parallax_e = np.sqrt(float(plx_e) ** 2 + _PLX_SYS_ERR**2)

        # Gaia GSP-Phot Teff is intentionally NOT extracted as a likelihood
        # prior. The photometric SED already constrains Teff via BC tables;
        # injecting GSP-Phot as a separate observable double-counts the
        # information and pins to the sampler-percentile width (formal σ
        # ≈ 1–5 K) which is ~100x narrower than the actual systematic vs
        # spectroscopy (Andrae+23 quotes ±100–200 K). Spectroscopic priors
        # used by LACHESIS are [Fe/H] (and optionally logg) — Teff is
        # output, not input.

        # FLAME radius — 5x error inflation (ARIADNE convention to avoid
        # over-constraining from crude FLAME estimates)
        rad = _col(ap, "Rad-Flame") if ap is not None else None
        if rad is not None:
            self._radius_flame = float(rad)
            lo = _col(ap, "b_Rad-Flame")
            hi = _col(ap, "B_Rad-Flame")
            if lo is not None and hi is not None:
                self._radius_flame_e = 5 * max(
                    abs(float(rad) - float(lo)), abs(float(hi) - float(rad))
                )

        # FLAME luminosity
        lum = _col(ap, "Lum-Flame") if ap is not None else None
        if lum is not None:
            self._luminosity = float(lum)
            lo = _col(ap, "b_Lum-Flame")
            hi = _col(ap, "B_Lum-Flame")
            if lo is not None and hi is not None:
                self._luminosity_e = max(
                    abs(float(lum) - float(lo)), abs(float(hi) - float(lum))
                )

        # FLAME mass
        mass_v = _col(ap, "Mass-Flame") if ap is not None else None
        if mass_v is not None:
            self._mass = float(mass_v)
            lo = _col(ap, "b_Mass-Flame")
            hi = _col(ap, "B_Mass-Flame")
            if lo is not None and hi is not None:
                self._mass_e = max(
                    abs(float(mass_v) - float(lo)), abs(float(hi) - float(mass_v))
                )

        # FLAME age
        age_v = _col(ap, "Age-Flame") if ap is not None else None
        if age_v is not None:
            self._age = float(age_v)
            lo = _col(ap, "b_Age-Flame")
            hi = _col(ap, "B_Age-Flame")
            if lo is not None and hi is not None:
                self._age_e = max(
                    abs(float(age_v) - float(lo)), abs(float(hi) - float(age_v))
                )

        # Gaia photometry
        for band in _CATALOGS["Gaia"].bands:
            mag_v = _col(main, band.col)
            err_v = _col(main, band.err_col)
            if _qc_mag(mag_v, err_v):
                self._add_mag(
                    band.pyphot,
                    float(mag_v),
                    float(err_v) if err_v is not None else 0.01,
                )

    # ── Gaia TAP crossmatch ──────────────────────────────────────

    def _query_crossmatches(self):
        """Query Gaia DR3 best_neighbour tables for external catalog IDs."""
        if self._gaia_id is None:
            return

        self._ids["Gaia"] = self._gaia_id
        tap_down = False
        xmatch_needed = []

        for cat_name, cat_def in _CATALOGS.items():
            if cat_name == "Gaia":
                continue
            if cat_name in self._ignore:
                self._ids[cat_name] = None
                continue
            if cat_def.xmatch_table is None:
                continue

            if tap_down:
                xmatch_needed.append(cat_name)
                continue

            query = (
                f"SELECT xmatch.original_ext_source_id "
                f"FROM gaiadr3.{cat_def.xmatch_table}_best_neighbour AS xmatch "
                f"WHERE xmatch.source_id = {int(self._gaia_id)}"
            )

            result = _tap_query(Gaia, query)
            if result is None:
                tap_down = True
                xmatch_needed.append(cat_name)
                self._ids[cat_name] = None
            elif len(result) > 0:
                self._ids[cat_name] = result[0][0]
            else:
                self._ids[cat_name] = None
                logger.info("No %s crossmatch for Gaia %d", cat_name, self._gaia_id)

        if xmatch_needed:
            logger.warning(
                "Gaia TAP down, falling back to VizieR XMatch for: %s", xmatch_needed
            )
            self._xmatch_fallback(xmatch_needed)

    def _xmatch_fallback(self, catalogs_needed):
        """Recover catalog IDs via VizieR XMatch when Gaia TAP is down."""
        region = CircleSkyRegion(self._coord, radius=self._search_radius)

        for cat_name in catalogs_needed:
            if cat_name in self._ignore:
                continue

            # Cone-only fallback (RAVE, SkyMapper — not on XMatch server)
            if cat_name in _CONE_ONLY_FALLBACK:
                viz_cat, id_cols = _CONE_ONLY_FALLBACK[cat_name]
                result = _with_timeout(
                    Vizier.query_region,
                    self._coord, radius=self._search_radius, catalog=viz_cat,
                )
                if not result or len(result[0]) == 0:
                    self._ids[cat_name] = None
                    continue
                result[0].sort("_r")
                row = result[0][0]
                found = next(
                    (_col(row, c) for c in id_cols if c in result[0].colnames),
                    None,
                )
                self._ids[cat_name] = found
                continue

            if cat_name not in _XMATCH_CONFIG:
                continue

            vizier_cat, id_cols = _XMATCH_CONFIG[cat_name]

            xm = _with_timeout(
                XMatch.query,
                cat1="vizier:I/355/gaiadr3",
                cat2=vizier_cat,
                max_distance=self._search_radius,
                area=region,
            )
            try:
                if xm is None or len(xm) == 0:
                    self._ids[cat_name] = None
                    continue

                xm.sort("angDist")
                row = xm[0]
                if "Source" in xm.colnames:
                    src_mask = xm["Source"] == self._gaia_id
                    if src_mask.sum() > 0:
                        row = xm[src_mask][0]

                # Pattern B: APASS — extract mags directly from XMatch row
                if cat_name == "APASS":
                    self._ids["APASS"] = "xmatch_done"
                    for band in _CATALOGS["APASS"].bands:
                        if band.pyphot in self._magnitudes:
                            continue
                        mag_v = _col(row, band.col)
                        err_v = _col(row, band.err_col)
                        if _qc_mag(mag_v, err_v):
                            err_f = float(err_v) if err_v is not None else 0.02
                            self._add_mag(band.pyphot, float(mag_v), err_f)
                    continue

                # Tycho-2: composite TYC1-TYC2-TYC3 ID
                if cat_name == "TYCHO2":
                    try:
                        tyc_id = (
                            f"{int(row['TYC1'])}-{int(row['TYC2'])}-{int(row['TYC3'])}"
                        )
                        self._ids["TYCHO2"] = tyc_id
                    except (KeyError, ValueError):
                        self._ids["TYCHO2"] = None
                    continue

                # Pattern A: extract ID from first matching column
                if id_cols is not None:
                    found = next(
                        (_col(row, c) for c in id_cols if c in xm.colnames),
                        None,
                    )
                    self._ids[cat_name] = found
                else:
                    self._ids[cat_name] = None

            except Exception as e:
                logger.warning("XMatch fallback failed for %s: %s", cat_name, e)
                self._ids[cat_name] = None

    # ── Bailer-Jones distance ────────────────────────────────────

    def _query_bailer_jones(self):
        """Query Bailer-Jones EDR3 geometric distance."""
        if self._gaia_id is None:
            return
        res = _with_timeout(
            Vizier.query_constraints,
            catalog="I/352/gedr3dis", Source=str(self._gaia_id),
        )
        if not res or len(res[0]) == 0:
            return
        try:
            row = res[0][0]
            dist = float(row["rgeo"])
            lo = dist - float(row["b_rgeo"])
            hi = float(row["B_rgeo"]) - dist
            self._distance = dist
            self._distance_e = max(lo, hi)
        except Exception as e:
            logger.warning("Bailer-Jones row parse failed: %s", e)

    # ── Photometry retrieval ─────────────────────────────────────

    def _fetch_photometry(self):
        """Dispatch photometry retrieval for all catalogs."""
        if self._gaia_id is None:
            return

        # Bulk VizieR cone search for standard catalogs
        viz_ids = [
            _CATALOGS[name].vizier_id
            for name in _STANDARD_CATALOGS
            if name not in self._ignore
            and _CATALOGS[name].vizier_id is not None
            and self._ids.get(name) is not None
        ]
        cats = {}
        if viz_ids:
            result = _with_timeout(
                Vizier.query_region,
                self._coord, radius=self._search_radius, catalog=viz_ids,
            )
            if result:
                cats = result

        for cat_name in _STANDARD_CATALOGS:
            if cat_name in self._ignore:
                continue
            ext_id = self._ids.get(cat_name)
            if ext_id is None:
                continue
            self._fetch_standard(cat_name, _CATALOGS[cat_name], cats, ext_id)

        # Proper-motion-propagated cone fallback for the IR catalogs. The bulk
        # cone above is centred on the input coordinates; for high-PM stars the
        # 2MASS (~1999) and AllWISE (~2010) sources sit many arcsec away and are
        # missed, and AllWISE frequently has no Gaia best_neighbour at all
        # (Gaia's own crossmatch drops the displaced source). Re-fetch any
        # still-missing 2MASS/WISE bands from a tight cone at the position
        # propagated to each catalog's epoch.
        for cat_name in ("2MASS", "Wise"):
            if cat_name in self._ignore:
                continue
            cat_def = _CATALOGS[cat_name]
            if all(b.pyphot in self._magnitudes for b in cat_def.bands):
                continue
            self._fetch_by_cone(cat_name, cat_def)

        # Jy-native far-IR catalogs (IRAS, AKARI, Herschel) -> IR-excess bands.
        if "far_ir" not in self._ignore:
            self._fetch_far_ir()
        if "Spitzer" not in self._ignore:
            try:
                self._fetch_spitzer()
            except Exception as e:
                logger.warning("Spitzer SEIP fetch failed: %s", e)

        if "TESS" not in self._ignore:
            self._fetch_tess()
        if "APASS" not in self._ignore:
            self._fetch_apass()
        if "SkyMapper" not in self._ignore:
            self._fetch_skymapper()
        if "GALEX" not in self._ignore:
            self._fetch_galex()
        if "MERMILLIOD" not in self._ignore:
            self._fetch_mermilliod()
        if "STROMGREN" not in self._ignore:
            self._fetch_stromgren()

        # Gaia XP synthetic gap-filling runs LAST: it needs the complete
        # observed band inventory to know which wavelengths are uncovered
        # (and first-write-wins means real photometry always beats it).
        if "XP" not in self._ignore:
            try:
                self._fetch_xp_synth()
            except Exception as e:
                logger.warning("XP synthetic gap-filling failed: %s", e)

        # DESI/BOSS optical spectro-photometry: the FAINT-end continuation
        # of the XP gap-filling ladder (G > 16, where XP stops firing).
        # DESI before BOSS: denser MWS sample, better-behaved fluxing.
        if "DESI" not in self._ignore:
            try:
                self._fetch_optspec("DESI")
            except Exception as e:
                logger.warning("DESI optspec fill skipped: %s", e)
        if "BOSS" not in self._ignore:
            try:
                self._fetch_optspec("BOSS")
            except Exception as e:
                logger.warning("BOSS optspec fill skipped: %s", e)

        # SPHEREx QR2 super-band spectro-photometry (independent of XP;
        # results are cached per source, so only the first Star pays the
        # extraction cost).
        if "SPHEREX" not in self._ignore:
            try:
                self._fetch_spherex()
            except Exception as e:
                logger.warning("SPHEREx extraction failed: %s", e)

        # CASSIS / Spitzer-IRS low-res super-band spectro-photometry (5-38 um,
        # all IR-excess bands). One cone search, cached per position.
        if "CASSIS" not in self._ignore:
            try:
                self._fetch_cassis()
            except Exception as e:
                logger.warning("CASSIS/IRS extraction failed: %s", e)

        # IUE/INES UV spectro-photometry: the GALEX anchor for bright stars
        # (GALEX saturates at FUV/NUV ~ 14; IUE observed exactly those).
        if "IUE" not in self._ignore:
            try:
                self._fetch_iue()
            except Exception as e:
                logger.warning("IUE/INES synthesis failed: %s", e)

    def _fetch_standard(self, cat_name, cat_def, cats, ext_id):
        """Fetch photometry from VizieR results, filtered by crossmatched ID."""
        if cat_def.vizier_id is None:
            return

        try:
            cat = cats[cat_def.vizier_id]
            cat.sort("_r")
        except (TypeError, KeyError):
            logger.info("No VizieR data for %s", cat_name)
            return

        # Filter by crossmatched ID
        if cat_name == "TYCHO2":
            try:
                tyc1, tyc2, tyc3 = str(ext_id).split("-")
                mask = (
                    (cat["TYC1"] == int(tyc1))
                    & (cat["TYC2"] == int(tyc2))
                    & (cat["TYC3"] == int(tyc3))
                )
            except (ValueError, KeyError):
                return
        elif cat_name == "GLIMPSE":
            tmass_id = self._ids.get("2MASS")
            if tmass_id is None:
                return
            id_cols = cat_def.xmatch_id_cols or ()
            col_name = next((c for c in id_cols if c in cat.colnames), None)
            if col_name is None:
                return
            mask = cat[col_name] == tmass_id
        else:
            id_cols = cat_def.xmatch_id_cols or ()
            col_name = next((c for c in id_cols if c in cat.colnames), None)
            if col_name is None:
                return
            mask = cat[col_name] == ext_id

        matched = cat[mask]
        if len(matched) == 0:
            return
        row = matched[0]

        # Catalog-level QC
        if cat_def.qc is not None and not cat_def.qc(row):
            logger.info("%s failed catalog QC", cat_name)
            return

        # WISE extended source check
        if cat_name == "Wise" and not _qc_wise_extended(row):
            logger.info("WISE source is extended, skipping")
            return

        for band in cat_def.bands:
            if band.pyphot in self._magnitudes:
                continue

            # Per-band QC
            if band.pyphot in _2MASS_BAND_IDX:
                if not _qc_2mass_band(row, _2MASS_BAND_IDX[band.pyphot]):
                    continue
            if band.pyphot in _WISE_BAND_IDX:
                if not _qc_wise_band(row, _WISE_BAND_IDX[band.pyphot]):
                    continue

            mag_v = _col(row, band.col)
            err_v = _col(row, band.err_col)
            if not _qc_mag(mag_v, err_v):
                continue

            err_f = float(err_v) if err_v is not None else 0.02
            self._add_mag(band.pyphot, float(mag_v), err_f)

    def _pm_available(self):
        """True when a finite Gaia proper motion is on hand (propagation to
        other catalog epochs is then trustworthy)."""
        try:
            return bool(self._pmra is not None and self._pmdec is not None
                        and np.isfinite(float(self._pmra))
                        and np.isfinite(float(self._pmdec)))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _match_sep_arcsec(cat, row):
        """Separation of a VizieR match row in arcsec from the ``_r`` column
        (unit taken from the table metadata; None when unavailable)."""
        try:
            unit = cat['_r'].unit or u.arcsec
            return float((float(row['_r']) * unit).to(u.arcsec).value)
        except (KeyError, TypeError, ValueError):
            return None

    def _propagated_coord(self, epoch_year):
        """Gaia position proper-motion-propagated to ``epoch_year`` (decimal
        year). Falls back to the input coordinate when PM is unavailable --
        callers that crossmatch epoch-distant catalogs must check
        :meth:`_pm_available` first (the far-IR fetcher skips loudly)."""
        if not self._pm_available():
            return self._coord
        ra = self._gaia_ra if self._gaia_ra is not None else self.ra
        dec = self._gaia_dec if self._gaia_dec is not None else self.dec
        kw = dict(
            ra=float(ra) * u.deg, dec=float(dec) * u.deg,
            pm_ra_cosdec=float(self._pmra) * u.mas / u.yr,
            pm_dec=float(self._pmdec) * u.mas / u.yr,
            obstime=Time(self._gaia_ref_epoch, format="decimalyear"),
        )
        if self._parallax is not None and float(self._parallax) > 0:
            kw["distance"] = (1000.0 / float(self._parallax)) * u.pc
        return SkyCoord(**kw).apply_space_motion(
            Time(epoch_year, format="decimalyear"))

    def _fetch_by_cone(self, cat_name, cat_def):
        """Fetch a catalog from a tight cone at the PM-propagated position.

        Fallback for catalogs missed by the primary path -- either because the
        Gaia best_neighbour table is empty (e.g. AllWISE for a high-PM star) or
        because the source moved out of the input-position cone. Matches by
        crossmatched ID when one is available, otherwise the nearest source to
        the propagated position; the same per-band QC as the primary path
        guards against false matches. Idempotent: never overwrites a band the
        primary path already attached.
        """
        if cat_def.vizier_id is None:
            return
        epoch = _CATALOG_EPOCH.get(cat_name)
        coord = (self._propagated_coord(epoch) if epoch is not None
                 else self._coord)
        result = _with_timeout(
            Vizier.query_region, coord,
            radius=self._search_radius, catalog=cat_def.vizier_id,
        )
        if not result:
            logger.info("No PM-propagated cone data for %s", cat_name)
            return
        try:
            cat = result[cat_def.vizier_id]
            cat.sort("_r")
        except (TypeError, KeyError):
            return

        # Prefer the crossmatched ID if we have one and it lands in the cone,
        # else the nearest source to the propagated position.
        row = cat[0]
        ext_id = self._ids.get(cat_name)
        if ext_id is not None:
            id_cols = cat_def.xmatch_id_cols or ()
            col_name = next((c for c in id_cols if c in cat.colnames), None)
            if col_name is not None:
                matched = cat[cat[col_name] == ext_id]
                if len(matched):
                    row = matched[0]

        # Catalog-level QC (same predicates as the primary path).
        if cat_def.qc is not None and not cat_def.qc(row):
            logger.info("%s failed catalog QC (cone fallback)", cat_name)
            return
        if cat_name == "Wise" and not _qc_wise_extended(row):
            logger.info("WISE source is extended (cone fallback), skipping")
            return

        for band in cat_def.bands:
            if band.pyphot in self._magnitudes:
                continue
            if band.pyphot in _2MASS_BAND_IDX:
                if not _qc_2mass_band(row, _2MASS_BAND_IDX[band.pyphot]):
                    continue
            if band.pyphot in _WISE_BAND_IDX:
                if not _qc_wise_band(row, _WISE_BAND_IDX[band.pyphot]):
                    continue
            mag_v = _col(row, band.col)
            err_v = _col(row, band.err_col)
            if band.flux_unit is not None:
                self._add_flux(band.pyphot, mag_v, err_v, band.flux_unit)
                continue
            if not _qc_mag(mag_v, err_v):
                continue
            err_f = float(err_v) if err_v is not None else 0.02
            self._add_mag(band.pyphot, float(mag_v), err_f)
        logger.info("Recovered %s via PM-propagated cone fallback", cat_name)

    def _fetch_far_ir(self):
        """Fetch the Jy-native far-IR catalogs (IRAS, AKARI, Herschel).

        These have no Gaia best_neighbour, report F_nu in Jy/mJy rather than
        magnitudes, and span epochs from 1983.5 (IRAS) to ~2011 (Herschel), so
        each is fetched positionally at its PM-propagated epoch. Their bands are
        IR-excess bands; missing/low-quality bands are simply skipped.
        """
        for cat_name, (cat_def, radius, max_sep) in _FAR_IR_CATALOGS.items():
            if cat_name in self._ignore:
                continue
            if all(b.pyphot in self._magnitudes for b in cat_def.bands):
                continue
            try:
                self._fetch_flux_catalog(cat_name, cat_def, radius, max_sep)
            except Exception as e:
                logger.warning("Far-IR fetch failed for %s: %s", cat_name, e)

    def _fetch_flux_catalog(self, cat_name, cat_def, radius, max_sep=None):
        """Fetch one Jy-native catalog at the PM-propagated catalog epoch.

        Handles per-band quality predicates, the IRAS percent-error convention,
        and multi-table catalogs (Herschel splits one VizieR catalog into one
        sub-table per band). Each accepted band is converted Jy->mag by
        :meth:`_add_flux`. ``max_sep`` (arcsec) rejects nearest-row matches
        beyond the catalog's positional accuracy -- field sources masquerading
        as the star inject fake IR excesses, the worst possible failure for
        the IRX endpoint.
        """
        epoch = _CATALOG_EPOCH.get(cat_name)
        if epoch is not None and not self._pm_available():
            # The star may sit arcmin from its catalog-epoch position
            # (IRAS is 33 yr before Gaia); nearest-row matching at the
            # unpropagated position would risk attaching a random field
            # source. Loud skip instead of a silent wrong answer.
            from termcolor import colored
            print(colored(
                '\t\t*** far-IR catalog %s SKIPPED: Gaia proper motion '
                'unavailable, cannot propagate to the %s catalog epoch. '
                'Re-run when the Gaia query succeeds, or pass the band '
                'manually via add_mag. ***' % (cat_name, epoch), 'yellow'))
            logger.warning("far-IR %s skipped: no proper motion", cat_name)
            return
        coord = (self._propagated_coord(epoch) if epoch is not None
                 else self._coord)
        result = _with_timeout(
            Vizier.query_region, coord,
            radius=radius * u.arcsec, catalog=cat_def.vizier_id,
        )
        if not result:
            logger.info("No far-IR data for %s", cat_name)
            return
        for band in cat_def.bands:
            if band.pyphot in self._magnitudes:
                continue
            tbl = band.table or cat_def.vizier_id
            try:
                cat = result[tbl]
            except (TypeError, KeyError):
                continue
            if len(cat) == 0:
                continue
            cat.sort("_r")
            row = cat[0]
            sep = self._match_sep_arcsec(cat, row)
            if max_sep is not None and sep is not None and sep > max_sep:
                from termcolor import colored
                print(colored(
                    '\t\t*** %s %s match REJECTED: nearest source is '
                    '%.1f" from the propagated position (> %.0f" catalog '
                    'accuracy) -- field source, not the star. ***'
                    % (cat_name, band.pyphot, sep, max_sep), 'yellow'))
                continue
            if band.qual is not None and not band.qual(row):
                continue
            flux = _col(row, band.col)
            err = _col(row, band.err_col)
            if band.err_is_percent and flux is not None and err is not None:
                try:
                    err = abs(float(flux)) * float(err) / 100.0
                except (TypeError, ValueError):
                    err = None
            self._add_flux(band.pyphot, flux, err, band.flux_unit)
        logger.info("Queried far-IR catalog %s", cat_name)

    def _fetch_spitzer(self):
        """Spitzer Enhanced Imaging Products (SEIP): IRAC 3.6-8.0 + MIPS 24.

        Not usable on VizieR (only a 1000-row excerpt); the full 42M-source
        list is IRSA catalog 'slphotdr4', queried via IRSA TAP. Fluxes are in
        uJy. Bright stars saturate IRAC, so per-band saturation (softsatflag)
        and detection (fluxtype==1) cuts are mandatory. PM-propagated to the
        cryo-mission epoch (~2007).
        """
        # (col prefix, pyphot, flux suffix, has soft-saturation flag)
        bands = [
            ("i1", "SPITZER_IRAC_36", "f_ap1", True),
            ("i2", "SPITZER_IRAC_45", "f_ap1", True),
            ("i3", "SPITZER_IRAC_58", "f_ap1", True),
            ("i4", "SPITZER_IRAC_80", "f_ap1", True),
            ("m1", "SPITZER_MIPS_24", "f_psf", False),
        ]
        if all(p in self._magnitudes for _, p, _, _ in bands):
            return
        try:
            from astroquery.ipac.irsa import Irsa
        except Exception:
            return
        coord = self._propagated_coord(2007.0)
        res = _with_timeout(
            Irsa.query_region, coord, catalog="slphotdr4",
            spatial="Cone", radius=self._search_radius,
        )
        if res is None or len(res) == 0:
            logger.info("No Spitzer SEIP data")
            return
        # Nearest source to the propagated position.
        if "ra" in res.colnames and "dec" in res.colnames:
            sep = coord.separation(
                SkyCoord(res["ra"], res["dec"], unit="deg"))
            row = res[int(np.argmin(sep))]
        else:
            row = res[0]
        for prefix, pyphot, fsfx, has_sat in bands:
            if pyphot in self._magnitudes:
                continue
            ftype = _col(row, f"{prefix}_fluxtype")
            if ftype is None or int(ftype) != 1:        # 1 = real >=3-sigma flux
                continue
            if has_sat:
                sat = _col(row, f"{prefix}_softsatflag")
                if sat is None or int(sat) != 0:        # saturated/nonlinear
                    continue
            flux = _col(row, f"{prefix}_{fsfx}")
            err = _col(row, f"{prefix}_d{fsfx}")
            self._add_flux(pyphot, flux, err, "uJy")
        logger.info("Queried Spitzer SEIP (IRSA slphotdr4)")

    def _fetch_tess(self):
        """Query TESS TIC via MAST, crossmatched on Gaia ID."""
        result = _with_timeout(
            Catalogs.query_region,
            self._coord, radius=self._search_radius, catalog="TIC",
        )
        if result is None or len(result) == 0:
            return

        result.sort("dstArcSec")
        gaia_mask = result["GAIA"] == str(self._gaia_id)
        matched = result[gaia_mask]
        if len(matched) == 0:
            return

        row = matched[0]
        if _col(row, "objType") != "STAR":
            return

        self._tic_id = int(row["ID"])
        kic = _col(row, "KIC")
        if kic is not None:
            self._kic_id = int(kic)

        band = _CATALOGS["TESS"].bands[0]
        if band.pyphot in self._magnitudes:
            return

        mag_v = _col(row, band.col)
        err_v = _col(row, band.err_col)
        if _qc_mag(mag_v, err_v):
            err_f = float(err_v) if err_v is not None else 0.01
            self._add_mag(band.pyphot, float(mag_v), err_f)

    def _fetch_apass(self):
        """Query APASS via Gaia TAP external.apassdr9 table."""
        apass_id = self._ids.get("APASS")
        if apass_id is None or apass_id == "xmatch_done":
            return

        query = (
            "SELECT apass.vmag, apass.e_vmag, "
            "apass.bmag, apass.e_bmag, "
            "apass.g_mag, apass.e_g_mag, "
            "apass.r_mag, apass.e_r_mag, "
            "apass.i_mag, apass.e_i_mag "
            "FROM external.apassdr9 AS apass "
            "INNER JOIN gaiadr3.apassdr9_best_neighbour AS xmatch "
            "ON apass.recno = xmatch.original_ext_source_id "
            f"WHERE xmatch.source_id = {int(self._gaia_id)}"
        )

        result = _tap_query(Gaia, query)
        if result is None or len(result) == 0:
            return

        row = result[0]
        for band in _CATALOGS["APASS"].bands:
            if band.pyphot in self._magnitudes:
                continue
            mag_v = _col(row, band.col)
            err_v = _col(row, band.err_col)
            if _qc_mag(mag_v, err_v):
                err_f = float(err_v) if err_v is not None else 0.02
                self._add_mag(band.pyphot, float(mag_v), err_f)

    def _fetch_skymapper(self):
        """Query SkyMapper DR2 via SkyMapper TAP service."""
        sm_id = self._ids.get("SkyMapper")
        if sm_id is None:
            return

        from astroquery.utils.tap.core import TapPlus

        tap = TapPlus(url="https://api.skymapper.nci.org.au/public/tap/")
        query = (
            "SELECT object_id, u_psf, e_u_psf, v_psf, e_v_psf, "
            "g_psf, e_g_psf, r_psf, e_r_psf, i_psf, e_i_psf, "
            "z_psf, e_z_psf, flags "
            f"FROM dr2.master WHERE object_id = {int(sm_id)}"
        )
        result = _tap_query(tap, query)
        if result is None or len(result) == 0:
            return

        row = result[0]
        if not _qc_skymapper(row):
            return

        for band in _CATALOGS["SkyMapper"].bands:
            if band.pyphot in self._magnitudes:
                continue
            mag_v = _col(row, band.col)
            err_v = _col(row, band.err_col)
            if _qc_mag(mag_v, err_v):
                err_f = float(err_v) if err_v is not None else 0.02
                self._add_mag(band.pyphot, float(mag_v), err_f)

    def _fetch_galex(self):
        """Query GALEX via VizieR XMatch with Gaia DR3.

        Falls back to a proper-motion-propagated cone against the GUVcat
        (II/335) when the J2016-position XMatch misses: GALEX's mean epoch is
        ~2007, so a high-PM star (e.g. a nearby WD at ~1 arcsec/yr) sits many
        arcsec from its Gaia position and the 3 arcsec match fails.
        """
        region = CircleSkyRegion(self._coord, radius=self._search_radius)
        xm = _with_timeout(
            XMatch.query,
            cat1="vizier:I/355/gaiadr3",
            cat2="vizier:II/312/ais",
            max_distance=self._search_radius,
            area=region,
        )
        try:
            if xm is not None and len(xm) > 0:
                xm.sort("angDist")
                src_mask = xm["Source"] == self._gaia_id
                if src_mask.sum() > 0:
                    row = xm[src_mask][0]
                    for band in _CATALOGS["GALEX"].bands:
                        if band.pyphot in self._magnitudes:
                            continue
                        if not _qc_galex_band(row, band.pyphot):
                            continue
                        mag_v = _col(row, band.col)
                        err_v = _col(row, band.err_col)
                        if _qc_mag(mag_v, err_v):
                            err_f = (float(err_v) if err_v is not None
                                     else 0.02)
                            self._add_mag(band.pyphot, float(mag_v), err_f)
            if ('GALEX_FUV' in self._magnitudes
                    and 'GALEX_NUV' in self._magnitudes):
                return
            # PM-propagated cone fallback (GUVcat II/335; FUVmag/NUVmag).
            coord = self._propagated_coord(2007.0)
            res = _with_timeout(
                Vizier.query_region, coord,
                radius=self._search_radius, catalog='II/335/galex_ais',
            )
            if not res:
                return
            cat = res['II/335/galex_ais']
            cat.sort('_r')
            row = cat[0]
            for pyphot, mc, ec in (("GALEX_FUV", "FUVmag", "e_FUVmag"),
                                   ("GALEX_NUV", "NUVmag", "e_NUVmag")):
                if pyphot in self._magnitudes:
                    continue
                if not _qc_galex_band(row, pyphot):
                    continue
                mag_v = _col(row, mc)
                err_v = _col(row, ec)
                if _qc_mag(mag_v, err_v):
                    err_f = float(err_v) if err_v is not None else 0.02
                    self._add_mag(pyphot, float(mag_v), err_f)
                    logger.info('GALEX %s recovered via PM-propagated cone',
                                pyphot)
        except Exception as e:
            logger.warning("GALEX XMatch query failed: %s", e)

    def _fetch_mermilliod(self):
        """Query Mermilliod Johnson UBV via VizieR XMatch.

        Color algebra: V direct, B = (B-V) + V, U = (U-B) + B.
        """
        region = CircleSkyRegion(self._coord, radius=5 * u.arcmin)
        xm = _with_timeout(
            XMatch.query,
            cat1="vizier:I/355/gaiadr3",
            cat2="vizier:II/168/ubvmeans",
            max_distance=3 * u.arcmin,
            area=region,
        )
        try:
            if xm is None or len(xm) == 0:
                return

            xm.sort("angDist")
            src_mask = xm["Source"] == self._gaia_id
            if src_mask.sum() == 0:
                return
            row = xm[src_mask][0]

            v = _col(row, "Vmag")
            v_e = _col(row, "e_Vmag")
            if not _qc_mag(v, v_e):
                return

            v, v_e = float(v), float(v_e) if v_e is not None else 0.0

            if v_e > 0 and "GROUND_JOHNSON_V" not in self._magnitudes:
                self._add_mag("GROUND_JOHNSON_V", v, v_e)

            bv = _col(row, "B-V")
            bv_e = _col(row, "e_B-V")
            if _qc_mag(bv, bv_e):
                bv = float(bv)
                bv_e = float(bv_e) if bv_e is not None else 0.0
                b = bv + v
                b_e = np.sqrt(v_e**2 + bv_e**2)
                if b_e > 0 and "GROUND_JOHNSON_B" not in self._magnitudes:
                    self._add_mag("GROUND_JOHNSON_B", b, b_e)

                ub = _col(row, "U-B")
                ub_e = _col(row, "e_U-B")
                if _qc_mag(ub, ub_e):
                    ub = float(ub)
                    ub_e = float(ub_e) if ub_e is not None else 0.0
                    u_mag = ub + b
                    u_e = np.sqrt(b_e**2 + ub_e**2)
                    if u_e > 0 and "GROUND_JOHNSON_U" not in self._magnitudes:
                        self._add_mag("GROUND_JOHNSON_U", u_mag, u_e)

        except Exception as e:
            logger.warning("Mermilliod XMatch query failed: %s", e)

    def _fetch_stromgren(self):
        """Query Stromgren from Paunzen+15 and Hauck+98 via VizieR XMatch.

        Index algebra: b=(b-y)+y, v=m1+2(b-y)+y, u=c1+2m1+3(b-y)+y.
        """
        xm_configs = [
            ("vizier:J/A+A/580/A23/catalog", self._search_radius),
            ("vizier:II/215/catalog", self._search_radius),
        ]
        region = CircleSkyRegion(self._coord, radius=self._search_radius)

        for cat2, max_dist in xm_configs:
            xm = _with_timeout(
                XMatch.query,
                cat1="vizier:I/355/gaiadr3",
                cat2=cat2,
                max_distance=max_dist,
                area=region,
            )
            try:
                if xm is None or len(xm) == 0:
                    continue

                xm.sort("angDist")
                src_mask = xm["Source"] == self._gaia_id
                if src_mask.sum() == 0:
                    continue
                row = xm[src_mask][0]

                y = _col(row, "Vmag")
                y_e = _col(row, "e_Vmag")
                if not _qc_mag(y, y_e):
                    continue

                y = float(y)
                y_e = float(y_e) if y_e is not None else 0.0

                by = _col(row, "b-y")
                by_e = _col(row, "e_b-y")
                m1 = _col(row, "m1")
                m1_e = _col(row, "e_m1")
                c1 = _col(row, "c1")
                c1_e = _col(row, "e_c1")

                if not all(
                    _qc_mag(x, xe) for x, xe in [(by, by_e), (m1, m1_e), (c1, c1_e)]
                ):
                    continue

                by, by_e = float(by), float(by_e) if by_e is not None else 0.0
                m1, m1_e = float(m1), float(m1_e) if m1_e is not None else 0.0
                c1, c1_e = float(c1), float(c1_e) if c1_e is not None else 0.0

                b_mag = by + y
                v_mag = m1 + 2 * by + y
                u_mag = c1 + 2 * m1 + 3 * by + y

                b_e = np.sqrt(by_e**2 + y_e**2)
                v_e = np.sqrt(m1_e**2 + 4 * by_e**2 + y_e**2)
                u_e = np.sqrt(c1_e**2 + 4 * m1_e**2 + 9 * by_e**2 + y_e**2)

                for name, m, e in [
                    ("STROMGREN_u", u_mag, u_e),
                    ("STROMGREN_v", v_mag, v_e),
                    ("STROMGREN_b", b_mag, b_e),
                    ("STROMGREN_y", y, y_e),
                ]:
                    if name not in self._magnitudes:
                        self._add_mag(name, m, e)

                break  # found in one catalog, skip the other

            except Exception as e:
                logger.warning("Stromgren XMatch failed for %s: %s", cat2, e)

    # ── Spectroscopic parameters ─────────────────────────────────

    # Gaia XP synthetic photometry, restricted to GaiaXPy's STANDARDIZED
    # systems (zero points tied to external references; the raw synthetic
    # systems carry ~1% un-corrected offsets). Every target band below is
    # already in ARIADNE's filter registry and already convolved in every
    # model grid, so this costs nothing downstream. Mag conventions match
    # mag_to_flux's per-band dispatch: SDSS_*/PS1_* are AB there and the
    # gaiaxpy SDSS/PS1 systems are AB-native; the rest are Vega on both
    # sides. u/U bands are deliberately absent: XP carries documented
    # systematics blueward of 400 nm.
    _XP_SYSTEMS = {
        # iteration order = fill priority (narrow/diagnostic systems first)
        'StromgrenStd': {'v': 'STROMGREN_v', 'b': 'STROMGREN_b',
                         'y': 'STROMGREN_y'},
        'JkcStd': {'B': 'GROUND_JOHNSON_B', 'V': 'GROUND_JOHNSON_V',
                   'R': 'GROUND_COUSINS_R', 'I': 'GROUND_COUSINS_I'},
        'SdssStd': {'g': 'SDSS_g', 'r': 'SDSS_r', 'i': 'SDSS_i',
                    'z': 'SDSS_z'},
        'Panstarrs1Std': {'gp': 'PS1_g', 'rp': 'PS1_r', 'ip': 'PS1_i',
                          'zp': 'PS1_z', 'yp': 'PS1_y'},
    }
    # QC-ONLY systems: never used to fill, only to validate observed catalog
    # bands against same-source XP synthetics (non-standardized accuracy is
    # ample at the >1 mag wrong-crossmatch threshold). SkyMapper's split
    # detections for high-PM stars were slipping through every other check
    # because it had no QC coverage (SCR 1845: i band 1 mag faint).
    _XP_QC_SYSTEMS = {
        'SkyMapper': {'u': 'SkyMapper_u', 'v': 'SkyMapper_v',
                      'g': 'SkyMapper_g', 'r': 'SkyMapper_r',
                      'i': 'SkyMapper_i', 'z': 'SkyMapper_z'},
    }
    _XP_GAP_LOGLAM = 0.05   # "covered": an observed band within 5% in log-lam
    _XP_MAX_G = 16.0        # XP hockey-stick regime begins around G ~ 16
    _XP_MIN_UM = 0.40       # documented BP systematics blueward of 400 nm
    _XP_MAX_ERR = 0.05      # mag; reject noisy synthetic bands
    _XP_SYS_FLOOR = 0.010   # mag; standardized-system absolute accuracy
    # For very red stars (mid-M and later) the blue-band synthetics are
    # outside the FGK calibration color range (Paunzen b-y <= 0.8) and in
    # XP's documented weak regime (sparse red-source standardization):
    # SCR 1845 (M8.5) got B/g/y fills 20-60% off, which tilted the whole
    # photosphere fit. Restrict fills to lam >= 0.6 um when BP-RP > 2.5.
    _XP_RED_BPRP = 2.5
    _XP_RED_MIN_UM = 0.60
    # The ultra-broad Gaia bands never count as wavelength coverage: G spans
    # 0.33-1.0 um, so its effective wavelength would spuriously "cover" the
    # V/r region (and BP/RP likewise g/i) for every Gaia star, defeating the
    # gap-filling entirely.
    _XP_COVERAGE_EXCLUDE = frozenset(
        ('Gaia_G', 'Gaia_BP', 'Gaia_RP',
         'GaiaDR2v2_G', 'GaiaDR2v2_BP', 'GaiaDR2v2_RP'))
    # Calibration onto ARIADNE's catalog conventions (synth - obs medians,
    # corrected additively; the measured robust scatter becomes the per-band
    # error floor where it exceeds _XP_SYS_FLOOR):
    # - Stromgren (135 Paunzen+15 stars, 2026-06): v -66 +/- 36 mmag,
    #   b -63 +/- 14, y -4 +/- 10 -- the GaiaXPy absolute Stromgren_Std
    #   calibration vs the index-reconstruction convention
    #   (v = m1 + 2(b-y) + y) ARIADNE's Stromgren catalogs use.
    # - JKC (98 Landolt-2013 standards, 2026-06): B +6 +/- 15, V +6 +/- 10
    #   mmag (near-circular: gaiaxpy JKC_Std is itself Landolt-standardized,
    #   so these measure the standardization residual). CAVEAT: B carries a
    #   +16 mmag/mag color trend redward -- the constant holds for the
    #   FGK-dominated calibration range only; the _XP_RED_BPRP / _XP_RED_MIN_UM
    #   gates already block B fills for very red stars.
    # - SDSS (Ivezic Stripe-82 standards, bright end r 14-15.5, 2026-06):
    #   z -10 +/- 12 mmag (a known GSPC red-band residual; the Stripe-82
    #   standards and ARIADNE's VizieR DR12 fetch are the same native-SDSS
    #   system, so the correction keeps fills and catalog rows mutually
    #   consistent); g/r/i offsets < 5 mmag (left at 0, floors only).
    # - PS1 (single NGP calibration field, DR1 MeanObject, 2026-06):
    #   g -6 +/- 9, i -5 +/- 6 mmag; other bands < 5 mmag. PS1 floors stay
    #   on _XP_SYS_FLOOR (their 6-9 mmag single-field scatter does not
    #   sample PS1's field-to-field zero-point variation).
    # None of these should be extrapolated past BP-RP ~ 2.4 (every system
    # drifts "synth fainter" redward; the red gates cover this).
    _XP_MAG_OFFSET = {'STROMGREN_v': +0.066, 'STROMGREN_b': +0.063,
                      'GROUND_JOHNSON_B': -0.006, 'GROUND_JOHNSON_V': -0.006,
                      'SDSS_z': +0.010, 'PS1_g': +0.006, 'PS1_i': +0.005}
    _XP_BAND_FLOOR = {'STROMGREN_v': 0.036, 'STROMGREN_b': 0.017,
                      'STROMGREN_y': 0.013,
                      'GROUND_JOHNSON_B': 0.016, 'GROUND_COUSINS_R': 0.013,
                      'GROUND_COUSINS_I': 0.013,
                      'SDSS_g': 0.014, 'SDSS_i': 0.012, 'SDSS_z': 0.013}

    def _fetch_xp_synth(self):
        """Fill optical coverage gaps with Gaia XP synthetic photometry.

        Generates standardized-system synthetic magnitudes from the XP
        continuous spectrum (GaiaXPy, full coefficient-covariance error
        propagation with the empirical error correction) and adds ONLY the
        registry bands whose wavelength is not already covered by observed
        photometry. Real measurements always win: this runs after every
        catalog fetcher and `_add_mag` is first-write-wins; the log-lambda
        gap rule additionally stops the synthetic bands from stacking up
        next to an observed band of a different system.
        """
        from termcolor import colored
        if self._gaia_id is None:
            return
        # Gaia photometry lives under the librarian-internal name (the
        # adapter renames to GaiaDR2v2_* later); check both defensively.
        g_entry = (self._magnitudes.get('Gaia_G')
                   or self._magnitudes.get('GaiaDR2v2_G'))
        if g_entry is None or g_entry[0] > self._XP_MAX_G:
            logger.info('XP gap-filling skipped: G magnitude missing or '
                        'fainter than %.1f', self._XP_MAX_G)
            return
        try:
            from gaiaxpy import generate, PhotometricSystem
        except ImportError:
            logger.warning('gaiaxpy not installed; XP gap-filling skipped '
                           '(pip install gaiaxpy)')
            return
        from ..phot_utils import get_effective_wavelength

        ps = [PhotometricSystem.Stromgren_Std, PhotometricSystem.JKC_Std,
              PhotometricSystem.SDSS_Std, PhotometricSystem.PanSTARRS1_Std,
              PhotometricSystem.Sky_Mapper]      # QC-only (never filled)
        df = _with_timeout(generate, [str(self._gaia_id)],
                           photometric_system=ps, error_correction=True,
                           save_file=False)
        if df is None or not len(df):
            logger.info('no XP continuous spectrum for %s; gap-filling '
                        'skipped', self._gaia_id)
            return
        row = df.iloc[0]

        # QC pass first: XP synthetics are epoch-2016 Gaia measurements of
        # the SAME source, so an observed catalog band that disagrees with
        # its own XP synthetic by > 1 mag is a wrong/split crossmatch (the
        # high-proper-motion disease: PS1/SDSS mean objects split fast
        # movers) or a blend -- no survey calibration or stellar variability
        # short of a Mira explains 1 mag. Drop it; the fill loop below then
        # treats the band as uncovered and the XP value takes its place.
        # Threshold is PM-aware: for ordinary stars 1.0 mag (nothing short
        # of a Mira explains more), but a star moving > 1"/yr smears across
        # the multi-epoch mean objects of ground surveys (SkyMapper/PS1/
        # SDSS split or trail its detections -- SCR 1845's SkyMapper riz
        # came out 0.45-0.97 mag faint and anchored the whole fit wrong),
        # while the XP synthetic is a single-epoch measurement of the same
        # source. 0.25 mag of disagreement is already damning there.
        _pm = np.hypot(float(getattr(self, '_pmra', None) or 0.0),
                       float(getattr(self, '_pmdec', None) or 0.0))
        qc_thr = 0.25 if _pm > 1000.0 else 1.0
        qc_maps = {**self._XP_SYSTEMS, **self._XP_QC_SYSTEMS}
        for sys_label, bands in qc_maps.items():
            for gband, target in bands.items():
                ent = self._magnitudes.get(target)
                synth = row.get(f'{sys_label}_mag_{gband}')
                if ent is None or synth is None or not np.isfinite(synth):
                    continue
                if abs(ent[0] - (float(synth)
                                 + self._XP_MAG_OFFSET.get(target, 0.0))) \
                        > qc_thr:
                    del self._magnitudes[target]
                    print(colored(
                        f'\t\t*** {target} catalog mag {ent[0]:.2f} is '
                        f'>{qc_thr:.2f} mag from its XP synthetic '
                        f'{float(synth):.2f} -- wrong/split crossmatch '
                        '(high-PM?); dropped. ***', 'yellow'))

        obs_loglam = []
        for b in self._magnitudes:
            if b in self._XP_COVERAGE_EXCLUDE:
                continue
            try:
                lam = get_effective_wavelength(b)
            except Exception:
                continue
            if np.isfinite(lam) and lam > 0:
                obs_loglam.append(np.log(lam))
        obs_loglam = np.asarray(obs_loglam)

        filled = []
        for sys_label, bands in self._XP_SYSTEMS.items():
            for gband, target in bands.items():
                if target in self._magnitudes:
                    continue
                mag = row.get(f'{sys_label}_mag_{gband}')
                if mag is None or not np.isfinite(mag):
                    continue
                err = row.get(f'{sys_label}_mag_error_{gband}')
                if err is None or not np.isfinite(err):
                    fl = row.get(f'{sys_label}_flux_{gband}')
                    fe = row.get(f'{sys_label}_flux_error_{gband}')
                    err = (1.0857 * fe / fl
                           if (fl is not None and fe is not None
                               and np.isfinite(fl) and np.isfinite(fe)
                               and fl > 0) else None)
                if err is None or not np.isfinite(err) \
                        or err > self._XP_MAX_ERR:
                    continue
                lam = get_effective_wavelength(target)
                min_um = self._XP_MIN_UM
                bp = self._magnitudes.get('Gaia_BP',
                                          self._magnitudes.get('GaiaDR2v2_BP'))
                rp = self._magnitudes.get('Gaia_RP',
                                          self._magnitudes.get('GaiaDR2v2_RP'))
                if bp is not None and rp is not None \
                        and (bp[0] - rp[0]) > self._XP_RED_BPRP:
                    min_um = self._XP_RED_MIN_UM
                if not np.isfinite(lam) or lam < min_um:
                    continue
                if obs_loglam.size and np.min(
                        np.abs(np.log(lam) - obs_loglam)) \
                        < self._XP_GAP_LOGLAM:
                    continue
                mag = float(mag) + self._XP_MAG_OFFSET.get(target, 0.0)
                floor = self._XP_BAND_FLOOR.get(target, self._XP_SYS_FLOOR)
                self._add_mag(target, mag, float(np.hypot(err, floor)))
                # A filled band covers its wavelength too -- stops a second
                # synthetic system from stacking a near-duplicate next to it.
                obs_loglam = np.append(obs_loglam, np.log(lam))
                filled.append(target)
        self._xp_filled = filled
        if filled:
            print(colored(
                '\t\t*** Gaia XP synthetic photometry filled '
                f'{len(filled)} uncovered band(s): {", ".join(filled)} '
                '(disable with ignore=["XP"]) ***', 'cyan'))

    # DESI/BOSS optical spectro-photometry fill: the faint-end continuation
    # of the XP ladder. Only fires for G > _OPTSPEC_MIN_G (XP owns the
    # bright end with calibrated standardized-system synthetics; fibre
    # spectro-photometry is not zero-point calibrated to the same level).
    _OPTSPEC_MIN_G = 16.0
    _OPTSPEC_MAX_ERR = 0.05
    # Per-(release, band) zero-points/floors: a calibration campaign per
    # data release (the XP-campaign template) populates these. Until then
    # offsets stay 0 and every fill carries the conservative system floor
    # measured live against PS1 (18 DESI-DR1 MWS stars, rms 0.070 mag,
    # per-band means within 0.06 -- a blue-to-red flux-cal tilt).
    _OPTSPEC_MAG_OFFSET = {}
    _OPTSPEC_BAND_FLOOR = {}
    _OPTSPEC_SYS_FLOOR = 0.07

    def _fetch_optspec(self, which):
        """Synthesize optical photometry from DESI/BOSS spectra (G > 16).

        Same gap rule and Gaia-band exclusion as the XP fill; only the
        synthetic-mag source differs (a real flux-calibrated spectrum
        convolved through the registry curves instead of GaiaXPy). Fills are
        recorded in ``_xp_filled`` -- they are spectrophotometry, so
        ``Fitter.spectra = False`` must drop them with the XP fills.
        """
        from termcolor import colored
        from . import _optspec
        from ..phot_utils import get_effective_wavelength
        g = (self._magnitudes.get('Gaia_G')
             or self._magnitudes.get('GaiaDR2v2_G'))
        if g is None or float(g[0]) <= self._OPTSPEC_MIN_G:
            return
        coord = self._propagated_coord(2017.0)   # DESI/BOSS epochs ~2015-21
        fetch = _optspec.fetch_desi if which == 'DESI' else _optspec.fetch_boss
        sid = f'{which.lower()}_{self._gaia_id}' if self._gaia_id else None
        spec = fetch(float(coord.ra.deg), float(coord.dec.deg),
                     source_id=sid)
        if spec is None:
            return
        wave, flux, ivar = spec
        # XP gap rule, verbatim: a band already observed/filled within
        # _XP_GAP_LOGLAM in log-lambda blocks a synthetic at that wavelength.
        obs_loglam = []
        for b in self._magnitudes:
            if b in self._XP_COVERAGE_EXCLUDE:
                continue
            try:
                lam = get_effective_wavelength(b)
            except Exception:
                continue
            if np.isfinite(lam) and lam > 0:
                obs_loglam.append(np.log(lam))
        obs_loglam = np.asarray(obs_loglam)
        syn = _optspec.synth_phot(wave, flux, ivar,
                                  _optspec.default_fill_bands())
        filled = []
        for band in sorted(syn, key=get_effective_wavelength):
            mag, err = syn[band]
            if band in self._magnitudes or err > self._OPTSPEC_MAX_ERR:
                continue
            lam = get_effective_wavelength(band)
            if obs_loglam.size and np.min(np.abs(np.log(lam) - obs_loglam)) \
                    < self._XP_GAP_LOGLAM:
                continue
            mag += self._OPTSPEC_MAG_OFFSET.get((which, band), 0.0)
            floor = self._OPTSPEC_BAND_FLOOR.get(
                (which, band), self._OPTSPEC_SYS_FLOOR)
            self._add_mag(band, float(mag), float(np.hypot(err, floor)))
            obs_loglam = np.append(obs_loglam, np.log(lam))
            filled.append(band)
        if filled:
            if not hasattr(self, '_xp_filled') or self._xp_filled is None:
                self._xp_filled = []
            self._xp_filled = list(self._xp_filled) + filled
            print(colored(
                f'\t\t*** {which} spectro-photometry filled '
                f'{len(filled)} band(s): {", ".join(filled)} '
                f'(disable with ignore=["{which}"]) ***', 'cyan'))

    _IRS_EPOCH = 2006.0          # mid Spitzer IRS cryo mission (2003-2009)

    def _fetch_cassis(self):
        """CASSIS/IRS low-res super-band photometry at the PM-propagated
        position. Extraction + binning live in ``_cassis``. All IRS bands are
        >= 5 um, so ``_add_mag`` routes them straight to the star's IR-excess
        set (config.irx_bands); none enter the fit."""
        from termcolor import colored
        from . import _cassis
        c = self._propagated_coord(self._IRS_EPOCH)
        res = _cassis.fetch_spectrum(float(c.ra.deg), float(c.dec.deg))
        if res is None:
            logger.info('CASSIS: no IRS low-res spectrum at this position')
            return
        lam, f, e = res
        added = []
        for band, (mag, mag_e, n) in _cassis.superband_mags(lam, f, e).items():
            if band not in self._magnitudes:
                self._add_mag(band, mag, mag_e)      # irx_bands -> excess set
                added.append(f'{band}({n})')
        self._cassis_added = added
        if added:
            print(colored(
                '\t\t*** CASSIS/IRS spectro-photometry added '
                f'{len(added)} IR super-band(s): {", ".join(added)} '
                '(disable with ignore=["CASSIS"]) ***', 'cyan'))

    # GALEX 10%-nonlinearity (saturation) limits, Morrissey et al. 2007:
    # FUV 13.73, NUV 13.85 -- rounded to 14 with margin. A catalog GALEX mag
    # BRIGHTER than this is unreliable; IUE replaces it.
    _IUE_SAT_FUV = 14.0
    _IUE_SAT_NUV = 14.0
    _IUE_EPOCH = 1987.0          # mid IUE mission (1978-1996)

    def _fetch_iue(self):
        """Synthesize GALEX FUV/NUV from IUE/INES spectra for bright stars.

        Fires only when a GALEX band is absent OR present but brighter than
        the saturation limit (then the catalog value is unreliable and the
        IUE synthesis REPLACES it, loudly). Unsaturated catalog GALEX is
        never touched."""
        from termcolor import colored
        from . import _iue
        fuv = self._magnitudes.get('GALEX_FUV')
        nuv = self._magnitudes.get('GALEX_NUV')
        need = {
            'GALEX_FUV': fuv is None or float(fuv[0]) < self._IUE_SAT_FUV,
            'GALEX_NUV': nuv is None or float(nuv[0]) < self._IUE_SAT_NUV,
        }
        if not any(need.values()):
            return
        c = self._propagated_coord(self._IUE_EPOCH)
        res = _iue.fetch_spectrum(float(c.ra.deg), float(c.dec.deg))
        if res is None:
            return
        syn = _iue.synth_galex(*res)
        for band, (mag, err) in syn.items():
            if not need.get(band):
                continue
            cur = self._magnitudes.get(band)
            if cur is not None:
                del self._magnitudes[band]
                print(colored(
                    f'\t\t*** {band} = {float(cur[0]):.2f} is brighter than '
                    'the GALEX saturation limit -- REPLACED by the IUE/INES '
                    f'synthesis ({mag:.2f} +/- {err:.2f}) ***', 'yellow'))
            else:
                print(colored(
                    f'\t\t*** {band} filled from IUE/INES: {mag:.2f} +/- '
                    f'{err:.2f} (disable with ignore=["IUE"]) ***', 'cyan'))
            self._add_mag(band, float(mag), float(err))

    # SPHEREx usable point-source window in Vega Ks (saturation flags begin
    # ~9; all-sky 5-sigma depth ~16.5 at 2 um).
    _SPX_KS_BRIGHT = 8.5
    _SPX_KS_FAINT = 16.5
    _SPX_EPOCH = 2025.9          # mid-QR2; PM offsets within +-0.55 yr << 1 px

    def _fetch_spherex(self):
        """SPHEREx QR2 super-band photometry at the PM-propagated position.

        Extraction + binning live in ``_spherex``; this guard-rails it: a
        Gaia id and a 2MASS Ks inside the usable brightness window are
        required (brighter saturates, fainter drowns). The four <= 2.4 um
        bands enter the fit; the red bands are routed to the IR-excess set
        by ``Star.add_mag``-equivalent logic downstream (irx_bands).
        """
        from termcolor import colored
        if self._gaia_id is None:
            return
        ks = self._magnitudes.get('2MASS_Ks')
        if ks is None:
            logger.info('SPHEREx skipped: no 2MASS Ks to check the '
                        'brightness window')
            return
        if not (self._SPX_KS_BRIGHT <= ks[0] <= self._SPX_KS_FAINT):
            logger.info('SPHEREx skipped: Ks=%.2f outside the usable window '
                        '(%.1f-%.1f)', ks[0], self._SPX_KS_BRIGHT,
                        self._SPX_KS_FAINT)
            return
        from . import _spherex
        c = self._propagated_coord(self._SPX_EPOCH)
        lam, f, e = _spherex.extract_spectrum(
            float(c.ra.deg), float(c.dec.deg), int(self._gaia_id))
        if not len(lam):
            logger.info('SPHEREx: no clean exposures for this position')
            return
        added = []
        for band, (mag, mag_e, n) in _spherex.superband_mags(lam, f, e).items():
            if band not in self._magnitudes:
                self._add_mag(band, mag, mag_e)
                added.append(f'{band}({n})')
        self._spherex_added = added
        if added:
            print(colored(
                '\t\t*** SPHEREx QR2 spectro-photometry added '
                f'{len(added)} super-band(s): {", ".join(added)} '
                '(disable with ignore=["SPHEREX"]) ***', 'cyan'))

    def _query_spectroscopic_priors(self):
        """Assemble spectroscopic priors (Teff, logg, [Fe/H]).

        Teff/logg come from the survey priority chain (first match wins):
        PASTEL > APOGEE DR17 > GALAH DR3 > RAVE DR6 > LAMOST DR11.

        [Fe/H] depends on ``feh_source``:
          * "hypatia" (default) — the Hypatia Catalog, a multi-study
            high-resolution-spectroscopy compilation whose spread gives a
            realistic σ; falls back to the survey-chain [Fe/H] on a Hypatia
            miss.
          * "survey" — use the survey-chain [Fe/H] (legacy behaviour).
        """
        base = None
        for query_fn, source_name in [
            (self._query_pastel, "PASTEL"),
            (self._query_apogee, "APOGEE_DR17"),
            (self._query_galah, "GALAH_DR3"),
            (self._query_rave, "RAVE_DR6"),
            (self._query_lamost, "LAMOST_DR11"),
        ]:
            result = query_fn()
            if result is not None:
                result["source"] = source_name
                base = result
                logger.info("Spectroscopic params from %s", source_name)
                break

        if self._feh_source == "hypatia":
            hyp = self._query_hypatia()
            if hyp is not None:
                if base is None:
                    base = {"teff": None, "teff_err": None,
                            "logg": None, "logg_err": None,
                            "source": None}
                base["feh"] = hyp["feh"]
                base["feh_err"] = hyp["feh_err"]
                base["feh_source"] = "Hypatia"
                base["feh_n_meas"] = hyp["n_meas"]
                base["feh_solarnorm"] = hyp["solarnorm"]
                base["feh_statistic"] = hyp["statistic"]
                base["feh_uncertainty"] = hyp["uncertainty"]
                logger.info("[Fe/H] from Hypatia: %.3f ± %.3f (n=%d, %s)",
                            hyp["feh"], hyp["feh_err"], hyp["n_meas"],
                            hyp["matched_name"])

        self._spectroscopic_params = base

    def _query_hypatia(self) -> dict | None:
        """Query the Hypatia Catalog for [Fe/H] using the configured options."""
        from ._hypatia import query_hypatia_feh
        names = []
        tmass = self._ids.get("2MASS") if self._ids else None
        if tmass:
            names.append(f"2MASS J{tmass}")
        return query_hypatia_feh(
            self._gaia_id, names,
            statistic=self._hypatia_statistic,
            uncertainty=self._hypatia_uncertainty,
            solarnorm=self._hypatia_solarnorm,
            floor=self._feh_floor,
        )

    def _query_apogee(self) -> dict | None:
        """Query APOGEE DR17 via VizieR (III/286).

        Crossmatches via 2MASS ID. Rejects entries with ASPCAPFLAG != 0.
        Uses [M/H] as proxy for [Fe/H] (close enough for priors).
        """
        tmass_id = self._ids.get("2MASS")
        if tmass_id is None:
            return None

        try:
            cat = _with_timeout(
                Vizier.query_constraints,
                catalog="III/286/allstars", **{"2MASS": str(tmass_id)},
            )
            if not cat or len(cat[0]) == 0:
                return None

            row = cat[0][0]

            # Reject flagged entries
            flag = _col(row, "ASPCAPFLAG")
            if flag is not None and int(flag) != 0:
                return None

            teff = _col(row, "Teff")
            teff_e = _col(row, "e_Teff")
            logg = _col(row, "logg")
            logg_e = _col(row, "e_logg")
            mh = _col(row, "__M_H_")
            mh_e = _col(row, "e__M_H_")

            if any(v is None or not np.isfinite(float(v)) for v in (teff, logg, mh)):
                return None

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 100.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.2,
                "feh": float(mh),
                "feh_err": float(mh_e) if mh_e is not None else 0.1,
            }
        except Exception as e:
            logger.warning("APOGEE DR17 query failed: %s", e)
            return None

    def _query_galah(self) -> dict | None:
        """Query GALAH DR3 via VizieR (J/MNRAS/506/150).

        Pre-matched to Gaia DR3 source_id. Rejects entries with quality flags.
        """
        if self._gaia_id is None:
            return None

        try:
            cat = _with_timeout(
                Vizier.query_constraints,
                catalog="J/MNRAS/506/150/catalog",
                Source=str(self._gaia_id),
            )
            if not cat or len(cat[0]) == 0:
                return None

            row = cat[0][0]

            # Quality flags: reject if flagged
            flag_sp = _col(row, "flag_sp")
            flag_fe_h = _col(row, "flag_fe_h")
            if flag_sp is not None and int(flag_sp) != 0:
                return None
            if flag_fe_h is not None and int(flag_fe_h) != 0:
                return None

            teff = _col(row, "Teff")
            teff_e = _col(row, "e_Teff")
            logg = _col(row, "logg")
            logg_e = _col(row, "e_logg")
            feh = _col(row, "fe_h")
            feh_e = _col(row, "e_fe_h")

            if any(v is None or not np.isfinite(float(v)) for v in (teff, logg, feh)):
                return None

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 100.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.2,
                "feh": float(feh),
                "feh_err": float(feh_e) if feh_e is not None else 0.1,
            }
        except Exception as e:
            logger.warning("GALAH DR3 query failed: %s", e)
            return None

    def _query_rave(self) -> dict | None:
        """Query RAVE DR6 for spectroscopic Teff, logg, [Fe/H]."""
        rave_id = self._ids.get("RAVE")
        if rave_id is None:
            return None

        try:
            cat = _with_timeout(
                Vizier.query_constraints,
                catalog="III/283/madera", ObsID=str(rave_id),
            )
            if not cat or len(cat[0]) == 0:
                return None

            row = cat[0][0]
            qual = _col(row, "Qual")
            if qual is not None and int(qual) == 1:
                return None

            return {
                "teff": float(row["TeffmC"]),
                "teff_err": float(row["e_Teffm"]),
                "logg": float(row["loggmC"]),
                "logg_err": float(row["e_loggm"]),
                "feh": float(row["[m/H]mC"]),
                "feh_err": float(row["e_[m/H]m"]),
            }
        except Exception as e:
            logger.warning("RAVE DR6 query failed: %s", e)
            return None

    def _query_lamost(self) -> dict | None:
        """Query LAMOST DR9 stellar parameters via VizieR (V/162).

        Crossmatches via positional cone search since LAMOST has no
        pre-matched Gaia source_id. Requires SNR_g > 30.
        """
        try:
            cats = _with_timeout(
                Vizier.query_region,
                self._coord, radius=self._search_radius, catalog="V/162",
            )
            if cats is None:
                return None
            if not cats or len(cats[0]) == 0:
                return None

            cats[0].sort("_r")
            row = cats[0][0]

            # SNR quality cut
            snrg = _col(row, "snrg")
            if snrg is not None and float(snrg) < 30:
                return None

            teff = _col(row, "Teff")
            teff_e = _col(row, "e_Teff")
            logg = _col(row, "logg")
            logg_e = _col(row, "e_logg")
            feh = _col(row, "__Fe_H_")
            feh_e = _col(row, "e__Fe_H_")

            if any(v is None for v in (teff, logg, feh)):
                return None

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 100.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.2,
                "feh": float(feh),
                "feh_err": float(feh_e) if feh_e is not None else 0.1,
            }
        except Exception as e:
            logger.warning("LAMOST DR11 query failed: %s", e)
            return None

    def _query_pastel(self) -> dict | None:
        """Query PASTEL catalog via VizieR (B/pastel).

        Literature compilation of spectroscopic parameters. Positional match.
        Uses conservative default errors since many entries lack formal errors.
        """
        try:
            cats = _with_timeout(
                Vizier.query_region,
                self._coord, radius=self._search_radius, catalog="B/pastel/pastel",
            )
            if not cats or len(cats[0]) == 0:
                return None

            cats[0].sort("_r")
            row = cats[0][0]

            teff = _col(row, "Teff")
            logg = _col(row, "logg")
            feh = _col(row, "__Fe_H_")

            if any(v is None or not np.isfinite(float(v)) for v in (teff, logg, feh)):
                return None

            teff_e = _col(row, "e_Teff")
            logg_e = _col(row, "e_logg")
            feh_e = _col(row, "e__Fe_H_")

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 150.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.3,
                "feh": float(feh),
                "feh_err": float(feh_e) if feh_e is not None else 0.15,
            }
        except Exception as e:
            logger.warning("PASTEL query failed: %s", e)
            return None

    def _query_extinction(self):
        """Query max line-of-sight Av from SFD dustmap.

        SFD is purely 2D (line-of-sight integrated), so the distance kwarg
        is intentionally not passed. The SFDQuery instance is cached at
        module scope to avoid reloading ~600 MB of FITS per Star.
        """
        try:
            sfd = _DUSTMAP_CACHE.get("sfd")
            if sfd is None:
                from dustmaps.sfd import SFDQuery
                sfd = SFDQuery()
                _DUSTMAP_CACHE["sfd"] = sfd
            coords = SkyCoord(self.ra, self.dec, unit=(u.deg, u.deg), frame='icrs')
            ebv = sfd(coords)
            self.Av = float(ebv) * _AV_PER_EBV_SFD
        except Exception as e:
            logger.warning("SFD extinction query failed: %s", e)
            self.Av = None

    # ── Helpers ──────────────────────────────────────────────────

    def _impute_zero_errors(self):
        """Impute errors for zero-error bands (matching ARIADNE).

        Computes the maximum relative flux error from bands with valid
        errors, adds 0.1, and assigns that relative error to zero-error
        bands. This mirrors ARIADNE's Star.extract_info() logic where
        mx_rel_er = max(flux_er / flux) + 0.1.
        """
        _LN10x04 = 0.4 * np.log(10)  # ≈ 0.9210

        good_rel = []
        for _band, (_mag, err) in self._magnitudes.items():
            if err > 0:
                good_rel.append(_LN10x04 * err)

        if not good_rel:
            return

        mx_rel = max(good_rel) + 0.1
        imputed = mx_rel / _LN10x04

        for band, (mag, err) in list(self._magnitudes.items()):
            if err <= 0:
                self._magnitudes[band] = (mag, imputed)

    def _add_mag(self, pyphot_name, mag, err):
        """Add magnitude. First-write-wins (catalog priority by query order)."""
        if pyphot_name not in self._magnitudes:
            self._magnitudes[pyphot_name] = (mag, err)

    def _add_flux(self, pyphot_name, flux, err, unit):
        """Add a flux-density measurement (Jy/mJy/uJy) by converting it to a
        magnitude via the band's effective wavelength and zero-flux, then
        deferring to :meth:`_add_mag`.

        The far-IR catalogs (IRAS, AKARI, Spitzer, Herschel) report F_nu in
        janskys rather than Vega/AB magnitudes; this is the single intake point
        that turns them into the librarian's internal (mag, err) form. The
        conversion round-trips exactly with the Star's mag->flux step
        (flux = zero_flux * 10**(-0.4 mag)).
        """
        from ..phot_utils import get_zero_flux, get_effective_wavelength
        if flux is None:
            return
        try:
            flux = float(flux)
        except (TypeError, ValueError):
            return
        if not np.isfinite(flux) or flux <= 0:
            return
        fac = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6,
               "µJy": 1e-6, "10-3Jy": 1e-3}.get(unit)
        if fac is None:
            logger.warning("Unknown flux unit %r for %s", unit, pyphot_name)
            return
        lam = get_effective_wavelength(pyphot_name)
        z0 = get_zero_flux(pyphot_name)
        if not (np.isfinite(lam) and np.isfinite(z0) and z0 > 0):
            return
        f_lambda = ((flux * fac) * u.Jy).to(
            u.erg / u.s / u.cm**2 / u.um,
            equivalencies=u.spectral_density(lam * u.um)).value
        mag = -2.5 * np.log10(f_lambda / z0)
        try:
            e = float(err)
        except (TypeError, ValueError):
            e = None
        # Flux error -> magnitude error: dm = 1.0857 * sigma_F / F.
        emag = (1.0857 * e / flux) if (e is not None and np.isfinite(e)
                                       and e > 0) else 0.05
        self._add_mag(pyphot_name, mag, emag)

    def _print_summary(self):
        import random
        from termcolor import colored
        from ..phot_utils import get_effective_wavelength
        c = random.choice(['red', 'green', 'blue', 'yellow', 'grey', 'magenta', 'cyan', 'white'])
        t2 = "\t\t"
        t3 = "\t\t\t"

        def _filter_wavelength(b):
            try:
                return get_effective_wavelength(b)
            except Exception:
                return 99.0

        # Photometry table sorted by wavelength (like ARIADNE)
        bands = sorted(
            self._magnitudes.keys(),
            key=_filter_wavelength,
        )
        print(colored(f"{t2}--- Retrieved photometry ---", c))
        print(colored(f"{t2}{' Filter':^20s}\t{'Magnitude':>9s}\t{'Uncertainty':>11s}", c))
        print(colored(f"{t2}{'':->20s}\t{'-':->9s}\t{'-':->11s}", c))
        for band in bands:
            mag, err = self._magnitudes[band]
            print(colored(f"{t2}{band:^20s}\t{mag: ^9.4f}\t{err: ^11.4f}", c))

        # Stellar params — new color (matching ARIADNE's display_star_fin)
        c = random.choice(['red', 'green', 'blue', 'yellow', 'grey', 'magenta', 'cyan', 'white'])
        if self._gaia_id:
            print(colored(f"{t3}Gaia DR3 ID : {self._gaia_id}", c))
        if self._tic_id:
            print(colored(f"{t3}TIC : {self._tic_id}", c))
        if self._kic_id:
            print(colored(f"{t3}KIC : {self._kic_id}", c))
        if self._teff is not None:
            print(colored(f"{t3}Gaia Effective temperature : ", c), end='')
            print(colored(f"{self._teff:.3f} +/- {self._teff_e:.3f}", c))
        if self._radius_flame is not None:
            print(colored(f"{t3}Gaia Stellar radius : ", c), end='')
            print(colored(f"{self._radius_flame:.3f} +/- {self._radius_flame_e:.3f}", c))
        if self._luminosity is not None:
            print(colored(f"{t3}Gaia Stellar Luminosity : ", c), end='')
            print(colored(f"{self._luminosity:.3f} +/- {self._luminosity_e:.3f}", c))
        if self._parallax is not None:
            print(colored(f"{t3}Gaia Parallax : ", c), end='')
            print(colored(f"{self._parallax:.3f} +/- {self._parallax_e:.3f}", c))
        if self._distance is not None:
            print(colored(f"{t3}Bailer-Jones distance : ", c), end='')
            print(colored(f"{self._distance:.3f} +/- {self._distance_e:.3f}", c))
        if hasattr(self, 'Av') and self.Av is not None:
            print(colored(f"{t3}Maximum Av : ", c), end='')
            print(colored(f"{self.Av:.3f}", c))
        print()
        print()

    # ── Properties ───────────────────────────────────────────────

    @property
    def parallax(self) -> tuple[float, float] | None:
        """(parallax_mas, error_mas) with Lindegren+21 correction."""
        if self._parallax is None:
            return None
        return (self._parallax, self._parallax_e)

    @property
    def teff(self) -> tuple[float, float] | None:
        """(Teff_K, error_K) from Gaia GSP-Phot."""
        if self._teff is None:
            return None
        return (self._teff, self._teff_e or 100.0)

    @property
    def radius(self) -> tuple[float, float] | None:
        """(R_Rsun, error_Rsun) from FLAME (5x inflated error)."""
        if self._radius_flame is None:
            return None
        return (self._radius_flame, self._radius_flame_e)

    @property
    def luminosity(self) -> tuple[float, float] | None:
        """(L_Lsun, error_Lsun) from FLAME."""
        if self._luminosity is None:
            return None
        return (self._luminosity, self._luminosity_e)

    @property
    def mass(self) -> tuple[float, float] | None:
        """(M_Msun, error_Msun) from FLAME."""
        if self._mass is None:
            return None
        return (self._mass, self._mass_e)

    @property
    def age(self) -> tuple[float, float] | None:
        """(age_Gyr, error_Gyr) from FLAME."""
        if self._age is None:
            return None
        return (self._age, self._age_e)

    @property
    def distance(self) -> tuple[float, float] | None:
        """(distance_pc, error_pc) from Bailer-Jones EDR3."""
        if self._distance is None:
            return None
        return (self._distance, self._distance_e)

    @property
    def magnitudes(self) -> dict[str, tuple[float, float]]:
        """{'pyphot_filter_name': (mag, error)}."""
        return dict(self._magnitudes)

    @property
    def gaia_id(self) -> int | None:
        return self._gaia_id

    @property
    def tic_id(self) -> int | None:
        return self._tic_id

    @property
    def spectroscopic_params(self) -> dict | None:
        """Spectroscopic params dict with source tag, or None.

        Keys: teff, teff_err, logg, logg_err, feh, feh_err, source.
        """
        return dict(self._spectroscopic_params) if self._spectroscopic_params else None

    @property
    def rave_params(self) -> dict | None:
        """RAVE DR6 spectroscopic params, or None. Backward-compatible."""
        if (
            self._spectroscopic_params is not None
            and self._spectroscopic_params.get("source") == "RAVE_DR6"
        ):
            return dict(self._spectroscopic_params)
        return None

