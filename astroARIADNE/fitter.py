"""Main driver of the fitting routine."""

__all__ = ['Fitter', 'dynesty_log_like', 'dynesty_loglike_bma', 'pt_dynesty']

import copy
import json
import logging
import os
import pickle
import time
import warnings

logger = logging.getLogger(__name__)
from multiprocessing import (Pool, set_start_method)
from tqdm import tqdm

import extinction
import astropy.units as u
import numpy as np
import scipy.stats as st
from numpy.random import choice
from astropy.constants import sigma_sb
from termcolor import colored

from .config import (filesdir, gridsdir, priorsdir, filter_names, colors,
                     iso_mask, iso_bands, grid_wave_coverage,
                     COOL_COMPANION_GRIDS, irx_bands)
from .error import *
# ``isochrones`` (and the MIST grids) are a heavy, optional dependency used
# only for the post-fit age/mass estimate (estimate_age=True). Import it
# lazily so the photosphere/IR-excess fit runs without it.
try:
    from .isochrone import estimate
except Exception:                       # isochrones not installed
    estimate = None
from .phot_utils import *
from .sed_library import *
from .utils import *

try:
    import dynesty
    from dynesty.utils import resample_equal

    bma_flag = True
except ModuleNotFoundError:
    wrn = 'Dynesty package not found.\n'
    wrn += 'Install dynesty with `pip install dynesty`'
    warnings.warn(wrn)
    bma_flag = False

set_start_method('fork', force=True)


# ---------------------------------------------------------------------------
# Portable NetCDF (de)serialization of the BMA result dict (scipy only, no
# arviz/xarray/netCDF4). Numeric arrays -> NetCDF variables; the nested result
# structure, scalars and strings -> a single JSON `ariadne_result` attribute.
# Round-trips losslessly; a version-independent replacement for BMA.pkl.
# ---------------------------------------------------------------------------
def _nc_dtype(arr):
    """numpy dtype -> NetCDF3 type code, or None if not a numeric array."""
    k = arr.dtype.kind
    if k == 'f':
        return 'f8' if arr.dtype.itemsize >= 8 else 'f4'
    if k in 'iub':            # ints/uints/bools -> int32 (NetCDF3 has no int64)
        return 'i4'
    return None               # strings/objects -> kept inline in the JSON


def _result_dict_to_nc(d, path):
    """Serialize a (possibly nested) result dict to NetCDF via scipy.io."""
    from scipy.io import netcdf_file
    arrays = {}                                   # var name -> ndarray

    def enc(o):
        if isinstance(o, dict):
            return {'__d__': {str(k): enc(v) for k, v in o.items()}}
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, (str, bool, int, float)) or o is None:
            return o
        arr = o if isinstance(o, np.ndarray) else (
            np.asarray(o) if isinstance(o, (list, tuple)) else None)
        if arr is not None and _nc_dtype(arr) is not None and arr.ndim >= 1 \
                and arr.size > 0:
            name = f'v{len(arrays)}'
            arrays[name] = arr
            return {'__a__': name, 'shape': list(arr.shape),
                    'dtype': str(arr.dtype)}
        if isinstance(o, (list, tuple)):          # non-numeric / ragged / empty
            return [enc(x) for x in o]
        if isinstance(o, np.ndarray):             # string/object array -> list
            return enc(o.tolist())
        return str(o)                             # last-resort fallback

    meta = enc(d)
    f = netcdf_file(path, 'w')
    f.format = 'ariadne-bma-1'
    f.ariadne_result = json.dumps(meta)
    for name, arr in arrays.items():
        nct = _nc_dtype(arr)
        dims = []
        for j, s in enumerate(arr.shape):
            dn = f'{name}_{j}'
            f.createDimension(dn, int(s))
            dims.append(dn)
        f.createVariable(name, nct, tuple(dims))[:] = arr.astype(nct)
    f.close()


def _result_nc_to_dict(path):
    """Inverse of :func:`_result_dict_to_nc`."""
    from scipy.io import netcdf_file
    f = netcdf_file(path, 'r', mmap=False)
    meta_raw = f.ariadne_result
    meta = json.loads(meta_raw.decode() if isinstance(meta_raw, bytes)
                      else meta_raw)
    arrays = {name: np.array(var[:]) for name, var in f.variables.items()}
    f.close()

    def dec(o):
        if isinstance(o, dict):
            if '__a__' in o:
                return arrays[o['__a__']].astype(o['dtype']).reshape(o['shape'])
            if '__d__' in o:
                return {k: dec(v) for k, v in o['__d__'].items()}
            return {k: dec(v) for k, v in o.items()}
        if isinstance(o, list):
            return [dec(x) for x in o]
        return o

    return dec(meta)


# Module-level globals consumed by the dynesty wrappers; set per fit in
# Fitter.initialize. Defaults keep single-star fits free of binary state.
binary = False
contrast = None
# True when the noise model is a single shared jitter term (jitter='shared'),
# so the prior transform builds one 'shared_noise' instead of per-band terms.
jitter_shared = False


def _fm07_rv(wave, a_v, r_v=3.1):
    """Fitzpatrick & Massa (2007) extinction with the ARIADNE call signature.

    ``extinction.fm07`` is fixed at Rv=3.1 and takes ``(wave, a_v, unit)`` rather
    than ``(wave, a_v, r_v)``. ARIADNE always calls laws as ``law(wave, Av, Rv)``,
    so this wrapper absorbs (and ignores) ``r_v`` and forwards to ``fm07``. It is
    a single module-level function on purpose: ``sed_library._ext_attenuation_base``
    memoises on the law object's identity, so a stable callable is required for
    the cache to hit.
    """
    return extinction.fm07(wave, a_v)


# Modified-blackbody convention for the disk hypotheses (Wyatt 2008): pure
# blackbody blueward of lambda0, B_lambda * (lambda0/lambda)^beta beyond it.
# Small grains emit inefficiently at lambda >> grain size, so a pure BB
# over-predicts the submm tail (PACS160/SPIRE/FIS bands).
_MODBB_BETA = 1.0
_MODBB_LAM0 = 210.0     # um

# Absolute flux-calibration floor for the IR-excess bands. Unlike the
# photosphere-fit bands (which receive a fitted jitter noise term, scale
# >= 0.05*flux, see create_priors) the IR-excess bands are routed OUT of the
# fit by Star and compared to the model raw. Catalogue statistical errors omit
# the instrument's absolute calibration uncertainty, which for the mid/far-IR
# photometers used here is at least a few percent (Spitzer MIPS-24 ~2-4%, WISE
# W1-W4 2.4-5.7%, AKARI ~3-7%). Without a floor a band that reports an
# unphysically tiny error (a MIPS-24 point at 0.06% blew up to ~1350 sigma)
# produces a spurious huge excess that dominates the model-selection chi^2.
# Floor the *observed* error at this fraction of the observed flux.
_IRX_ABS_CAL_FLOOR = 0.02

# Substellar companion radius ceiling (Rsun) for the IR-excess companion
# hypothesis on a COMPACT (WD / sdB) primary. A companion detectable ONLY as IR
# excess -- unresolved, not dominating the optical (the host still fits as a
# clean WD) -- must be a brown dwarf or giant planet: a luminous M dwarf would
# swamp the red optical and is the JOINT-BINARY regime, not an IR excess. ~1.5
# R_Jup is the substellar / young-giant radius plateau (BDs and giant planets
# pile up at ~1 R_Jup, young/inflated up to ~1.3-1.5); larger R2 is stellar and
# physically excluded. This is a PHYSICAL prior bound. Without it the compact-
# primary branch left R2 open to 1 Rsun, letting the companion build a luminous
# pseudo-star out of any red residual -- the 64% WD-companion over-call, with
# R2 railing to 0.5-0.72 Rsun. Validated WD+BD benchmarks sit well under it
# (GD 1400 L6 ~0.10 Rsun, WD 0137-349 BD). (MS primaries cap at R1 itself.)
_IRX_WD_R2_MAX = 0.15

# Per-band significance threshold for the "detected excess" flag (n_significant,
# has_excess >= 2 bands). 3 sigma is the conservative default that keeps the
# single-band false-positive rate low. Lowering it captures marginal excesses
# but raises false positives and (being a hard cut) only relocates the
# threshold-flip instability for bands sitting near the new value.
_IRX_SIG_THRESH = 3.0

# Disk-vs-companion is broadband-DEGENERATE for a compact (WD) primary when the
# excess is confined to the short-IR (<= W2, 4.6 um): warm circumstellar dust
# and a cool companion are indistinguishable without a longer-wavelength band to
# break the SED shape -- the Owens+2023 result (they needed NIR spectroscopy).
# So a WD whose quantified excess ENDS AT W2 (reddest band <= _IRX_RESOLVE_UM)
# is reported 'unresolved' rather than given a (coin-flip) disk/companion
# verdict. A band redward of this (W3 12 um and beyond, also Spitzer IRAC 5.8/8)
# breaks the degeneracy. At WISE all-sky depth only ~8% of WDs reach it, so this
# honestly bounds the classifiable WD sample. (MS primaries are NOT gated: a
# substellar companion can't compete with the host's brightness there, so a
# short-IR MS excess is unambiguously a disk.)
_IRX_RESOLVE_UM = 5.0

# Far-IR confusion noise (1 sigma, mJy/beam) by pyphot band. These are MEASURED
# instrument extragalactic confusion limits -- not tuned thresholds: a far-IR
# point near the confusion floor carries an irreducible uncertainty its
# catalogue error omits (the large beam is filled by unresolved background
# sources). Added in quadrature to the observed error, this lets the Bayesian
# evidence discount confusion-limited points CONTINUOUSLY -- no distance / SNR /
# wavelength cut. Distance enters only implicitly (a distant source is fainter,
# hence nearer the floor); nearby bright disks (eps Eri, eta Crv) sit far above
# it and are untouched. Refs: SPIRE Nguyen+2010; PACS Berta+2011; MIPS Dole+2004.
# NOTE: this is the extragalactic term only. Cirrus (position-dependent, scales
# with the local far-IR background) is the dominant contaminant for distant
# in-plane sources and is not yet folded in.
_IRX_CONF_MJY = {
    'HERSCHEL_SPIRE_PSW': 5.8, 'HERSCHEL_SPIRE_PMW': 6.3, 'HERSCHEL_SPIRE_PLW': 6.8,
    'HERSCHEL_PACS_BLUE': 0.15, 'HERSCHEL_PACS_GREEN': 0.27, 'HERSCHEL_PACS_RED': 0.92,
    'SPITZER_MIPS_70': 0.9, 'SPITZER_MIPS_160': 40.0,
}


def _irx_confusion_sigma(band, lam_um):
    """Far-IR confusion noise for a band as internal F_lambda (erg/s/cm^2/um);
    0 for bands without a confusion entry. Reuses the librarian's Jy->F_lambda
    conversion so it lands in the exact internal units."""
    c = _IRX_CONF_MJY.get(str(band))
    if not c or not np.isfinite(lam_um) or float(lam_um) <= 0:
        return 0.0
    from astropy import units as u
    return float(((c * 1e-3) * u.Jy).to(
        u.erg / u.s / u.cm ** 2 / u.um,
        equivalencies=u.spectral_density(float(lam_um) * u.um)).value)


def _modbb_factor(lam_um):
    """Emission-efficiency factor of the modified blackbody at lam_um."""
    lam = np.asarray(lam_um, float)
    return np.where(lam > _MODBB_LAM0, (_MODBB_LAM0 / lam) ** _MODBB_BETA, 1.0)


def _disk_lmax(T):
    """Empirical ceiling on the disk fractional luminosity L_dust/L_* vs dust
    temperature. Cold/warm debris disks reach ~1e-1 only in extreme
    (giant-impact) systems; sustained HOT dust is observed only at hot-exozodi
    levels (~1e-4, Kral+2017). Log-linear decline from 1e-1 at <=500 K to 1e-4
    at 1500 K. Without this joint prior, a luminous companion photosphere can
    masquerade as an astrophysically impossible hot, ultra-luminous 'disk'
    (e.g. a T-dwarf secondary fit as a 1245 K, L/L*=0.16 blackbody)."""
    T = np.asarray(T, float)
    frac = ((np.log10(np.clip(T, 500.0, 1500.0)) - np.log10(500.0))
            / (np.log10(1500.0) - np.log10(500.0)))
    return 10.0 ** (-1.0 - 3.0 * frac)


def _modbb_bolo_frac(T):
    """Bolometric fraction int(B_mod)/int(B): corrects the Omega <-> L_dust
    relation for the flux the modification removes from the submm tail."""
    from scipy.constants import h, c, k as kB
    lam = np.geomspace(1.0, 5000.0, 400) * 1e-6        # m
    with np.errstate(over='ignore'):
        B = (2 * h * c ** 2 / lam ** 5) / np.expm1(h * c / (lam * kB * T))
    return float(np.trapz(B * _modbb_factor(lam * 1e6), lam)
                 / np.trapz(B, lam))


def _fit_excess_blackbody(waves_um, excess, excess_err, dist_pc, lum_lsun):
    """Characterize an IR excess as a single-temperature (debris-disk) modified
    blackbody (beta=1 beyond 210 um, the field-standard convention).

    Fits ``F_exc(lambda) = Omega * B_mod(T)`` to the positive excess fluxes
    (``F_obs - F_model`` in erg/s/cm^2/um at the band wavelengths), then derives
    the dust temperature ``T`` and the fractional excess luminosity
    ``L_dust/L_*`` from the bolometric integral of the modified blackbody.

    Returns ``{t_dust, l_dust_over_lstar, n_bands, fit_chi2}`` or ``None`` when
    fewer than two positive bands are available (or distance/luminosity unknown)
    -- a single band cannot pin both temperature and solid angle.
    """
    from scipy.constants import h, c, k as kB, sigma as sigma_sb  # SI
    w = np.asarray(waves_um, float)
    F = np.asarray(excess, float)
    E = np.asarray(excess_err, float)
    m = (F > 0) & np.isfinite(F) & np.isfinite(E) & (E > 0)
    if m.sum() < 2 or not dist_pc or dist_pc <= 0 or not lum_lsun \
            or lum_lsun <= 0:
        return None
    lam = w[m] * 1e-6                                  # m
    F, E = F[m], E[m]
    mod_fac = _modbb_factor(w[m])

    def _planck_per_um(T):
        # SI B_lambda [W/m^2/m/sr] -> erg/s/cm^2/um/sr (x1e7 x1e-4 x1e-6)
        B = (2 * h * c ** 2 / lam ** 5) / np.expm1(h * c / (lam * kB * T))
        return B * 1e-3 * mod_fac

    best = None                                        # (chi2, T, Omega)
    for T in np.geomspace(30.0, 1500.0, 400):
        B = _planck_per_um(T)
        wts = 1.0 / E ** 2
        Omega = float(np.sum(F * B * wts) / np.sum(B * B * wts))  # sr (>0)
        if Omega <= 0:
            continue
        chi2 = float(np.sum(((F - Omega * B) * np.sqrt(wts)) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, float(T), Omega)
    if best is None:
        return None
    chi2, T, Omega = best
    f_dust = Omega * sigma_sb * T ** 4 / np.pi * 1e3 * _modbb_bolo_frac(T)
    d_cm = dist_pc * 3.0856775815e18
    l_dust = 4.0 * np.pi * d_cm ** 2 * f_dust          # erg/s
    l_star = lum_lsun * 3.828e33                        # erg/s
    return dict(t_dust=T, l_dust_over_lstar=float(l_dust / l_star),
                omega_sr=Omega, n_bands=int(m.sum()), fit_chi2=chi2)


def _bb_fit_teff(w_um, F, E):
    """Raw blackbody Teff fit to band fluxes.

    Fits ``F_lambda = A * B_lambda(T)`` with the scale ``A`` optimal
    analytically at each ``T`` of a log-spaced scan, with one 4-sigma
    clipping pass so chromospheric UV or a mid-IR excess cannot drag the
    estimate. Returns ``(T, keep_mask)`` or ``None`` with fewer than 3
    usable bands.
    """
    from scipy.constants import h, c, k as kB  # SI
    w = np.asarray(w_um, float)
    F = np.asarray(F, float)
    E = np.asarray(E, float)
    good = np.isfinite(F) & np.isfinite(E) & (F > 0) & (E > 0)
    if good.sum() < 3:
        return None
    lam = w * 1e-6                             # m

    def _best_T(msk):
        # Fit in log-flux with RELATIVE-error weights. Linear-flux 1/sigma^2
        # weighting hands control to the faintest points (their absolute
        # errors are tiny), so five garbage catalog mags from a split
        # high-PM crossmatch can out-vote 2MASS+WISE; relative weighting is
        # scale-free. The calibration anchors are fit with flat 2% relative
        # errors, so this also puts both sides of the calibration on the
        # same convention.
        lamv, lnF = lam[msk], np.log(F[msk])
        wts = (F[msk] / E[msk]) ** 2           # 1/sigma_rel^2
        B0 = 2 * h * c ** 2 / lamv ** 5
        x0 = h * c / (lamv * kB)
        wsum = np.sum(wts)
        best = None                            # (chi2, T, A)
        for T in np.geomspace(1000.0, 60000.0, 600):
            B = B0 / np.expm1(x0 / T)
            if not np.all(B > 0):              # Planck underflow at scan edge
                continue
            lnB = np.log(B)
            lnA = float(np.sum(wts * (lnF - lnB)) / wsum)
            chi2 = float(np.sum(wts * (lnF - lnA - lnB) ** 2))
            if not np.isfinite(chi2):
                continue
            if best is None or chi2 < best[0]:
                best = (chi2, float(T), np.exp(lnA))
        return best

    best = _best_T(good)
    if best is None:
        return None
    # One clipping pass: drop bands deviating > 4 sigma from the first fit
    # (chromospheric GALEX UV on G dwarfs, IR excesses, bad catalogue points).
    _, T, A = best
    B = (2 * h * c ** 2 / lam ** 5) / np.expm1(h * c / (lam * kB * T))
    resid = np.where(good, np.abs(F - A * B) / np.where(E > 0, E, np.inf),
                     np.inf)
    keep = good & (resid < 4.0)
    if 3 <= keep.sum() < good.sum():
        refit = _best_T(keep)
        if refit is not None:
            return refit[1], keep
    return T, good


# Calibration anchors for the gate estimate. Blackbody fits to real stellar
# SEDs are biased: cool (by up to ~600 K around 4000-5000 K, from
# line/molecular blanketing) through the FGK range, and HOT (by ~+2000 K at
# 15000 K, once the optical/NIR bands slide onto the Rayleigh-Jeans tail)
# above ~8000 K -- both large enough to misgate at the 4000 and 16000 K
# boundaries. The bias is removed by fitting the same blackbody to BT-Settl
# synthetic photometry in the star's own band set at these anchor Teffs and
# inverting the curve; the anchor range spans every gate boundary
# (3000/4000/7000/15000/16000 K) with margin on both sides.
_BB_GATE_ANCHORS = (1200., 1600., 2000., 2400., 2800., 3000., 3200., 3500.,
                    4000., 4500., 5000., 5500., 6000., 6500., 7000., 8000.,
                    10000., 12000., 14000., 16000., 20000.)

# Bands blueward of this are excluded from the gate fit entirely: the Balmer
# jump makes the BB-vs-star bias non-monotonic in u (verified: an SDSS_u
# detection flips the 4000-4500 K calibration), and GALEX UV is chromospheric
# on late-type stars anyway. 0.4 um keeps B and g, drops u and GALEX.
_BB_GATE_MIN_UM = 0.4


def _bb_gate_teff(star):
    """Blackbody Teff estimate for grid gating, from the star's photometry.

    The Teff gates need *some* estimate to work with: when the librarian
    finds no catalogue Teff (``star.temp`` is 0/None), neither extreme is
    acceptable -- 0 K reads as ultra-cool and drops every warm grid, while
    skipping the gates multi-counts the BT family and admits
    regime-inappropriate grids. This estimate is used **only** for gating,
    never as a prior.

    The raw blackbody fit is calibrated on the fly against BT-Settl
    synthetic photometry generated for the very bands that survived the
    fit's clipping pass (see ``_BB_GATE_ANCHORS``), making the estimate
    accurate to ~100 K through the gate boundaries regardless of which
    bands the star happens to have. Bands blueward of ``_BB_GATE_MIN_UM``
    are excluded (Balmer jump / chromospheric UV). The calibrated value is
    clamped into the anchor range, so a Rayleigh-Jeans-degenerate fit
    railing the T scan cannot return an unphysical gate value. Returns
    Teff in K, or ``None`` with fewer than 3 usable bands.
    """
    m = (star.used_filters == 1) & \
        (np.asarray(star.wave, float) >= _BB_GATE_MIN_UM)
    w = np.asarray(star.wave, float)[m]        # um
    F = np.asarray(star.flux, float)[m]
    E = np.asarray(star.flux_er, float)[m]
    raw = _bb_fit_teff(w, F, E)
    if raw is None:
        return None
    t_bb, keep = raw
    # Calibration is best-effort: any failure falls back to the raw fit.
    try:
        from .sed_library import load_grid, get_interpolated_flux
        bands = np.asarray(star.filter_names)[m][keep]
        g = load_grid('btsettl')
        anchors, t_anchor_bb = [], []
        for t_true in _BB_GATE_ANCHORS:
            fl = np.asarray(get_interpolated_flux(t_true, 4.5, 0.0, bands, g),
                            dtype=float)
            r = _bb_fit_teff(w[keep], fl, 0.02 * fl)
            # Discard railed anchor fits (scan-edge temperatures): an anchor
            # whose SED is hopeless in this band set (e.g. a 1200 K model
            # observed only in the optical) produces garbage, not signal.
            if r is not None and 1100.0 < r[0] < 55000.0:
                anchors.append(t_true)
                t_anchor_bb.append(r[0])
        t_anchor_bb = np.asarray(t_anchor_bb)
        anchors = np.asarray(anchors)
        if len(anchors) >= 4:
            # Only the bias curve AROUND t_bb has to be invertible -- distant
            # anchors can be garbage (wrong-regime band sets) or carry kinks
            # (Rayleigh-Jeans-degenerate hot fits) without affecting the
            # local inversion. Start from the anchor whose blackbody
            # temperature is closest to t_bb and expand while strictly
            # increasing.
            i0 = int(np.argmin(np.abs(t_anchor_bb - t_bb)))
            lo = i0
            while lo > 0 and t_anchor_bb[lo - 1] < t_anchor_bb[lo]:
                lo -= 1
            hi = i0
            while hi + 1 < len(anchors) and \
                    t_anchor_bb[hi + 1] > t_anchor_bb[hi]:
                hi += 1
            if hi - lo + 1 >= 3:
                # np.interp clamps at both segment ends: a railed/degenerate
                # raw fit gates as the segment's hottest (or coolest) anchor
                # instead of a scan-edge temperature.
                return float(np.interp(t_bb, t_anchor_bb[lo:hi + 1],
                                       anchors[lo:hi + 1]))
    except Exception:
        logger.warning('blackbody gate calibration failed; using the raw '
                       'blackbody Teff', exc_info=True)
    return t_bb


# Host-temperature ceiling for the joint primary+companion marginalisation in
# the IRX companion hypothesis. Above this the host (FGK/WD) is spectrally
# distinct from any cool companion, so the frozen-primary residual fit is
# unbiased and keeps better null specificity; below it (mid/late-M and cooler)
# the host can absorb the companion and the joint fit is needed. ~4000 K spans
# all M dwarfs while excluding K and warmer. Override via the env var
# ARIADNE_COOL_HOST_GATE (e.g. 0 to force the frozen-primary path everywhere).
COOL_HOST_GATE_K = float(os.environ.get('ARIADNE_COOL_HOST_GATE', 4000.0))


def _excess_model_selection(names, waves_um, excess, excess_err, cool_interps,
                            z, av, dist_pc, lum_lsun, use_norm, av_law,
                            teff_primary, fit_names=None, fit_waves=None,
                            fit_errs=None, rad_primary=None, prim_ig=None,
                            logg_primary=None, teff_primary_err=None,
                            rad_primary_err=None, exc_obs=None, fit_obs=None):
    """Bayesian model selection on the IR excess.

    Compares three hypotheses for the excess fluxes by their Bayesian evidence:
    no excess (null, 0 parameters), a **debris-disk blackbody** (free dust
    temperature ``T`` and fractional luminosity ``L_dust/L_*``), and a **cool
    unresolved companion** (free ``Teff2`` and radius ``R2``, a grid photosphere
    sharing the primary's z/Av). The evidence integrals are done by direct 2-D
    quadrature over physical priors -- at two dimensions this is exact and
    noise-free, so it is preferable to nested sampling (which would only add
    Monte-Carlo error). This is the evidence-based analogue of ARIADNE's BMA,
    replacing a chi^2 point comparison: posterior model probabilities fall out
    as ``Z_i / sum Z`` (equal model priors), with the common data normalisation
    cancelling. Also returns marginal parameter posteriors (mean +/- std).

    Priors: log-uniform ``T`` in [30, 1500] K (silicates sublimate hotter) and
    ``L_dust/L_*`` in [1e-7, 1e-1]; uniform ``Teff2`` in
    [2300 K, min(grid, T_primary)] (a companion is cooler than the primary) and
    log-uniform ``R2`` in [0.03, 1] R_sun. Returns ``None`` if <2 finite bands
    or distance/luminosity are unknown.
    """
    from scipy.constants import h as H, c as C, k as KB, sigma as SB
    w = np.asarray(waves_um, float)
    F = np.asarray(excess, float)
    E = np.asarray(excess_err, float)
    sel = np.isfinite(F) & np.isfinite(E) & (E > 0)
    if sel.sum() < 2 or not dist_pc or dist_pc <= 0 \
            or not lum_lsun or lum_lsun <= 0:
        return None
    names = np.asarray(names)[sel]
    w, F, E = w[sel], F[sel], E[sel]
    lam = w * 1e-6
    inv2 = 1.0 / E ** 2
    d_cm = dist_pc * 3.0856775815e18
    l_star = lum_lsun * 3.828e33

    chi2_null = float(np.sum(F ** 2 * inv2))             # model = 0

    # Fit-band consistency penalty. The photosphere fit converged on the fit
    # bands, so any flux an excess component adds there would have appeared as
    # a fit residual: each hypothesis pays sum((c_b/sigma_b)^2) over the fit
    # bands for the flux c_b it predicts in them. Without this, a companion
    # constrained only by W3/W4 can wander to (Teff2, R2) combinations that
    # wildly over-predict J/H/K -- the bright-companion bias. (The rigorous
    # treatment of a decisive companion remains the joint two-component fit.)
    have_fit = (fit_names is not None and fit_waves is not None
                and fit_errs is not None and len(np.atleast_1d(fit_names)) > 0)
    if have_fit:
        fn_b = np.asarray(fit_names)
        fw = np.asarray(fit_waves, float)
        fe_b = np.asarray(fit_errs, float)
        okf = np.isfinite(fw) & np.isfinite(fe_b) & (fe_b > 0)
        fn_b, fw, fe_b = fn_b[okf], fw[okf], fe_b[okf]
        have_fit = fw.size > 0
        flam_fit = fw * 1e-6
        finv2 = 1.0 / fe_b ** 2

    # Disk grid: log-uniform (T, L_dust/L_*), modified blackbody (beta=1 beyond
    # 210 um). Omega(T,L) inverts L_dust = 4 d^2 Omega sigma T^4 * X(T) * 1e3,
    # X(T) the modified bolometric fraction.
    mod_irx = _modbb_factor(w)
    mod_fit = _modbb_factor(fw) if have_fit else None

    def _bmod(lam_m, T, fac):
        with np.errstate(over='ignore'):
            B = (2 * H * C ** 2 / lam_m ** 5) \
                / np.expm1(H * C / (lam_m * KB * T)) * 1e-3
        return B * fac                                     # erg/s/cm^2/um/sr

    # L is sampled finely (0.006 dex): the likelihood is sharp in amplitude
    # (flux ~ L), and a coarse L grid quantizes the evidence -- a nested
    # two-component model would then win on interpolation alone.
    Tg = np.geomspace(30.0, 1500.0, 64)
    Lg = np.geomspace(1e-7, 1e-1, 1024)
    Xg = np.array([_modbb_bolo_frac(T) for T in Tg])
    chi2_disk = np.empty((Tg.size, Lg.size))
    omega_grid = np.empty((Tg.size, Lg.size))
    for i, T in enumerate(Tg):
        B = _bmod(lam, T, mod_irx)
        Omega = Lg * l_star / (4 * d_cm ** 2 * SB * T ** 4 * 1e3 * Xg[i])
        mod = Omega[:, None] * B[None, :]
        chi2_disk[i] = np.sum((F[None, :] - mod) ** 2 * inv2[None, :], axis=1)
        if have_fit:
            Bf = _bmod(flam_fit, T, mod_fit)
            chi2_disk[i] += Omega ** 2 * float(np.sum(Bf ** 2 * finv2))
        # Joint (T, L) prior: fractional luminosities above the empirical
        # ceiling are outside the disk hypothesis' support.
        chi2_disk[i, Lg > _disk_lmax(T)] = np.inf
        omega_grid[i] = Omega

    # Two-temperature disk (warm + cold modified blackbodies, the asteroid- +
    # Kuiper-belt analogue pair): a 4-D quadrature on coarser axes, activated
    # only when the band count can justify the two extra parameters (>=5
    # quantified excess bands); the evidence comparison handles the Occam
    # penalty. T_warm > T_cold breaks the labeling degeneracy.
    chi2_disk2 = None
    Tflat = Lflat = Oflat = None
    if F.size >= 5:
        Tg2 = np.geomspace(30.0, 1500.0, 24)
        Lg2 = np.geomspace(1e-7, 1e-1, 36)
        X2 = np.array([_modbb_bolo_frac(T) for T in Tg2])
        ncomb = Tg2.size * Lg2.size
        Mc = np.empty((ncomb, F.size))
        Pf2 = np.empty((ncomb, fw.size)) if have_fit else None
        Oflat = np.empty(ncomb)
        Tflat = np.repeat(Tg2, Lg2.size)
        Lflat = np.tile(Lg2, Tg2.size)
        k = 0
        for i, T in enumerate(Tg2):
            B = _bmod(lam, T, mod_irx)
            Bf = _bmod(flam_fit, T, mod_fit) if have_fit else None
            Om = Lg2 * l_star / (4 * d_cm ** 2 * SB * T ** 4 * 1e3 * X2[i])
            for j in range(Lg2.size):
                Mc[k] = Om[j] * B
                if have_fit:
                    Pf2[k] = Om[j] * Bf
                Oflat[k] = Om[j]
                k += 1
        chi2_disk2 = np.full((ncomb, ncomb), np.inf)
        valid2 = Lflat <= _disk_lmax(Tflat)      # same joint (T, L) ceiling
        for i in range(ncomb):
            if not valid2[i]:
                continue
            ordered = (Tflat[i] > Tflat) & valid2            # warm i, cold j
            if not ordered.any():
                continue
            mod2 = Mc[i][None, :] + Mc[ordered]
            c2 = np.sum((F[None, :] - mod2) ** 2 * inv2[None, :], axis=1)
            if have_fit:
                c2 += np.sum((Pf2[i][None, :] + Pf2[ordered]) ** 2
                             * finv2[None, :], axis=1)
            chi2_disk2[i, ordered] = c2

    # Companion BMA: each cool grid contributes a 2-D (Teff2 x R2) evidence at
    # the primary's [M/H]; the companion-hypothesis evidence is the across-grid
    # mean (equal grid priors) and the recovered Teff2/R2 are evidence-weighted
    # -- the IR-excess analogue of ARIADNE's photosphere BMA. Per-grid flux is
    # computed once per Teff2 (rad=1) and scaled by R2^2. A grid extended into
    # the L/T regime (cool BT-Settl) plus the dedicated BD grids (ATMO, Sonora)
    # lets a substellar companion outrun the free-blackbody disk, which would
    # otherwise absorb a BD photosphere as a too-hot, unphysical "disk".
    # Companion radius prior: an unresolved coeval companion to a
    # MAIN-SEQUENCE primary cannot be larger than the primary (MS mass-radius
    # is monotonic) -- without the cap, a tilted photosphere baseline lets
    # the quadrature build a luminous pseudo-companion out of the deficit
    # (SCR 1845 produced R2 = 0.114 against a 0.069 Rsun primary). The cap is
    # floored at 0.12 Rsun (~1.2 R_Jup): substellar companions are
    # degeneracy-supported, so a T dwarf at 0.09-0.10 Rsun is genuinely
    # larger than a late-M primary -- a literal R2 <= R1 against SCR 1845's
    # 0.07-0.10 Rsun primary excludes the true T6 solution and flips the
    # verdict to "disk". Compact (WD/sdB) primaries: a BD companion is genuinely
    # LARGER than the primary, so R2 <= R1 is wrong -- but leaving the range open
    # to 1 Rsun (the old behaviour) let the companion build a luminous pseudo-
    # star (R2 -> 0.5-0.72 Rsun) out of any red residual: the 64% WD-companion
    # over-call. The physical bound is SUBSTELLAR (an IR-excess-only companion to
    # a WD that doesn't dominate the optical must be a BD/giant planet; a
    # luminous dM is the joint-binary regime), so cap at _IRX_WD_R2_MAX.
    r2_hi = 1.0
    if rad_primary is not None and np.isfinite(rad_primary):
        if rad_primary > 0.05:
            r2_hi = max(float(rad_primary), 0.12)   # MS primary: cap at R1
        else:
            r2_hi = _IRX_WD_R2_MAX                   # compact: substellar ceiling
    R2g = np.geomspace(0.03, r2_hi, 64)
    t_hi_max = float(teff_primary) if teff_primary else 6000.0
    # --- Joint primary marginalisation (de-biases the companion Teff2) ---
    # The frozen-primary residual fit lets a COOL host photosphere absorb the
    # companion: the base SED fit, run on the near-IR fit bands the companion
    # also lights up, inflates the primary radius to soak up that flux, so the
    # residual the companion grid then fits is biased and Teff2 pins (~1700 K on
    # an M8 host, regardless of truth). Here the primary (teff1, rad1) is instead
    # re-marginalised against the FULL band set jointly with the companion: the
    # optical fit bands anchor the primary (it cannot inflate without breaking
    # them) and the companion explains the IR excess. rad1 enters the flux
    # linearly (model ~ rad1^2), so its amplitude A1 is profiled in closed form
    # per node; teff1 is gridded. Debris disks are far-IR (orthogonal to the
    # primary's optical/near-IR), so this marginalisation factorises for them
    # into a common volume V_prim -- multiplied into z_null/z_disk/z_disk2 below,
    # it cancels in disk-vs-disk ratios and only re-weights disk-vs-companion.
    # Falls back to the frozen-primary residual fit when the primary grid or the
    # observed fluxes are unavailable (e.g. the WD/Koester path).
    #
    # GATED TO COOL HOSTS (teff_primary < COOL_HOST_GATE_K): absorption only
    # happens when the host photosphere overlaps the companion in temperature
    # (both cool), i.e. mid/late-M and cooler primaries. On WARM (FGK) and WD
    # hosts the cool companion is spectrally distinct, the frozen primary cannot
    # absorb it, and the extra (teff1, rad1) freedom of the joint fit only adds
    # noise-fitting flexibility -- which costs NULL specificity (a clean solar
    # twin drifting to 'ambiguous') and perturbs WD+BD companion Teff2. So warm
    # and WD hosts stay on the validated frozen-primary path; only cool hosts,
    # where the bias is real, get the joint marginalisation.
    joint = (prim_ig is not None and have_fit and exc_obs is not None
             and fit_obs is not None and teff_primary
             and np.isfinite(teff_primary) and float(teff_primary) > 0
             and float(teff_primary) < COOL_HOST_GATE_K)
    V_prim = 1.0
    Tg1 = np.array([float(teff_primary) if teff_primary else 0.0])
    def _scalar_err(e, default):
        # posterior uncertainties are stored as asymmetric (lower, upper) pairs;
        # collapse to a single positive scalar (the wider side) for the grid span.
        try:
            arr = np.atleast_1d(np.asarray(e, float))
            arr = arr[np.isfinite(arr) & (arr > 0)]
            return float(np.max(arr)) if arr.size else float(default)
        except (TypeError, ValueError):
            return float(default)

    if joint:
        obs_e = np.asarray(exc_obs, float)[sel]          # obs flux, excess bands
        obs_f = np.asarray(fit_obs, float)[okf]          # obs flux, fit bands
        te = float(teff_primary)
        ts = _scalar_err(teff_primary_err, 0.03 * te)
        pta = np.asarray(prim_ig.index_columns[1], float)   # primary Teff axis
        t1_lo = max(te - 4 * ts, float(np.min(pta)))
        t1_hi = min(te + 4 * ts, float(np.max(pta)))
        Tg1 = (np.linspace(t1_lo, t1_hi, 11) if t1_hi > t1_lo + 1.0
               else np.array([te]))
        lg1 = float(logg_primary) if logg_primary is not None else 4.5
        zg1 = float(z)
        pcol_e = np.array([b in prim_ig.column_index for b in names])
        pcol_f = np.array([b in prim_ig.column_index for b in fn_b])
        Pe = np.zeros((Tg1.size, names.size))            # primary unit flux,
        Pf = np.zeros((Tg1.size, fn_b.size))             # excess + fit bands
        for k, t1 in enumerate(Tg1):
            th = (np.array([t1, lg1, zg1, 1.0, av]) if use_norm
                  else np.array([t1, lg1, zg1, dist_pc, 1.0, av]))
            if pcol_e.any():
                ve = np.asarray(model_grid(th, names[pcol_e], w[pcol_e],
                                           prim_ig, use_norm, av_law)).ravel()
                Pe[k, pcol_e] = np.where(np.isfinite(ve), ve, 0.0)
            if pcol_f.any():
                vf = np.asarray(model_grid(th, fn_b[pcol_f], fw[pcol_f],
                                           prim_ig, use_norm, av_law)).ravel()
                Pf[k, pcol_f] = np.where(np.isfinite(vf), vf, 0.0)
        # rad1^2 (A1) clamp from the primary radius posterior.
        if rad_primary is not None and np.isfinite(rad_primary) \
                and rad_primary > 0:
            r1 = float(rad_primary)
            r1s = _scalar_err(rad_primary_err, 0.2 * r1)
            a1_lo = max(r1 - 4 * r1s, 0.3 * r1) ** 2
            a1_hi = (r1 + 4 * r1s) ** 2
        else:
            a1_lo, a1_hi = 1e-6, 1e6
        # Primary-only fit-band chi^2 profile over teff1 (A1 profiled) -> V_prim
        # (the primary nuisance volume the disk/null hypotheses also integrate).
        den_f = np.sum(Pf ** 2 * finv2, axis=1)          # per teff1 node
        num0_f = np.sum(obs_f[None, :] * Pf * finv2, axis=1)
        chi2_p = np.full(Tg1.size, np.inf)
        for k in range(Tg1.size):
            if den_f[k] <= 0:
                continue
            a1 = min(max(num0_f[k] / den_f[k], a1_lo), a1_hi)
            chi2_p[k] = float(np.sum((obs_f - a1 * Pf[k]) ** 2 * finv2))
        c_fit = float(np.min(chi2_p)) if np.isfinite(chi2_p).any() else 0.0
        V_prim = float(np.mean(np.exp(-(chi2_p - c_fit) / 2)))
        # Pre-summed primary terms for the joint A1 profile (teff1-only).
        num0 = (np.sum(obs_e[None, :] * Pe * inv2, axis=1)
                + np.sum(obs_f[None, :] * Pf * finv2, axis=1))
        den = den_f + np.sum(Pe ** 2 * inv2, axis=1)
        # Re-optimised NULL (the R2 -> 0 limit of the joint companion), on the
        # SAME [(fit - c_fit) + excess] scale as the companion chi^2. The null
        # MUST get the same primary (teff1, rad1) re-optimisation the companion
        # does -- otherwise the companion beats a clean photosphere purely by
        # re-fitting the primary the frozen null cannot (a false-positive
        # companion on null hosts; VB10 went none -> ambiguous without this).
        a1n = np.clip(np.where(den > 0, num0 / np.where(den > 0, den, 1.0),
                               np.nan), a1_lo, a1_hi)
        chi2_null_re = np.full(Tg1.size, np.inf)
        for k in range(Tg1.size):
            if not np.isfinite(a1n[k]):
                continue
            cf_k = float(np.sum((obs_f - a1n[k] * Pf[k]) ** 2 * finv2))
            ce_k = float(np.sum((obs_e - a1n[k] * Pe[k]) ** 2 * inv2))
            chi2_null_re[k] = (cf_k - c_fit) + ce_k

    comp_per_grid = []   # (gname, T2g, Tg1, chi2[nT1, 64, 64])
    for gname_g, ig in cool_interps:
        la, ta, fa = ig.index_columns
        t_lo = float(np.min(ta))
        t_hi = min(float(np.max(ta)), t_hi_max)
        if t_hi <= t_lo + 1.0:
            continue
        # Evaluate on exact grid nodes: z snapped to the nearest [M/H] node
        # (the companion is coeval -- the primary's z -- and the BD grids'
        # 0.5-dex feh resolution dwarfs the snap), logg at 5.0 (canonical field
        # BD) snapped to a node. Off-node values would put nonzero interpolation
        # weight on the sparse cubes' missing nodes (e.g. Sonora's non-solar
        # quarter-step logg holes) and NaN the whole grid out of the BMA.
        z_g = float(fa[int(np.argmin(np.abs(np.asarray(fa) - z)))])
        lg_g = float(la[int(np.argmin(np.abs(np.asarray(la) - 5.0)))])
        T2g = np.linspace(t_lo, t_hi, 64)
        chi2 = np.full((Tg1.size, T2g.size, R2g.size), np.inf)
        names_all = np.concatenate([names, fn_b]) if have_fit else names
        w_all = np.concatenate([w, fw]) if have_fit else w
        # Bands the BD grids carry no column for (they stop at ~30 um; the
        # far-IR set was added later) are modeled by their asymptotes, which
        # are exact for a cool photosphere: pure Rayleigh-Jeans at
        # lam >= 8 um (F_lambda ~ lam^-4, anchored on the reddest finite
        # band) and Wien-dead (~0) at lam < 1 um. Without this, a SPIRE
        # detection KeyErrors the whole quantification. Mid-range gaps stay
        # NaN and skip the node (none exist in the shipped grids).
        in_cols = np.array([b in ig.column_index for b in names_all])
        names_in, w_in = names_all[in_cols], w_all[in_cols]
        for i, T2 in enumerate(T2g):
            theta = (np.array([T2, lg_g, z_g, 1.0, av]) if use_norm
                     else np.array([T2, lg_g, z_g, dist_pc, 1.0, av]))
            funit_all = np.full(len(names_all), np.nan)
            funit_all[in_cols] = np.asarray(
                model_grid(theta, names_in, w_in, ig, use_norm,
                           av_law)).ravel()
            bad = ~np.isfinite(funit_all)
            if bad.any():
                fin = np.isfinite(funit_all) & (funit_all > 0)
                if fin.any():
                    iref = int(np.argmax(np.where(fin, w_all, -np.inf)))
                    rj = funit_all[iref] * (w_all[iref] / w_all) ** 4
                    funit_all = np.where(bad & (w_all >= 8.0), rj, funit_all)
                funit_all = np.where(
                    ~np.isfinite(funit_all) & (w_all < 1.0), 0.0, funit_all)
            funit = funit_all[:len(names)]
            if not np.all(np.isfinite(funit)) or np.all(funit <= 0):
                continue
            if joint:
                # Companion fit-band flux (rad=1); zero where the cool grid has
                # no column (already RJ/Wien-filled above for far-IR/blue).
                cf_b = funit_all[len(names):] if have_fit \
                    else np.zeros(fn_b.size)
                cf_b = np.where(np.isfinite(cf_b), cf_b, 0.0)
                R2sq = R2g ** 2
                cP = (np.sum(funit[None, :] * Pe * inv2, axis=1)
                      + np.sum(cf_b[None, :] * Pf * finv2, axis=1))   # per t1
                for k in range(Tg1.size):
                    if not np.isfinite(den[k]) or den[k] <= 0:
                        continue
                    a1 = (num0[k] - R2sq * cP[k]) / den[k]          # per R2
                    a1 = np.clip(a1, a1_lo, a1_hi)
                    me = a1[:, None] * Pe[k][None, :] \
                        + R2sq[:, None] * funit[None, :]
                    mf = a1[:, None] * Pf[k][None, :] \
                        + R2sq[:, None] * cf_b[None, :]
                    ce = np.sum((obs_e[None, :] - me) ** 2 * inv2[None, :],
                                axis=1)
                    cfit = np.sum((obs_f[None, :] - mf) ** 2 * finv2[None, :],
                                  axis=1)
                    chi2[k, i] = (cfit - c_fit) + ce      # Delta-fit + excess
            else:
                mod = (R2g ** 2)[:, None] * funit[None, :]
                c0 = np.sum((F[None, :] - mod) ** 2 * inv2[None, :], axis=1)
                if have_fit:
                    cf = funit_all[len(names):]
                    okc = np.isfinite(cf) & (cf > 0)
                    if okc.any():
                        c0 = c0 + (R2g ** 4) * float(
                            np.sum(cf[okc] ** 2 * finv2[okc]))
                chi2[0, i] = c0
        if np.isfinite(chi2).any():
            comp_per_grid.append((gname_g, T2g, Tg1, chi2))
    comp_ok = len(comp_per_grid) > 0

    # Evidence = mean likelihood over the (prior-uniform) grids; subtract the
    # global min chi^2 for numerical stability (cancels in the ratios).
    chi2_comp_min = min((float(np.min(c)) for _, _, _, c in comp_per_grid),
                        default=np.inf)
    chi2_d2_min = (float(np.min(chi2_disk2)) if chi2_disk2 is not None
                   else np.inf)
    # In joint mode the null is the re-optimised primary (R2 -> 0 companion);
    # otherwise the frozen-primary residual. V_prim is the primary (teff1, rad1)
    # nuisance volume the companion integrates explicitly (its teff1 axis); the
    # far-IR disk hypotheses do not interact with the primary, so they pick it
    # up as a common factor.
    null_joint = joint and np.isfinite(chi2_null_re).any()
    null_ref = float(np.min(chi2_null_re)) if null_joint else chi2_null
    m0 = min(null_ref, float(np.min(chi2_disk)), chi2_comp_min, chi2_d2_min)
    if null_joint:
        z_null = float(np.mean(np.exp(-(chi2_null_re - m0) / 2)))
    else:
        z_null = float(np.exp(-(chi2_null - m0) / 2)) * V_prim
    with np.errstate(invalid='ignore'):
        wd = np.where(np.isfinite(chi2_disk),
                      np.exp(-(chi2_disk - m0) / 2), 0.0)
    # Mean over the prior SUPPORT (the L<=L_max(T) region), not the full grid.
    z_disk = float(wd.sum() / max(np.isfinite(chi2_disk).sum(), 1)) * V_prim
    z_disk2 = 0.0
    wd2 = None
    if chi2_disk2 is not None and np.isfinite(chi2_disk2).any():
        ordered = np.isfinite(chi2_disk2)
        wd2 = np.where(ordered, np.exp(-(chi2_disk2 - m0) / 2), 0.0)
        z_disk2 = float(wd2.sum() / ordered.sum()) * V_prim   # prior half mean
    # Per-grid companion evidence; the hypothesis evidence is their mean. The
    # companion chi^2 is [teff1, Teff2, R2]; the mean over all three axes is the
    # primary-marginalised companion evidence (Occam-commensurable with V_prim).
    z_comp_g = [float(np.mean(np.exp(-(c - m0) / 2)))
                for _, _, _, c in comp_per_grid]
    z_comp = float(np.mean(z_comp_g)) if comp_ok else 0.0
    tot = z_null + z_disk + z_disk2 + z_comp
    p_none, p_disk, p_comp = z_null / tot, z_disk / tot, z_comp / tot
    p_disk2 = z_disk2 / tot

    wdn = wd / wd.sum()
    TT, LL = np.meshgrid(Tg, Lg, indexing='ij')
    t_dust = float(np.sum(wdn * TT))
    ld = float(np.sum(wdn * LL))
    disk = dict(
        t_dust=t_dust, t_dust_err=float(np.sqrt(np.sum(wdn * (TT - t_dust) ** 2))),
        l_dust_over_lstar=ld,
        l_dust_over_lstar_err=float(np.sqrt(np.sum(wdn * (LL - ld) ** 2))),
        omega_sr=float(np.sum(wdn * omega_grid)))
    disk2 = None
    if wd2 is not None and wd2.sum() > 0:
        w2n = wd2 / wd2.sum()
        wi = w2n.sum(axis=1)            # marginal over the warm component
        wj = w2n.sum(axis=0)            # marginal over the cold component
        # MAP pair alongside the marginal means: when a component is weakly
        # constrained its marginal-mean (T, Omega) combination is NOT a real
        # posterior solution (it can badly violate the data); anything that
        # DISPLAYS the 2-T model must use the *_map values.
        im = np.unravel_index(int(np.argmin(chi2_disk2)), chi2_disk2.shape)
        disk2 = dict(
            t_warm=float(np.sum(wi * Tflat)),
            l_warm=float(np.sum(wi * Lflat)),
            omega_warm=float(np.sum(wi * Oflat)),
            t_cold=float(np.sum(wj * Tflat)),
            l_cold=float(np.sum(wj * Lflat)),
            omega_cold=float(np.sum(wj * Oflat)),
            t_warm_map=float(Tflat[im[0]]),
            l_warm_map=float(Lflat[im[0]]),
            omega_warm_map=float(Oflat[im[0]]),
            t_cold_map=float(Tflat[im[1]]),
            l_cold_map=float(Lflat[im[1]]),
            omega_cold_map=float(Oflat[im[1]]))
    companion = None
    if comp_ok and z_comp > 0:
        # Per-grid posterior (within-grid mean/var), then BMA across grids by
        # evidence. The reported error folds the within-grid spread and the
        # across-grid disagreement -- the model uncertainty the BMA exposes.
        gp = []   # (gname, t2, t2var, r2, r2var, z_g, samples)
        _rng = np.random.default_rng(0)
        for (gname_g, T2g, _Tg1, c), zg in zip(comp_per_grid, z_comp_g):
            # c is [teff1, Teff2, R2]; marginalise the primary teff1 axis to get
            # the (Teff2, R2) posterior weights for this grid.
            wc = np.exp(-(c - m0) / 2).sum(axis=0)
            s = float(wc.sum())
            if not np.isfinite(s) or s <= 0:
                continue
            wcn = wc / s
            T2M, R2M = np.meshgrid(T2g, R2g, indexing='ij')
            t2 = float(np.sum(wcn * T2M))
            r2 = float(np.sum(wcn * R2M))
            # Correlated (Teff2, R2) posterior samples off the quadrature
            # weights -- T2 and R2 are strongly anticorrelated along the
            # constant-flux manifold, so independent marginal draws (mean +/-
            # std) badly over-disperse anything visualizing the posterior.
            idx = _rng.choice(wcn.size, size=40, p=wcn.ravel())
            samples = [[float(T2M.ravel()[i]), float(R2M.ravel()[i])]
                       for i in idx]
            gp.append((gname_g, t2, float(np.sum(wcn * (T2M - t2) ** 2)),
                       r2, float(np.sum(wcn * (R2M - r2) ** 2)), zg, samples))
        if gp:
            # MAP pair (global chi^2 minimum across grids / teff1 / Teff2 / R2):
            # the actual best-fit (Teff2, R2). The marginal MEAN below drifts
            # toward the grid centre (~1700 K) for a weakly-detected companion
            # -- a prior artefact, not a solution -- so quote teff2_map (like the
            # 2-T disk *_map values) when sigma/Teff is large.
            t2_map = r2_map = float('nan')
            _bestc = np.inf
            for (gname_g, T2g, _Tg1, c) in comp_per_grid:
                mi = int(np.argmin(c))
                cv = float(c.ravel()[mi])
                if cv < _bestc:
                    _bestc = cv
                    _, _i, _j = np.unravel_index(mi, c.shape)
                    t2_map, r2_map = float(T2g[_i]), float(R2g[_j])
            gw = np.array([g[5] for g in gp], float)
            gw = gw / gw.sum()
            t2a = np.array([g[1] for g in gp])
            r2a = np.array([g[3] for g in gp])
            t2v = np.array([g[2] for g in gp])
            r2v = np.array([g[4] for g in gp])
            teff2 = float(np.sum(gw * t2a))
            rad2 = float(np.sum(gw * r2a))
            companion = dict(
                teff2=teff2,
                teff2_map=t2_map,
                teff2_err=float(np.sqrt(np.sum(gw * (t2v + (t2a - teff2) ** 2)))),
                rad2=rad2,
                rad2_map=r2_map,
                rad2_err=float(np.sqrt(np.sum(gw * (r2v + (r2a - rad2) ** 2)))),
                n_grids=len(gp),
                per_grid={g[0]: dict(teff2=g[1],
                                     teff2_err=float(np.sqrt(g[2])),
                                     rad2=g[3],
                                     rad2_err=float(np.sqrt(g[4])),
                                     weight=float(wi),
                                     samples=g[6])
                          for g, wi in zip(gp, gw)})

    # Classification: the two disk variants jointly oppose the companion.
    p_dtot = p_disk + p_disk2
    pm = max(p_none, p_dtot, p_comp)
    if p_none == pm:
        cls = 'none'
    elif p_dtot >= p_comp and p_dtot / (p_dtot + p_comp) > 0.75:
        cls = 'disk'
    elif p_comp > p_dtot and p_comp / (p_dtot + p_comp) > 0.75:
        cls = 'companion'
    else:
        cls = 'ambiguous'
    z_dtot = z_disk + z_disk2
    lnbf = (float(np.log(z_dtot) - np.log(z_comp))
            if (z_dtot > 0 and z_comp > 0) else float('nan'))
    return dict(p_none=p_none, p_disk=p_disk, p_disk2=p_disk2,
                p_companion=p_comp, excess_class=cls, disk=disk, disk2=disk2,
                companion=companion, lnbf_disk_companion=lnbf,
                n_bands=int(sel.sum()))


class Fitter:
    """The Fitter class handles the fitting routines and parameter estimation.

    Examples
    --------
    The fitter isn't instantiaded with any arguments, rather you instantiate a
    Fitter object, then you set up the configurations and finally you
    initialize the object by running the initialize method.
    >>> f = Fitter()
    >>> f.star = s  # s must be a valid Star object.
    >>> f.initialize()

    Attributes
    ----------
    star : Star
        The ``Star`` object to fit.
    setup : list
        Sampler configuration. The first element selects the engine
        (``'dynesty'``); the remaining elements are
        ``[nlive, dlogz, bound, sample, threads, dynamic, walks]``.
    out_folder : str
        Folder where results and plots are written. Defaults to the star name.
    verbose : bool
        Program verbosity. Default ``True``.
    norm : bool
        If ``True``, fit a single normalisation constant instead of radius and
        distance; the radius is recovered afterwards from the Gaia parallax.
    grid : str
        Atmosphere grid for a single-grid fit (e.g. ``'phoenix'``,
        ``'btsettl'``, ``'ck04'``, ``'kurucz'``, ``'coelho'``, ``'bosz'``,
        ``'sphinx'``, ``'tlusty'``, ``'newera'``, ``'koester'``). ``'koester'``
        is a DA white-dwarf grid (standalone; pins [Fe/H]=0 and uses uniform
        WD Teff/logg priors).
    bma : bool
        If ``True``, run Bayesian Model Averaging across several grids. This
        loads every model interpolator and fits all of them, so it is slower.
    models : list
        The grids to average over when ``bma`` is ``True``.
    sequential : bool
        If ``True``, fit the BMA grids one after another. See also
        ``n_grid_jobs`` to fit grids concurrently. Default ``True``.
    n_grid_jobs : int
        Number of BMA grids fit concurrently, one core each. Default ``1``
        (sequential). Each job loads its own grid, so size this to the
        machine's core count and memory.
    n_samples : int or None
        Number of samples drawn from the BMA-averaged posterior. ``None`` uses
        the combined chain length.
    walks : int
        Number of ``rwalk`` MCMC steps per dynesty proposal. Default ``25``;
        also settable as ``setup[7]``.
    av_law : str
        Extinction law. Default ``'fitzpatrick'``.
    estimate_logg : bool
        If ``True``, treat log g as a derived quantity via the isochrone step.
        Default ``False``.
    estimate_age : bool
        If ``True``, run the MIST isochrone fit for age, mass and EEP after the
        SED fit. Default ``True``.
    isochrone_dlogz : float
        Evidence tolerance for the MIST isochrone age/mass fit (the dominant
        cost of a BMA run). Lower is more precise but slower. Default ``0.01``.
    prior_setup : dict or None
        Manual prior specification consumed by ``create_priors_from_setup``.
    experimental : bool
        Enable experimental code paths. Default ``False``.

    """

    colors = colors

    def __init__(self):

        # Default values for attributes
        self._interpolators = []
        self._grids = []
        self.out_folder = None
        self.verbose = True
        self.star = None
        self.setup = ['dynesty']
        self.norm = False
        self.grid = 'phoenix'
        self.estimate_logg = False
        self.estimate_age = True
        # Excess-noise (jitter) model. ``True`` (default): one jitter term per
        # band — the most flexible buffer for underestimated/heterogeneous
        # catalogue errors and model inadequacy, but N nuisance dimensions.
        # ``'shared'``: a single jitter term broadcast across all bands — keeps
        # the buffer (so the likelihood stays smooth) while collapsing those N
        # dimensions to 1. ``False``: fix all jitter to 0 (catalogue errors
        # only) — NOT recommended: removing the buffer makes the likelihood
        # stiff (slower, not faster) and biases the stellar parameters.
        self.jitter = True
        self.av_law = 'fitzpatrick'
        self.n_samples = None
        self.bma = False
        # Default BMA ensemble (override via ``f.models``). Listing every grid
        # is safe: ``initialize`` Teff-gates each one to its valid regime, so a
        # grid inappropriate for the star is skipped automatically (e.g. SPHINX
        # only contributes below 4000 K, CK04/Kurucz only above 4000 K,
        # BT-NextGen/BT-Cond only below 4000 K). Coelho (optical-only) and
        # TLUSTY (hot-star, needs its own uniform prior) are deliberately left
        # out — they are standalone-only and must be requested explicitly.
        self.models = ['phoenix', 'btsettl', 'newera', 'btnextgen', 'btcond',
                       'btdusty', 'kurucz', 'ck04', 'bosz', 'sphinx',
                       'stagger', 'coelho']
        self.prior_setup = None
        self.sequential = True
        self.experimental = False
        # Unresolved-binary / SB2 mode (opt-in, off by default). When True the
        # forward model sums two stellar components sharing distance, extinction
        # and metallicity (teff2/logg2/rad2 added). REQUIRES a flux-ratio
        # constraint -- ``contrast`` is a list of (filter_name, delta_mag,
        # sigma) tuples (|m2 - m1| in an ARIADNE filter) -- or the secondary is
        # degenerate. Use ``add_contrast`` or set ``contrast`` directly; the
        # binary gate in initialize enforces no-norm + a parallax + >=1 contrast.
        self.binary = False
        # IR-excess characterization: computed only by the fit_irx() endpoint.
        # The default fit endpoints never run it (the irx bands are still
        # routed out of the fit either way).
        self._do_irx = False
        self._force_wd = False        # set by fit_wd(); forces the WD path
        self.contrast = None
        # Spectro-photometry switch: when False, bands derived from spectra
        # (Gaia XP synthetic fills, SPHEREx super-bands) are dropped from the
        # fit at initialize() -- catalog photometry only.
        self._spectra = True
        # Number of MCMC steps per proposal for dynesty's rwalk sampler.
        self.walks = 25
        # Number of BMA model grids to fit concurrently, each on a single
        # core. 1 (default) keeps the original sequential behaviour. Set this
        # to len(models) to run every grid at once. CAUTION: this spawns
        # ``n_grid_jobs`` processes, so choose a value your machine's core
        # count and memory can handle (each grid loads its own model grid).
        self.n_grid_jobs = 1
        # Evidence tolerance for the MIST isochrone age/mass nested sampling
        # (the dominant cost of a BMA run). Lower is more precise but slower;
        # the isochrone fit itself is parallelised across ``threads``.
        self.isochrone_dlogz = 0.01
        # Activate the typo guard now that all default attributes exist. After
        # this, assigning a *new* public attribute (e.g. ``f.bmaa = True`` for
        # ``f.bma``) raises instead of silently creating a dead attribute and
        # running the wrong kind of fit. See ``__setattr__``.
        self._guard_active = True

    # Public attributes set after __init__ (by initialize(), the fit methods
    # and the BMA worker). Allow-listed so the typo guard does not reject
    # ARIADNE's own internal writes.
    _ALLOWED_DYNAMIC = frozenset({
        'coordinator', 'default_priors', 'fixed', 'ndim', 'out',
        'prior_sources', 'priors', 'sampler', 'start', 'binary_comparison',
    })

    def __setattr__(self, name, value):
        """Guard against silently mistyped config attributes.

        Private (``_``-prefixed) names, declared ``@property`` setters, the
        public attributes created in ``__init__``, and the known post-init
        attributes (``_ALLOWED_DYNAMIC``) are all allowed. A *new* public name
        assigned after construction (e.g. ``f.bmaa`` for ``f.bma``) raises
        ``AttributeError`` rather than becoming a dead attribute. The guard is
        inactive until ``_guard_active`` is set at the end of ``__init__`` so
        construction, copying and unpickling (which start from a fresh
        ``__dict__``) are unaffected.
        """
        if (name.startswith('_')
                or not self.__dict__.get('_guard_active', False)
                or name in self.__dict__
                or name in self._ALLOWED_DYNAMIC
                or isinstance(getattr(type(self), name, None), property)):
            super().__setattr__(name, value)
            return
        raise AttributeError(
            f"{type(self).__name__!r} has no configuration attribute "
            f"{name!r}. Check for a typo (e.g. 'bma' not 'bmaa'). Valid "
            f"config options: av_law, binary, bma, contrast, estimate_age, "
            f"estimate_logg, experimental, grid, isochrone_dlogz, jitter, "
            f"models, n_grid_jobs, n_samples, norm, out_folder, prior_setup, "
            f"sequential, setup, star, verbose, walks."
        )

    @property
    def star(self):
        """Star to fit for."""
        return self._star

    @star.setter
    def star(self, star):
        self._star = star

    @property
    def setup(self):
        """Set up options."""
        return self._setup

    @setup.setter
    def setup(self, setup):
        err_msg = 'The setup has to contain at least the fitting engine'
        err_msg += f', which must be dynesty.\nThe setup was {setup}'
        if len(setup) < 1:
            InputError(err_msg).__raise__()
        self._setup = setup
        self._engine = setup[0]
        defaults = False
        if len(setup) == 1:
            defaults = True
        if self._engine == 'multinest':
            InputError(
                "The 'multinest' engine has been removed (deprecated since "
                "1.5.0). Use engine='dynesty' (the default), which also "
                "supports Bayesian Model Averaging."
            ).__raise__()
        if self._engine == 'dynesty':
            if defaults:
                self._nlive = 500
                self._dlogz = 0.5
                self._bound = 'multi'
                self._sample = 'rwalk'
                self._threads = 1
                self._dynamic = False
            else:
                self._nlive = setup[1]
                self._dlogz = setup[2]
                self._bound = setup[3]
                self._sample = setup[4]
                self._threads = setup[5]
                self._dynamic = setup[6]
                # Optional 8th element: number of rwalk MCMC steps per
                # proposal. Omitted -> keep the current value (default 25),
                # so existing 7-element setups are unaffected.
                if len(setup) > 7:
                    self.walks = setup[7]

    @property
    def norm(self):
        """Bool to decide if a normalization constant will be fitted.

        Set as True to not fit for radius and distance and fit for a
        normalization constant. After the fit a radius is calculated using
        the Gaia parallax.
        """
        return self._norm

    @norm.setter
    def norm(self, norm):
        if type(norm) is not bool:
            InputError('norm must be True or False.').__raise__()
        self._norm = norm

    @property
    def grid(self):
        """Model grid selected."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        assert type(grid) == str
        self._grid = grid
        self._interpolator = load_grid(grid)

    @property
    def bma(self):
        """Bayesian Model Averaging (BMA).

        Set to True if BMA is wanted. This loads every model grid interpolator
        and fits an SED to all of them, so the runtime will be slower!
        """
        return self._bma

    @bma.setter
    def bma(self, bma):
        self._bma = bma if bma_flag else False

    @property
    def models(self):
        """Models to be used in BMA.

        Defaults to the full Teff-gated ensemble
        ``['phoenix', 'btsettl', 'btnextgen', 'btcond', 'btdusty', 'kurucz',
        'ck04', 'bosz', 'sphinx', 'stagger', 'coelho']``; ``initialize``
        skips any grid outside its valid temperature range for the target
        (gating on the catalogue Teff, or on a calibrated blackbody fit to
        the photometry when none exists), plus any grid that cannot
        synthesize one of the star's observed bands. Assign a list to
        override which grids are candidates; the gates still apply.
        """
        return self._bma_models

    @models.setter
    def models(self, mods):
        self._bma_models = mods

    @property
    def sequential(self):
        """Set to True to make BMA sequentially instead of parallel."""
        return self._sequential

    @sequential.setter
    def sequential(self, sequential):
        self._sequential = sequential

    @property
    def n_samples(self):
        """Set number of samples for BMA."""
        return self._nsamp

    @n_samples.setter
    def n_samples(self, nsamp):
        self._nsamp = nsamp

    @property
    def verbose(self):
        """Program verbosity. Default is True."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if type(verbose) is not bool:
            InputError('Verbose must be True or False.').__raise__()
        self._verbose = verbose

    @property
    def out_folder(self):
        """Output folder.

        If none is provided the default will be the starname.
        """
        return self._out_folder

    @out_folder.setter
    def out_folder(self, out_folder):
        if type(out_folder) is not str and out_folder is not None:
            err_msg = 'Output folder must be an address or None.'
            InputError(err_msg).__raise__()
        self._out_folder = out_folder

    @property
    def av_law(self):
        """Select extinction law."""
        return self._av_law

    @av_law.setter
    def av_law(self, law):
        laws = [
            'cardelli',
            'odonnell',
            'calzetti',
            'fitzpatrick',
            'fm07'
        ]
        law = law.lower()
        if law not in laws:
            err_msg = f'Extinction law {law} not recognized.'
            err_msg += 'Available extinction  laws are: `cardelli`, `odonnell`'
            err_msg += ', `calzetti`, `fitzpatrick`, and `fm07`'
            InputError(err_msg).__raise__()
        law_f = None
        if law == laws[0]:
            law_f = extinction.ccm89
        if law == laws[1]:
            law_f = extinction.odonnell94
        if law == laws[2]:
            law_f = extinction.calzetti00
        if law == laws[3]:
            law_f = extinction.fitzpatrick99
        if law == laws[4]:
            law_f = _fm07_rv  # fixed Rv=3.1; see module-level wrapper
        self._av_law = law_f

    @property
    def spectra(self):
        """Use spectro-photometry-derived bands (Gaia XP synthetic fills,
        SPHEREx super-bands)? Default True; set False for catalog-only fits.
        """
        return self._spectra

    @spectra.setter
    def spectra(self, val):
        self._spectra = bool(val)

    def _drop_spectra_bands(self):
        """Drop every spectro-photometry-derived band from the fit.

        Covers the Gaia XP synthetic fills (tracked by Star.xp_filled
        provenance) and any SPHEREX_* band. Same shallow-copy pattern as
        ``_apply_grid_coverage``: the user's Star is left untouched.
        """
        xp = set(getattr(self.star, 'xp_filled', ()) or ())
        names = self.star.filter_names
        used = self.star.used_filters == 1
        drop = used & np.array(
            [(n in xp) or n.startswith('SPHEREX_') for n in names])
        if not drop.any():
            return
        s = copy.copy(self.star)
        s.used_filters = self.star.used_filters.copy()
        s.used_filters[drop] = 0
        s.filter_mask = np.where(s.used_filters == 1)[0]
        self.star = s
        print(colored(
            '\t\t*** Fitter.spectra = False: dropped '
            f'{int(drop.sum())} spectro-photometric band(s): '
            f'{", ".join(names[drop])} ***', 'yellow'))

    def _apply_grid_coverage(self):
        """Drop bands outside the current grid's native wavelength coverage.

        Some grids (e.g. Coelho 2014) only cover the optical; their redder
        photometry was extrapolated when the grid was built and is unreliable.
        For a standalone fit we therefore exclude those bands. The user's Star
        is left untouched: a shallow copy with a restricted ``used_filters`` /
        ``filter_mask`` is substituted, so every downstream consumer
        (dimension count, priors, likelihood, output) sees the reduced set.
        """
        cov = grid_wave_coverage.get(self._grid.lower())
        if cov is None:
            return
        used = self.star.used_filters == 1
        outside = used & (self.star.wave > cov)
        if not outside.any():
            return
        dropped = self.star.filter_names[outside]
        s = copy.copy(self.star)
        s.used_filters = self.star.used_filters.copy()
        s.used_filters[outside] = 0
        s.filter_mask = np.where(s.used_filters == 1)[0]
        self.star = s
        logger.warning(
            "Grid '%s' only covers <= %.2f um; excluding %d out-of-coverage "
            "band(s) from the fit: %s", self._grid, cov, len(dropped),
            ', '.join(dropped))

    def add_contrast(self, filt, delta_mag, sigma):
        """Add a flux-ratio (delta-magnitude) constraint for a binary fit.

        ``filt`` is an ARIADNE filter name where a companion contrast was
        measured (speckle/AO/Gaia), ``delta_mag`` = |m2 - m1| > 0 in that band,
        ``sigma`` its 1-sigma uncertainty. At least one such constraint is
        required to enable ``binary`` mode (the secondary is otherwise
        degenerate). Multiple bands of differing colour additionally constrain
        the secondary Teff.
        """
        if self.contrast is None:
            self.contrast = []
        self.contrast.append((str(filt), float(delta_mag), float(sigma)))

    def _prep_contrast(self):
        """Validate ``self.contrast`` and pack it into the (cfilts, dmags,
        sigmas) array triple the likelihood consumes. The filter array is built
        once so the grid-interpolation cache keys on its identity.
        """
        cols = set(self._interpolator.column_index.keys())
        cfilts, dmags, sigmas = [], [], []
        for filt, dmag, sig in self.contrast:
            filt = str(filt)
            if filt not in cols:
                raise ValueError(
                    f"contrast filter {filt!r} is not a column in the model "
                    f"grid; use an ARIADNE filter name present in the grid.")
            cfilts.append(filt)
            dmags.append(float(dmag))
            sigmas.append(float(sig))
        return (np.array(cfilts), np.array(dmags), np.array(sigmas))

    def _add_binary_dilution(self, out):
        """Per-band host-flux fraction f1/(f1+f2) from the binary posterior.

        This is the Furlan & Howell planet-radius dilution / third-light
        correction -- the quantity exoplanet radius-vetting needs. Distance and
        the shared extinction cancel in the ratio, so the dilution in a band is
        ``f1*rad1^2 / (f1*rad1^2 + f2*rad2^2)``. Stored per fitted band in
        ``out['binary_dilution']`` (posterior samples). Also flags the secondary
        as prior-dominated when its Teff scatter is large -- a single-band
        contrast typically leaves teff2 weakly constrained, in which case the
        dilution is still reliable but teff2 should not be quoted.
        """
        ps = out['posterior_samples']
        t1, g1, z, r1 = (np.asarray(ps['teff']), np.asarray(ps['logg']),
                         np.asarray(ps['z']), np.asarray(ps['rad']))
        t2, g2, r2 = (np.asarray(ps['teff2']), np.asarray(ps['logg2']),
                      np.asarray(ps['rad2']))
        bands = self.star.filter_names[self.star.filter_mask]
        n = len(t1)
        host = np.empty((n, len(bands)))
        for i in range(n):
            f1 = get_interpolated_flux(
                t1[i], g1[i], z[i], bands, interpolator) * r1[i] ** 2
            f2 = get_interpolated_flux(
                t2[i], g2[i], z[i], bands, interpolator) * r2[i] ** 2
            host[i] = f1 / (f1 + f2)
        out['binary_dilution'] = {str(b): host[:, j]
                                  for j, b in enumerate(bands)}
        # Also expose each band's dilution as a posterior trace so the BMA
        # averager (which iterates posterior_samples) folds it across grids.
        for j, b in enumerate(bands):
            out['posterior_samples'][f'dilution_{b}'] = host[:, j]
        frac = float(np.std(t2) / np.median(t2)) if np.median(t2) else np.inf
        out['binary_secondary_unconstrained'] = bool(frac > 0.15)
        if self.verbose:
            med = ', '.join(
                f'{b} {np.median(host[:, j]):.3f}'
                for j, b in enumerate(bands))
            print(colored('\t\t\tBinary host-flux fraction (median): '
                          + med, 'cyan'))
            if out['binary_secondary_unconstrained']:
                print(colored(
                    '\t\t\tWARNING: secondary Teff is prior-dominated '
                    '(sigma/Teff > 15%); the dilution is reliable but teff2 is '
                    'weakly constrained. Add a second contrast band of '
                    'differing colour to constrain it.', 'yellow'))

    def _evidence_fit(self, binary, grids, subfolder):
        """Run an auxiliary SED-only fit (no contrast) and return per-grid
        log-evidence. Used by the model comparison; reuses this fit's star,
        sampler settings and primary priors. For ``binary=True`` the degenerate
        (contrast-free) binary is allowed via ``_skip_contrast_gate``.
        """
        sub = Fitter()
        sub.star = self.star
        sub.setup = list(self._setup)
        sub.norm = False
        sub._av_law = self._av_law            # resolved law (bypass setter)
        sub.binary = binary
        sub._skip_contrast_gate = True        # SED-only, no contrast term
        sub.estimate_logg = self.estimate_logg
        sub.estimate_age = False              # not needed for the evidence
        sub.n_samples = self.n_samples
        sub.n_grid_jobs = self.n_grid_jobs
        sub.prior_setup = {
            k: v for k, v in self.prior_setup.items()
            if k in ('teff', 'logg', 'z', 'dist', 'rad', 'Av', 'norm')}
        sub.out_folder = self.out_folder.rstrip('/') + '/' + subfolder
        sub.verbose = False
        if self._bma:
            sub.bma = True
            sub.models = grids
        else:
            sub.bma = False
            sub.grid = self._grid
        sub.initialize()
        if self._bma:
            sub.fit_bma()
        else:
            sub.fit_dynesty()
        ev = {}
        for g in grids:
            try:
                with open(f'{sub.out_folder}/{g}_out.pkl', 'rb') as fh:
                    ev[g] = float(pickle.load(fh)['global_lnZ'])
            except (FileNotFoundError, KeyError):
                pass
        return ev

    def _binary_model_comparison(self):
        """Does the broadband SED alone favour a binary? Reports, per grid, the
        SED-only Bayesian evidence of the two-component vs single-star model
        (``delta_lnZ > 0`` favours the binary).

        BOTH models condition on the same data -- the photometry, NOT the
        contrast. Including the contrast would trivially favour the binary (it
        is a companion detection), so this asks the separate, well-posed
        question of whether the SED shows two temperatures on its own. If
        ``delta_lnZ <= 0`` the secondary is established by the contrast rather
        than the SED, so trust the dilution but not the SED-derived teff2. The
        contrast still does its job in the main fit (the dilution). Roughly
        triples the run time; a failure here is logged, not fatal.
        """
        grids = list(self._bma_models) if self._bma else [self._grid]
        print(colored('\t\t*** SED-ONLY BINARY-vs-SINGLE EVIDENCE ***', 'cyan'))
        try:
            z_single = self._evidence_fit(False, grids, 'single_star_comparison')
            z_binary = self._evidence_fit(True, grids, 'binary_sed_comparison')
        except Exception as e:
            print(colored(f'\t\t\tmodel comparison failed ({e}); the binary '
                          f'fit is unaffected.', 'yellow'))
            return

        comp = {}
        for g in grids:
            if g not in z_binary or g not in z_single:
                continue
            zb, zs = z_binary[g], z_single[g]
            comp[g] = {'lnZ_binary_sed': zb, 'lnZ_single': zs,
                       'delta_lnZ': zb - zs}
            verdict = ('SED favours binary' if zb - zs > 0
                       else 'SED alone does not require binary')
            print(colored(f'\t\t\t{g}: ln Z  binary {zb:.2f}  single {zs:.2f}'
                          f'  ->  Delta {zb - zs:+.2f}  ({verdict})', 'cyan'))
        self.binary_comparison = comp
        for g, c in comp.items():
            p = f'{self.out_folder}/{g}_out.pkl'
            try:
                with open(p, 'rb') as fh:
                    o = pickle.load(fh)
                o['binary_vs_single'] = c
                with open(p, 'wb') as fh:
                    pickle.dump(o, fh)
            except FileNotFoundError:
                pass

    def initialize(self):
        """Initialize the fitter.

        To be run only after every input is added.
        This function calculates the number of dimensions, runs the prior
        creation, creates output directory, initializes coordinators and sets
        up global variables.
        """
        global prior_dict, coordinator, fixed, order, star, use_norm, av_law
        global binary, contrast, jitter_shared
        jitter_shared = (getattr(self, 'jitter', True) == 'shared')
        self.start = time.time()
        err_msg = 'No star is detected. Please create an instance of Star.'
        if self.star is None:
            er = InputError(err_msg)
            er.log(self.out + '/output.log')
            er.__raise__()

        # Unresolved-binary mode gate. The secondary is degenerate without a
        # flux-ratio constraint, so refuse to run without one; it also needs the
        # explicit rad/dist (non-norm) model and a parallax-based distance.
        binary = bool(self.binary)
        contrast = None
        if binary:
            if self.norm:
                raise ValueError('binary mode requires norm=False (explicit '
                                 'rad/dist), not the normalisation parametrisation.')
            if not self.contrast and not getattr(self, '_skip_contrast_gate',
                                                  False):
                raise ValueError('binary mode requires at least one flux-ratio '
                                 'constraint: Fitter.add_contrast(filter, '
                                 'delta_mag, sigma) or set Fitter.contrast.')
            if self.star.dist is None or self.star.dist == -1:
                raise ValueError('binary mode requires a parallax-based distance '
                                 '(shared between the two components).')
            # contrast is optional only for the internal SED-only comparison fit
            # (_skip_contrast_gate); a normal binary fit always has one.
            contrast = self._prep_contrast() if self.contrast else None
        # Spectro-photometry opt-out: Fitter.spectra = False drops every band
        # whose value came from a spectrum rather than a catalog measurement
        # (Gaia XP synthetic fills; SPHEREx super-bands when they land).
        if not self.spectra:
            self._drop_spectra_bands()
        # For a standalone fit with a grid of limited wavelength coverage,
        # exclude out-of-coverage bands (e.g. Coelho's extrapolated IR). Not
        # applied in BMA, where every grid must share the same band set.
        if not self._bma:
            self._apply_grid_coverage()
        star = self.star
        if not self._bma:
            global interpolator
            interpolator = self._interpolator
        use_norm = self.norm
        # Extinction law
        av_law = self._av_law

        # Declare order of parameters.
        if not self.norm:
            order = np.array(
                [
                    'teff', 'logg', 'z',
                    'dist', 'rad', 'Av'
                ]
            )
            if binary:
                order = np.append(order, ['teff2', 'logg2', 'rad2'])
        else:
            order = np.array(['teff', 'logg', 'z', 'norm', 'Av'])

        # Create output directory
        if self.out_folder is None:
            self.out_folder = self.star.starname + '/'
        create_dir(self.out_folder)

        self.star.save_mags(self.out_folder + '/')

        # Single-grid coverage pre-pass. A single-grid ensemble (e.g. a WD fit
        # via fit_wd, which forces koester) has no cross-grid evidence
        # comparison, so a fit band the grid cannot synthesize must be
        # DESELECTED -- dropped from the fit -- rather than dropping the only
        # grid, which would leave the band-coverage guard below with an empty
        # ensemble and abort the fit (a WD carrying a J-PLUS narrow band that
        # koester has no column for, JPLUS_J0515, did exactly this). Done before
        # the noise-term count so ndim stays consistent. (Multi-grid BMA keeps
        # the strict guard: evidences are only comparable when every member fits
        # the same band set.)
        if self._bma and len(self._bma_models) == 1:
            _fin = getattr(self.load_interpolator(self._bma_models[0].lower()),
                           'finite_bands', None)
            if _fin is not None:
                _fn = self.star.filter_names
                _miss = [str(b) for b, u in zip(_fn, self.star.used_filters)
                         if u == 1 and str(b) not in _fin]
                if _miss:
                    for _b in _miss:
                        self.star.used_filters[_fn == _b] = 0
                    print(colored(
                        f'\t\t*** single-grid fit ({self._bma_models[0]}): '
                        f'deselected {len(_miss)} uncovered fit band(s) '
                        f'{", ".join(_miss)} (no grid column). ***', 'yellow'))

        # Parameter coordination.
        # Order for the parameters are:
        # teff, logg, z, dist, rad, Av, noise
        # or
        # teff, logg, z, norm, Av, noise
        npars = 6 if not self.norm else 5
        if binary:
            npars += 3  # teff2, logg2, rad2
        # Noise model: one jitter term per band by default, or a single shared
        # jitter term when ``jitter == 'shared'``.
        npars += 1 if getattr(self, 'jitter', True) == 'shared' else self.star.used_filters.sum()
        npars = int(npars)
        self.coordinator = np.zeros(npars)  # 1 for fixed params
        self.fixed = np.zeros(npars)
        coordinator = self.coordinator
        fixed = self.fixed

        # Setup priors.
        self.default_priors = self._default_priors()
        self.create_priors_from_setup()
        prior_dict = self.priors

        # Fast / bulk mode: deactivate the per-band excess-noise (jitter) terms
        # by fixing every ``*_noise`` parameter to 0. This removes one sampling
        # dimension per band (the jitter terms dominate the dimensionality of a
        # well-covered SED) and makes the likelihood fall back to the catalogue
        # errors alone — no likelihood change needed, the fixed-parameter
        # machinery handles it. ``get_ndim`` below subtracts these from ndim.
        if not self.jitter:
            for i, p in enumerate(order):
                if str(p).endswith('_noise'):
                    self.coordinator[i] = 1
                    self.fixed[i] = 0.0

        # Get dimensions.
        self.ndim = self.get_ndim()

        # warnings
        if len(self._setup) == 1:
            logger.warning('Using default setup values')

        # BMA settings
        # if BMA is used, load all interpolators requested.
        if self._bma:
            if self.n_samples is None:
                self.n_samples = 'max'

            if self.star.offline:
                if self.star.temp is None:
                    self.star.temp = 4001
                off_msg = 'Offline mode assumes that the stellar'
                off_msg += ' temperature is greater than 4000 K'
                off_msg += '. If you believe this is not the case then please '
                off_msg += 'add a temperature to the Star constructor. '
                off_msg += f'The input temperature is {self.star.temp}'
                print(colored(off_msg, 'yellow'))

            # The Teff gates are not optional -- they exist so the ensemble
            # cannot multi-count one model family (the BT triplet is
            # identical above 2600 K by construction) or admit
            # regime-inappropriate grids.
            # When no catalogue Teff exists (no Gaia GSP-Phot ->
            # star.temp is 0 or None), gate on a blackbody fit to the
            # observed photometry instead: accurate to a few hundred K, far
            # tighter than any gate boundary needs.
            # 0, None and the offline -1 sentinel all mean "no estimate".
            gate_temp = self.star.temp
            if not gate_temp or gate_temp < 0:
                gate_temp = _bb_gate_teff(self.star)
                if gate_temp:
                    msg = '\t\t*** No catalogue Teff available: grid gating '
                    msg += 'uses a blackbody fit to the photometry, '
                    msg += f'Teff ~ {gate_temp:.0f} K (override by passing '
                    msg += 'temp to the Star constructor) ***'
                    print(colored(msg, 'yellow'))
                else:
                    # No catalogue Teff and no photometry to estimate one
                    # from; assume a warm star (the historical offline-mode
                    # default).
                    gate_temp = 4001.0
                    msg = '\t\t*** No Teff estimate possible: grid gating '
                    msg += 'assumes Teff > 4000 K (override by passing temp '
                    msg += 'to the Star constructor) ***'
                    print(colored(msg, 'yellow'))
            for mod in self._bma_models:
                # BT-NextGen and BT-Cond are identical to BT-Settl (and each
                # other) by construction above ~2600 K -- verified on the
                # genuine SVO data (2026-06): 18-20% apart at 2000 K,
                # converged at 2600 K. They participate only in the late-M/L
                # regime where the dust treatments actually differ.
                if (mod.lower() in ['btcond', 'btnextgen'] and
                    gate_temp > 2600) or \
                        (mod.lower() in ['ck04', 'kurucz'] and
                         gate_temp < 4000) or \
                        (mod.lower() == 'coelho' and gate_temp < 3000) or \
                        (mod.lower() == 'newera' and
                         (gate_temp < 2300 or gate_temp > 12000)) or \
                        (mod.lower() == 'bosz' and
                         (gate_temp < 2800 or gate_temp > 16000)) or \
                        (mod.lower() == 'sphinx' and gate_temp > 4000) or \
                        (mod.lower() == 'btdusty' and gate_temp > 2600) or \
                        (mod.lower() == 'stagger' and
                         (gate_temp < 4000 or gate_temp > 7000)) or \
                        (mod.lower() == 'tlusty' and gate_temp < 15000):
                    continue

                df = self.load_interpolator(mod.lower())

                # A grid that cannot synthesize one of the star's observed
                # bands (NaN at every node -- e.g. STAGGER's POLLUX spectra
                # lack GALEX UV and everything past ~13 um) returns -inf for
                # every likelihood call: a full sampler run wasted on a
                # structural zero weight. Skip it loudly instead. The band set
                # cannot be reduced per-grid in BMA mode -- evidences are only
                # comparable when every member fits the same bands.
                fin = getattr(df, 'finite_bands', None)
                if fin is not None:
                    used = self.star.filter_names[self.star.used_filters == 1]
                    miss = [b for b in used if b not in fin]
                    if miss:
                        print(colored(
                            f'\t\t*** {mod} dropped from the BMA ensemble: '
                            f'no grid coverage for fitted band(s) '
                            f'{", ".join(miss)} -- its evidence would be '
                            'zero. ***', 'yellow'))
                        continue

                self._interpolators.append(df)
                self._grids.append(mod)
            thr = self._threads if self._sequential else len(
                self._interpolators)
        else:
            thr = self._threads
        en = 'Bayesian Model Averaging' if self._bma else self._engine
        display_routine(en, self._nlive, self._dlogz, self.ndim, self._bound,
                        self._sample, thr, self._dynamic)
        self.show_priors()

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = 6 if not self.norm else 5
        if self.binary:
            ndim += 3  # teff2, logg2, rad2
        ndim += 1 if getattr(self, 'jitter', True) == 'shared' else self.star.used_filters.sum()
        ndim -= self.coordinator.sum()
        return int(ndim)

    @staticmethod
    def _gaia_color_absmag(star):
        """Gaia BP-RP colour and absolute G magnitude from the star.

        Returns ``(bp_rp, abs_g)``, either being ``None`` when its inputs are
        missing (no Gaia photometry, or no parallax/distance). This is the robust
        cool-dwarf signal: M dwarfs almost always have Gaia BP/RP photometry even
        when they lack a Gaia Teff/luminosity estimate.
        """
        try:
            fn = list(star.filter_names)
            mags = star.mags
            bp = mags[fn.index('GaiaDR2v2_BP')]
            g = mags[fn.index('GaiaDR2v2_G')]
            rp = mags[fn.index('GaiaDR2v2_RP')]
        except (ValueError, AttributeError, KeyError, IndexError, TypeError):
            return None, None
        bp_rp = (bp - rp) if (bp and rp) else None
        abs_g = None
        dist = getattr(star, 'dist', None)
        if g and dist and dist > 0:
            abs_g = g + 5.0 - 5.0 * np.log10(dist)
        return bp_rp, abs_g

    @staticmethod
    def _is_wd_host(star):
        """True when the star sits on the Gaia white-dwarf cooling sequence
        (faint absolute G, blue colour: M_G > 10 and BP-RP < 1.5).

        A WD photosphere is Koester, whose spectra stop at ~3 um, so its
        IR-excess bands are predicted by Rayleigh-Jeans extrapolation (the WD
        continuum is pure RJ there). The main-sequence deep-IR ensemble would
        mis-fit the WD and mischaracterize a cool-companion excess as warm
        dust (WD 0137-349, GD 1400). The colour cut excludes the M-dwarf
        sequence, which is comparably faint but much redder.
        """
        bp_rp, abs_g = Fitter._gaia_color_absmag(star)
        return bool(abs_g is not None and abs_g > 10.0
                    and bp_rp is not None and bp_rp < 1.5)

    @staticmethod
    def _cool_dwarf_flags(temp, lum, bp_rp=None, abs_g=None):
        """Classify a star for regime-aware priors.

        Returns ``(is_cool, is_cool_dwarf)``. The FGK population priors mishandle
        M dwarfs (Teff pulled ~600 K hot into a hot-giant degeneracy; logg left
        wide), so cool stars get an M/late-K Teff prior and cool *dwarfs* a tight
        main-sequence logg prior.

        Primary signal is Gaia BP-RP colour (>1.8 -> M/late-K) and absolute G
        magnitude (>8 -> on the main sequence, excluding red giants), because M
        dwarfs frequently lack the Gaia Teff/luminosity estimates. Falls back to
        the Teff estimate (< 4000 K) and luminosity (< 0.5 Lsun) when colour /
        parallax are unavailable (e.g. non-Gaia input).
        """
        if bp_rp is not None:
            is_cool = bp_rp > 1.8
        else:
            # temp > 0 (not just != 0) so the -1 / -999 "no Gaia data" sentinels
            # are treated as unknown, NOT as cool (-1 < 4000 would misfire).
            is_cool = temp is not None and temp > 0 and temp < 4000
        if abs_g is not None:
            dwarfish = abs_g > 8.0
        elif lum is not None and lum > 0:
            dwarfish = lum < 0.5
        else:
            dwarfish = False
        return is_cool, bool(is_cool and dwarfish)

    def _default_priors(self):
        global order
        defaults = dict()

        # Initialize prior source tracking
        if not hasattr(self, 'prior_sources'):
            self.prior_sources = {}

        # Resolve spectroscopic parameters: prefer spectroscopic_params,
        # fall back to rave_params for backward compatibility.
        spec = getattr(self.star, 'spectroscopic_params', None)
        if spec is None:
            spec = getattr(self.star, 'rave_params', None)
        spec_source = spec.get('source', 'spectroscopic') if spec else None

        # Per-field null-safe helper: a spectroscopic field is usable only if
        # the spec dict exists and both the value and its error are not None.
        # This handles partial spec dicts (e.g. Hypatia [Fe/H]-only results
        # where teff/logg are None) by falling back per-field to the
        # population/default prior rather than crashing.
        def _spec_has(field):
            return (spec is not None and
                    spec.get(field) is not None and
                    spec.get(field + '_err') is not None)

        # Regime detection for cool stars: the FGK population priors are wrong
        # for M dwarfs (Teff pulled ~600 K hot into a hot-giant degeneracy; logg
        # left wide open). Use the Gaia Teff/luminosity estimates to switch in
        # M-dwarf-appropriate Teff/logg priors below. Spectroscopic priors, when
        # present, still take precedence over these.
        _bp_rp, _abs_g = self._gaia_color_absmag(self.star)
        is_cool, is_cool_dwarf = self._cool_dwarf_flags(
            getattr(self.star, 'temp', None), getattr(self.star, 'lum', None),
            _bp_rp, _abs_g)

        _grid_l = (getattr(self, '_grid', '') or '').lower()
        # A WD host in fit_irx is a koester-only BMA (self._grid is unset), so
        # also honour the _wd_host flag -- otherwise the WD gets the FGK/MS
        # Teff and logg priors and the fit rails off the koester grid edges.
        if getattr(self, '_wd_host', False):
            _grid_l = 'koester'
        # Teff: standalone grids that sit outside the FGK population prior need
        # their own wide uniform prior. Koester white dwarfs span 5000-80000 K;
        # TLUSTY hot stars 15-55 kK. The FGK population prior assigns ~zero
        # probability there, so it cannot reach either regime.
        if _grid_l == 'koester':
            defaults['teff'] = st.uniform(loc=5000, scale=75000)  # 5-80 kK (WD)
            self.prior_sources['teff'] = 'koester_wd_uniform'
            logger.info('Using uniform Teff prior for the Koester WD grid')
        elif _grid_l == 'tlusty':
            defaults['teff'] = st.uniform(loc=15000, scale=40000)  # 15-55 kK
            self.prior_sources['teff'] = 'tlusty_uniform'
            logger.info('Using uniform Teff prior for the TLUSTY hot-star grid')
        # Teff: spectroscopic star-specific or population prior
        elif _spec_has('teff'):
            defaults['teff'] = st.norm(loc=spec['teff'],
                                       scale=spec['teff_err'])
            self.prior_sources['teff'] = spec_source
            logger.info('Using %s Teff prior (star-specific)', spec_source)
        elif is_cool:
            # M / late-K dwarf range; uniform avoids imposing the FGK centre.
            defaults['teff'] = st.uniform(loc=2300, scale=1900)  # 2300-4200 K
            self.prior_sources['teff'] = 'mdwarf_uniform'
            logger.info('Using cool-star uniform Teff prior (Gaia Teff < 4000 K)')
        else:
            with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
                defaults['teff'] = pickle.load(jar)
            self.prior_sources['teff'] = 'rave_population'
            logger.info('Using population Teff prior (no spectroscopic match)')

        # logg: spectroscopic, isochrone, or population prior
        if _grid_l == 'koester':
            defaults['logg'] = st.uniform(loc=6.5, scale=3.0)  # 6.5-9.5 (WD)
            self.prior_sources['logg'] = 'koester_wd_uniform'
            logger.info('Using uniform logg prior for the Koester WD grid')
        elif _spec_has('logg'):
            defaults['logg'] = st.norm(loc=spec['logg'],
                                       scale=spec['logg_err'])
            self.prior_sources['logg'] = spec_source
            logger.info('Using %s logg prior (star-specific)', spec_source)
        elif is_cool_dwarf:
            # M dwarfs sit tightly on the main sequence; this replaces both the
            # wide uniform default and the (typically uninformative, ~3 dex)
            # isochrone estimate, which is what let broadband fits drift to the
            # hot-giant solution.
            defaults['logg'] = st.norm(loc=5.0, scale=0.25)
            self.prior_sources['logg'] = 'mdwarf_main_sequence'
            logger.info('Using M-dwarf main-sequence logg prior N(5.0, 0.25)')
        elif self.star.get_logg:
            defaults['logg'] = st.norm(
                loc=self.star.logg, scale=self.star.logg_e)
            self.prior_sources['logg'] = 'isochrone'
            logger.info('Using isochrone logg estimate')
        else:
            defaults['logg'] = st.uniform(loc=3.5, scale=2.5)
            self.prior_sources['logg'] = 'uniform_default'
            logger.info('Using population logg prior')

        # [Fe/H]: spectroscopic or population prior
        if _grid_l == 'koester':
            # White dwarfs have no metallicity axis; the Koester grid carries a
            # single dummy z=0 node, so pin z to it. The trilinear interpolator
            # handles the resulting length-1 third axis as long as z is fixed at
            # exactly that node (z index is 2 in both norm and non-norm orders).
            z_idx = 2
            self.coordinator[z_idx] = 1
            self.fixed[z_idx] = 0.0
            defaults['z'] = None
            self.prior_sources['z'] = 'koester_fixed_zero'
            logger.info('Pinning [Fe/H]=0 for the Koester WD grid (no z axis)')
        elif _spec_has('feh'):
            defaults['z'] = st.norm(loc=spec['feh'], scale=spec['feh_err'])
            self.prior_sources['z'] = spec_source
            logger.info('Using %s [Fe/H] prior (star-specific)', spec_source)
        else:
            defaults['z'] = st.norm(loc=-0.125, scale=0.234)
            self.prior_sources['z'] = 'population'
            logger.info('Using population [Fe/H] prior')
        # Distance prior setup.
        if not self._norm:
            if self.star.dist != -1:
                _d_scale = 3 * self.star.dist_e
                defaults['dist'] = st.truncnorm(
                    a=-self.star.dist / _d_scale, b=1e100,
                    loc=self.star.dist, scale=_d_scale,
                )
                self.prior_sources['dist'] = 'gaia'
            else:
                defaults['dist'] = st.uniform(loc=1, scale=3000)
                self.prior_sources['dist'] = 'uniform_default'
            # Radius prior setup.
            if _grid_l == 'koester':
                # White dwarfs are ~0.004-0.02 R_sun; the main-sequence floor of
                # 0.05 R_sun would exclude every WD radius. Use a degenerate
                # range instead.
                defaults['rad'] = st.uniform(loc=0.003, scale=0.047)  # 0.003-0.05
                self.prior_sources['rad'] = 'koester_wd_uniform'
            else:
                defaults['rad'] = st.uniform(loc=0.05, scale=100)
                self.prior_sources['rad'] = 'uniform_default'
        # Normalization prior setup.
        else:
            # up = 1 / 1e-30
            # defaults['norm'] = st.truncnorm(a=0, b=up, loc=1e-20, scale=1e-10)
            defaults['norm'] = st.uniform(loc=0, scale=1e-20)
            self.prior_sources['norm'] = 'uniform_default'
        # Extinction prior setup. Within the Local Bubble (~70 pc) extinction
        # is negligible and a free Av only absorbs blue model mismatch and
        # biases Teff hot, so the DEFAULT prior auto-pins Av=0 for d < 70 pc.
        # A user-supplied Av prior (via prior_setup) bypasses this entirely.
        d_est = getattr(self.star, 'dist', None)
        _av_is_default = (getattr(self, 'prior_setup', None) is None
                          or (isinstance(self.prior_setup, dict)
                              and self.prior_setup.get('Av', 'default')
                              == 'default'))
        auto_pin = bool(d_est and 0 < d_est < 70 and self.star.Av != 0.
                        and _av_is_default)
        if auto_pin:
            print(colored(
                '\t\t*** d = %.1f pc < 70 pc: Av auto-pinned to 0 (override '
                'by setting an explicit Av prior) ***' % d_est, 'yellow'))
        if self.star.Av == 0. or auto_pin:
            av_idx = 4 if self._norm else 5
            self.coordinator[av_idx] = 1
            self.fixed[av_idx] = 0
            defaults['Av'] = None
            self.prior_sources['Av'] = ('auto_pinned_d<70pc' if auto_pin
                                        else 'fixed_zero')
        else:
            if self.star.Av_e is None:
                defaults['Av'] = st.uniform(loc=0, scale=self.star.Av)
                self.prior_sources['Av'] = 'dustmap_uniform'
            else:
                mu = self.star.Av
                sig = self.star.Av_e
                low = 0
                up = 100
                b, a = (up - mu) / sig, (low - mu) / sig
                defaults['Av'] = st.truncnorm(loc=mu, scale=sig, a=a, b=b)
                self.prior_sources['Av'] = 'dustmap'
        # Noise model prior setup. Per-band jitter by default; a single shared
        # jitter term when ``jitter == 'shared'`` -- the likelihood broadcasts
        # the one term across all bands, keeping the error/model-adequacy buffer
        # while collapsing N nuisance dimensions to 1.
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        # Jitter prior scale: 10x the catalog error, floored at 5% of the
        # band flux. The jitter absorbs MODEL inadequacy, whose size does not
        # care how precise the catalog is -- scaling with sigma_cat alone
        # prior-forbids millimag bands (Gaia G at 0.003 mag -> 3% ceiling)
        # from absorbing the ~5-10% systematics every atmosphere grid carries
        # for cool stars, and the likelihood then tilts the photosphere to
        # nail the precise bands at the expense of the rest (SCR 1845).
        if getattr(self, 'jitter', True) == 'shared':
            mflx, merr = float(np.median(flxs)), float(np.median(errs))
            sc = max(merr * 10, 0.05 * mflx)
            defaults['shared_noise'] = st.truncnorm(
                loc=0, scale=sc, a=0, b=(1 - mflx) / sc)
            order = np.append(order, 'shared_noise')
        else:
            for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
                p_ = get_noise_name(filt) + '_noise'
                sigma = max(flx_e * 10, 0.05 * flx)
                b = (1 - flx) / sigma
                defaults[p_] = st.truncnorm(loc=0, scale=sigma, a=0, b=b)
                order = np.append(order, p_)
        return defaults

    def create_priors_from_setup(self):
        """Create priors from the manual setup."""
        prior_dict = dict()
        keys = self.prior_setup.keys()
        noise = []
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        if getattr(self, 'jitter', True) == 'shared':
            noise = ['shared_noise']
        else:
            for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
                p_ = get_noise_name(filt) + '_noise'
                noise.append(p_)
        prior_out = 'Parameter\tPrior\tValues\n'
        if 'norm' in keys and ('rad' in keys or 'dist' in keys):
            er = PriorError('rad or dist', 1)
            er.log(self.out_folder + '/output.log')
            er.__raise__()
        for k in keys:
            if type(self.prior_setup[k]) == str:
                if self.prior_setup[k] == 'default':
                    prior_dict[k] = self.default_priors[k]
                    prior_out += k + '\tdefault\n'
                if self.prior_setup[k].lower() == 'rave':
                    # RAVE prior only available for teff and logg. It's already
                    # the default for [Fe/H]
                    if k == 'logg' or k == 'teff':
                        if k == 'teff':
                            with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
                                prior_dict[k] = pickle.load(jar)
                        else:  # k == 'logg'
                            with open(priorsdir + '/logg_ppf.pkl', 'rb') as jar:
                                prior_dict[k] = pickle.load(jar)
                        prior_out += k + '\tRAVE\n'

            else:
                prior = self.prior_setup[k][0]
                if prior == 'fixed':
                    value = self.prior_setup[k][1]
                    idx = np.where(k == order)[0]
                    self.coordinator[idx] = 1
                    self.fixed[idx] = value
                    prior_out += k + '\tfixed\t{}\n'.format(value)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_fixed'
                if prior == 'normal':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    prior_dict[k] = st.norm(loc=mu, scale=sig)
                    prior_out += k + '\tnormal\t{}\t{}\n'.format(mu, sig)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_normal'
                if prior == 'truncnorm':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    low = self.prior_setup[k][3]
                    up = self.prior_setup[k][4]
                    b, a = (up - mu) / sig, (low - mu) / sig
                    prior_dict[k] = st.truncnorm(a=a, b=b, loc=mu, scale=sig)
                    prior_out += k
                    prior_out += '\ttruncatednormal\t{}\t{}\t{}\t{}\n'.format(
                        mu, sig, low, up)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_truncnorm'
                if prior == 'uniform':
                    low = self.prior_setup[k][1]
                    up = self.prior_setup[k][2]
                    prior_dict[k] = st.uniform(loc=low, scale=up - low)
                    prior_out += k + '\tuniform\t{}\t{}\n'.format(low, up)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_uniform'
        for par in noise:
            prior_dict[par] = self.default_priors[par]
        # Binary secondary shares the primary's resolved priors. Sharing the
        # exact prior keeps the teff1>=teff2 swap in the prior transform
        # measure-preserving (and gives a sensible default the user need not
        # restate). prior_out already records the primary entries.
        if self.binary:
            for tgt, src in (('teff2', 'teff'), ('logg2', 'logg'),
                             ('rad2', 'rad')):
                if src in prior_dict:
                    prior_dict[tgt] = prior_dict[src]
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[tgt] = self.prior_sources.get(src)
        ff = open(self.out_folder + '/prior.dat', 'w')
        ff.write(prior_out)
        ff.close()
        del ff
        self.priors = prior_dict
        pass

    def show_priors(self, return_string=False):
        """Display priors in statistical notation.

        Parameters
        ----------
        return_string : bool, optional
            If True, return the formatted string instead of printing.
            Default: False

        Returns
        -------
        str or None
            Formatted prior information if return_string=True, else None
        """
        global order
        _T = '\t\t\t'
        c = np.random.choice(colors)

        # Physical parameters to display (skip noise priors)
        physical_params = []
        for param_name in self.priors.keys():
            if param_name.endswith('_noise'):
                continue
            physical_params.append(param_name)

        # Format each prior
        prior_strs = {}
        for param_name in physical_params:
            prior_obj = self.priors[param_name]
            param_idx = np.where(order == param_name)[0]
            if len(param_idx) > 0 and self.coordinator[param_idx[0]] == 1:
                fixed_value = self.fixed[param_idx[0]]
                prior_str = f"Fixed({fixed_value:.4f})"
            else:
                prior_str = self._format_prior_notation(param_name, prior_obj)
            prior_strs[param_name] = prior_str

        # Sort parameters in logical order
        param_order = ['teff', 'logg', 'z', 'dist', 'rad', 'norm', 'Av']
        sorted_params = [p for p in param_order if p in physical_params]
        sorted_params.extend([p for p in physical_params if p not in param_order])

        # Build table
        lines = []
        lines.append(f'{_T}{"Parameter":24s}  {"Prior"}')
        lines.append(f'{_T}{"-" * 52}')
        for param_name in sorted_params:
            lines.append(f'{_T}{param_name:24s}  {prior_strs[param_name]}')

        output = '\n'.join(lines)

        if return_string:
            return output
        else:
            print(colored(output, c))
            print()
            return None

    def _format_prior_notation(self, param_name, prior_obj):
        """Format scipy.stats distribution in statistical notation.

        Parameters
        ----------
        param_name : str
            Parameter name (e.g., 'teff', 'logg')
        prior_obj : scipy.stats distribution
            The prior distribution object

        Returns
        -------
        str
            Formatted statistical notation
        """
        # RAVE population prior: loaded as an InterpolatedUnivariateSpline
        # from teff_ppf.pkl / logg_ppf.pkl. Detect by type since manual
        # prior setups don't populate prior_sources.
        if 'Spline' in type(prior_obj).__name__:
            return "RAVE (population)"
        if hasattr(self, 'prior_sources') and param_name in self.prior_sources:
            if self.prior_sources[param_name] == 'rave_population':
                return "RAVE (population)"

        # Frozen scipy distributions expose .dist (the class) and .kwds/.args
        # (the frozen parameters). Use the underlying class name to identify
        # the distribution.
        if hasattr(prior_obj, 'dist'):
            inner_name = type(prior_obj.dist).__name__
            kwds = dict(getattr(prior_obj, 'kwds', {}) or {})
            args = list(getattr(prior_obj, 'args', ()) or ())
        else:
            inner_name = type(prior_obj).__name__
            kwds, args = {}, []

        def _pop(key, idx, default=0.0):
            if key in kwds:
                return kwds[key]
            if idx < len(args):
                return args[idx]
            return default

        try:
            if 'norm_gen' in inner_name and 'truncnorm' not in inner_name:
                mu = _pop('loc', 0)
                sigma = _pop('scale', 1, 1.0)
                return f"N({mu:.3f}, {sigma:.3f})"

            if 'uniform_gen' in inner_name:
                a = _pop('loc', 0)
                scale = _pop('scale', 1, 1.0)
                b = a + scale
                return f"U({a:.2f}, {b:.2f})"

            if 'truncnorm_gen' in inner_name:
                a_std = _pop('a', 0)
                b_std = _pop('b', 1)
                mu = _pop('loc', 2)
                sigma = _pop('scale', 3, 1.0)
                a_act = mu + a_std * sigma
                b_act = mu + b_std * sigma
                a_str = "-inf" if a_std < -1e10 else f"{a_act:.2f}"
                b_str = "inf" if b_std > 1e10 else f"{b_act:.2f}"
                return f"TN({mu:.2f}, {sigma:.2f}, [{a_str}, {b_str}])"

            return inner_name

        except (AttributeError, TypeError):
            return f"{inner_name} (custom)"

    def fit(self):
        """Run fitting routine."""
        self.fit_dynesty()
        if self.binary:
            self._binary_model_comparison()
        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time,
            self.out_folder, self._engine, self.norm, grid=self._grid)
        pass

    # -- Operating-mode endpoints -------------------------------------------
    # Four explicit, non-overlapping entry points:
    #   fit_bma()    -- the default: BMA photosphere fit, NO IR-excess.
    #   fit_single() -- one explicit grid (koester, tlusty, coelho, ...).
    #   fit_binary() -- the SB2 two-component fit.
    #   fit_irx()    -- BMA + IR-excess characterization, deep-IR ensemble
    #                   auto-selected (user override respected, validated).
    # The three new ones are one-call: they initialize() themselves.

    def fit_single(self, grid=None):
        """Fit one explicit model grid (no BMA, no IR-excess).

        The endpoint for the standalone-only grids (koester, tlusty, coelho)
        or any deliberate single-grid run. ``grid`` overrides ``self.grid``.
        """
        if grid is not None:
            self.grid = grid
        self.binary = False
        self.bma = False
        self._do_irx = False
        self.initialize()
        self.fit()

    def fit_binary(self, bma=False):
        """Fit the unresolved-binary (SB2) two-component model.

        Requires the binary configuration (contrast constraints etc.) to be
        set up beforehand; see the SB2 section of the README. With
        ``bma=True`` the two-component fit is model-averaged over the
        ensemble, else it runs on ``self.grid``.
        """
        self.binary = True
        self.bma = bool(bma)
        self._do_irx = False
        self.initialize()
        if bma:
            self.fit_bma()
        else:
            self.fit()

    def fit_wd(self, models=None, excess_from_um=1.0):
        """White-dwarf SED fit + IR-excess characterization (the WD endpoint).

        Fits the photosphere with the Koester DA grid under WD-appropriate
        priors (Teff 5-80 kK, logg 6.5-9.5) and characterizes any IR excess --
        an unresolved cool companion or a debris disk -- by Rayleigh-Jeans
        extrapolation of the WD continuum past Koester's ~3 um reach. Every
        band redward of ``excess_from_um`` (default 1.0 um, the WD/companion
        crossover) joins the excess set; the WD is fit on its UV-optical
        photometry, where Koester is excellent.

        ``fit_irx`` also auto-detects a WD host from the Gaia HR position and
        routes here; ``fit_wd`` is the deliberate, always-WD endpoint.
        """
        self._force_wd = True
        return self.fit_irx(models=models, excess_from_um=excess_from_um)

    def fit_irx(self, models=None, route_mid_ir=False, excess_from_um=None):
        """BMA fit + IR-excess characterization -- the endpoint that
        quantifies and classifies IR excess.

        The ensemble is auto-selected for IR-excess work: among the candidate
        fit grids, find the deepest IR reach any grid achieves and keep EVERY
        grid that achieves it -- flux range is conserved first, member count
        maximized second, so the common (intersection) reach is clipped as
        little as possible. Passing ``models`` overrides the selection
        entirely and is respected to the full extent -- but grids with no
        IR-excess capability at all (e.g. phoenix, coelho) make the run fail
        loudly rather than silently clip the reach to nothing.

        ``excess_from_um`` sets the fit/excess wavelength boundary: every band
        redward of it joins the quantified-excess set. Default 4.4 um (the W2
        block); ``route_mid_ir=True`` is shorthand for 2.75 um; a WD host
        (auto-detected, or via fit_wd) uses 1.0 um. Pass any value for full
        control (e.g. 2.2 to route H/J/K to the excess for a veiled companion).

        A WD host is detected from the Gaia HR position and dispatched to the
        Koester photosphere + WD priors automatically; fit_wd() is the
        explicit WD endpoint.
        """
        from .sed_library import grid_irx_reach, load_grid
        from .phot_utils import get_effective_wavelength

        def _reach_um(g):
            # Reddest EXCESS band the grid can model (grid_irx_reach returns
            # every in-coverage finite band, so intersect with irx_bands --
            # otherwise an IRX-blind grid like phoenix scores its reddest FIT
            # band and slips past validation).
            try:
                bands = grid_irx_reach(g, load_grid(g)) & set(irx_bands)
            except Exception:
                return 0.0
            w = [get_effective_wavelength(b) for b in bands]
            w = [x for x in w if np.isfinite(x)]
            return max(w) if w else 0.0

        # White-dwarf host: the photosphere is Koester (the only WD grid),
        # whose spectra stop at ~3 um. Koester has zero physical IR-excess
        # reach, but a WD's IR continuum is exactly Rayleigh-Jeans, so the
        # excess bands beyond 3 um are predicted by RJ extrapolation in
        # _compute_irx. Auto-detected from the Gaia HR position, or forced by
        # the dedicated fit_wd() endpoint.
        self._wd_host = (self._is_wd_host(self.star)
                         or getattr(self, '_force_wd', False))
        if self._wd_host:
            self.models = list(models) if models is not None else ['koester']
            _wd_reach = max((get_effective_wavelength(b) for b in irx_bands),
                            default=0.0)
            reaches = {g: (_wd_reach if g == 'koester' else _reach_um(g))
                       for g in self.models}
            print(colored(
                '\t\t*** WD host (Gaia HR position): photosphere = Koester; '
                'IR-excess bands predicted by Rayleigh-Jeans extrapolation '
                '(the WD continuum is pure RJ beyond ~3 um). ***', 'cyan'))
        elif models is not None:
            bad = [g for g in models if _reach_um(g) <= 0.0]
            if bad:
                raise ValueError(
                    'fit_irx: the following grids have no IR-excess reach and '
                    f'would clip the quantified band set to nothing: {bad}. '
                    'Remove them from `models` (or use fit_bma() if you do '
                    'not want IR-excess characterization).')
            self.models = list(models)
            reaches = {g: _reach_um(g) for g in self.models}
        else:
            candidates = ['btsettl', 'btnextgen', 'btcond', 'btdusty',
                          'sphinx', 'newera', 'bosz', 'ck04', 'kurucz',
                          'stagger', 'phoenix']
            reaches = {g: _reach_um(g) for g in candidates}
            deepest = max(reaches.values())
            # keep every grid achieving the deepest reach (2% tolerance for
            # near-identical band edges); Teff gating happens in initialize().
            self.models = [g for g in candidates
                           if reaches[g] >= 0.98 * deepest]
        sel = ', '.join('%s(%.0fum)' % (g, reaches[g]) for g in self.models)
        print(colored('\t\t*** IRX ensemble: %s ***' % sel, 'cyan'))
        if self._norm:
            # Explicit degradation notice: without a physical distance the
            # excess can be detected but not characterized.
            print(colored(
                '\t\t*** fit_irx in norm mode: no physical distance, so the '
                'disk L_dust/L_* and the companion (Teff2, R2) BMA are '
                'UNAVAILABLE; per-band excess significances are still '
                'computed. ***', 'yellow'))
        self.binary = False
        self.bma = True
        self._do_irx = True
        # Band routing. In an IRX fit the W2 block (>= 4.4 um: W2, IRAC 4.5,
        # SPHEREx 4.65/4.90) ALWAYS joins the excess set: it is where a cool
        # companion peaks, so leaving it in the fit lets the photosphere
        # absorb exactly the signal fit_irx exists to quantify (SCR 1845's
        # T6 verdict went ambiguous this way). The photosphere stays anchored
        # through SPHEREX_4300 (4.4 um). Regular SED endpoints (fit_bma,
        # fit_single, fit_binary) are untouched -- W2 is a fit band there.
        # OPT-IN route_mid_ir=True widens routing to the full 2.75-5 um
        # block: maximal companion-vs-disk shape leverage (the W1-W2 CH4
        # color), at the cost of measuring everything beyond 2.4 um against
        # an extrapolated baseline -- any tilt becomes high-significance
        # pseudo-excess (SCR 1845 produced p=1.0 false companions this way
        # when the baseline carried bugs). Enable only when the <= 2.4 um
        # photosphere is demonstrably clean (XP + SPHEREx NIR anchored).
        # Excess-set routing is a single wavelength rule: every band redward of
        # `excess_from_um` joins the quantified-excess set (the photosphere is
        # fit blueward of it). The three built-in modes are just defaults of
        # this one knob, and the user can set any threshold (e.g. route H/J/K
        # to the excess for a heavily veiled NIR companion):
        #   - normal fit_irx : 4.4 um (the W2 block -- W2/IRAC 4.5/SPHEREx red)
        #   - route_mid_ir   : 2.75 um (the full W1-W2 CH4-colour block)
        #   - WD host        : 1.0 um (a hot WD + cool companion cross over at
        #                      ~1 um; Koester owns UV-optical, the companion
        #                      owns >= 1 um, RJ-extrapolated past Koester's
        #                      ~3 um reach -- this also moves every SPHEREx
        #                      band out of the fit so Koester is never
        #                      band-guarded).
        if excess_from_um is None:
            excess_from_um = (1.0 if self._wd_host
                              else 2.75 if route_mid_ir else 4.4)
        ex_bands = [str(b) for b in self.star.filter_names
                    if np.isfinite(get_effective_wavelength(str(b)))
                    and get_effective_wavelength(str(b)) >= excess_from_um]
        self._route_irx_extra(ex_bands, anchor_um=excess_from_um)
        self.initialize()
        # initialize() applies the Teff gate and the band-coverage guard, so
        # the surviving ensemble can be smaller than the roster announced
        # above (e.g. btnextgen/btcond drop out for any star above 4000 K).
        if list(self._grids) != list(self.models):
            print(colored('\t\t*** IRX ensemble after Teff gate / coverage '
                          'guard: %s ***' % ', '.join(self._grids), 'cyan'))
        self.fit_bma()

    # Fit bands that fit_irx(route_mid_ir=True) re-routes to the excess set:
    # the full 2.75-5 um block, moved together so the fit/excess boundary
    # stays a wavelength rule rather than an instrument rule.
    _IRX_EXTRA_BANDS = ('SPHEREX_2750', 'SPHEREX_3050', 'SPHEREX_3350',
                        'WISE_RSR_W1', 'SPITZER_IRAC_36', 'SPHEREX_3700',
                        'SPHEREX_4050', 'SPHEREX_4300', 'SPITZER_IRAC_45',
                        'WISE_RSR_W2', 'SPHEREX_4650', 'SPHEREX_4900')

    # The W2 block (>= 4.4 um), routed to the excess set on EVERY fit_irx
    # call: the peak of a cool companion's SED. Regular SED endpoints keep
    # these as fit bands.
    _IRX_W2_BANDS = ('SPITZER_IRAC_45', 'WISE_RSR_W2',
                     'SPHEREX_4650', 'SPHEREX_4900')

    def _route_irx_extra(self, bands=None, anchor_um=2.4):
        """Move the given fit bands to the quantified-excess set (fit_irx
        endpoint only). Shallow-copy contract: the user's Star is untouched.
        """
        if bands is None:
            bands = self._IRX_EXTRA_BANDS
        star = self.star
        names = star.filter_names
        move = np.isin(names, np.asarray(bands)) & \
            (star.used_filters == 1)
        if not move.any():
            return
        s = copy.copy(star)
        s.used_filters = star.used_filters.copy()
        s.irx_used_filters = star.irx_used_filters.copy()
        s.used_filters[move] = 0
        s.irx_used_filters[move] = 1
        s.filter_mask = np.where(s.used_filters == 1)[0]
        s.irx_filter_mask = np.where(s.irx_used_filters == 1)[0]
        self.star = s
        print(colored(
            '\t\t*** fit_irx: routed %d band(s) to the excess set '
            '(the photosphere is fit from <= %.1f um): %s ***'
            % (int(move.sum()), anchor_um, ', '.join(names[move])), 'cyan'))

    def fit_bma(self):
        """Perform the fit with different models and the average the output.

        Only works with dynesty.
        """
        if not self._grids:
            if self._bma and getattr(self, '_bma_models', None):
                raise RuntimeError(
                    'fit_bma: every requested grid was dropped by the Teff '
                    'gate or the band-coverage guard for this star (see the '
                    'notices above). Choose grids valid for this regime, or '
                    'pass an explicit temp to the Star constructor if the '
                    'gate estimate looks wrong.')
            raise RuntimeError(
                'fit_bma: no grids were loaded at initialize(). If you are '
                'using the manual flow, set `f.bma = True` BEFORE calling '
                'initialize() (the one-call endpoints fit_irx/fit_binary '
                'handle this for you).')
        if len(self.star.filter_names[self.star.filter_mask]) <= 5:
            print(colored('\t\t\tNOT ENOUGH POINTS TO MAKE THE FIT! !', 'red'))
            return

        global interpolator, _BMA_FITTER

        n_jobs = min(max(int(self.n_grid_jobs), 1), len(self._grids))
        if n_jobs > 1:
            # Across-grid parallelism: the grids are independent fits, so run
            # several at once (one core each). Each worker writes its own
            # ``_out.pkl``, which we read back below for averaging.
            ncpu = os.cpu_count() or 1
            if n_jobs > ncpu:
                logger.warning(
                    'n_grid_jobs=%d exceeds the %d available CPU cores; '
                    'this will oversubscribe the machine.', n_jobs, ncpu)
            _BMA_FITTER = self
            print(f'\t\t\tFITTING {len(self._grids)} MODELS, '
                  f'{n_jobs} CONCURRENTLY')
            with Pool(n_jobs) as pool:
                for gr, err in pool.map(_bma_grid_worker,
                                        range(len(self._grids))):
                    if err is not None:
                        dump_out = self.out_folder + '/' + gr + '_DUMP.pkl'
                        DynestyError(dump_out, gr, err).__raise__()
        else:
            for intp, gr in zip(self._interpolators, self._grids):
                interpolator = intp
                self.grid = gr
                out_file = self.out_folder + '/' + gr + '_out.pkl'
                print('\t\t\tFITTING MODEL : ' + gr)
                try:
                    self.fit_dynesty(out_file=out_file)
                except ValueError as e:
                    dump_out = self.out_folder + '/' + gr + '_DUMP.pkl'
                    pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                    DynestyError(dump_out, gr, e).__raise__()
                    continue

        # Now that the fitting finished, read the outputs and average
        # the posteriors
        outs = []
        for g in self._grids:
            in_folder = f'{self.out_folder}/{g}_out.pkl'
            outs.append(in_folder)
            # with open(in_folder, 'rb') as out:
            #     outs.append(pickle.load(out))
        c = np.random.choice(self.colors)
        avgd = self.bayesian_model_average(outs, self._grids, self._norm,
                                           self.n_samples, c)
        self.save_bma(avgd)

        if self.binary:
            self._binary_model_comparison()

        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time, self.out_folder,
            'Bayesian Model Averaging', self.norm)
        pass

    def _bma_dynesty(self, intp, grid):
        global interpolator
        interpolator = intp

        # Parallel parallelized routine experiment
        if self.experimental:
            if self._dynamic:
                with Pool(self._threads) as executor:
                    sampler = dynesty.DynamicNestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz_init=self._dlogz,
                                       nlive_batch=self._nlive,
                                       wt_kwargs={'pfrac': .95})
            else:
                with Pool(self._threads) as executor:
                    sampler = dynesty.NestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz=self._dlogz)

        elif self._dynamic:
            sampler = dynesty.DynamicNestedSampler(
                dynesty_loglike_bma, pt_dynesty, self.ndim,
                bound=self._bound, sample=self._sample, logl_args=([intp])

            )
            sampler.run_nested(dlogz_init=self._dlogz,
                               nlive_init=self._nlive,
                               wt_kwargs={'pfrac': .95})
        else:
            try:
                self.sampler = dynesty.NestedSampler(
                    dynesty_loglike_bma, pt_dynesty, self.ndim,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample,
                    logl_args=([intp])
                )
                self.sampler.run_nested(dlogz=self._dlogz)
            except Error:
                dump_out = self.out_folder + '/' + grid + '_DUMP.pkl'
                pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                er = DynestyError(dump_out, grid)
                er.log(self.out + '/output.log')
                er.__raise__()

        results = self.sampler.results
        out_file = self.out_folder + '/' + grid + '_out.pkl'
        self.save(out_file, results=results)
        pass

    def fit_dynesty(self, out_file=None):
        """Run dynesty."""
        # Set up some globals
        global mask, flux, flux_er, filts, wave
        mask = star.filter_mask
        flux = star.flux[mask]
        flux_er = star.flux_er[mask]
        filts = star.filter_names[mask]
        wave = star.wave[mask]
        if self._dynamic:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.DynamicNestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample,
                        pool=executor, walks=self.walks,
                        queue_size=self._threads
                    )
                    self.sampler.run_nested(dlogz_init=self._dlogz,
                                            nlive_init=self._nlive,
                                            wt_kwargs={'pfrac': 1})
            else:
                self.sampler = dynesty.DynamicNestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=self.walks,
                    bound=self._bound, sample=self._sample

                )
                self.sampler.run_nested(dlogz_init=self._dlogz,
                                        nlive_init=self._nlive,
                                        wt_kwargs={'pfrac': 1})
        else:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.NestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample,
                        pool=executor, walks=self.walks,
                        queue_size=self._threads,
                    )
                    self.sampler.run_nested(dlogz=self._dlogz)
            else:
                self.sampler = dynesty.NestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=self.walks,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample
                )
                self.sampler.run_nested(dlogz=self._dlogz)
        results = self.sampler.results
        if out_file is None:
            out_file = f'{self.out_folder}/{self._grid}_out.pkl'
        self.save(out_file, results=results)
        pass

    def save(self, out_file, results=None):
        """Save the dynesty output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ: The global evidence.
        lnZerr: The global evidence error.
        posterior_samples: A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed: An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit: The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star: The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)
        engine: The fitting engine used (Dynesty)

        Also creates a log file with the best fit parameters and 1 sigma
        error bars.

        """
        out = dict()
        logdat = '#Parameter\tmedian\tupper\tlower\t3sig_CI\n'
        log_out = self.out_folder + '/' + 'best_fit.dat'
        lnz, lnzer, posterior_samples = self.dynesty_results(results)

        mask = self.star.filter_mask

        # Save global evidence
        out['dynesty'] = results
        out['global_lnZ'] = lnz
        out['global_lnZerr'] = lnzer

        # Create raw samples holder

        out['posterior_samples'] = dict()
        j = 0
        k = 0  # filter counter
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                samples = posterior_samples[:, j]
                if 'noise' in param:
                    filt = self.star.filter_names[mask][k]
                    flx = self.star.flux[mask][k]
                    _, samples = flux_to_mag(flx, samples, filt)
                    k += 1
                out['posterior_samples'][param] = samples
                j += 1
            else:
                out['posterior_samples'][param] = self.fixed[i]

        # Save loglike, priors and posteriors.

        out['posterior_samples']['loglike'] = np.zeros(
            posterior_samples.shape[0]
        )

        # If normalization constant was fitted, derive the radius from the
        # distance. With no distance (2-parameter source) there is no radius:
        # set it NaN so mass and luminosity follow as NaN (unmeasurable) instead
        # of crashing the code that expects a 'rad' sample.
        if use_norm:
            if star.dist != -1:
                rad = self._get_rad(
                    out['posterior_samples']['norm'], star.dist, star.dist_e
                )
            else:
                rad = np.full_like(out['posterior_samples']['norm'], np.nan)
            out['posterior_samples']['rad'] = rad

        # Create a distribution of masses.

        logg_samp = out['posterior_samples']['logg']
        rad_samp = out['posterior_samples']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        out['posterior_samples']['grav_mass'] = mass_samp

        # Create a distribution of luminosities.

        teff_samp = out['posterior_samples']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        out['posterior_samples']['lum'] = lum_samp

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['posterior_samples']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            out['posterior_samples']['AD'] = ad_samp

        for i in range(posterior_samples.shape[0]):
            theta = build_params(
                posterior_samples[i, :], flux, flux_er, filts, coordinator,
                fixed, self.norm)
            out['posterior_samples']['loglike'][i] = log_likelihood(
                theta, flux, flux_er, wave, filts, interpolator, self.norm,
                av_law, binary=self.binary, contrast=contrast)

        if self.binary:
            self._add_binary_dilution(out)

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.
        # do only if not bma

        if not self.bma:
            out['best_fit_averaged'] = dict()
            out['uncertainties_averaged'] = dict()
            out['confidence_interval_averaged'] = dict()
            best_theta = np.zeros(order.shape[0])

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if 'noise' in param:
                        continue
                    samp = out['posterior_samples'][param]

                    if param == 'z':
                        logdat = out_filler(samp, logdat, param, '[Fe/H]', out)
                    elif param == 'norm':
                        logdat = out_filler(samp, logdat, param, '(R/D)^2',
                                            out, fmt='e')
                        if star.dist != 1:
                            logdat = out_filler(
                                out['posterior_samples']['rad'], logdat, 'rad',
                                'R', out
                            )
                    else:
                        logdat = out_filler(samp, logdat, param, param, out)
                else:
                    logdat = out_filler(
                        0, logdat, param, param, out, fixed=self.fixed[i]
                    )
                best_theta[i] = out['best_fit_averaged'][param]

            # Add derived mass to best fit dictionary.

            samp = out['posterior_samples']['grav_mass']
            logdat = out_filler(
                samp, logdat, 'grav_mass', 'grav_mass', out
            )

            # Add derived luminosity to best fit dictionary.

            samp = out['posterior_samples']['lum']
            logdat = out_filler(samp, logdat, 'lum', 'lum', out)

            # Add derived angular diameter to best fit dictionary.

            if not use_norm:
                samp = out['posterior_samples']['AD']
                logdat = out_filler(samp, logdat, 'AD', 'AD', out)

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if 'noise' not in param:
                        continue
                    samp = out['posterior_samples'][param]
                    logdat = out_filler(samp, logdat, param,
                                        param, out, fmt='f')

            # Fill in best loglike, prior and posterior.

            out['best_fit_averaged']['loglike'] = log_likelihood(
                best_theta, flux, flux_er, wave,
                filts, interpolator, self.norm, av_law,
                binary=self.binary, contrast=contrast
            )

            # Spectral type
            # Load Mamajek spt table
            mamajek_spt = np.loadtxt(
                filesdir + '/mamajek_spt.dat', dtype=str, usecols=[0])
            mamajek_temp = np.loadtxt(
                filesdir + '/mamajek_spt.dat', usecols=[1])

            # Find spt
            spt_idx = np.argmin(abs(mamajek_temp - out['best_fit_averaged']['teff']))
            spt = mamajek_spt[spt_idx]
            out['spectral_type'] = spt

        # Utilities for plotting.

        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['engine'] = self._engine
        out['norm'] = self.norm
        out['model_grid'] = self.grid
        out['av_law'] = av_law
        out['binary'] = self.binary

        # Fit-health diagnostics (additive; never changes the fit). Grid axes
        # come from the interpolator the fit used (free, no reload); the
        # grid-edge check is skipped gracefully if they can't be obtained.
        grid_axes = None
        _interp = globals().get('interpolator', None)
        if _interp is not None and hasattr(_interp, 'index_columns'):
            try:
                ii = _interp.index_columns          # (logg, teff, z)
                grid_axes = {'logg': ii[0], 'teff': ii[1], 'z': ii[2]}
            except Exception:
                grid_axes = None
        out['diagnostics'] = self._fit_health(
            results, param_samples=out['posterior_samples'],
            grid_axes=grid_axes)
        if self.verbose:
            self._print_fit_health(out['diagnostics'], self.grid)

        if not self.bma:
            with open(log_out, 'w') as logfile:
                logfile.write(logdat)
        try:
            out['irx'] = (self._compute_irx(out, [self._grid], av_law)
                          if getattr(self, '_do_irx', False) else {})
        except Exception as e:                       # never break the save
            logger.warning('IR-excess quantification skipped: %s', e)
            out['irx'] = {}
        pickle.dump(out, open(out_file, 'wb'))
        pass

    def _irx_theta(self, d):
        """Pack a best-fit / posterior-sample dict into the physical theta
        vector model_grid expects, in the same order the fit used."""
        if self._norm:
            order = ['teff', 'logg', 'z', 'norm', 'Av']
        else:
            order = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
            if self.binary:
                order += ['teff2', 'logg2', 'rad2']
        return np.array([float(d[k]) for k in order])

    def _compute_irx(self, out, grids, av_law):
        """Quantify IR excess on the bands every in-play grid reaches.

        The star's IR-excess bands (``config.irx_bands``, routed out of the
        photosphere fit by ``Star``) are compared to the model predicted at the
        best fit. Reach gating follows the BMA rule: a band is *quantified* only
        if every in-play grid reaches it -- the common-reach set is the
        intersection of the grids' ``finite_bands``. Bands the star has but that
        lie beyond the common reach are recorded plot-only (model = NaN), so the
        SED can still show them.

        The model flux and its uncertainty per band are the (evidence-weighted,
        for BMA) posterior-predictive: draws pick a grid by weight and a
        posterior sample from that grid, then evaluate the model on that grid.
        Reported per band: model, model_err, observed, excess = obs - model and
        its significance sigma = excess / sqrt(obs_err^2 + model_err^2). A
        '_summary' entry lists the quantified, plot-only and >=3 sigma bands.
        """
        star = self.star
        mask = np.asarray(getattr(star, 'irx_filter_mask', []), dtype=int)
        if mask.size == 0:
            return {}
        names = np.asarray(star.filter_names)
        irx_filt = names[mask]
        irx_wave = np.asarray(star.wave)[mask]
        irx_obs = np.asarray(star.flux)[mask]
        irx_obs_e = np.asarray(star.flux_er)[mask]

        interps = {g: load_grid(g) for g in grids}
        wd = getattr(self, '_wd_host', False)
        rj_ref_band, rj_ref_wl = None, None
        if wd:
            # WD photosphere: every excess band is quantifiable via RJ
            # extrapolation. Anchor RJ on the reddest FIT band Koester reaches
            # (<= its ~3 um coverage) -- the photosphere there is already deep
            # in the Rayleigh-Jeans tail, so F_lambda ~ lambda^-4 is exact.
            common = set(str(f) for f in irx_filt)
            kg = next((i for g, i in interps.items() if g == 'koester'), None)
            if kg is not None:
                fm = np.asarray(star.filter_mask, int)
                fb = np.asarray(star.filter_names)[fm]
                fw = np.asarray(star.wave)[fm]
                # anchor RJ on the reddest FIT band Koester models RELIABLY;
                # for a hot WD even ~0.8 um is already in the RJ tail (peak <<
                # 0.3 um). Cap at 2.5 um: the Koester model spectra stop at
                # ~3 um, so band fluxes redward of that are unreliable (W2 comes
                # out ~10x high, IRAC-4.5 = 0) and must not anchor the tail.
                cand = [(float(w), str(b)) for b, w in zip(fb, fw)
                        if str(b) in kg.finite_bands
                        and np.isfinite(w) and 0.6 <= w <= 2.5]
                if cand:
                    rj_ref_wl, rj_ref_band = max(cand)
        else:
            common = set.intersection(*[grid_irx_reach(g, i)
                                        for g, i in interps.items()])

        # Posterior mixture: (grid, weight, posterior_samples) per member.
        if out.get('weights') and out.get('originals'):
            members = [(g, float(out['weights'].get(g, 0.0)),
                        out['originals'][g])
                       for g in grids if g in out['originals']]
        else:
            members = [(grids[0], 1.0, out['posterior_samples'])]
        wsum = sum(w for _, w, _ in members) or 1.0
        members = [(g, w / wsum, ps) for g, w, ps in members]

        rng = np.random.default_rng(0)
        ndraw = 256
        gpick = rng.choice(len(members), size=ndraw,
                           p=[w for _, w, _ in members])

        result = {}
        for f, wv, fo, fe in zip(irx_filt, irx_wave, irx_obs, irx_obs_e):
            f = str(f)
            if f not in common:
                result[f] = dict(wave=float(wv), obs=float(fo),
                                 obs_err=float(fe), model=float('nan'),
                                 model_err=float('nan'), excess=float('nan'),
                                 sigma=float('nan'), quantified=False)
                continue
            band = np.array([f])
            wl = np.array([float(wv)])
            draws = np.empty(ndraw)
            for d in range(ndraw):
                g, _w, ps = members[gpick[d]]
                n = len(ps['teff'])
                j = int(rng.integers(n))
                # fixed parameters (e.g. a pinned Av) are stored as scalars, not
                # posterior arrays -- take the scalar; index the rest by draw.
                theta = self._irx_theta(
                    {k: (ps[k][j] if hasattr(ps[k], '__len__') else ps[k])
                     for k in ps})
                try:
                    m = model_grid(theta, band, wl, interps[g], self._norm,
                                   av_law, binary=self.binary)
                    val = float(np.asarray(m).ravel()[0])
                except (KeyError, ValueError, IndexError):
                    # the grid carries no column for this band (e.g. Koester
                    # has none of the >3 um IR-excess bands) -> RJ below.
                    val = float('nan')
                if (wd and rj_ref_band is not None
                        and (not np.isfinite(val) or float(wv) > 2.9)):
                    # Band beyond Koester's ~3 um spectral reach -> RJ-extrapolate
                    # the WD photosphere from the reddest reliably-modeled band.
                    # Force this for ALL bands > 2.9 um even when the grid returns
                    # a (finite) value: the Koester band fluxes are unreliable
                    # there (W2 ~10x high, IRAC-4.5 = 0). The WD continuum is pure
                    # Rayleigh-Jeans there (F_lam ~ lam^-4), so this is exact.
                    mref = model_grid(theta, np.array([rj_ref_band]),
                                      np.array([rj_ref_wl]), interps[g],
                                      self._norm, av_law, binary=self.binary)
                    vref = float(np.asarray(mref).ravel()[0])
                    if np.isfinite(vref) and vref > 0:
                        val = vref * (rj_ref_wl / float(wv)) ** 4
                draws[d] = val
            model = float(np.nanmean(draws))
            model_err = float(np.nanstd(draws))
            excess = float(fo) - model
            # Floor the observed error at the absolute-calibration level before
            # forming the significance (see _IRX_ABS_CAL_FLOOR): the raw
            # catalogue error can be unphysically small for bright IR points.
            obs_e_eff = max(float(fe), _IRX_ABS_CAL_FLOOR * float(fo))
            # Add the far-IR confusion noise (see _IRX_CONF_MJY): the catalogue
            # error omits the unresolved background filling a large far-IR beam,
            # so a confusion-limited point would otherwise look spuriously
            # significant and could drive a false cold (2-T) component.
            obs_e_eff = float(np.hypot(obs_e_eff,
                                       _irx_confusion_sigma(f, float(wv))))
            denom = float(np.hypot(obs_e_eff, model_err))
            sigma = excess / denom if denom > 0 else float('nan')
            result[f] = dict(wave=float(wv), obs=float(fo), obs_err=float(fe),
                             obs_err_eff=obs_e_eff, model=model,
                             model_err=model_err, excess=excess,
                             sigma=sigma, quantified=True)
        q = {k: v for k, v in result.items() if v['quantified']}
        sig = sorted(k for k, v in q.items()
                     if np.isfinite(v['sigma']) and v['sigma'] >= _IRX_SIG_THRESH)
        # Aggregate deviation from the photosphere across the IR bands.
        chi2 = float(np.nansum(
            [v['sigma'] ** 2 for v in q.values() if np.isfinite(v['sigma'])]))
        # Characterize the excess by Bayesian model selection over three
        # hypotheses -- no excess (null), a debris-disk blackbody, and a cool
        # unresolved companion -- returning posterior model probabilities and
        # marginal parameter posteriors (the evidence-based analogue of BMA).
        bf = out.get('best_fit_averaged', {})
        waves = [v['wave'] for v in q.values()]
        exc = [v['excess'] for v in q.values()]
        errs = [float(np.hypot(v.get('obs_err_eff', v['obs_err']),
                               v['model_err']))
                for v in q.values()]
        ms = {}
        if members:
            gname = max(members, key=lambda mm: mm[1])[0]   # highest weight
            # Cool-grid ensemble for the companion BMA: the dedicated BD grids
            # (ATMO, Sonora) plus the fit grid when it reaches the cool regime
            # (e.g. cool-extended BT-Settl). Each is evaluated at the primary's
            # [M/H]; grids that fail to load are simply skipped.
            cool = []
            fitg = interps[gname]
            if float(np.min(fitg.index_columns[1])) < 2300.0:
                cool.append((gname, fitg))
            for _bd in COOL_COMPANION_GRIDS:
                try:
                    cool.append((_bd, load_grid(_bd)))
                except Exception as _e:
                    logger.warning('cool companion grid %s unavailable: %s',
                                   _bd, _e)
            # Fit-band budget for the consistency penalty: photometric error
            # inflated by the FITTED per-band noise terms (mag-space jitter ->
            # flux via mag_to_flux, the likelihood's own convention). Raw
            # photometric errors alone over-penalize -- M-dwarf fits carry
            # several-sigma model systematics that the noise terms absorb.
            fmask = getattr(star, 'filter_mask', None)
            fkw = {}
            if fmask is not None and len(fmask):
                fn_fit = star.filter_names[fmask]
                fe_fit = np.asarray(star.flux_er[fmask], float)
                infl = np.zeros(len(fn_fit))
                for _i, _b in enumerate(fn_fit):
                    nz = bf.get(get_noise_name(_b) + '_noise',
                                bf.get('shared_noise'))
                    try:
                        infl[_i] = float(mag_to_flux(
                            float(star.mags[fmask][_i]), float(nz), _b)[1])
                    except (TypeError, ValueError):
                        infl[_i] = 0.0
                fkw = dict(fit_names=fn_fit, fit_waves=star.wave[fmask],
                           fit_errs=np.sqrt(fe_fit ** 2 + infl ** 2),
                           fit_obs=np.asarray(star.flux[fmask], float))
            # Joint primary marginalisation inputs (de-bias the companion Teff2
            # on cool hosts): the primary grid, its logg, and the teff/rad
            # posterior widths let the companion path re-fit (teff1, rad1)
            # against the full SED instead of inheriting the absorbed photosphere.
            unc = out.get('uncertainties_averaged', {}) or {}
            ms = _excess_model_selection(
                list(q), waves, exc, errs, cool,
                float(bf.get('z', 0.0)), float(bf.get('Av', 0.0)),
                getattr(star, 'dist', None), bf.get('lum'), self._norm, av_law,
                bf.get('teff'), rad_primary=bf.get('rad'),
                prim_ig=interps.get(gname), logg_primary=bf.get('logg'),
                teff_primary_err=unc.get('teff'), rad_primary_err=unc.get('rad'),
                exc_obs=[v['obs'] for v in q.values()], **fkw) or {}
            # Compact-primary composite demotion: a companion whose radius RAILS
            # at the substellar cap (_IRX_WD_R2_MAX) is too luminous to be a
            # brown dwarf -- it is a luminous (dM) companion (the joint-binary
            # regime) or a blend/contaminant, NOT a clean IR-excess substellar
            # companion. Relabel it 'composite' so it leaves the companion
            # sample (the WD over-call: R2 pinned at the cap). A genuine BD/L/T
            # companion (e.g. GD 1400, R2 ~ 0.08) sits well below the cap and is
            # untouched; MS primaries (cap = R1) never reach this branch.
            _rp = bf.get('rad')
            if (ms.get('excess_class') == 'companion' and _rp is not None
                    and np.isfinite(float(_rp)) and float(_rp) <= 0.05):
                _r2 = (ms.get('companion') or {}).get('rad2')
                if (_r2 is not None and np.isfinite(float(_r2))
                        and float(_r2) >= 0.95 * _IRX_WD_R2_MAX):
                    ms = dict(ms)
                    ms['excess_class'] = 'composite'
                    ms['composite_reason'] = (
                        'companion R2 railed at the substellar cap '
                        '(%.3f Rsun): excess too luminous for a brown dwarf -- '
                        'luminous (dM) companion or blend, not a clean IR-excess '
                        'substellar companion' % float(_r2))
            # Compact-primary degeneracy gate (see _IRX_RESOLVE_UM): a WD whose
            # quantified excess ends at W2 cannot be classified disk-vs-companion
            # in broadband -- report 'unresolved' instead of a coin-flip verdict.
            if (ms.get('excess_class') in ('companion', 'disk', 'disk2',
                                           'composite')
                    and _rp is not None and np.isfinite(float(_rp))
                    and float(_rp) <= 0.05):
                _reddest = max((v['wave'] for v in q.values()), default=0.0)
                if float(_reddest) <= _IRX_RESOLVE_UM:
                    ms = dict(ms)
                    ms['excess_class'] = 'unresolved'
                    ms['unresolved_reason'] = (
                        'excess confined to <= W2 (reddest quantified band '
                        '%.1f um): disk vs companion is broadband-degenerate '
                        'without a W3+ band' % float(_reddest))
        # Least-squares single-temperature blackbody fit to the measured
        # excess. This is the curve used by the SED+IRX model overlay: it
        # threads the excess by construction, whereas the evidence-marginalized
        # `disk` posterior-mean omega is skewed high by its log-uniform prior
        # (the integrated L_dust/L_star is robust; the curve shape is not).
        disk_bb = _fit_excess_blackbody(
            waves, exc, errs, getattr(star, 'dist', None), bf.get('lum'))
        result['_summary'] = dict(
            quantified=sorted(q),
            plot_only=sorted(k for k, v in result.items()
                             if k != '_summary' and not v['quantified']),
            significant=sig,
            n_significant=len(sig),
            # Detection claim: >= 2 independent bands at >= 3 sigma. A single
            # significant band is recorded as a caution, not a detection --
            # one band is exactly what a wrong far-IR crossmatch or a noisy
            # W4 produces (GJ 1214 showed a 48x PACS-100 "excess" this way).
            # excess_class stays the evidence verdict over ALL quantified
            # bands (characterization is reported separately from detection).
            has_excess=bool(len(sig) >= 2),
            single_band_caution=(sig[0] if len(sig) == 1 else None),
            excess_chi2=chi2,
            excess_class=ms.get('excess_class'),
            p_none=ms.get('p_none'),
            p_disk=ms.get('p_disk'),
            p_companion=ms.get('p_companion'),
            lnbf_disk_companion=ms.get('lnbf_disk_companion'),
            disk=ms.get('disk'),
            disk2=ms.get('disk2'),
            p_disk2=ms.get('p_disk2'),
            disk_bb=disk_bb,
            # Explicit degradation marker: without a physical distance +
            # luminosity (norm mode) the excess is detected per band but not
            # characterizable -- say so instead of silently returning Nones.
            degraded=(None if (getattr(star, 'dist', None) and bf.get('lum'))
                      else 'no physical distance/luminosity (norm mode): '
                           'per-band significances only; disk L_dust/L_* and '
                           'companion BMA unavailable'),
            companion=ms.get('companion'),
            common_reach_um=(float(np.nanmax([v['wave'] for v in q.values()]))
                             if q else float('nan')))
        return result

    def save_bma(self, avgd):
        """Save BMA output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ : The global evidences.
        posterior_samples : A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed : An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit : The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star : The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)

        Also creates a log file with the best fit parameters, 1 sigma
        error bars and 3 sigma CIs.

        """
        out = dict()
        logdat_samples = '#Parameter\tmedian\tupper\tlower\t3sig_low\t3sig_up\n'
        logdat_average = '#Parameter\tmedian\tupper\tlower\t3sig_low\t3sig_up\n'
        log_out_samples = f'{self.out_folder}/best_fit_sample.dat'
        log_out_average = f'{self.out_folder}/best_fit_average.dat'
        prob_out = f'{self.out_folder}/model_probabilities.dat'
        synth_out = f'{self.out_folder}/synthetic_fluxes.dat'

        # Save global evidence of each model.
        out['lnZ'] = avgd['evidences']

        # Save original samples.
        out['originals'] = avgd['originals']

        # Save weights.
        out['weights'] = avgd['weights']

        # Create raw samples holder
        out['weighted_samples'] = dict()
        out['weighted_average'] = dict()
        j = 0
        for i, par in enumerate(order):
            if not self.coordinator[i]:
                out['weighted_samples'][par] = avgd['weighted_samples'][par]
                out['weighted_average'][par] = avgd['weighted_average'][par]
                j += 1
            else:
                out['weighted_samples'][par] = self.fixed[i]
                out['weighted_average'][par] = self.fixed[i]
        # Carry over the binary per-band dilution traces (not in `order`).
        for par in avgd['weighted_samples']:
            if par.startswith('dilution_'):
                out['weighted_samples'][par] = avgd['weighted_samples'][par]
                out['weighted_average'][par] = avgd['weighted_average'][par]

        # If normalization constant was fitted, derive the radius from the
        # distance; with no distance there is no radius (NaN -> mass/lum NaN).
        if use_norm:
            if star.dist != -1:
                rad_sampled = self._get_rad(
                    out['weighted_samples']['norm'], star.dist, star.dist_e
                )
                rad_averageed = self._get_rad(
                    out['weighted_average']['norm'], star.dist, star.dist_e
                )
            else:
                rad_sampled = np.full_like(
                    out['weighted_samples']['norm'], np.nan)
                rad_averageed = np.nan
            out['weighted_samples']['rad'] = rad_sampled
            out['weighted_average']['rad'] = rad_averageed

        # Create a distribution of masses

        logg_samp = out['weighted_samples']['logg']
        logg_average = out['weighted_average']['logg']
        rad_samp = out['weighted_samples']['rad']
        rad_average = out['weighted_average']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        mass_average = self._get_mass(logg_average, rad_average)
        out['weighted_samples']['grav_mass'] = mass_samp
        out['weighted_average']['grav_mass'] = mass_average

        # Create a distribution of luminosities.

        teff_samp = out['weighted_samples']['teff']
        teff_average = out['weighted_average']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        lum_average = self._get_lum(teff_average, rad_average)
        out['weighted_samples']['lum'] = lum_samp
        out['weighted_average']['lum'] = lum_average

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['weighted_samples']['dist']
            dist_average = out['weighted_average']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            ad_average = self._get_angular_diameter(rad_average, dist_average)
            out['weighted_samples']['AD'] = ad_samp
            out['weighted_average']['AD'] = ad_average

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.

        out['best_fit_samples'] = dict()
        out['uncertainties_samples'] = dict()
        out['confidence_interval_samples'] = dict()
        out['best_fit_averaged'] = dict()
        out['uncertainties_averaged'] = dict()
        out['confidence_interval_averaged'] = dict()
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if 'noise' in param:
                    continue
                samp = out['weighted_samples'][param]
                sampw = out['weighted_average'][param]

                if param == 'z':
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                '[Fe/H]', out, method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                '[Fe/H]', out,
                                                method='averaged')
                elif param == 'norm':
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                '(R/D)^2', out, fmt='e',
                                                method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                '(R/D)^2', out, fmt='e',
                                                method='averaged')
                    if star.dist != 1:
                        logdat_samples = out_filler(
                            out['weighted_samples']['rad'], logdat_samples,
                            'rad', 'R', out, method='samples'
                        )
                        logdat_average = out_filler(
                            out['weighted_average']['rad'], logdat_average,
                            'rad', 'R', out, method='averaged'
                        )

                else:
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                param, out, method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                param, out, method='averaged')
            else:
                logdat_samples = out_filler(
                    0, logdat_samples, param, param, out, fixed=self.fixed[i],
                    method='samples'
                )
                logdat_average = out_filler(
                    0, logdat_average, param, param, out, fixed=self.fixed[i],
                    method='averaged'
                )

        # Add derived mass to best fit dictionary.

        samp = out['weighted_samples']['grav_mass']
        sampw = out['weighted_average']['grav_mass']
        logdat_samples = out_filler(samp, logdat_samples, 'grav_mass',
                                    'grav_mass', out, method='samples')
        logdat_average = out_filler(sampw, logdat_average, 'grav_mass',
                                    'grav_mass', out, method='averaged')

        # Add derived luminosity to best fit dictionary.

        samp = out['weighted_samples']['lum']
        sampw = out['weighted_average']['lum']
        logdat_samples = out_filler(samp, logdat_samples, 'lum', 'lum', out,
                                    method='samples')
        logdat_average = out_filler(sampw, logdat_average, 'lum', 'lum', out,
                                    method='averaged')

        # Add derived angular diameter to best fit dictionary.

        if not use_norm:
            samp = out['weighted_samples']['AD']
            sampw = out['weighted_average']['AD']
            logdat_samples = out_filler(samp, logdat_samples, 'AD', 'AD', out,
                                        method='samples')
            logdat_average = out_filler(sampw, logdat_average, 'AD', 'AD', out,
                                        method='averaged')

        # Add estimated age to best fit dictionary. This is done with the wider
        # sampled distribution instead of the averaged one in order to save time

        if self.estimate_age and estimate is None:
            print(colored('\t\t*** estimate_age skipped: the optional '
                          '`isochrones` dependency is not installed. ***',
                          'yellow'))
        if self.estimate_age and estimate is not None:
            age_samp, mass_samp, eep_samp = self._run_estimate_age(
                out['best_fit_samples'],
                out['uncertainties_samples'],
                c=choice(self.colors)
            )
            # Create new thingy for MIST samples. Sadly now everything done before
            # this update will be incompatible :(
            out['mist_samples'] = dict()
            out['mist_samples']['age'] = age_samp
            out['mist_samples']['iso_mass'] = mass_samp
            out['mist_samples']['eep'] = eep_samp
            logdat_samples = out_filler(age_samp, logdat_samples, 'age', 'age', out,
                                        method='samples')
            logdat_samples = out_filler(mass_samp, logdat_samples, 'iso_mass',
                                        'iso_mass', out, method='samples')
            logdat_samples = out_filler(eep_samp, logdat_samples, 'eep', 'eep', out,
                                        method='samples')
            # Ugly... but faster than doing the kde 2x times...
            age = out['best_fit_samples']['age']
            age_unc = out['uncertainties_samples']['age']
            age_ci = out['confidence_interval_samples']['age']
            iso_mass = out['best_fit_samples']['iso_mass']
            iso_mass_unc = out['uncertainties_samples']['iso_mass']
            iso_mass_ci = out['confidence_interval_samples']['iso_mass']
            eep = out['best_fit_samples']['eep']
            eep_unc = out['uncertainties_samples']['eep']
            eep_ci = out['confidence_interval_samples']['eep']
            out['best_fit_averaged']['age'] = age
            out['best_fit_averaged']['iso_mass'] = iso_mass
            out['best_fit_averaged']['eep'] = eep
            out['uncertainties_averaged']['age'] = age_unc
            out['uncertainties_averaged']['iso_mass'] = iso_mass_unc
            out['uncertainties_averaged']['eep'] = eep_unc
            out['confidence_interval_averaged']['age'] = age_ci
            out['confidence_interval_averaged']['iso_mass'] = iso_mass_ci
            out['confidence_interval_averaged']['eep'] = eep_ci
            logdat_average += f'age\t{age:.4f}\t'
            logdat_average += f'{age_unc[1]:.4f}\t{age_unc[0]:.4f}\t'
            logdat_average += f'{age_ci[0]:.4f}\t{age_ci[1]}\n'
            logdat_average += f'iso_mas\t{iso_mass:.4f}\t'
            logdat_average += f'{iso_mass_unc[1]:.4f}\t{iso_mass_unc[0]:.4f}\t'
            logdat_average += f'{iso_mass_ci[0]:.4f}\t{iso_mass_ci[1]}\n'
            logdat_average += f'eep\t{eep:.4f}\t'
            logdat_average += f'{eep_unc[1]:.4f}\t{eep_unc[0]:.4f}\t'
            logdat_average += f'{eep_ci[0]:.4f}\t{eep_ci[1]}\n'
        else:
            out['mist_samples'] = {}
        ###
        probdat = ''

        max_prob = 0
        max_prob_mod = ''
        for k in avgd['weights'].keys():
            if avgd['weights'][k] > max_prob:
                max_prob = avgd['weights'][k]
                max_prob_mod = k
            probdat += f'{k}_probability\t{avgd["weights"][k]:.4f}\n'

        # Get synthetic mag and fluxes for highest probability model.
        synth_mod = 'btsettl'
        intp = self.load_interpolator(synth_mod)
        # Broadcast to a common length: a pinned axis (e.g. Koester's single
        # [Fe/H] node, or a fixed parameter) is stored as a scalar, which would
        # otherwise break len()/zip over the posterior samples.
        ogteff = np.atleast_1d(avgd['originals'][max_prob_mod]['teff'])
        oglogg = np.atleast_1d(avgd['originals'][max_prob_mod]['logg'])
        ogfeh = np.atleast_1d(avgd['originals'][max_prob_mod]['z'])
        n = max(ogteff.size, oglogg.size, ogfeh.size)
        ogteff = np.broadcast_to(ogteff, (n,))
        oglogg = np.broadcast_to(oglogg, (n,))
        ogfeh = np.broadcast_to(ogfeh, (n,))
        fluxes = np.zeros((len(oglogg), len(filter_names)))
        for i, t, g, z in zip(range(len(oglogg)), ogteff, oglogg, ogfeh):
            fluxes[i, :] = get_interpolated_flux(t, g, z, filter_names, intp)
        synthdat = 'Filter\tFlux\n'
        for i, filt in enumerate(filter_names):
            samp = fluxes[:, i]
            # xx, pdf = estimate_pdf(samp)
            # cdf = estimate_cdf(samp, hdr=True)
            b, _, _ = credibility_interval(samp, alpha=1)
            synthdat += f'{filt}\t{b:.6e}\n'

        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if 'noise' not in param:
                    continue
                samp = out['weighted_samples'][param]
                sampw = out['weighted_average'][param]
                logdat_samples = out_filler(samp, logdat_samples, param, param,
                                            out, fmt='f', method='samples')
                logdat_average = out_filler(sampw, logdat_average, param, param,
                                            out, fmt='f', method='averaged')

        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['norm'] = self.norm
        out['engine'] = 'Bayesian Model Averaging'
        out['av_law'] = av_law
        out['binary'] = self.binary

        # Spectral type

        # Load Mamajek spt table
        mamajek_spt, mamajek_temp = np.loadtxt(f'{filesdir}/mamajek_spt.dat',
                                               dtype=str, usecols=[0, 1],
                                               unpack=True)

        mamajek_temp = mamajek_temp.astype(float)

        # Find spt
        spt_idx = np.argmin(
            abs(mamajek_temp - out['best_fit_averaged']['teff']))
        spt = mamajek_spt[spt_idx]
        out['spectral_type'] = spt
        out_file = f'{self.out_folder}/BMA.pkl'
        with open(log_out_samples, 'w') as logfile:
            logfile.write(logdat_samples)
        with open(log_out_average, 'w') as logfile:
            logfile.write(logdat_average)
        with open(prob_out, 'w') as logfile:
            logfile.write(probdat)
        with open(synth_out, 'w') as logfile:
            logfile.write(synthdat)
        try:
            out['irx'] = (self._compute_irx(out, list(self._grids), av_law)
                          if getattr(self, '_do_irx', False) else {})
        except Exception as e:                       # never break the save
            logger.warning('IR-excess quantification skipped: %s', e)
            out['irx'] = {}
        pickle.dump(out, open(out_file, 'wb'))
        self.out = out
        # NetCDF is an optional ecosystem-interop export (needs arviz). Don't
        # let a missing optional dependency or an export hiccup discard the
        # primary BMA.pkl result that was just written.
        nc_file = f'{self.out_folder}/ariadne_result.nc'
        try:
            self.to_netcdf(nc_file)
        except Exception as e:
            logger.warning('NetCDF export skipped (%s); BMA.pkl is written.', e)

    @staticmethod
    def bayesian_model_average(outputs, grids, norm, nsamples, c='white'):
        """Perform Bayesian Model Averaging.

        Parameters
        ----------
        outputs: array_like
            An array or list of output pickle files from the individual model
            fits.
        grids: array_like
            An array or list of the model grids used to model.
        norm: bool
            Flag indicating if the normalization factor was fit for.
        nsamples: int
            The number of samples to sample from the averaged distribution.
        c: str, optional
            Termcolor color.

        """
        evidences = []
        post_samples = []
        model_posteriors = []
        # Read and extract model posterior information.
        for o in outputs:
            with open(o, 'rb') as f:
                model_posteriors.append(pickle.load(f))
        # Extract the evidences of each model.
        for o in model_posteriors:
            evidences.append(o['global_lnZ'])
            post_samples.append(o['posterior_samples'])
        # Resolve the sample count. ``n_samples`` defaults to the sentinel
        # 'max' (set in ``initialize`` when the user leaves it unset); interpret
        # it as the largest per-model posterior length. Without this, 'max'
        # reaches ``np.zeros``/``choice`` below and raises "'str' object cannot
        # be interpreted as an integer" — so a BMA run used to crash at the
        # averaging step unless n_samples was set explicitly.
        if not isinstance(nsamples, (int, np.integer)):
            nsamples = max(len(v) for o in post_samples for v in o.values()
                           if hasattr(v, '__len__'))
        # Per-model fit-health diagnostics + evidence errors (additive).
        per_model_diag = {}
        evidence_err = {}
        for o in model_posteriors:
            g = o['model_grid']
            if 'diagnostics' in o:
                per_model_diag[g] = o['diagnostics']
            evidence_err[g] = o.get('global_lnZerr', float('nan'))
        # Convert evidences to weights/probabilities via a softmax of the
        # log-evidences. Shift by the MAX (not the min) so the largest exponent
        # is 0: softmax is shift-invariant, so the weights are unchanged in the
        # normal regime, but shifting by the min lets exp() overflow to inf ->
        # nan when one grid's log-evidence dominates (delta lnZ > ~709).
        evidences = np.array(evidences)
        weights = evidences - evidences.max()
        weights = [np.exp(e) / np.exp(weights).sum() for e in weights]
        weights = np.array(weights)
        # We're not averaging these
        ban = ['loglike', 'priors', 'posteriors']
        if norm:  # if normalization was used, we won't average the radius
            ban.append('rad')
        # Create an output dictionary for the averaged samples.
        out = dict()
        out['originals'] = dict()
        out['weights'] = dict()
        # Populate the dict with the probabilities and individual model
        # posteriors
        for i, o in enumerate(model_posteriors):
            out['weights'][o['model_grid']] = weights[i]
            out['originals'][o['model_grid']] = o['posterior_samples']
        out['weighted_samples'] = dict()
        out['weighted_average'] = dict()

        print(colored('\t\t*** AVERAGING POSTERIOR SAMPLES ***', c))
        for k in tqdm(post_samples[0].keys()):
            if k in ban:  # Skip things that are not main parameters.
                continue
            try:  # Skip fixed params.
                len(post_samples[0][k])
            except TypeError:
                continue
            # A normalization fit with no distance has no radius/mass/luminosity
            # (all-NaN samples); a KDE can't be built on them, so carry the NaN
            # through rather than crash.
            if np.all(np.isnan(np.asarray(post_samples[0][k], dtype=float))):
                out['weighted_samples'][k] = np.full(nsamples, np.nan)
                out['weighted_average'][k] = np.full(nsamples, np.nan)
                continue
            traces = []
            extended_weights = []
            out['weighted_samples'][k] = np.zeros(nsamples)
            out['weighted_average'][k] = np.zeros(nsamples)
            for i, o in enumerate(post_samples):
                # This is for the weighted sampling.
                traces.append(o[k])
                extended_weights.append(np.ones(len(o[k])) * weights[i])
                # This is for the weighted averaging.
                # The weighted averaging consists of taking the weighted
                # average of the posterior samples, supersampled to maked them
                # coincide in length.
                weighted_samples = choice(o[k], nsamples) * weights[i]
                out['weighted_average'][k] += weighted_samples
            # Do the weighted sampling. For this we're going to estimate the
            # KDE of the 'master' posterior built by taking random samples
            # from each model where the number of samples is proportional
            # to the model's relative probability. This is equivalent
            # to taking the KDE of each model and then do the weighted average
            avg_kde = st.gaussian_kde(np.concatenate(traces),
                                   weights=np.concatenate(extended_weights))
            out['weighted_samples'][k] = avg_kde.resample(nsamples)[0]
        # Now we save the evidences.
        out['evidences'] = dict()
        for e, g in zip(evidences, grids):
            out['evidences'][g] = e
        out['evidences_err'] = evidence_err
        out['per_model_diagnostics'] = per_model_diag
        # Surface per-grid fit health to the user.
        if per_model_diag:
            print(colored('\t\t*** PER-GRID FIT HEALTH ***', c))
            for g in grids:
                if g in per_model_diag:
                    Fitter._print_fit_health(per_model_diag[g], g)
        return out

    @staticmethod
    def dynesty_results(results):
        """Extract posterior samples, global evidence and its error."""
        weights = np.exp(results['logwt'] - results['logz'][-1])
        posterior_samples = resample_equal(results.samples, weights)
        lnz = results.logz[-1]
        lnzer = results.logzerr[-1]
        return lnz, lnzer, posterior_samples

    @staticmethod
    def _fit_health(results, param_samples=None, grid_axes=None,
                    edge_frac=0.02):
        """Compute additive fit-health diagnostics for a single grid fit.

        Purely diagnostic — it never alters the fit. Returns a dict with:

        - ``logz`` / ``logzerr`` : the grid's log-evidence and its error.
        - ``eff_samples`` : Kish effective sample size of the importance
          weights (a low value flags an under-sampled / unreliable posterior).
        - ``dlogz_final`` : the last log-evidence increment (a convergence
          proxy; small means the run had stopped accumulating evidence).
        - ``edge_flags`` : a list of teff/logg/z parameters whose posterior
          median sits within ``edge_frac`` of a grid boundary, i.e. the fit is
          pegged against the edge of the model grid and the (tight) error bar
          is not trustworthy. Empty when ``grid_axes`` is unavailable.

        Parameters
        ----------
        results : dynesty results
            The sampler results object.
        param_samples : dict, optional
            ``{param_name: samples}`` (e.g. the saved ``posterior_samples``),
            used for the grid-edge check.
        grid_axes : dict, optional
            ``{'teff': arr, 'logg': arr, 'z': arr}`` of grid node values.
        edge_frac : float
            Fraction of the grid span that counts as "at the edge".
        """
        diag = {}
        try:
            diag['logz'] = float(results.logz[-1])
            diag['logzerr'] = float(results.logzerr[-1])
        except Exception:
            diag['logz'] = float('nan')
            diag['logzerr'] = float('nan')
        try:
            logwt = np.asarray(results['logwt'])
            w = np.exp(logwt - results.logz[-1])
            w = w / w.sum()
            diag['eff_samples'] = int(round(1.0 / np.sum(w ** 2)))
        except Exception:
            diag['eff_samples'] = None
        try:
            diag['dlogz_final'] = float(results.logz[-1] - results.logz[-2])
        except Exception:
            diag['dlogz_final'] = None

        flags = []
        if param_samples is not None and grid_axes is not None:
            for p, axis in grid_axes.items():
                s = param_samples.get(p)
                if s is None:
                    continue
                s = np.atleast_1d(np.asarray(s, dtype=float))
                if s.size < 2:          # fixed parameter, nothing to flag
                    continue
                axis = np.asarray(axis, dtype=float)
                lo, hi = float(axis.min()), float(axis.max())
                span = hi - lo
                if span <= 0:
                    continue
                med = float(np.median(s))
                if med <= lo + edge_frac * span:
                    flags.append({'param': p, 'edge': 'lower',
                                  'median': med, 'bound': lo})
                elif med >= hi - edge_frac * span:
                    flags.append({'param': p, 'edge': 'upper',
                                  'median': med, 'bound': hi})
        diag['edge_flags'] = flags
        return diag

    @staticmethod
    def _print_fit_health(diag, grid_name=None):
        """Print a one-line fit-health summary plus any grid-edge warnings."""
        tag = f' — {grid_name}' if grid_name else ''
        lnz = diag.get('logz', float('nan'))
        lnzerr = diag.get('logzerr', float('nan'))
        neff = diag.get('eff_samples')
        neff_s = 'n/a' if neff is None else str(neff)
        print(colored(
            f'\t\t\tFit health{tag}: log Z = {lnz:.3f} +/- {lnzerr:.3f} | '
            f'eff. samples {neff_s}', 'cyan'))
        for f in diag.get('edge_flags', []):
            print(colored(
                f"\t\t\t  WARNING: {f['param']} posterior pegged at the "
                f"{f['edge']} grid edge (median {f['median']:.4g}, "
                f"bound {f['bound']:.4g}) — error bar may be unreliable.",
                'yellow'))

    @staticmethod
    def _get_mass(logg, rad):
        """Calculate mass from logg and radius."""
        # Solar logg = 4.437
        # g = g_Sol * M / R**2
        mass = logg + 2 * np.log10(rad) - 4.437
        mass = 10 ** mass
        return mass

    @staticmethod
    def _get_lum(teff, rad):
        sb = sigma_sb.to(u.solLum / u.K ** 4 / u.solRad ** 2).value
        L = 4 * np.pi * rad ** 2 * sb * teff ** 4
        return L

    @staticmethod
    def _get_rad(samples, dist, dist_e):
        """Calculate radius from the normalization constant and distance."""
        norm = samples
        # Create a synthetic distribution for distance.
        # N = (R / D) ** 2
        b, a = (np.inf - dist) / dist_e, (0 - dist) / dist_e
        d = st.truncnorm(loc=dist, scale=dist_e, a=a, b=b).rvs(
            size=norm.shape[0])
        n = np.sqrt(norm)
        r = n * d  # This is in pc
        r *= u.pc.to(u.solRad)  # Transform to Solar radii
        return r

    @staticmethod
    def _get_angular_diameter(rad, dist):
        diameter = 2 * rad
        ad = (diameter / (dist * u.pc.to(u.solRad))) * u.rad.to(u.marcsec)
        return ad

    @staticmethod
    def load_interpolator(model):
        """Load a grid interpolator by name."""
        return load_grid(model)

    def _run_estimate_age(self, bf, unc, c='white'):
        """Estimate age using MIST isochrones.

        Parameters
        ----------
        bf: dict
            A dictionary with the best fit parameters.
        unc: dict
            A dictionary with the uncertainties.

        """
        print(
            colored(
                '\t\t*** ESTIMATING AGE AND MASS USING MIST ISOCHRONES ***', c
            )
        )
        params = dict()  # params for isochrones.
        for i, k in enumerate(order):
            if k == 'logg' or 'noise' in k:
                continue
            if k == 'teff':
                par = 'Teff'
            if k == 'z':
                par = 'feh'
            if k == 'dist':
                par = 'distance'
            if k == 'rad':
                par = 'radius'
            if k == 'Av':
                par = 'AV'
            if not self.coordinator[i]:
                if k != 'norm':
                    params[par] = (bf[k], max(unc[k]))
                if k == 'norm' and self.star.dist != -1:
                    params['distance'] = (self.star.dist, self.star.dist_e)
                if par == 'distance':
                    err = max(unc[k])
                    params['parallax'] = (1000 / bf[k], 1000 * err / bf[k] ** 2)
            else:
                continue

        params['mass'] = (bf['grav_mass'], max(unc['grav_mass']))
        if self.star.lum != 0 and self.star.lum_e != 0:
            params['logL'] = (np.log10(bf['lum']),
                              abs(np.log10(max(unc['lum']))))
        mags = self.star.mags[iso_mask == 1]
        mags_e = self.star.mag_errs[iso_mask == 1]
        used_bands = []
        for m, e, b in zip(mags, mags_e, iso_bands):
            if m != 0:
                params[b] = (m, e)
                used_bands.append(b)

        age_samp, mass_samp, eep_samp = estimate(
            used_bands, params, logg=False,
            out_folder=self.out_folder,
            threads=self._threads, dlogz=self.isochrone_dlogz)

        return age_samp, mass_samp, eep_samp

    def to_dict(self):
        """Return BMA results as a structured dict matching the ecosystem spec.

        Returns a dictionary with keys ``posterior``, ``observed_data``, and
        ``sample_stats``, each containing numpy arrays ready for conversion to
        ``arviz.InferenceData``.

        Raises
        ------
        RuntimeError
            If the fitter has not been run yet (``self.out`` is not populated).
        """
        if not hasattr(self, 'out') or self.out is None:
            raise RuntimeError(
                'No results available. Run the fitter before calling to_dict().'
            )
        out = self.out
        ws = out.get('weighted_samples', {})

        # Determine number of draws from any free parameter
        n_draws = None
        for v in ws.values():
            if hasattr(v, '__len__') and len(v) > 1:
                n_draws = len(v)
                break
        if n_draws is None:
            n_draws = 100_000  # fallback

        def _to_chain(arr_or_scalar):
            """Reshape to (1, n_draws) for single-chain arviz convention."""
            if hasattr(arr_or_scalar, '__len__') and len(arr_or_scalar) > 1:
                return np.array(arr_or_scalar).reshape(1, -1)
            return np.full((1, n_draws), float(arr_or_scalar))

        # Map internal names → ecosystem spec names
        param_map = {
            'teff': 'Teff',
            'logg': 'logg',
            'z': 'feh',
            'rad': 'radius',
            'lum': 'luminosity',
            'dist': 'distance',
            'Av': 'Av',
        }

        posterior = {}
        for internal, spec in param_map.items():
            if internal in ws:
                posterior[spec] = _to_chain(ws[internal])

        # Observed photometry
        mask = self.star.filter_mask
        observed_data = {
            'wavelength': self.star.wave[mask],  # already in micron
            'flux': self.star.flux[mask],
            'flux_err': self.star.flux_er[mask],
        }

        # Model evidence and weights
        weights = out.get('weights', {})
        lnZ = out.get('lnZ', {})

        model_names = list(weights.keys())
        weight_vals = np.array([weights[m] for m in model_names])
        lnZ_vals = np.array([lnZ.get(m, 0.0) for m in model_names])
        bma_log_evidence = float(np.sum(weight_vals * lnZ_vals))

        sample_stats = {
            'log_evidence': bma_log_evidence,
            'model_names': model_names,
            'model_weights': weight_vals,
        }

        # Per-model posteriors from originals
        originals = out.get('originals', {})
        model_posteriors = {}
        for model_name, model_samples in originals.items():
            n_model = None
            for v in model_samples.values():
                if hasattr(v, '__len__') and len(v) > 1:
                    n_model = len(v)
                    break
            if n_model is None:
                continue

            def _to_model_chain(a, nd=n_model):
                if hasattr(a, '__len__') and len(a) > 1:
                    return np.array(a).reshape(1, -1)
                return np.full((1, nd), float(a))

            mp = {}
            for internal, spec in param_map.items():
                if internal in model_samples:
                    mp[spec] = _to_model_chain(model_samples[internal])
            if 'loglike' in model_samples:
                mp['log_likelihood'] = _to_model_chain(
                    model_samples['loglike'])
            model_posteriors[model_name] = mp

        # MIST isochrone samples
        mist = out.get('mist_samples', {})
        mist_posterior = {}
        for k in ('age', 'iso_mass', 'eep'):
            if k in mist and hasattr(mist[k], '__len__') and len(mist[k]) > 1:
                mist_posterior[k] = _to_chain(mist[k])

        # Filter names and bandwidths for SED reconstruction
        observed_data['filter_names'] = np.array(
            self.star.filter_names[mask].tolist()
            if hasattr(self.star.filter_names[mask], 'tolist')
            else list(self.star.filter_names[mask])
        )
        observed_data['bandwidths'] = self.star.bandpass[mask]

        # Model SED fluxes (injected by worker from SEDPlotter)
        model_sed = {}
        if 'model_sed' in out:
            ms = out['model_sed']
            if 'model_flux' in ms:
                model_sed['model_flux'] = np.array(ms['model_flux'])

        # Summary statistics for quick display
        summary = {}
        for src_key in ('best_fit_averaged', 'uncertainties_averaged',
                        'confidence_interval_averaged'):
            if src_key in out:
                summary[src_key] = {}
                for k, v in out[src_key].items():
                    if hasattr(v, '__len__'):
                        summary[src_key][k] = np.array(
                            [float(x) for x in v])
                    else:
                        summary[src_key][k] = float(v)

        return {
            'posterior': posterior,
            'observed_data': observed_data,
            'sample_stats': sample_stats,
            'model_posteriors': model_posteriors,
            'mist_posterior': mist_posterior,
            'model_sed': model_sed,
            'summary': summary,
        }

    def to_netcdf(self, path):
        """Export the full BMA result to a portable NetCDF (``.nc``) file.

        Written with ``scipy.io`` only -- no arviz/xarray/netCDF4 dependency.
        Numeric arrays (posteriors, weighted samples, SED fluxes, evidences)
        become NetCDF variables; the nested result structure, scalars, strings
        and the IR-excess classification ride in a single JSON ``ariadne_result``
        attribute. The file round-trips losslessly through
        :meth:`Fitter.read_netcdf` and is a drop-in, version-independent
        replacement for ``BMA.pkl``.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``'ariadne_result.nc'``).
        """
        out = getattr(self, 'out', None)
        if out is None:
            raise RuntimeError('No result to export; run the fit first.')
        _result_dict_to_nc(out, path)
        logger.info(f'Wrote BMA result to {path}')

    @staticmethod
    def read_netcdf(path):
        """Read a result written by :meth:`to_netcdf` back into a dict.

        Inverse of :meth:`to_netcdf`; returns the same structure that was
        pickled to ``BMA.pkl``, with the numeric arrays restored from the
        NetCDF variables.
        """
        return _result_nc_to_dict(path)


#####################
# Dynesty wrappers


# Set by fit_bma before forking the grid-parallel Pool; read by the workers.
_BMA_FITTER = None


def _bma_grid_worker(index):
    """Fit one BMA model grid single-threaded and write its ``_out.pkl``.

    Runs in a forked worker process: the star photometry, priors, coordinator
    and other module globals are inherited from the parent, so only the
    per-grid interpolator is swapped in. The fit result is delivered via the
    on-disk pickle (the parent reads it back for averaging), so nothing needs
    to be pickled back to the parent. Returns ``(grid, None)`` on success or
    ``(grid, error)`` on failure.
    """
    global interpolator
    fitter = _BMA_FITTER
    intp = fitter._interpolators[index]
    grid = fitter._grids[index]
    out_file = f'{fitter.out_folder}/{grid}_out.pkl'
    # Forked processes inherit the parent's RNG state; reseed so each grid
    # draws independently.
    np.random.seed()
    interpolator = intp
    fitter._threads = 1   # grids are the unit of parallelism; no inner Pool
    fitter._grid = grid   # used by save() for out['model_grid']
    try:
        fitter.fit_dynesty(out_file=out_file)
    except ValueError as e:
        dump_out = f'{fitter.out_folder}/{grid}_DUMP.pkl'
        pickle.dump(fitter.sampler.results, open(dump_out, 'wb'))
        return grid, e
    return grid, None


def dynesty_loglike_bma(cube, interpolator):
    """Dynesty log likelihood wrapper for BMA."""
    theta = build_params(cube, coordinator, fixed, use_norm)
    return log_likelihood(theta, star, interpolator, use_norm, av_law)


def dynesty_log_like(cube):
    """Dynesty log likelihood wrapper."""
    theta = build_params(
        cube, flux, flux_er, filts, coordinator, fixed, use_norm
    )
    return log_likelihood(theta, flux, flux_er, wave,
                          filts, interpolator, use_norm, av_law,
                          binary=binary, contrast=contrast)


def pt_dynesty(cube):
    """Dynesty prior transform."""
    return prior_transform_dynesty(cube, flux, flux_er, filts, prior_dict,
                                   coordinator, use_norm, binary=binary,
                                   shared=jitter_shared)
