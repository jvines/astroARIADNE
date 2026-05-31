"""Gaia XP (BP/RP) low-resolution spectrophotometry -> ARIADNE pseudo-bands.

ARIADNE fits band-convolved fluxes, not spectra. This module turns a target's
Gaia DR3 XP_CONTINUOUS spectrum into synthetic fluxes through a small set of
feature-targeted pseudo-bandpasses, so XP can enter the existing band-flux
likelihood as additional "filters" (the model grids must be re-convolved through
the same passbands; see notebooks/build_model_grids.py and ``XP_PSEUDO_BANDS``).

Pipeline (all verified against real data):
  1. retrieve XP_CONTINUOUS coefficients + covariance via astroquery
     (``Gaia.load_data(retrieval_type='XP_CONTINUOUS')``);
  2. reconstruct the sampled spectrum with GaiaXPy's ``calibrate`` *fed the
     coefficient DataFrame directly* — GaiaXPy's own re-fetch path 500s on
     anonymous list queries, so we never let it re-download;
  3. integrate the spectrum through the pseudo-bands with pyphot, the same
     machinery the grid builder uses.

Design notes (from validation):
  * Pseudo-bands are >=15-20 nm wide. Narrower windows approach XP's effective
    resolution, where the synthetic flux becomes sensitive to reconstruction
    sampling (a ~0.2% systematic appears at 12 nm). 15-20 nm stays safely above.
  * Bands target logg/[Fe/H]/molecular features (the information XP actually adds
    over broadband) plus continuum anchors, all inside XP's 336-1020 nm span.
  * XP samples are strongly correlated; keep the band count small (~10) and use a
    dedicated XP jitter parameter in the fit. Do NOT add hundreds of fine bands.
"""
import numpy as np

# name, central wavelength [nm], FWHM [nm]. All >=15 nm (resolution-safe) and
# inside XP coverage (336-1020 nm).
XP_PSEUDO_BANDS = [
    ('XP_Gband',    430.0, 20.0),   # CH G-band  -> logg / [Fe/H]
    ('XP_bluecont', 470.0, 30.0),   # blue pseudo-continuum anchor
    ('XP_Mgb',      518.0, 20.0),   # Mg b / MgH -> logg
    ('XP_greencont',570.0, 30.0),   # green continuum anchor
    ('XP_CaH',      638.0, 20.0),   # CaH        -> M-dwarf gravity
    ('XP_TiO705',   705.0, 25.0),   # TiO        -> M-dwarf Teff / [M/H]
    ('XP_redcont',  780.0, 30.0),   # red continuum anchor
    ('XP_TiO843',   843.0, 20.0),   # TiO band edge
    ('XP_CaT',      860.0, 25.0),   # Ca II triplet -> logg
    ('XP_NIRcont',  960.0, 30.0),   # near-IR continuum anchor
]

# erg/s/cm^2/AA  <-  W/nm/m^2  (XP calibrated flux units)
_WNM_M2_TO_ERG_S_CM2_AA = 1e7 / 1e4 / 10.0

_FILTER_CACHE = None


def _build_pseudo_filters():
    """Build (and cache) pyphot Filter objects for the pseudo-bands.

    Constructed in Angstrom to match pyphot's native internal unit; Gaussian
    transmission, photon-counting (``dtype='photon'``), matching the SED grids.
    """
    global _FILTER_CACHE
    if _FILTER_CACHE is not None:
        return _FILTER_CACHE
    from pyphot import Filter
    out = []
    for name, c_nm, fwhm_nm in XP_PSEUDO_BANDS:
        sig_aa = (fwhm_nm / 2.3548200450309493) * 10.0
        c_aa = c_nm * 10.0
        wl = np.linspace(c_aa - 5 * sig_aa, c_aa + 5 * sig_aa, 400)
        trans = np.exp(-0.5 * ((wl - c_aa) / sig_aa) ** 2)
        out.append((name, Filter(wl, trans, name=name, dtype='photon',
                                 unit='Angstrom')))
    _FILTER_CACHE = out
    return out


def fetch_xp_continuous(source_id, data_release='Gaia DR3'):
    """Retrieve XP_CONTINUOUS coefficients for a Gaia source_id.

    Returns the single-row pandas DataFrame of BP/RP coefficients, errors and
    correlations, or ``None`` if the source has no XP spectrum.
    """
    from astroquery.gaia import Gaia
    res = Gaia.load_data(ids=[int(source_id)], retrieval_type='XP_CONTINUOUS',
                         data_release=data_release, format='csv', verbose=False)
    if not res:
        return None
    key = next(iter(res))
    tab = res[key]
    tab = tab[0] if isinstance(tab, list) else tab
    df = tab.to_pandas()
    return df if len(df) else None


def xp_pseudo_fluxes(source_id, data_release='Gaia DR3'):
    """Synthetic fluxes through the XP pseudo-bands for a Gaia source_id.

    Returns ``(names, fluxes, flux_errors)`` in erg/s/cm^2/AA (matching the SED
    grid columns / pyphot zero-point convention), or ``None`` if the source has
    no XP spectrum. ``flux_errors`` are propagated from GaiaXPy's per-sample flux
    errors (treated as independent — an approximation; the full XP covariance is
    available in the coefficient DataFrame for a future upgrade).
    """
    df = fetch_xp_continuous(source_id, data_release=data_release)
    if df is None:
        return None
    import gaiaxpy
    from pyphot.ezunits import unit
    spec, sampling = gaiaxpy.calibrate(df, save_file=False)
    wl_aa = np.asarray(sampling, float) * 10.0
    F = np.asarray(spec['flux'].iloc[0], float) * _WNM_M2_TO_ERG_S_CM2_AA
    Ferr = np.asarray(spec['flux_error'].iloc[0], float) * _WNM_M2_TO_ERG_S_CM2_AA
    wq = unit.Quantity(wl_aa, 'angstrom')
    fq = unit.Quantity(F, 'erg/s/cm**2/angstrom')
    # error: integrate (F+sigma) through each band and difference, a cheap
    # independent-sample approximation of the band-flux uncertainty.
    feq = unit.Quantity(F + Ferr, 'erg/s/cm**2/angstrom')

    def _val(r):
        return float(r.magnitude) if hasattr(r, 'magnitude') else float(r)

    names, fluxes, errs = [], [], []
    for name, filt in _build_pseudo_filters():
        f0 = _val(filt.get_flux(wq, fq))
        f1 = _val(filt.get_flux(wq, feq))
        names.append(name)
        fluxes.append(f0)
        errs.append(abs(f1 - f0))
    return names, np.array(fluxes), np.array(errs)
