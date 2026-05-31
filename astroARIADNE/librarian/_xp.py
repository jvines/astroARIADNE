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

# erg/s/cm^2/um  <-  W/nm/m^2  (XP calibrated flux -> ARIADNE grid-column units).
#   W -> erg/s : 1e7   |   /m^2 -> /cm^2 : 1e-4   |   /nm -> /um : *1e3
_WNM_M2_TO_ERG_S_CM2_UM = 1e7 * 1e-4 * 1e3

_FILTER_CACHE = None


def _build_pseudo_filters():
    """Build (and cache) pyphot UnitFilter objects for the pseudo-bands.

    Built as ``UnitFilter`` with a unit-quantity wavelength in micron — the same
    class and unit convention as ``pyphot.get_library()`` filters and the
    erg/cm^2/s/um flux ARIADNE feeds. (A plain ``Filter`` with a ``unit='um'``
    *string* mis-scales the photon-counting integral by ~1e4 — verified against a
    library filter — so the Quantity form is required.) Gaussian transmission,
    photon-counting.
    """
    global _FILTER_CACHE
    if _FILTER_CACHE is not None:
        return _FILTER_CACHE
    from pyphot.sandbox import UnitFilter
    from pyphot.ezunits import unit
    out = []
    for name, c_nm, fwhm_nm in XP_PSEUDO_BANDS:
        sig = (fwhm_nm / 2.3548200450309493) * 1e-3   # nm FWHM -> um sigma
        c = c_nm * 1e-3                               # nm -> um
        wl = np.linspace(c - 5 * sig, c + 5 * sig, 400)
        trans = np.exp(-0.5 * ((wl - c) / sig) ** 2)
        out.append((name, UnitFilter(unit.Quantity(wl, 'um'), trans,
                                     name=name, dtype='photon')))
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

    Returns ``(names, fluxes, flux_errors)`` in erg/s/cm^2/um (matching the SED
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
    wl_um = np.asarray(sampling, float) * 1e-3                  # nm -> um
    F = np.asarray(spec['flux'].iloc[0], float) * _WNM_M2_TO_ERG_S_CM2_UM
    Ferr = np.asarray(spec['flux_error'].iloc[0], float) * _WNM_M2_TO_ERG_S_CM2_UM
    wq = unit.Quantity(wl_um, 'um')
    fq = unit.Quantity(F, 'erg/cm**2/s/um')
    # error: integrate (F+sigma) through each band and difference, a cheap
    # independent-sample approximation of the band-flux uncertainty.
    feq = unit.Quantity(F + Ferr, 'erg/cm**2/s/um')

    # Mirror the grid builder exactly: get_flux(...).to('erg/cm**2/s/um'). The
    # ``.to()`` is essential — pyphot's raw get_flux normalises to its internal
    # wavelength unit (Angstrom), a uniform 1e4 offset; ``.to('...um')`` puts the
    # synthetic flux on the SAME scale as the grid columns.
    def _val(r):
        return float(r.to('erg/cm**2/s/um').magnitude) if hasattr(r, 'to') \
            else (float(r.magnitude) if hasattr(r, 'magnitude') else float(r))

    names, fluxes, errs = [], [], []
    for name, filt in _build_pseudo_filters():
        f0 = _val(filt.get_flux(wq, fq))
        f1 = _val(filt.get_flux(wq, feq))
        names.append(name)
        fluxes.append(f0)
        errs.append(abs(f1 - f0))
    return names, np.array(fluxes), np.array(errs)
