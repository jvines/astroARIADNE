# Changelog

All notable changes to astroARIADNE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-05-31

### Added
- Koester DA white-dwarf grid (`grid='koester'`; Koester 2010, via SVO). A
  perfectly rectilinear 82×13 (Teff 5000–80000 K, log g 6.5–9.5) cube with no
  metallicity axis — built with a single pinned `[Fe/H]=0` node that the
  trilinear interpolator handles as a degenerate third axis. Runs standalone
  with white-dwarf priors (uniform Teff/log g, a WD-appropriate radius range
  reaching below the main-sequence 0.05 R_sun floor, `[Fe/H]` fixed), and a 3 µm
  red cutoff (`grid_wave_coverage['koester']`) drops WISE/Spitzer. Lets ARIADNE
  fit isolated white dwarfs / WD components, a regime no other grid reaches.
  Validated on the DA standard GD 71. Build with `notebooks/download_koester.py`
  + `build_model_grids.py koester`.
- PHOENIX-NewEra grid support (`grid='newera'`; Husser et al. 2025), the
  modern-opacity ACES successor (rebuilt line lists, `[M/H]` to −4.0),
  BMA-capable across the full FGKM range. The spectra are too large to ship, so
  the grid is wired but not pre-built: stage and build it with
  `notebooks/download_newera.py` + `build_model_grids.py newera` (the builder
  pins the α=0 slice). The reader is verified against a real NewEra HSR file.
- BMA now ships with a default model ensemble (`phoenix`, `btsettl`,
  `btnextgen`, `btcond`, `kurucz`, `ck04`, `bosz`, `sphinx`), so `Fitter.bma =
  True` no longer requires hand-setting `Fitter.models` — a BMA run with no
  `models` set previously raised. SPHINX is now part of that default, Teff-gated
  to the M-dwarf regime (< 4000 K) by the existing per-grid temperature filter,
  so it contributes to cool-star fits and is silently dropped for warmer stars.
  Coelho (optical-only) and TLUSTY (hot-star) remain standalone-only and must be
  requested explicitly. Listing every grid stays safe because each is gated to
  the temperature range where it is valid.
- Unresolved-binary / SB2 fitting mode (`Fitter.binary = True`). The forward
  model sums two stellar components (`teff2/logg2/rad2` added) sharing distance,
  extinction and metallicity, with `teff1 >= teff2` enforced in the prior
  transform. Requires a flux-ratio constraint via
  `Fitter.add_contrast(filter, delta_mag, sigma)` — the secondary is otherwise
  degenerate, so the mode refuses to run without one (and needs the explicit
  rad/dist parametrisation + a parallax). Reports the per-band host-flux
  fraction `f1/(f1+f2)` — the Furlan & Howell planet-radius dilution / third-
  light correction — in `out['binary_dilution']`, and flags a prior-dominated
  secondary Teff. A single contrast band pins the dilution but leaves teff2
  weakly constrained; two bands of differing colour recover the secondary
  (validated to ~1% on synthetic data). Works single-grid or with BMA (the
  dilution is then grid-systematics-aware). Automatically reports a fair
  *SED-only* single-vs-binary Bayesian evidence comparison in
  `out['binary_vs_single']` — does the photometry alone favour two temperatures,
  independent of the contrast detection (delta_lnZ > 0 favours the binary) — and
  the SED plot overlays both components and their sum. Single-star fits are
  byte-identical (the mode is fully gated).
- Regime-aware priors for M dwarfs. Cool stars (Gaia BP-RP > 1.8, or Teff
  < 4000 K) now get an M/late-K uniform Teff prior instead of the FGK
  population prior, and cool dwarfs (absolute G > 8, i.e. on the main sequence)
  a tight `N(5.0, 0.25)` log g prior. This fixes a systematic in which M dwarfs
  were pulled ~600 K hot into a spurious hot-giant solution (GJ 1214 with
  default priors: log g 1.7 -> 5.0, Teff 3800 -> 2940 K). Detection is
  colour-primary because M dwarfs frequently lack a Gaia Teff/luminosity
  estimate; spectroscopic priors still take precedence.
- `Star.from_gaia(source_id)` and `Star.from_name(name)` classmethods that
  resolve coordinates (Gaia DR3 TAP / Simbad name resolution) and build the
  `Star` — the documented "I have a Gaia ID" entry point no longer requires
  hand-entering RA/Dec.
- Fitzpatrick & Massa (2007) extinction law (`av_law='fm07'`).
- Fit-health diagnostics: each fit now records per-grid log-evidence ± error,
  effective sample size, a convergence proxy, and grid-edge-pegging flags
  (when a teff/logg/[Fe/H] posterior is railed against a grid boundary, so the
  error bar is unreliable). Stored in the output (`diagnostics` /
  `per_model_diagnostics`) and printed after each fit.
- Typo guard on `Fitter` config: assigning an unknown public attribute (e.g.
  `f.bmaa = True` instead of `f.bma`) now raises `AttributeError` instead of
  silently creating a dead attribute and running the wrong kind of fit.
- **IR-excess measurement**, reach-gated across grids. The mid/far-IR bands are
  treated as excess bands (not photosphere-fit bands) and quantified post-fit:
  per band, the model vs observed flux and a significance
  `(obs − model) / √(obs_err² + model_err²)` against the evidence-weighted
  posterior-predictive model. A band is quantified only where *every* in-play
  grid reaches it — single fit → that grid; BMA → the intersection of the
  ensemble's (reliability-capped) reach — and redder bands are shown plot-only.
  `out['irx']['_summary']` aggregates this into catalogue columns: an excess χ²,
  the number of ≥3σ bands, a `has_excess` flag, and — when ≥2 positive bands are
  available — a single-temperature blackbody characterization of the excess
  (`t_dust` and the fractional excess luminosity `l_dust_over_lstar`, the
  debris-disk product). Stored per band in `out['irx']`.
- The atmosphere grids are **reconvolved out to the mid/far-IR**, each to its own
  true spectral reach (BT-Settl/NextGen/Cond ~150 µm, TLUSTY to SPIRE 250 µm,
  CK04/Kurucz 160 µm, SPHINX ~70 µm, BOSZ to MIPS 24). A transmission-area NaN
  guard self-clamps every band to where the spectrum actually covers it, so each
  grid carries the common band set with NaN beyond its reach.
- A **custom-filter library** (`Datafiles/custom_filters.h5`, built from the SVO
  Filter Profile Service by `notebooks/build_custom_filters.py`) supplies the
  mid/far-IR passbands pyphot lacks — Spitzer MIPS 24/70/160, AKARI/IRC S9W &
  L18W, IRAS 12/25/60/100, Herschel SPIRE PLW — in their energy (bolometer)
  convention, loaded alongside the pyphot library by `phot_utils`.
- The librarian now fetches **AllWISE W3 (12 µm) and W4 (22 µm)** and routes them
  to the IR-excess set, so excess is measured automatically for any star with
  AllWISE photometry (`qph` upper-limits rejected, Vega mags like W1/W2).

### Changed
- Bayesian Model Averaging weights: the evidence softmax now shifts by the
  maximum log-evidence instead of the minimum, preventing `exp()` overflow to
  `nan` weights when one grid dominates by a large margin (Δln Z ≳ 709).
- **Model grids are now stored as HDF5 flux cubes** (`<Name>.h5`) instead of
  pandas-pickled DataFrames (`<Name>_DF.pkl`). The interpolator reads the cube
  directly, so the grids no longer depend on the pandas/pickle version they were
  written with and survive numpy/pandas upgrades; pandas is no longer imported
  on the fit path.

### Removed
- **The `'multinest'` sampling engine has been removed** (deprecated in 1.5.0).
  It only ever supported single-grid fits, never BMA; `dynesty` is the sole
  engine and is faster. `engine='multinest'` now raises a clear error, and
  `pymultinest` is no longer imported. **Breaking change** for any code that
  passed `engine='multinest'`.
- The legacy human-readable `model_grid_<Name>.dat` grid tables (~90 MB) are
  gone; `Star.load_grid` now reads the grid axes from the HDF5 cube instead.

### Fixed
- Single-grid fits crashed at the end-of-run summary: `end()` read the output
  back as `{engine}_out.pkl` (e.g. `dynesty_out.pkl`) while `save()` writes it
  as `{grid}_out.pkl`. `end()` now reads the grid-named pickle. The results were
  always written correctly; only the final summary raised.
- `_make_transform` (prior transform) read only keyword arguments, so a
  positionally-constructed prior (`truncnorm(a, b, loc, scale)`) raised a
  `KeyError` and `norm(mu, sigma)` silently became a standard normal. It now
  normalises parameters via `_parse_args`, supporting both styles exactly.
- `SEDPlotter` could not plot fits from the BOSZ, SPHINX or TLUSTY grids: the
  interpolator dispatch was missing those grids (`AttributeError`), and `SED()`
  had no fallback for grids without cached spectra. The interpolator path now
  covers all grids, and `SED()` skips the continuous model spectrum with a
  warning (still drawing photometry and synthetic fluxes) when a grid has no
  spectra available.

## [1.5.0] - 2026-05-23

### Added
- **New photometry/astrometry librarian** (now a `librarian/` package): Gaia
  DR3 `best_neighbour` cross-matching with a VizieR XMatch fallback,
  per-catalogue quality-control, robust coordinate→`source_id` resolution via a
  Gaia TAP cone query, and **Hypatia-Catalog [Fe/H]** spectroscopic priors. The
  `Star` lookup and prior construction were adapted accordingly, including
  per-field null-safe spectroscopic priors (a Hypatia [Fe/H]-only result no
  longer requires a spectroscopic Teff/logg). Filter names are bridged to the
  ARIADNE convention (Gaia DR3 → `GaiaDR2v2_*`).
- Three new model grids:
  - **BOSZ** (Mészáros et al. 2024) — MARCS+ATLAS9 atmospheres synthesised with
    Synspec, Teff 2800–16000 K, full 0.05–32 µm coverage. Added to the BMA set
    (temperature-gated to its validity range); brings independent
    MARCS/Synspec physics to the FGK regime.
  - **SPHINX-II** (Iyer et al. 2025) — M-dwarf grid (Teff 2000–4000 K), gray
    cloud log κ = −29 / C/O = 0.5 fiducial slice. Standalone (`f.grid = 'sphinx'`).
  - **TLUSTY** OSTAR2002 + BSTAR2006 (Lanz & Hubeny) — NLTE hot O/B stars
    (Teff 15000–55000 K). Standalone; uses a wide uniform Teff prior since the
    FGK population prior cannot reach the hot regime.
- Tunable sampler/parallelism knobs (defaults preserve prior behaviour):
  - `walks` — number of `rwalk` MCMC steps per proposal (also `setup[7]`).
  - `n_grid_jobs` — number of BMA model grids fit concurrently (one core each).
  - `isochrone_dlogz` — evidence tolerance for the MIST isochrone age/mass fit.
- `grid_wave_coverage` in `config.py`: standalone fits with a limited-coverage
  grid (e.g. Coelho) now exclude out-of-coverage bands instead of using
  extrapolated fluxes.
- Blackbody quality-control overlay on the raw SED plot: the photometry QC
  blackbody fit (temperature + flagged bands) is persisted on the `Star` and
  drawn on `SED_no_model.png`, with QC-flagged bands outlined in red so SED
  outliers (blends, bad cross-matches, IR excess) can be inspected visually.

### Changed
- **Major fitting speedups** (~12–25× on the benchmark BMA run):
  - Prior transform now uses closed-form inverse CDFs instead of scipy frozen
    `.ppf` (~80× faster per call — the dominant cost).
  - `nogil` numba trilinear interpolator replaces the per-call DFInterpolator
    path (~18× faster, bit-identical for in-bounds inputs).
  - Parameter order, prior transforms and the extinction curve are cached on
    object identity (extinction is linear in Av → precomputed once).
  - The MIST isochrone age/mass fit is parallelised across `threads`.

### Fixed
- `KeyError` in the non-BMA `save()`/`end()` path (`best_fit_averaged` keys).
- dynesty `queue_size` off-by-one left one pool worker idle.

### Deprecated
- The `'multinest'` sampling engine is **deprecated** and will be **removed
  entirely in v2.0**. It only ever supported single-grid fits (not BMA), and
  `dynesty` is now both faster and the only engine exercised by the BMA path.
  Use `engine='dynesty'` (the default).

## [1.4.6] - 2026-04-07

### Fixed
- Photometry QC outlier detection used a monochromatic Planck function with
  incorrect `lambda^-3` scaling and no zero-point calibration, causing false
  positives in NIR/MIR bands (2MASS, WISE). Replaced with `_synth_mags()`
  that uses the correct `lambda^-5` Planck formula and per-filter zero-point
  fluxes (Vega or AB as appropriate for PS1, SDSS, GALEX).

## [1.4.5] - 2026-04-05

### Added
- BMA-weighted gray sampled spectra on SED plot: when running in Bayesian Model
  Averaging mode, `plot_SED()` now draws 100 gray model spectra sampled from each
  grid's posterior proportionally to its BMA weight, visualizing the combined
  model + parameter uncertainty behind the best-fit spectrum.

### Changed
- Rewrote `end()` output in `utils.py` as a LACHESIS-style columnar table with
  proper units (Teff (K), log(g) (dex), [Fe/H] (dex), etc.), column header,
  separator rule, and a dedicated noise-parameter block.
- Model weights now display as a proper table with probabilities **and** logZ
  per grid, replacing the bare `phoenix probability : 0.0360` lines.
- `show_priors()` restyled to match the new LACHESIS table format (no more
  box-drawing characters) and is now called automatically from `initialize()`.

### Fixed
- `_format_prior_notation()` previously looked at
  `type(prior_obj).__name__`, which always returns `rv_continuous_frozen`
  for frozen scipy distributions. Switched to inspecting
  `type(prior_obj.dist).__name__` and pulling parameters from `.kwds`/`.args`.
- RAVE population priors (stored as `InterpolatedUnivariateSpline` from
  `teff_ppf.pkl`) now display as `RAVE (population)` instead of the raw
  scipy class name, even when using `create_priors_from_setup`.
- Removed three redundant local `colors = [...]` lists in `utils.py` /
  `fitter.py` that were shadowing the shared list imported from `config.py`.

## [1.4.3] - 2026-04-04

### Added
- Spectroscopic prior support from APOGEE DR17, GALAH DR3, and LAMOST DR5,
  with automatic survey priority cascade (APOGEE > GALAH > LAMOST > RAVE).
- `save_bma` now automatically writes `ariadne_result.nc` (arviz InferenceData)
  to the output folder alongside the existing pickle and `.dat` files.

### Fixed
- Multiprocessing crash on macOS with Python 3.12+ caused by `spawn` start
  method breaking the module-level globals that dynesty callbacks depend on.
  Forced `fork` unconditionally.
- `to_netcdf()` was never called and `self.out` was never populated by
  `save_bma`, making the netCDF export path dead code since 1.4.0.
- `pyphot` compatibility: replaced `.magnitude` (pint API) with `.value`
  (astropy API) in `phot_utils.py` to match current pyphot releases.
- Cached `pyphot.get_library()` at module level to avoid redundant HDF5 reads
  (~3 per filter per star).

## [1.4.1] - 2026-03-25

### Changed
- `to_dict()` / `to_netcdf()` now include MIST isochrone posterior samples
  (age, iso_mass, eep), filter names, filter bandwidths, best-fit model SED
  fluxes, and summary statistics (best_fit_averaged, uncertainties_averaged,
  confidence_interval_averaged). The `.nc` file is now a fully self-contained
  data product for both downstream tools and frontend visualization.
- Model SED fluxes can be injected via `fitter.out['model_sed']` before
  calling `to_netcdf()`, allowing the host application to include the
  best-fit model evaluated at observed filter wavelengths.

## [1.4.0] - 2026-03-24

### Added
- `Fitter.to_dict()` — export BMA results as a structured dictionary matching
  the ecosystem output spec (posterior samples, observed photometry, model weights).
- `Fitter.to_netcdf(path)` — export BMA results as an `arviz.InferenceData`
  object in netCDF4 format, the canonical inter-tool format for the
  ARIADNE → LACHESIS → PROTEUS pipeline.
- `arviz` added as a dependency.

## [1.3.4] - 2026-03-24

### Fixed
- `_get_gaia_id()` Vizier query excluded the `_r` (angular distance) column
  when restricting to `columns=['Source']`, causing `ValueError` on sort.
  Added `'+_r'` to the column list so the nearest-source lookup works correctly.

## [1.3.3] - 2026-03-15

### Added
- Precomputed spectra cache download from Zenodo for faster first-run setup.

### Fixed
- Excluded `spectra_cache.h5` from package distribution to keep PyPI package small.

## [1.3.2] - 2026-03-02

### Added
- Automatic fallback to VizieR XMatch / cone search when the Gaia TAP service is
  unavailable (503, 502, 500, or timeout). Affected catalogs: 2MASS, WISE, Pan-STARRS,
  SDSS, TYCHO2, APASS, RAVE, SkyMapper.

### Changed
- Migrated all `print()` / direct console output to Python's `logging` module across
  `error.py`, `fitter.py`, and `plotter.py`, allowing host applications to control
  verbosity via standard log configuration.
- `gaia_params()` now uses VizieR cone search instead of a direct Gaia TAP query,
  removing the dependency on Gaia TAP for basic stellar parameter lookup.
- Distance query (`_get_distance()`) rewritten to use direct VizieR constraints on
  the Bailer-Jones EDR3 catalog — simpler and more reliable.
- Gaia ID retrieval (`_get_gaia_id()`) switched from async Gaia cone search to VizieR
  cone search.

### Fixed
- Parallax error messages now distinguish between a masked value (no astrometric
  solution) and a non-positive measured parallax.

## [1.3.1] - 2025-12-28

### Fixed
- Matplotlib 3.1+ compatibility issue in corner plot generation (tick.label → tick.label1)

## [1.3.0] - 2025-12-28

### Added
- Gaia DR3 support - Complete upgrade from Gaia DR2 to DR3
- Direct TAP queries for APASS photometry via Gaia external.apassdr9
- Direct TAP queries for SkyMapper DR2 photometry
- RAVE DR6 stellar parameter priors support

### Fixed
- APASS photometry retrieval broken by Vizier removing recno field (Issue #73)
- SkyMapper photometry mismatch between DR2 crossmatch and DR1.1 catalog
- 2MASS Vizier column name compatibility issues
- IndexError when star exists in Gaia DR3 but not in DR2
- Catalog crossmatch robustness improvements

### Changed
- Updated Gaia query system to use DR3 tables and best_neighbour crossmatches
- Replaced Vizier queries with direct TAP service queries for APASS and SkyMapper
- Modernized README installation instructions

## [1.2.1] - 2024-12-01

### Fixed
- 2MASS Vizier column name compatibility
- Modernized for Python 3.11+

## [1.1.2] - 2024-11-01

### Fixed
- Temporary workaround for APASS recno issue (disabled APASS retrieval)

## [1.1.1] - 2024-08-01

### Fixed
- Deprecation warning
- Updated README examples

### Changed
- Deepened residual panel on SED plots
- Made error bars more apparent in sigma terms
