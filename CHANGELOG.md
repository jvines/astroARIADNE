# Changelog

All notable changes to astroARIADNE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
