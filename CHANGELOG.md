# Changelog

All notable changes to astroARIADNE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
