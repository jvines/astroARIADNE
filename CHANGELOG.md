# Changelog

All notable changes to astroARIADNE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.0] - 2026-11-20

### Added
- **Predicted photometry for every ARIADNE band.** Single-grid and BMA fits now
  write `synthetic_magnitudes.dat` and `synthetic_fluxes.dat` with the observed
  flux and magnitude (1-sigma errors from up to 3000 posterior draws through the
  forward model, distance/radius/extinction-scaled) of the best-fit SED in all
  bands, flagging the bands used in the fit, and store them in the pickle as
  `synthetic_photometry`. `synthetic_fluxes.dat` now holds observed-frame fluxes
  (previously BMA-only surface fluxes) (#88).
- **Predict the best-fit SED in any filter.** `Fitter.synthetic_filters`
  takes pyphot library filter names, `pyphot.Filter` objects or
  `(name, wavelength_um, transmission[, 'energy'])` tuples. Filters that are
  not ARIADNE grid bands are convolved once against the BT-Settl spectra of the
  spectra cache and pushed through the same posterior draws as the grid bands,
  giving fluxes, Vega and AB magnitudes with 1-sigma errors in the new
  `synthetic_user_filters.dat` (also appended to `synthetic_magnitudes.dat` /
  `synthetic_fluxes.dat` with `observed=0`, and stored in the pickle as
  `synthetic_user_filters`). Without the cache those filters are skipped with
  one warning naming the fetch command (#88).
- **`fit_bma` reuses existing per-grid fits.** A `<grid>_out.pkl` already in
  `out_folder` is loaded instead of re-running dynesty, so the BMA can be
  recomputed for a new `models` list (e.g. to re-plot without a bad grid)
  without refitting (#56). A pickle is reused only if the photometry, the
  norm vs dist/rad layout, the priors, the sampler settings, the extinction
  law and the fixed parameters all match the current setup; otherwise that
  grid is refitted with a note saying why. Outputs written before this
  release don't record priors or settings and are reused with a note saying
  so. Set `Fitter.refit = True` to refit every grid.
- **Gaia blend QC: unresolved neighbours no longer pass silently (#55).**
  Two checks, after Contardo & Hogg 2024 (arXiv:2403.18941, Sec. 2 and 3.1),
  switched off together with `Star(contamination_qc=False)`:
  - *Cross-match uniqueness.* The Gaia best-neighbour queries now also read
    `number_of_mates` and `number_of_neighbours`. A 2MASS or AllWISE
    counterpart with `number_of_mates > 0` is shared with other Gaia sources
    (an unresolved blend), and its bands are dropped; `number_of_neighbours
    > 1` keeps the bands and warns. Results are in `star.xmatch_flags`. When
    the ESA archive is unreachable the check is skipped with a warning.
  - *Neighbour flux contamination.* One Gaia DR3 cone (20") gives, per band,
    the fractional contaminating flux
    `c = sum 10^(-0.4 (G_nb - G_target)) exp(-sep^2 / 2 sigma^2)`, with sigma
    from a per-catalogue PSF FWHM table (`PSF_FWHM_ARCSEC` in
    `librarian/_contamination.py`; Gaia, TESS, Tycho-2 and Stromgren bands
    are not checked). Bands with `c > 0.01` are flagged and those with
    `c > 0.05` (about 0.05 mag) are removed from the fit, each with a warning
    naming the neighbours. The values are in `star.contamination`; thresholds
    are `Star(contamination_flag=..., contamination_exclude=...)`. Gaia G is
    the flux proxy, with no colour term.
- **HST WFC3 and ACS/WFC bands (#78).** 44 bands (WFC3/UVIS and WFC3/IR
  wide, medium, long-pass and extremely wide filters, plus ACS/WFC) are grid
  bands now, reconvolved from each grid's raw spectra, so HST photometry can
  be added with `Star.add_mag` and enters the fit like any other band. The
  names are the SVO ones (`HST_WFC3_F814W`, `HST_ACS_WFC_F606W`, ...); see
  `filters.md`.
- **Per-band magnitude systems.** HST photometry is published in Vega, ST or
  AB magnitudes, so `Star(..., mag_system={'HST_WFC3_F814W': 'AB'})` and
  `Star.add_mag(mag, err, band, system='ST')` say which one a magnitude is in.
  HST bands default to Vega and every other band keeps its usual system
  (AB for SDSS, PS1, SkyMapper and GALEX, Vega otherwise). The choice is used
  everywhere magnitudes become fluxes and back: the fit, the outlier clip,
  the plots and the synthetic magnitudes.

### Fixed
- **BOSZ and Coelho radii were too large by sqrt(4 pi) and sqrt(pi).** The
  BOSZ and Coelho grids held fluxes 1/(4 pi) and 1/pi of every other grid's,
  and nothing corrected for it, so a fit to either inflated the radius
  (BOSZ by 3.5x, Coelho by 1.8x). Both are on the common flux scale now.
- **Coelho, BT-Settl, BT-NextGen and BT-Cond nodes hold one spectrum each.**
  Coelho took the alpha-enhanced spectrum at the 1177 nodes where both
  variants exist, and the BT grids whichever [alpha/Fe] variant was written
  last at their 617 multi-variant nodes. Each node now holds the solar-alpha
  Coelho model and the standard-alpha BT model ([alpha/Fe] = 0 at
  [Fe/H] >= 0, +0.2 at -0.5, +0.4 below), the same as ARIADNE v2.
- **BT-NextGen and BT-Cond were copies of BT-Settl.** Both pickles held
  BT-Settl's fluxes, so a BMA over the three BT grids counted one model
  three times. They now hold the BT-NextGen and BT-Cond models.
- **The outlier clip used Vega zero points for SkyMapper.**
  `Star.clip_outlier_magnitudes` turned SkyMapper (AB) magnitudes into fluxes
  as if they were Vega, so those bands were judged against a shifted SED. It
  now uses the same per-band zero points as the fit.
- **Multi-threaded fits no longer die with `NameError: name 'flux' is not
  defined`** on machines where the worker processes are spawned (macOS
  default) (#58). The fitter's and the isochrone's pools now always fork,
  through their own fork context, instead of relying on a process-wide
  `set_start_method('fork', force=True)` at import that any later import or
  user code could undo. Importing astroARIADNE no longer changes the global
  start method. Diagnosed by @alexishtz and @JenniferBurt (PR #81).
- **The default `norm` prior no longer excludes nearby stars.** The fixed
  U(0, 1e-20) ceiling on (R/D)^2 capped R/D at 1e-10, railing any star
  nearer than ~225 pc x (R / R_sun) against the bound and returning a wrong
  Teff and radius with no warning (#90). With a distance, the ceiling is
  now (100.05 R_sun / D_min)^2, D_min = D - 5 sigma floored at 1 pc.
- **Fits whose posterior is railed against a prior bound now warn.** A
  parameter is flagged when its 3-sigma interval runs into a finite prior
  bound (the gap is under 1% of the interval's width); prior-dominated
  posteriors and zero floors are not flagged. Previously nothing warned.
- **The default `norm` ceiling without a distance is 1e-16** (was 1e-20),
  which covers a 1 R_sun star out to ~2 pc.
- **`show_priors` prints very small or large bounds in scientific
  notation;** `U(0, 1e-20)` used to print as `U(0.00, 0.00)`.
- **A user-supplied extinction no longer crashes `initialize()`.**
  `Star(Av=..., Av_e=...)` dropped `Av_e`, so `_default_priors` raised
  `AttributeError` for any star beyond the 70 pc Av auto-pin (#89). `Av_e`
  is now stored, and with it the Av prior is the truncated normal
  N(Av, Av_e) on [0, 100] instead of U(0, Av).
  A user-supplied Av is now labelled `user_*` in `prior_sources` instead of
  `dustmap*`; `Av_e=0` pins Av at the given value, and a negative `Av_e`
  is rejected.

## [1.5.6] - 2026-11-19

### Fixed
- **SkyMapper magnitudes are converted as AB.** `mag_to_flux` and
  `flux_to_mag` sent every SkyMapper band (u, v, g, r, i, z) through the
  Vega zero point, but SkyMapper DR2 magnitudes are AB. The fluxes came out
  too faint, worse in the redder bands, and tilted every fit that used them.

## [1.5.5] - 2026-11-13

### Fixed
- **False empty answers from the CfA mirror are no longer cached.** The
  mirror sometimes answers "nothing here" for a source it holds, with no
  error and nothing to tell it from a real empty (HIP 66924's 2MASS cone,
  2026-11-13; the row came back minutes later). astroquery cached those and
  replayed them on every rerun, so stars silently lost 2MASS, WISE and
  Strömgren. An empty answer from a mirror is now never cached and is asked
  once more; poisoned cache entries are purged when next read.
- **The mirror's overload answer is a database error.** "ERROR : temporary
  unavailable" / "(pipe) Request has been interrupted" arrive as HTTP 200 and
  read as empty; they are now retried like the other database errors and
  never cached. Permanent errors ("Table ... does not exist") are not.
- **Malformed VOTables from the mirror are retried** like a database error
  and never cached, instead of failing the query.
- **Blank rows from the mirror are dropped.** The mirror puts an all-empty
  row first in some answers (`Source=0` on `I/355/gaiadr3` and
  `I/352/gedr3dis`, `RAJ2000=0` on `II/246/out`); code reading row 0 lost the
  Gaia astrometry, the Bailer-Jones distance or the 2MASS bands.
- **TIC from VizieR when MAST fails.** MAST's catalogue service stalled on
  2026-11-13 (one TIC query took 401 s, the next failed after 90 s), so every
  star lost its TESS band. TIC-8 is now read from VizieR (IV/39/tic82,
  through the mirror failover) when MAST fails, matched on the Gaia ID.

## [1.5.4] - 2026-11-12

### Added
- **Failover for every CDS call.** A CDS Strasbourg outage takes down VizieR
  and xMatch together, and previously only the Gaia queries had a fallback.
  Every VizieR query is now retried on the CfA mirror
  (`vizier.cfa.harvard.edu`; override with `CDS_VIZIER_MIRRORS`), and xMatch
  (GALEX, Mermilliod, Strömgren and the catalogue-ID recovery) is reproduced
  locally from two VizieR cone queries. A host that fails is skipped for 10
  minutes (`CDS_DOWN_SECONDS`). The module is `astroARIADNE/librarian/_cds.py`.

### Fixed
- The CfA mirror returns shifted, corrupt values for default (`-out=*`) and
  explicit column lists, so requests sent to a mirror always ask for
  `-out.all`. A mirror database error arrives as an empty HTTP 200; it is
  retried (`CDS_MIRROR_TRIES`, default 5) and then raised, so the caller takes
  its own fallback instead of reading "not in catalogue".
- While CDS is down, an empty FLAME or Bailer-Jones answer from the mirror is
  confirmed with the ESA archive: CfA answers `Source=<id>` on `I/355/paramp`
  with an empty table even when it holds the row.
- A dead xMatch host is detected by a 10 s TCP probe; astroquery's own table
  check ignores `XMatch.TIMEOUT` and waited 75 s, past the query budget.
- Error responses that astroquery cached during an outage are deleted instead
  of being replayed for a week.

## [1.5.3] - 2026-11-11

### Added
- **ESA Gaia archive fallback.** When a VizieR query fails, the Gaia DR3 source
  row, the FLAME parameters and the Bailer-Jones distance are read from the ESA
  Gaia archive instead (`gaiadr3.gaia_source`,
  `gaiadr3.astrophysical_parameters`, `external.gaiaedr3_distance`), so a CDS
  outage no longer leaves stars without parallax, Gaia photometry or distance.
  An empty VizieR answer still means the source has no FLAME or distance entry;
  only a failed query goes to ESA. Gaia magnitude errors include the Riello et
  al. (2021) zero-point term, matching VizieR's `e_Gmag`.

### Fixed
- ESA archive queries, including the Gaia ID cone search, retry as async jobs
  when the sync job is rejected. ESA rejects sync jobs when its sync queue is
  full, which made the Gaia ID lookup fail outright.

## [1.5.2] - 2026-10-23

### Fixed
- `Fitter.to_netcdf` (called automatically by `save_bma` at the end of
  `fit_bma`) failed on arviz < 1.0 with `ValueError: unable to infer dtype on
  variable 'posterior'`. It passed all groups to `az.from_dict` in one dict,
  which only arviz >= 1.0 accepts, and arviz 1.x needs Python >= 3.12, so every
  Python 3.11 install was affected. Both arviz APIs are now supported. `BMA.pkl`
  was always written before the failure.

### Changed
- Minimum arviz raised from 0.19.0 to 0.23.4. arviz < 0.22 fails at import with
  matplotlib 3.11.

## [1.5.1] - 2026-05-31

### Fixed
- BMA model weights: the evidence softmax now shifts by the maximum
  log-evidence instead of the minimum, preventing `exp()` overflow to `nan`
  weights when one grid dominates by a large margin (Δln Z ≳ 709).
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
