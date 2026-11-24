# Changelog

All notable changes to astroARIADNE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Excess-model colours in the plot settings.** `plot_settings.dat` takes
  `irx_comp_color`, `irx_comp_draw_color`, `irx_disk_color`,
  `irx_disk_alt_color`, `irx_provisional_color` and `binary_comp_color`;
  they were hard-coded, as were `legend_fontsize` and `legend_loc` (the
  IRX_SED excess-model legend; `none` omits it) and `sed_draws` (gray posterior spectra on a BMA SED). Settings
  files without these keys keep the old values.
  `figsize` and `hr_figsize` accept non-integer inches, so a figure can be
  sized to a journal column.
- **Selectable companion injection model for injection-recovery tests.**
  `fitter.companion_injection_flux(..., model='sonora')` builds an injected
  companion from Sonora Bobcat by default (no longer BT-Settl, which shares
  its PHOENIX physics with the BT-family recovery grids). Above Bobcat's
  2400 K ceiling it falls back to ATMO 2020 and reports the model it used;
  `model='btsettl'` reproduces the old injections. The new
  `Fitter.irx_companion_grids` overrides the companion recovery grids, so
  the injection grid can be held out.
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
    the flux proxy, with no colour term. Results are saved in the Star's
    NetCDF file.
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

### Changed
- **IR-excess error budget now includes survey calibration and model
  uncertainty (`irx_error_budget='covariant'`, the default).** The excess
  likelihood used the catalogue error floored at 2% and the
  evidence-weighted posterior-predictive spread of the photosphere, which
  gave |ln B| of several hundred on three bands. Each excess band now adds
  its survey's absolute calibration uncertainty in quadrature (WISE, Spitzer
  IRAC/MIPS/IRS, AKARI IRC/FIS, IRAS, Herschel PACS/SPIRE and SPHEREx values
  from the instrument papers; 2% for any other band). The photosphere-model
  term uses the equal-weight spread of the ensemble grids, because grid
  evidences from the optical/NIR say nothing about the mid/far-IR.
  Calibration is modelled at the instrument level. Each instrument's
  absolute scale is fully correlated across its bands and the band-to-band
  remainder is independent, using the documented split for IRAS, SPIRE and
  SPHEREx. The evidence uses the full covariance matrix.
  `irx_error_budget='independent'` treats calibration as independent per
  band. `Fitter.irx_error_budget = 'legacy'` or
  `fit_irx(error_budget='legacy')` reproduces the earlier numbers exactly.
  SPHEREx calibration follows the SPHEREx Explanatory Supplement v2.0
  (1.4% in detector bands 1–4, 2.2% in bands 5–6).
  See docs/irx.md, "Error budget".

### Fixed
- **SPHEREx VARIANCE plane read from the wrong bytes in all but the first
  exposure of a layout variant.** The tile-compressed FLAGS heap varies in
  size per file (three D3 files of one l2b-v27 variant put VARIANCE at
  28376640 / 28221120 / 28108800 B), but the cached layout only re-checked the
  IMAGE offset. Aperture fluxes never read VARIANCE and are unaffected;
  aperture errors and the variance-weighted ePSF fit were wrong (up to 8.5%
  per exposure on TIC-275647095) and depended on run order. A cached layout
  with compressed FLAGS now re-reads the FLAGS header in the same request as
  the IMAGE header (still one round trip) and places VARIANCE after that
  file's own heap. The SPHEREx cache moves to `v5` and the PSF cache to
  `psf-v3`, so previously cached stars re-extract.
- **SPHEREx super-bands are now checked for Gaia neighbours inside the
  extraction aperture (#55).** The blend QC skipped SPHEREx, yet the bands
  are our own 3-px (18.45") aperture photometry and nothing in the aperture
  path subtracts a neighbour: on Gaia DR3 5377927940883353216 a neighbour
  19.95" away with 24% of the target's G flux sat at 3.2 px, and all 11
  super-bands came out 13-23% too bright while 2MASS, WISE and Gaia agreed
  to 2%. After the SPHEREx extraction, a 40" Gaia DR3 cone now gives per
  super-band `c = sum (F_nb / F_target) f_ap(sep) / f_ap(0)`, where `f_ap`
  is the fraction of a point source's flux inside the extraction aperture
  at that separation, tabulated per super-band from the mission's R7 ePSF
  (`SPHEREX_APERTURE_FRACTION` in `librarian/_contamination.py`; 0.99 at
  the centre, 0.30 at 3.25 px, 0 beyond 6.5 px). The flux ratio uses Gaia
  RP where both stars have it, else G. The thresholds are those of the
  other bands (flag at 0.01, remove at 0.05), and the values join
  `star.contamination` with the aperture radius in place of a PSF FWHM.
  Measured through the aperture, that star would lose all 11 super-bands at
  c = 0.096-0.099. Bands measured by the crowded-field forced ePSF fit,
  which already models the catalogued neighbours, are recorded as
  `deblended` and kept (that star is one: its 2MASS neighbours predict
  0.19 of its flux in the aperture, over the 0.02 trigger).
- **The Stagger grid is rebuilt from bolometrically verified spectra.**
  POLLUX serves the wrong spectrum for 34 of the 181 STAGGER-SPEC models
  (all >= 5700 K: each integrates to sigma T^4 of a different model, some
  duplicated), and the shipped grid, synthesized from the spectra cache, had
  them all, plus a flat 2.8% flux deficit and node values taken at each
  model's actual Teff (up to 156 K from the node). Each node now takes its
  own verified model, carried to the nominal Teff along its (logg, [Fe/H])
  sequence; nodes whose model is corrupted are holes (35 of 181) rather than
  wrong spectra. Stagger fluxes move by a median +3%.
- **BT-Settl nodes hold one, standard-alpha, spectrum.** Where AGSS2009 /
  CIFIST2011 ship several [alpha/Fe] variants of a node (617 nodes at logg 4.0
  and 4.5), the cube held whichever variant the builder happened to write
  last; it now holds the standard one ([alpha/Fe] = 0 at [Fe/H] >= 0, +0.2 at
  -0.5, +0.4 below). And at 2600-2700 K, [Fe/H] != 0, the fill that patches
  the solar-only cool shelf had overwritten 100 genuine metal-poor models with
  the solar one in every band but the Roman ones; those are the genuine models
  again. BT-NextGen and BT-Cond are built from SVO's standard-alpha spectra
  and had neither problem.
- **The cool [Fe/H] fill now covers the Roman and IRS columns.** Those columns
  were added after the fill and were NaN at every filled node (1310 BT-Settl,
  540 BT-NextGen, 710 BT-Cond), which brought back the interpolation wall the
  fill removes for cool, non-solar fits using them.
- **The outlier clip used the wrong zero points for SkyMapper and Stromgren.**
  `Star.clip_outlier_magnitudes` turned SkyMapper (AB) magnitudes into fluxes
  with Vega zero points and applied the Stromgren standard-system offsets
  with the opposite sign to the fit, so those bands were judged against a
  shifted SED. It now uses the same per-band zero points as the fit.
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
  estimate; spectroscopic priors still take precedence. The uniform Teff prior's
  lower edge is not a fixed value — it tracks the coolest node the in-play grids
  reach (capped at 2300 K), so an L/T primary reaches below the M-dwarf floor
  when the cloudy/brown-dwarf grids are in the ensemble (see *Changed*).
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
  available — **Bayesian model selection** over three hypotheses for the excess:
  no excess (null), a **debris-disk blackbody** (`t_dust`, fractional excess
  luminosity `l_dust_over_lstar`), and a **cool unresolved companion** (a grid
  photosphere of free `teff2`/radius). Each hypothesis' Bayesian evidence is
  integrated by direct 2-D quadrature over physical priors (exact at this
  dimensionality), giving posterior model probabilities `p_none` / `p_disk` /
  `p_companion`, an `excess_class`, the disk-vs-companion log Bayes factor, and
  marginal parameter posteriors (mean ± std). This is the evidence-based
  analogue of ARIADNE's BMA — the Occam-penalised null wins on noise, where a χ²
  comparison would force a spurious disk or companion. Stored per band in
  `out['irx']`; `SEDPlotter(ir_excess=True)` overlays the fitted disk blackbody
  and labels the classification.
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
- **BT-Dusty rebuilt from the SVO Theory Server** — the cloudy (max-dust) BT
  limit for the L/T transition, now spanning Teff 1000–6000 K, log g 0.5–6.0, 11
  metallicities (`[M/H]` −4.0…+0.5) and the full 0–1000 µm reach (the previous
  grid was solar-metallicity only with the far-IR NaN). The cloudless BT members
  cannot reproduce a dusty L dwarf's flattened NIR; BT-Dusty now wins the
  photosphere BMA for cloudy L0-type primaries.
- **Sonora Bobcat grid** (Marley et al. 2021) rebuilt from the source spectra
  (Zenodo 5063476) across 0.4–50 µm, filling the optical gap the previous Sonora
  cube left NaN. Cloud-free L/T/Y photospheres for the substellar-companion
  hypothesis.
- **Gaia XP synthetic photometry gap-filling**, plus **SPHEREx** QR2 (1–5 µm),
  **CASSIS / Spitzer-IRS** (5–38 µm) and **IUE/INES** (UV) super-band
  spectro-photometry — each fetched by the librarian by default (disable per
  source with `ignore=['XP'|'SPHEREX'|'CASSIS'|'IUE']`) to bridge catalogue gaps
  and cross-check the broadband SED. Filled bands are provenance-tagged.
- **Roman Space Telescope readiness (staged).** The eight WFI imaging broadbands
  (`ROMAN_F062`…`ROMAN_F213`, real SVO curves ≈ R/z/Y/J/H/(H-K)/Ks + the F146
  wide band) and seven `ROMANP_*` P127-prism super-bands (0.75–1.80 µm, R~100,
  gap-placed between the imaging centres so the prism adds spectral shape — the
  1.4 µm H₂O trough no WFI band covers — rather than re-measuring the broadband
  flux) are in the filter library and convolved into every grid, and are in
  `filter_names` as fit bands (< 2.3 µm, never IR-excess). They are **data-
  staged and inert**: no catalogue feeds them yet — the WFI/prism science
  archives publish ~1–2 yr post-launch — so they never touch a fit today, but
  the curves + grid columns + the `librarian/_roman.py` binning are in place so a
  Roman fit works the day photometry lands. Roman's 0.11″ resolution is the draw
  for ARIADNE's cool/substellar regime: it deblends the close pairs (e.g. the
  J0446/J0949 binaries) that 2MASS/WISE/SPHEREx merge. The Roman columns are
  **raw-reconvolved from the model spectra** for 15 of the 16 grids (rebuilt
  through the full 108-band filter set; the BT-family cool L/T/Y extension is
  preserved by merging the raw standard nodes with the species-derived cool
  nodes); Stagger keeps cache-derived columns (its raw spectra are not available)
  and NewEra fills in from its streamed re-build. Build with
  `notebooks/build_custom_filters.py` + a `build_model_grids.py` reconvolution
  (`augment_grids_roman.py` is the cache-derived fallback, exact for these
  low-resolution bands).
- **RUWE binarity flag.** The librarian fetches the Gaia DR3 RUWE and the fit
  summary warns when RUWE > 1.4 — the unresolved-binary / perturbed-astrometry
  regime where the parallax (and hence the radius and luminosity) is unreliable.
- **Bolometric flux `f_bol` as a standard output.** Every fit now reports the
  bolometric flux at Earth, F_bol = σ_SB·Teff⁴·(R/d)² (erg/s/cm²), as a derived
  quantity beside the luminosity — in the printed summary, the `best_fit*.dat`
  tables, `to_dict()` (`bolometric_flux`), the NetCDF export and `BMA.pkl`, with
  full posterior uncertainties. Because it depends only on (R/d)², F_bol is
  distance-independent and is reported even in norm mode (no parallax), where the
  luminosity is undefined — the direct SED observable behind the bolometric-flux
  / IRFM effective temperature. ARIADNE's model flux integrates to σ_SB·Teff⁴ by
  construction, so this is the full bolometric flux (no truncated-SED tail).
  Validated against ε Eri (F_bol ≈ 1.06×10⁻⁶ erg/s/cm²).
- **Predicted synthetic photometry (`synthetic_magnitudes.dat`).** Every fit now
  writes the best-fit SED's *observed* synthetic photometry — flux **and
  magnitude**, with 1σ errors — for the full ARIADNE filter set, so you can read
  off a magnitude the star lacks (a missing G, K, W1, …) from the bands it has.
  Each band is flagged observed (used in the fit) or predicted. The posterior is
  propagated through the forward model (distance/radius/extinction-scaled) on the
  BT-Settl grid, and mags are in each band's native Vega/AB system. Validated to
  reproduce the fitted-band catalogue magnitudes to ≈0.02 mag; the un-fit bands
  are genuine predictions (consistent to <0.01 mag between single-grid and BMA
  runs). Stored in `out['synthetic_photometry']` and `synthetic_magnitudes.dat`;
  `synthetic_fluxes.dat` now holds these observed fluxes (previously surface
  fluxes). WD/hot-subdwarf fits are a caveat — the BT-Settl photosphere is only a
  proxy there.

### Changed
- `gaiaxpy>=2.1.5` is now a declared dependency (the XP gap-fill imported it
  without declaring it). The ESA Gaia archive 3.15 (Nov 2026) rejects the
  `data_structure='raw'` DataLink request gaiaxpy 2.1.4 sends, so on 2.1.4
  installs the XP gap-fill silently filled nothing.
- Bayesian Model Averaging weights: the evidence softmax now shifts by the
  maximum log-evidence instead of the minimum, preventing `exp()` overflow to
  `nan` weights when one grid dominates by a large margin (Δln Z ≳ 709).
- **Model grids are now stored as HDF5 flux cubes** (`<Name>.h5`) instead of
  pandas-pickled DataFrames (`<Name>_DF.pkl`). The interpolator reads the cube
  directly, so the grids no longer depend on the pandas/pickle version they were
  written with and survive numpy/pandas upgrades; pandas is no longer imported
  on the fit path.
- **The cool-star Teff prior floors at the ensemble's grid reach** instead of a
  hard 2300 K (M9), so L/T primaries are no longer pinned at the M-dwarf edge and
  a cloudy solution can win where the data support it — validated on an L0
  Peter-Pan-disk host (Δln Z = +34.7 for the cool, cloudy fit, which also flipped
  a false companion classification to the correct disk). Warm-only ensembles are
  unchanged (capped at 2300 K).
- **IR-excess companion hypothesis** now enforces L₂ ≤ L₁ (a companion cannot
  outshine the primary). The `SEDPlotter(ir_excess=True)` disk overlay is trimmed
  to the excess region (no longer painted over the photosphere line) and the SED
  upper y-limit is keyed to the best-fit model rather than the gray posterior
  draws.
- **Spectra cache republished as v2.0 on Zenodo** (record 21251184, concept DOI
  10.5281/zenodo.19141565) with all 16 ensemble grids and the rebuilt BT-Dusty /
  Sonora cubes; `fetch.py` pins the new record ID and SHA-256.

### Removed
- **The `'multinest'` sampling engine has been removed** (deprecated in 1.5.0).
  It only ever supported single-grid fits, never BMA; `dynesty` is the sole
  engine and is faster. `engine='multinest'` now raises a clear error, and
  `pymultinest` is no longer imported. **Breaking change** for any code that
  passed `engine='multinest'`.
- The legacy human-readable `model_grid_<Name>.dat` grid tables (~90 MB) are
  gone; `Star.load_grid` now reads the grid axes from the HDF5 cube instead.
- `arviz` and `h5netcdf` are no longer dependencies. `ariadne_result.nc` is now
  written with `scipy.io` and holds the same dict as `BMA.pkl`; read it with
  `Fitter.read_netcdf`. 1.x files (arviz InferenceData) are not readable this way.

### Fixed
- The modified-blackbody disk model crashed with numpy >= 2.4, which removed
  `np.trapz`. `_modbb_bolo_frac` now integrates with `scipy.integrate.trapezoid`.
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
- A CDS outage no longer strips the photometry. Every VizieR query now fails
  over to a VizieR mirror (CfA by default, `CDS_VIZIER_MIRRORS` to change).
  CDS xMatch calls (APASS/2MASS/WISE/PS1/SDSS/Tycho-2 ID recovery, GALEX,
  Mermilliod UBV, Strömgren) are reproduced locally from two mirror cone
  queries when xMatch is down. A failed host is skipped for 10 minutes
  (`CDS_DOWN_SECONDS`), and error responses that astroquery cached during an
  outage are deleted instead of being replayed for a week.

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
