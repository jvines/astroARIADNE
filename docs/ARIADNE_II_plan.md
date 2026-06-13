# ARIADNE II — planning notes

Working document for a possible "ARIADNE II" paper + v2.0 release. Not user-facing
docs; a decision record. Nothing here is committed-to yet.

## The reframed thesis (after the X1 go/no-go)

**ARIADNE II = trustworthy logg and [Fe/H] with model-systematics-aware
uncertainties, enabled by Gaia XP — NOT "better radii".**

The 2022 paper's own admission is that ARIADNE constrains [Fe/H] poorly and
broadband logg is weak. Three things line up on exactly those parameters:

1. Inter-grid (model) spread is *largest* for logg/[Fe/H]/Av (see X1 below).
2. Gaia XP's information content (molecular bands, G-band/MgH/Ca II, Balmer
   region) is exactly the logg/[Fe/H]-sensitive signal broadband integrates away.
3. The defensible novelty (BMA across grids folding model systematics into the
   error bar) matters most where the grids actually disagree — logg/[Fe/H].

The headline is the *combination*: an ab-initio, training-set-independent
(Teff, logg, **[Fe/H]**) estimate whose uncertainties include the inter-grid
model systematic — which no XP tool currently does (GSP-Phot uses hard
best-library argmax; every ML method collapses systematics into one training
label system).

## X1 go/no-go result (first cut — anecdotal, ONE star, 4 grids)

Decomposition of output uncertainty into inter-grid (model/BMA) spread vs
within-grid (statistical + data) width, from existing BMA pickles
(benchmark FGK star, grids phoenix/btsettl/kurucz/ck04).
`ratio = inter_grid_std / typical_within_std`. Consistent across 3 runs:

| param | ratio (3 runs) | reading |
|-------|----------------|---------|
| logg  | 0.52 / 0.53 / 0.61 | model spread ~half the statistical width — LARGE |
| Av    | 0.30 / 0.23 / 0.37 | substantial |
| teff  | 0.31 / 0.19 / 0.25 | moderate |
| z     | 0.23 / 0.14 / 0.15 | modest |
| dist  | 0.12 / 0.18 / 0.14 | small (parallax-dominated) |
| rad   | 0.07 / 0.18 / 0.17 | SMALL — radius is grid-robust |

Striking standalone figure: per-grid logg medians on the *same star* swing
~0.5–0.7 dex (PHOENIX ~4.7–4.9 vs Kurucz ~4.1). That is the paper's money plot.

**Verdict: conditional GO.** Model-systematics term is real and non-trivial
(decisively for logg), so the paper isn't dead — but it redirects the framing
away from radius (which barely moves) toward logg/[Fe/H].

### Caveats that the real X1 must address
- One bright FGK star is anecdote, not evidence. Need a PANEL.
- The 4 grids are the correlated PHOENIX/Kurucz family → this inter-grid spread
  likely UNDERESTIMATES true model error. (Helps the "bars are too tight if you
  ignore model choice" argument; does NOT prove the combined bars are calibrated.)
- Radius being grid-robust is honest bad news for the "better exoplanet host
  radii" angle. Don't sell that.

## What XP actually entails (from the integration study)

**Architectural fact (verified):** ARIADNE fits band-convolved fluxes through a
strictly diagonal Gaussian (`fast_loglik`, sed_library.py:334-339); grids are
tables of fluxes through ~48 filters; the fit never touches a spectrum. XP is a
covariant low-res spectrum → not a drop-in.

**Chosen path: Option A — XP as synthetic pseudo-bands.**
GaiaXPy synthesizes fluxes through K~10–20 pseudo-bandpasses → each becomes a new
"filter" column → re-convolve grids through them → librarian fetches XP → the
existing likelihood/BMA/sampler are untouched. Reuses ARIADNE's whole
architecture. (Option B, a direct covariant-spectrum likelihood, makes ARIADNE a
worse GSP-Phot — rejected.)

**The covariance caveat is the whole ballgame:** XP samples are heavily
correlated; the diagonal likelihood has no off-diagonal term. Must use K small
(~10–20 whitened/broad bands), a dedicated XP-jitter parameter, and mask
<400 / >1000 nm. Naively dumping a fine spectrum in as hundreds of pseudo-bands
overweights XP ~10× and gives confidently-wrong posteriors.

**What XP buys:** nothing on the Teff–Av lever arm (XP 330–1050 nm is interior to
ARIADNE's GALEX→WISE span) or the bolometric tail. Real gains: cool stars
(M/late-K molecular bands SPHINX-II can model) and [Fe/H]/logg. Redundant for
bright low-Av FGK hosts (where real spectra usually already exist).

## Effort / sequence to a prototype (Option A)

1. Librarian XP retrieval (`Gaia.load_data(..., retrieval_type='XP_CONTINUOUS')`,
   pin Huang+2024 corrected coefficients). *small*
2. Pseudo-band definition + GaiaXPy synthesis (real in-range passbands + a few
   feature-targeted windows); validate units vs ARIADNE physical flux. *small–mod*
3. Rebuild ONE grid (PHOENIX) through the K new passbands; extend
   `config.filter_names`; add XP jitter param. *moderate (compute)*
4. End-to-end smoke test on CALSPEC / interferometric-radius / M-dwarf /
   metal-poor stars; confirm XP doesn't dominate; check pulls. *small*
5. Roll out to all grids; BMA calibration validation (the paper's core evidence).
   *moderate*
6. Scope demonstration sample to where XP wins + a redundancy control of bright
   FGK. *analysis*

Steps 1–4 ≈ a focused work-week + compute. Steps 5–6 are the science (months).

## Open decisions for Jose
- Paper spine: XP+validation fused (recommended) vs validation-only vs XP-only.
- Real X1 panel: assemble ~20–30 benchmark stars (FGK + M + metal-poor) with
  external truth (APOGEE logg/[Fe/H], CALSPEC, interferometric radii) and re-fit
  with the full 8-grid set + controlled freeze-parallax / inflate-zeropoint
  perturbations to attribute the within-grid budget.
- Whether v2.0 ships the platform engineering (X3/X4 catalogue runner) too, or
  v2 is purely the science and the platform is a separate track.

## Provenance
- X1 decomposition script: `/tmp/x1_decompose.py` (reads BMA `originals`/`weights`).
- XP integration study: 4-agent workflow, run wf_e5a71afa-f64.
- All numbers above are from existing `test_star_*/BMA.pkl` runs, one benchmark
  star — explicitly NOT a result, a go/no-go probe.

## Feature ideas to develop (seeded 2026-06-03)

Two catalogue-oriented outputs to grow from things ARIADNE already half-does.
Framing is a seed for Jose to sharpen, not a committed design.

### 1. Infrared excess — from visualization to a quantitative product
Today ARIADNE only *visualizes* IR excess (`add_mag` + `SEDPlotter(ir_excess=True)`;
the raw-SED blackbody QC already circles >5σ outliers). Make it a measured,
automated catalogue column:
- **Per-band excess significance** from the BMA posterior model:
  `(F_obs − F_model)/σ` for the redward bands (WISE W1–W4, Spitzer, Herschel),
  with σ folding in the inter-grid/BMA spread. Flag significant *positive* excess.
- **Aggregate statistic**: fractional excess luminosity `L_IR/L_*` and/or an
  excess χ² over the IR bands → one catalogue flag + magnitude.
- **Characterize it**: fit the residual either as a cool blackbody (debris-disk
  dust, T~100–500 K → disk T and `L_dust/L_*`) *or* as a cool photospheric
  companion (reuse the new binary two-component machinery). Disk-vs-companion
  discrimination is the actual science (single-T greybody vs a stellar SED).
- **Hooks**: automated debris-disk / dusty-WD / unresolved-cool-companion
  flagging at catalogue scale. Dusty white dwarfs (DZ + disk excess) tie
  straight into the new Koester grid.
- **Gotchas**: WISE blends/contamination drive false positives (the red-circle
  QC helps); upper limits need real handling; trustworthy only within the grids'
  IR coverage.

### 2. Bolometric flux as a first-class output
ARIADNE already integrates the model SED to a luminosity (`lum` = 4πd²·F_bol),
so F_bol is computed but not exposed. Surface **F_bol (erg/s/cm²) with posterior
uncertainties** directly:
- **Why**: F_bol is *distance-independent* — a direct SED observable, unlike L.
- **Killer app**: F_bol + an angular diameter θ (interferometry, or the Gaia
  radius) → a near-empirical Teff via Stefan–Boltzmann (the bolometric-flux /
  IRFM Teff), a model-light cross-check on the SED Teff. Also gives per-band
  bolometric corrections and a transferable column for external cross-validation.
- **How**: integrate each posterior draw's full model spectrum (the spectra
  cache ARIADNE already builds for plotting) over λ → an F_bol distribution.
- **Gotcha — the bolometric tail** (cf. the XP note above): flux outside the
  observed bands (UV for hot stars, IR for cool) comes from the model
  extrapolation, bounded by grid coverage (Koester stops at 3 µm, Coelho 0.9 µm).
  Report the tail fraction so F_bol's model-dependence is explicit per star.

Both are literally catalogue columns (an IR-excess flag/strength + F_bol and an
empirical Teff), so they slot into the "ARIADNE II = the catalogue" framing
rather than the (now-parked) XP thesis above.

---

## Spectra-ingestion modules (tasks #20-23) — design pass, 2026-06-12

All four follow the established ruling: spectra enter as **gap-filling
synthetic photometry in bounded band sets**, never as sampled spectra. The XP
gap-filler and the SPHEREx super-bands are the two existing instances of the
pattern; each module below is one of those two shapes.

### Recommended order

**1. DESI MWS + SDSS-V DR19 BOSS (#23) — XP-shaped, no registry change.**
Optical spectra (DESI: 360-980 nm, R~2000-5000, ~4M stellar spectra in DR1
via SPARCL; BOSS: 365-1040 nm) synthesized through the EXISTING optical band
curves (SDSS griz, PS1, Johnson-Cousins). Two roles, both proven by XP:
(a) gap filler for **G > 16** — exactly where the XP fill cuts off, so the
two modules are complementary by construction (DESI MWS targets 16 < r < 19);
(b) spectrum-vs-catalog QC for split-crossmatch detection, reusing the
existing QC machinery verbatim. Needs an offsets/floors calibration campaign
per release (the XP campaign template). No new bands, no grid work, no yuki.

**2. CASSIS / Spitzer-IRS (#20) — SPHEREx-shaped, registry +~10 bands.**
Low-res IRS 5.2-38 um spectra (~16k objects with staring obs, including most
bright debris-disk hosts — i.e. the IRX validation roster). Super-band
top-hats in the IRS range, e.g. IRS_5500/6300/7500/9000 (silicate)/11000/
13000/16000/19000/24000/30000 — final windows to be set against the SL/LL
order edges and the silicate features. All >= 5 um -> irx_bands (excess set)
per the wavelength rule. This turns a 2-4-band excess into a 10-15-band
spectral shape with the 10-um silicate feature as an unambiguous
disk-vs-companion discriminator. Cost: the full band-extension cycle
(custom_filters + grid columns via cache augment + yuki raw pass + tests);
the SPHEREX_1050 addition just rehearsed the whole pipeline in an afternoon.

**3. IUE / INES (#21) — niche but unique.**
1150-3350 A spectra of ~10^4 bright stars. Value: the UV anchor for hot
stars and WD companions where GALEX SATURATES (bright-star UV is otherwise
unobtainable). Shape: synthesize the existing GALEX FUV/NUV bands from the
IUE spectrum (no registry change for a first pass; optional 2-3 UV top-hats
later). Gate to fits whose photosphere reaches the UV (koester/tlusty/
hot kurucz). Small module, clear niche.

**4. AKARI IRC slitless (#22) — DEPRIORITIZE, likely drop.**
2.5-5 um slitless spectra of bright stars. SPHEREx now covers the same range
deeper, all-sky, with our extraction validated end-to-end. The only residual
niche is stars too bright for SPHEREx (saturation, Ks < 8.5 window edge) —
worth a one-day look only after 1-3 ship, or drop.
