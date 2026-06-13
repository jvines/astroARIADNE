# Theoretical background

This page sets out the model **astroARIADNE** actually fits — the SED forward
model, the likelihood, the priors, the nested-sampling evidence, and the
Bayesian Model Averaging (BMA) that combines several atmosphere grids. The
notation matches the implementation in
{doc}`sed_library <api>`; equation references point at the functions that
evaluate them.

## The spectral energy distribution

A star's spectral energy distribution (SED) is its emergent flux as a function
of wavelength, $F_\lambda$. We never observe $F_\lambda$ directly: a broadband
photometric measurement in filter $i$ with transmission $T_i(\lambda)$ returns
the band-integrated flux

$$
F_i = \frac{\int F_\lambda\, T_i(\lambda)\, \lambda\, d\lambda}
            {\int T_i(\lambda)\, \lambda\, d\lambda},
$$

i.e. a photon-counting (energy) average of $F_\lambda$ over the bandpass. A set
of magnitudes across many filters samples the SED at a handful of effective
wavelengths. Fitting the SED means finding the stellar (and nuisance)
parameters whose predicted band fluxes reproduce those measurements.

`phot_utils` handles the conversion from catalogue magnitudes to the flux
densities $F_i^{\mathrm{obs}}$ (with uncertainties $\sigma_i$) that the
likelihood compares against, using per-filter zero-points and effective
wavelengths.

## The forward model

The flux predicted in band $i$ is built from a **stellar atmosphere grid**.
Atmosphere models (PHOENIX, BT-Settl, Castelli–Kurucz, BOSZ, …) give the
surface flux of a star as a function of effective temperature $T_{\mathrm{eff}}$,
surface gravity $\log g$ and metallicity $[\mathrm{Fe/H}]$. ARIADNE stores each
grid **pre-convolved through every supported bandpass**, so a grid node is a
vector of band fluxes $f_i(T_{\mathrm{eff}}, \log g, [\mathrm{Fe/H}])$ rather
than a spectrum. Arbitrary parameter values are obtained by **trilinear
interpolation** in $(\log g, T_{\mathrm{eff}}, [\mathrm{Fe/H}])$
(`get_interpolated_flux`).

Three physical effects turn the surface flux into the flux received at Earth:

1. **Geometric dilution.** A star of radius $R$ at distance $d$ subtends a solid
   angle $\propto (R/d)^2$, so the received flux scales by $(R/d)^2$.
2. **Interstellar extinction.** Dust attenuates the flux by
   $10^{-0.4\,A(\lambda)}$. ARIADNE uses a fixed reddening law with
   $R_V = 3.1$, parameterised by the $V$-band extinction $A_V$. Because the law
   is linear in $A_V$ at fixed $R_V$, the per-band attenuation curve is computed
   once at $A_V = 1$ and raised to the power $A_V$ inside `model_grid`.
3. **Filter integration**, already baked into the grid.

The model band flux is therefore (`model_grid`)

$$
F_i^{\mathrm{model}} =
  \left(\frac{R}{d}\right)^{2}
  f_i(T_{\mathrm{eff}}, \log g, [\mathrm{Fe/H}])\;
  10^{-0.4\, A_V\, k_i},
\qquad
k_i \equiv \frac{A(\lambda_i)}{A_V},
$$

with $R$ and $d$ both in solar radii (the distance is converted from parsecs
internally). In **normalisation mode** (`use_norm=True`) the degenerate factor
$(R/d)^2$ is replaced by a single free scale $N$:
$F_i^{\mathrm{model}} = N\, f_i\, 10^{-0.4 A_V k_i}$. This is the right choice
when no reliable parallax is available, at the cost of not constraining $R$ and
$d$ separately.

### Parameters

The fitted vector $\theta$ is

$$
\theta = \bigl(T_{\mathrm{eff}},\ \log g,\ [\mathrm{Fe/H}],\
\underbrace{d,\ R}_{\text{or } N},\ A_V,\ \{s_i\}\bigr),
$$

where $\{s_i\}$ are **per-band excess-noise (jitter) terms** described below.
Any parameter can be fixed; the "coordinator" array tracks which entries are
free vs. held constant (`build_params`).

## The likelihood

Band measurements are treated as independent Gaussians. To absorb
underestimated catalogue errors and mild model inadequacy, each band carries an
excess-noise term $s_i$ added in quadrature to its reported error. The
log-likelihood is (`log_likelihood` / `fast_loglik`)

$$
\ln \mathcal{L}(\theta) =
-\frac{1}{2} \sum_i \left[
  \frac{\bigl(F_i^{\mathrm{obs}} - F_i^{\mathrm{model}}(\theta)\bigr)^2}
       {\sigma_i^2 + s_i^2}
  + \ln\!\bigl(2\pi (\sigma_i^2 + s_i^2)\bigr)
\right].
$$

The $\ln(2\pi(\sigma_i^2+s_i^2))$ term is essential: without it the sampler
would drive every $s_i \to \infty$ to flatten the residuals. Non-finite
evaluations (out-of-grid interpolation returns `NaN`) are mapped to a large
negative value so the sampler rejects them.

## Priors

Bayes' theorem gives the posterior over parameters for a fixed grid $G$,

$$
p(\theta \mid D, G) =
\frac{\mathcal{L}(\theta)\, \pi(\theta)}{Z_G},
\qquad
Z_G = \int \mathcal{L}(\theta)\, \pi(\theta)\, d\theta,
$$

where $\pi(\theta)$ is the prior and $Z_G$ the **evidence** (marginal
likelihood). ARIADNE's defaults:

- **$T_{\mathrm{eff}}$** — a population prior built from a RAVE-derived
  temperature distribution (an empirical spline), or a spectroscopic prior when
  one is supplied.
- **$\log g$** — a star-specific spectroscopic prior when available, otherwise a
  MIST isochrone estimate, otherwise broad uniform bounds
  $\mathcal{U}(3.5, 6.0)$.
- **$[\mathrm{Fe/H}]$** — a spectroscopic prior when available, otherwise a
  population normal $\mathcal{N}(-0.125, 0.234)$.
- **spectroscopic source chain** — star-specific $T_{\mathrm{eff}}/\log g$ are
  taken from the first survey that resolves the target,
  PASTEL > APOGEE DR17 > GALAH DR3 > RAVE DR6 > LAMOST DR11; $[\mathrm{Fe/H}]$
  defaults to the Hypatia Catalog (Hinkel et al. 2014), falling back to that
  same survey chain on a Hypatia miss.
- **distance** — the Bailer-Jones geometric distance posterior.
- **radius / normalisation, $A_V$** — uniform, with $A_V$ capped at the
  line-of-sight galactic maximum.
- **jitter $s_i$** — broad, weakly informative.

### Cool dwarfs

The RAVE-derived $T_{\mathrm{eff}}$ spline and the broad $\log g$ bounds above
are calibrated on FGK stars and misbehave for M dwarfs: the temperature prior
biases them several hundred K hot, and the wide gravity prior leaves a spurious
low-$\log g$ (giant) mode open. ARIADNE handles this automatically in two nested
steps (`_default_priors` / `_cool_dwarf_flags`):

- A **cool star** — Gaia colour $\mathrm{BP}-\mathrm{RP} > 1.8$, or Gaia
  $T_{\mathrm{eff}} < 4000\ \mathrm{K}$ when colour is unavailable — gets a
  uniform M/late-K temperature prior,
  $T_{\mathrm{eff}} \sim \mathcal{U}(2300, 4200)\ \mathrm{K}$, in place of the
  FGK population spline.
- A **cool dwarf** — a cool star that is also on the main sequence, flagged by
  absolute magnitude $M_G > 8$ (or luminosity $< 0.5\,L_\odot$ as a fallback) —
  *additionally* gets a tight main-sequence gravity prior,
  $\log g \sim \mathcal{N}(5.0, 0.25)$, which is what closes the giant mode.

A star-specific spectroscopic $T_{\mathrm{eff}}/\log g$ (from the survey chain
above) always overrides the automatic choice.

Nested sampling draws from the prior by mapping a unit-cube variate $u\in[0,1]$
through the inverse CDF (`prior_transform_dynesty`). ARIADNE substitutes
closed-form inverse CDFs for the uniform/normal/truncated-normal cases, which is
exact and far faster than evaluating `scipy` frozen-distribution `.ppf` on every
proposal.

## Nested sampling and the evidence

ARIADNE samples with [`dynesty`](https://dynesty.readthedocs.io). Nested
sampling reparameterises the evidence integral over the **prior mass**
$X(\lambda) = \int_{\mathcal{L}(\theta) > \lambda} \pi(\theta)\, d\theta$,
turning the multi-dimensional integral into a one-dimensional one,

$$
Z = \int_0^1 \mathcal{L}(X)\, dX,
$$

and accumulates it by repeatedly replacing the lowest-likelihood live point
with a higher-likelihood draw from the constrained prior. It returns **both**
the evidence $Z$ (with an uncertainty) and a set of posterior samples as a
by-product — which is exactly what BMA needs. Sampling stops when the estimated
remaining evidence falls below the `dlogz` tolerance.

The evidence is the quantity that penalises unnecessary model complexity (an
automatic Occam factor): a grid that fits well *without* fine-tuning its
parameters earns a higher $Z$ than one that only fits over a sliver of its prior
volume.

## Bayesian Model Averaging

No single atmosphere grid is correct everywhere — they differ in line lists,
convection treatment, opacities and wavelength coverage. Conditioning on one
grid hides that systematic uncertainty. BMA instead treats the **choice of
grid** as another thing to marginalise over.

Run the fit independently against each grid $G_k$ in the BMA set, obtaining an
evidence $Z_k$ and a posterior $p(\theta \mid D, G_k)$. With equal prior
probability across the $M$ grids, $P(G_k) = 1/M$, the **posterior probability of
each grid** is

$$
w_k \equiv P(G_k \mid D)
= \frac{Z_k\, P(G_k)}{\sum_j Z_j\, P(G_j)}
= \frac{Z_k}{\sum_j Z_j}.
$$

In practice these are computed from the log-evidences and a numerically stable
shift, $w_k \propto \exp(\ln Z_k - \min_j \ln Z_j)$, then normalised
(`bayesian_model_average`). The model-averaged posterior of any quantity
$\vartheta$ is the evidence-weighted mixture of the per-grid posteriors,

$$
p(\vartheta \mid D) = \sum_{k=1}^{M} w_k\; p(\vartheta \mid D, G_k).
$$

ARIADNE realises this two ways: a **weighted resampling** of all per-grid chains
into a master posterior (then summarised via KDE), and a **weighted average** of
the per-grid posteriors supersampled to a common length. Either way the reported
parameter uncertainties absorb the spread *between* grids, not just the
statistical scatter *within* one — the central reason to prefer BMA over picking
a single best-fitting atmosphere.

```{note}
BMA only makes sense when the grids being averaged share the same prior volume
and observable set, so their evidences are comparable. ARIADNE enforces this by
Teff-gating each grid to the regime where it is valid: the default ensemble
(PHOENIX, BT-Settl/NextGen/Cond/Dusty, Kurucz, Castelli-Kurucz, BOSZ, SPHINX,
STAGGER and Coelho) is filtered per target, so e.g. SPHINX contributes
only below 4000 K, the BT dust variants (BT-NextGen, BT-Cond, BT-Dusty) only
below 2600 K -- above the dust-condensation regime they are identical to
BT-Settl by construction (NextGen/Cond, verified on the source data) or
differ only by opacity vintage (Dusty) -- Kurucz/Castelli-Kurucz only above
4000 K, and STAGGER only between 4000 and 7000 K. Grids that cannot satisfy the
shared-prior-volume requirement are kept out of the average and run on their
own instead: TLUSTY (hot stars, which need their own uniform Teff prior) and
Koester (white dwarfs, with WD-specific priors and no metallicity axis).
```

## Unresolved binaries (two-component SED)

An unresolved binary contributes the **summed** flux of two stars that share a
distance, line-of-sight extinction and (to the precision the photometry can
distinguish) metallicity. ARIADNE's binary mode replaces the single-component
forward model with

$$
F_i^{\mathrm{model}} =
\left[\left(\frac{R_1}{d}\right)^{2} f_i(T_1, \log g_1, z)
    + \left(\frac{R_2}{d}\right)^{2} f_i(T_2, \log g_2, z)\right]
10^{-0.4\, A_V\, k_i},
$$

adding $(T_2, \log g_2, R_2)$ to $\theta$ while $d$, $A_V$ and $z$ stay shared
(`model_grid`, `binary=True`). The two components are exchange-symmetric, which
would split the posterior into two mirror modes; ARIADNE breaks the degeneracy by
**folding the prior** to enforce $T_1 \ge T_2$ — the prior transform swaps the
two component blocks whenever a draw violates the ordering
(`prior_transform_dynesty` / `_binary_swap_indices`). Folding (rather than
rejecting) keeps the prior volume — and therefore the evidence — clean, and it
requires the two components to share priors, which the defaults guarantee.

### The contrast constraint

Photometry alone barely constrains a faint secondary, so binary mode **requires**
at least one flux-ratio measurement — a magnitude difference
$\Delta m_i = |m_2 - m_1|$ in some band (speckle/AO imaging, or Gaia). Each is a
Gaussian term appended to the log-likelihood (`contrast_loglike`),

$$
\ln\mathcal{L}_{\Delta m} =
-\frac{1}{2}\sum_i
\left(\frac{\Delta m_i^{\mathrm{model}} - \Delta m_i}{\sigma_{\Delta m_i}}\right)^2,
\qquad
\Delta m_i^{\mathrm{model}} =
2.5\left|\log_{10}\!\left[\left(\frac{R_2}{R_1}\right)^{2}
\frac{f_i(T_2,\log g_2,z)}{f_i(T_1,\log g_1,z)}\right]\right|,
$$

in which the shared distance and extinction cancel — the contrast measures the
intrinsic flux ratio. A **single** contrast band pins the band dilution but
leaves $T_2$ and $R_2$ on a degenerate ridge; ARIADNE flags a prior-dominated
secondary as `binary_secondary_unconstrained` when the recovered $T_2$ posterior
scatter exceeds 15 % of its median ($\sigma_{T_2}/\mathrm{med}(T_2) > 0.15$),
the usual signature of a single-band fit. Two bands of different colour break the
degeneracy.

### Did the SED need two stars?

The contrast likelihood will always reward adding a second component, so a fit
that includes it cannot be used to decide *whether* the source is a binary.
ARIADNE answers that question with a separate, fair comparison: it re-fits both
the single-star and the binary model on the **photometry only** (no contrast
term), and reports the evidence difference

$$
\Delta\ln Z = \ln Z_{\mathrm{binary}}^{\mathrm{SED}} - \ln Z_{\mathrm{single}},
$$

in `binary_vs_single` (one such dict per grid, written into each
`{grid}_out.pkl`). $\Delta\ln Z > 0$ means the broadband SED *by itself* prefers
two temperatures; the automatic Occam factor of the evidence (see above)
penalises the three extra parameters, so a positive value is a genuine detection
rather than a consequence of the larger model.

### Dilution

The quantity most often wanted downstream is the **host-flux fraction** in each
band — the fraction of the *received* (radius-weighted) flux contributed by the
primary,

$$
D_i = \frac{R_1^{2} f_i(T_1,\log g_1,z)}
           {R_1^{2} f_i(T_1,\log g_1,z) + R_2^{2} f_i(T_2,\log g_2,z)},
$$

the Furlan & Howell third-light correction needed to recover a transiting
planet's true radius from a diluted transit depth (the common distance and
extinction cancel in the ratio). It is reported per band in
`out['binary_dilution']`, with full posterior traces (`dilution_<band>`), and
under BMA those traces are averaged across grids so the dilution carries the
model-systematic spread.
