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
- **$\log g$, $[\mathrm{Fe/H}]$** — spectroscopic priors when available
  (APOGEE > GALAH > LAMOST > RAVE > Hypatia), otherwise broad bounds.
- **distance** — the Bailer-Jones geometric distance posterior.
- **radius / normalisation, $A_V$** — uniform, with $A_V$ capped at the
  line-of-sight galactic maximum.
- **jitter $s_i$** — broad, weakly informative.

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
and observable set, so their evidences are comparable. Limited-coverage or
intrinsically standalone grids (e.g. Coelho, SPHINX-II, TLUSTY) are run on their
own rather than mixed into the average — see {doc}`guide`.
```
