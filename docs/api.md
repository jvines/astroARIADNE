# API Reference

The package splits into a **high-level interface** — the four classes you
instantiate directly — and a **computational core** of module-level functions
that implement the forward model, photometry, isochrone fit and helpers. The
core is documented here so the model in {doc}`theory` can be traced straight to
the code that evaluates it.

## High-level interface

### Fitter

```{eval-rst}
.. autoclass:: astroARIADNE.fitter.Fitter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Star

```{eval-rst}
.. autoclass:: astroARIADNE.star.Star
   :members:
   :undoc-members:
   :show-inheritance:
```

### SEDPlotter

```{eval-rst}
.. autoclass:: astroARIADNE.plotter.SEDPlotter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Librarian

The photometry/astrometry retrieval layer. `Librarian` resolves a target and
gathers catalogue photometry; `adapt_librarian` bridges its result to the
`Star`/`Fitter` parameter conventions.

```{eval-rst}
.. autoclass:: astroARIADNE.librarian.Librarian
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: astroARIADNE.librarian._adapter.adapt_librarian
```

## Computational core

### sed_library — model, likelihood and priors

The forward model and Gaussian likelihood evaluated on every nested-sampling
proposal, plus the unit-cube prior transforms. See {doc}`theory` for the
equations these implement.

```{eval-rst}
.. automodule:: astroARIADNE.sed_library
   :members:
   :undoc-members:
```

### phot_utils — photometric conversions

Magnitude/flux conversions, zero-points and bandpass lookups used to turn
catalogue magnitudes into the flux densities the likelihood compares against.

```{eval-rst}
.. automodule:: astroARIADNE.phot_utils
   :members:
   :undoc-members:
```

### isochrone — MIST age/mass estimation

The post-fit step that maps the sampled (Teff, logg, [Fe/H], radius) posterior
onto MIST tracks for age, mass and EEP.

```{eval-rst}
.. automodule:: astroARIADNE.isochrone
   :members:
   :undoc-members:
```

### utils — helpers

Credibility intervals, KDE summaries, and the formatted output routines.

```{eval-rst}
.. automodule:: astroARIADNE.utils
   :members:
   :undoc-members:
```
