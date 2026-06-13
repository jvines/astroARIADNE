# astroARIADNE

**astroARIADNE** (spectrAl eneRgy dIstribution bAyesian moDel averagiNg fittEr)
characterizes stellar atmospheres by fitting broadband photometry to grids of
stellar atmosphere models with nested sampling, and combines several model
grids through Bayesian Model Averaging (BMA).

It assembles the SED automatically from catalog photometry and gap-fills it
with synthetic photometry from Gaia XP, SPHEREx, DESI/BOSS, Spitzer-IRS and IUE
(see {doc}`data_sources`), fits through four explicit endpoints —
`fit_bma`, `fit_single`, `fit_binary` and `fit_irx` — and characterizes any
infrared excess by Bayesian model selection (see {doc}`irx`).

```{toctree}
:maxdepth: 2
:caption: Contents

installation
theory
usage
data_sources
output
irx
examples
customization
filters
api
changelog
citing
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
