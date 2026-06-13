# ARIADNE examples

One ready-to-run script per operating mode — run any of them and you should
reproduce the corresponding figure in the documentation!

| Script | Mode | Star | What it demonstrates |
|---|---|---|---|
| `bma_basic.py` | `fit_bma()` | WASP-19 | The default: BMA over the full Teff-gated ensemble |
| `single_coelho.py` | `fit_single('coelho')` | tau Ceti | Optical-only grid + automatic IR exclusion + Av auto-pin (<70 pc) |
| `single_tlusty.py` | `fit_single('tlusty')` | tau Sco | Hot-star NLTE grid + passing a parallax when Gaia has none |
| `single_koester.py` | `fit_single('koester')` | GD 71 | White-dwarf mode with automatic WD priors |
| `irx_disk.py` | `fit_irx()` | eps Eri | IR-excess: cold debris disk detected and characterized |
| `irx_companion.py` | `fit_irx()` | SCR 1845-6357 | IR-excess: unresolved T6 brown-dwarf companion classified as such |

All scripts fetch their photometry through the librarian (internet required)
and write into a folder named after the star in your working directory.
Expect a few minutes each — most of it is catalog queries, the fits
themselves are fast.
