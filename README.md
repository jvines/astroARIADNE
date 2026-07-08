# ARIADNE (spectrAl eneRgy dIstribution bAyesian moDel averagiNg fittEr)

[![Documentation Status](https://readthedocs.org/projects/astroariadne/badge/?version=latest)](https://astroariadne.readthedocs.io/en/latest/?badge=latest)

## Characterize stellar atmospheres easily!
**ARIADNE** Is a code written in Python 3.11+ designed to fit broadband
photometry to different stellar atmosphere models automatically using Nested
Sampling algorithms.

📖 **Documentation:** <https://astroariadne.readthedocs.io>


# Installation

<!-- DOC:INSTALLATION-START -->

**ARIADNE** requires Python 3.11 or higher and uses modern Python packaging standards (PEP 517/518).

## Quick Install (Recommended)

Install the latest stable version from PyPI:

```bash
pip install astroariadne
```

## ARIADNE 2 (pre-release)

ARIADNE 2 — the IR-excess (disk vs. companion) module, the L/T-capable cool
ensemble, the rebuilt BT-Dusty/Sonora grids, and the XP/SPHEREx/CASSIS
spectro-photometry fills — is **not yet on PyPI**. Install it, all model grids
included, straight from the development branch:

```bash
pip install git+https://github.com/jvines/astroARIADNE@v2
```

Then pull the spectra cache once for SED plotting (see *Spectra Cache* below):

```bash
python -c "from astroARIADNE.fetch import fetch_spectra_cache; fetch_spectra_cache()"
```

## Development Install

To install from source for development or to use the latest unreleased features:

```bash
git clone https://github.com/jvines/astroARIADNE.git
cd astroARIADNE
pip install -e .
```

The `-e` flag installs in editable mode, allowing you to modify the source code.

## Dependencies

All required dependencies are automatically installed via pip. The main packages include:

- Numpy (<https://numpy.org/>)
- Scipy (<https://www.scipy.org/>)
- Pandas (<https://pandas.pydata.org/>)
- numba (<http://numba.pydata.org/>)
- astropy (<https://astropy.readthedocs.io/en/stable/>)
- astroquery (<https://astroquery.readthedocs.io/en/latest/>)
- regions (<https://astropy-regions.readthedocs.io/en/latest/index.html>)
- PyAstronomy (<https://pyastronomy.readthedocs.io/en/latest/>)
- corner (<https://corner.readthedocs.io/en/latest/>)
- tqdm (<https://tqdm.github.io/>)
- matplotlib (<https://matplotlib.org/>)
- termcolor (<https://pypi.org/project/termcolor/>)
- extinction (<https://extinction.readthedocs.io/en/latest/>)
- pyphot (<http://mfouesneau.github.io/docs/pyphot/>)
- dustmaps (<https://dustmaps.readthedocs.io/en/latest/>) [**NEEDS CONFIGURING AND DOWNLOADING OF DUSTMAPS**]
- dynesty (<https://dynesty.readthedocs.io/en/latest/>)
- isochrones (<https://isochrones.readthedocs.io/en/latest/>) [**NEEDS EXTRA SETUP WITH `nosetests isochrones`**]

### Special Installations

#### dustmaps Configuration

After installing ARIADNE, you must download and configure dustmaps:

```python
# Download SFD dust map
import dustmaps.sfd
dustmaps.sfd.fetch()

# Download Bayestar dust map
import dustmaps.bayestar
dustmaps.bayestar.fetch()
```

See the [dustmaps documentation](https://dustmaps.readthedocs.io/en/latest/) for other available dust maps.

#### isochrones Setup

The isochrones package requires additional setup after installation:

```bash
# This downloads required stellar evolution models
python -c "from isochrones import get_bc_grid; get_bc_grid('mist')"
```

Or run the test suite which handles setup automatically:
```bash
nosetests isochrones
```

## Platform Support

**ARIADNE** has been tested on:
- **Linux** (fully supported)
- **macOS** (fully supported, tested up to Catalina and later)
- **Windows** (limited support - healpy dependency may cause issues, see [healpy/healpy#25](https://github.com/healpy/healpy/issues/25))

## Build System

ARIADNE uses `pyproject.toml` for modern, declarative package configuration. Dependency installation and version management are handled automatically by pip/setuptools.

## Model Spectra for SED Plotting

Plotting the SED model is optional — you can run the fitting code without any
model spectra. If you want SED plots, there are two options:

### Spectra Cache (Recommended)

Download the pre-computed spectra cache (~3.2 GB) from Zenodo. This contains
every model grid in the ensemble — Phoenix v2, BT-Settl (+ its cool
extension), BT-NextGen, BT-Cond, BT-Dusty, Castelli & Kurucz, Kurucz, Coelho,
BOSZ, SPHINX, TLUSTY, STAGGER, Koester, ATMO 2020, Sonora and PHOENIX-NewEra —
broadened to R=1500 and resampled to a common wavelength grid. This is all you
need for `plot_SED()`. The v2.0 cache lives at
[Zenodo record 21251184](https://doi.org/10.5281/zenodo.21251184) (the concept
DOI [10.5281/zenodo.19141565](https://doi.org/10.5281/zenodo.19141565) always
resolves to the latest version); `fetch_spectra_cache()` downloads it and
verifies the SHA-256.

```python
from astroARIADNE.fetch import fetch_spectra_cache
fetch_spectra_cache()
```

Or from the command line:

```bash
python -c "from astroARIADNE.fetch import fetch_spectra_cache; fetch_spectra_cache()"
```

Once downloaded the cache is found automatically. To use a cache stored
elsewhere — a shared/NAS copy, or one you have extended with additional grids
(e.g. BOSZ, SPHINX, TLUSTY) — point ARIADNE at it with the
`ARIADNE_SPECTRA_CACHE` environment variable:

```bash
export ARIADNE_SPECTRA_CACHE=/path/to/spectra_cache.h5
```

### Full Model Grids (Optional)

If you need the raw high-resolution model spectra (e.g. for custom broadening
or direct spectral analysis), you can download the full grids. Note that these
total several hundred GB.

| Model        | Link           |
| ------------- |:-------------:|
| Phoenix v2      | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/> |
| Phoenix v2   wavelength file   | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits> |
| BT-Models      | <http://svo2.cab.inta-csic.es/theory/newov2/>  |
| Castelli & Kurucz | <http://ssb.stsci.edu/cdbs/tarfiles/synphot3.tar.gz>      |
| Kurucz 1993 | <http://ssb.stsci.edu/cdbs/tarfiles/synphot4.tar.gz>  |

The wavelength file for the Phoenix model has to be placed in the root folder
of the PHOENIXv2 models.

To tell **ARIADNE** where to find the models, set the `ARIADNE_MODELS`
environment variable:

```bash
export ARIADNE_MODELS='/path/to/Models_Dir/'
```

You can add this to your `.bash_profile` or `.bashrc` so you don't have to
export every time.

The directory structure should look like this:

```
Models_Dir
│
└───BTCond
│   │
│   └───CIFIST2011
│
└───BTNextGen
│	 │
│	 └───AGSS2009
│
└───BTSettl
│	 │
│	 └───AGSS2009
│
└───Castelli_Kurucz
│	 │
│	 └───ckm05
│	 │
│	 └───ckm10
│	 │
│	 └───ckm15
│	 │
│	 └───ckm20
│	 │
│	 └───ckm25
│	 │
│	 └───ckp00
│	 │
│	 └───ckp02
│	 │
│	 └───ckp05
│
└───Kurucz
│	 │
│	 └───km01
│	 │
│	 └───km02
│	 │
│	 └───km03
│	 │
│	 └───km05
│	 │
│	 └───km10
│	 │
│	 └───km15
│	 │
│	 └───km20
│	 │
│	 └───km25
│	 │
│	 └───kp00
│	 │
│	 └───kp01
│	 │
│	 └───kp02
│	 │
│	 └───kp03
│	 │
│	 └───kp05
│	 │
│	 └───kp10
│
└───PHOENIXv2
	 │
     └─── WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
	 └───Z-0.0
	 │
	 └───Z-0.5
	 │
	 └───Z-1.0
	 │
	 └───Z-1.5
	 │
	 └───Z-2.0
	 │
	 └───Z+0.5
	 │
	 └───Z+1.0
```

#### Notes:
- The Phoenix v2 models with alpha enhancements are unused
- BT-models are BT-Settl, BT-Cond, and BT-NextGen

<!-- DOC:INSTALLATION-END -->

# How to use?

<!-- DOC:USAGE-START -->

## Stellar information setup

To use **ARIADNE** start by setting up the stellar information, this is done by
importing the Star module.

```python
from astroARIADNE.star import Star
```

After importing, a star has to be defined.

Stars are defined in **ARIADNE** by their RA and DEC in degrees, a name, and
optionally the Gaia DR3 source id, for example:

```python
starname = 'WASP-19'
ra = 148.41676021592826
dec = -45.65910531582427
gaia_id = 5411736896952029568

s = Star(starname, ra, dec, g_id=gaia_id)
```
The starname is used purely for user identification later on, and the 
`gaia_id` is provided to make sure the automatic photometry retrieval collects
the correct magnitudes, otherwise **ARIADNE** will try and get the `gaia_id` by
itself using a cone search centered around the RA and DEC.

**Convenience constructors.** If you already have the Gaia DR3 source id — or
just the star's name — you can skip entering RA/Dec by hand and let **ARIADNE**
resolve the coordinates for you:

```python
s = Star.from_gaia(5411736896952029568)   # coordinates from Gaia DR3 TAP
s = Star.from_name('WASP-19')             # coordinates resolved via Simbad
```

Executing the previous block will start the photometry and stellar parameter
retrieval routine. **ARIADNE** will query Gaia DR2 for an estimate on the
temperature, radius, parallax and luminosity for display as preliminar
information, as it's not used during the fit, and prints them along with its
TIC, KIC IDs if any of those exist, its Gaia DR3 ID, and maximum line-of-sight
extinction Av:

```
                        Gaia DR2 ID : 5411736896952029568
			Gaia Effective temperature : 5458.333 +/- 109.667
			Gaia Stellar radius : 1.001 +/- 0.195
			Gaia Stellar Luminosity : 0.802 +/- 0.009
			Gaia Parallax : 3.751 +/- 0.024
			Bailer-Jones distance : 265.144 +/- 0.620
			Maximum Av : 0.581
```

If you already know any of those values, you can override the search for them by
providing them in the Star constructor with their respective uncertainties.
Likewise if you already have the magnitudes and wish to override the on-line
search, you can provide a dictionary where the keys are the filters and values
are the mag, mag_err tuples.

If you want to check the retrieved magnitudes you can call the `print_mags`
method from Star:

```python
s.print_mags()
```

This will print the filters used, magnitudes and uncertainties. For WASP-19 this
would look like this:

```
		       Filter       	Magnitude	Uncertainty
		--------------------	---------	-----------
		    SkyMapper_u     	 14.1050 	  0.0070
		    SkyMapper_v     	 13.7370 	  0.0060
		  GROUND_JOHNSON_B  	 16.7920 	  0.1820
		    SkyMapper_g     	 12.4320 	  0.0030
		    GaiaDR2v2_BP    	 12.5227 	  0.0017
		  GROUND_JOHNSON_V  	 16.0100 	  0.0000
		    SkyMapper_r     	 12.0630 	  0.0050
		    GaiaDR2v2_G     	 12.1088 	  0.0005
		    SkyMapper_i     	 11.8740 	  0.0050
		    GaiaDR2v2_RP    	 11.5532 	  0.0014
		    SkyMapper_z     	 11.8610 	  0.0080
		      2MASS_J       	 10.9110 	  0.0260
		      2MASS_H       	 10.6020 	  0.0220
		      2MASS_Ks      	 10.4810 	  0.0230
		    WISE_RSR_W1     	 10.4360 	  0.0230
		    WISE_RSR_W2     	 10.4940 	  0.0200
```
**Note:**  **ARIADNE** automatically prints and saves the used magnitudes and
filters to a file.

The way the photometry retrieval works is that Gaia DR2 crossmatch catalogs are
queried for the Gaia ID, these crossmatch catalogs exist for ALL-WISE, APASS,
Pan-STARRS1, SDSS, 2MASS and Tycho-2, so finding photometry relies on these
crossmatches. For example, if we were to analyze NGTS-6, there are Pan-STARRS1
photometry which **ARIADNE** couldn't find due to the Pan-STARRS1 source not
being identified in the Gaia DR2 crossmatch, in this case if you wanted to add
that photometry manually, you can do so by using the `add_mag` method from
Star, for example, if you wanted to add the PS1_r mag to our `Star` object 
you would do:

```python
s.add_mag(13.751, 0.032, 'PS1_r')
```

If for whatever reason **ARIADNE** found a bad photometry point, and you needed
to remove it, you can invoke the `remove_mag` method. For example, you wanted
to remove the TESS magnitude due to it being from a blended source, you can just
run

```python
s.remove_mag('NGTS')
```

In the specific example of WASP-19, we see that GROUND_JOHNSON_B and GROUND_JOHNSON_V
are likely not the correct photometry. Instead the correct ones are 13.054 and 12.248,
respectively.
We can correct this mistake:
```python
s.remove_mag('GROUND_JOHNSON_B')
s.remove_mag('GROUND_JOHNSON_V')
s.add_mag(13.054, 0.048, 'GROUND_JOHNSON_B')
s.add_mag(12.248, 0.069, 'GROUND_JOHNSON_V')
```

And a new call to `s.print_mags()` would yield:
```
		       Filter       	Magnitude	Uncertainty
		--------------------	---------	-----------
		    SkyMapper_u     	 14.1050 	  0.0070
		    SkyMapper_v     	 13.7370 	  0.0060
		  GROUND_JOHNSON_B  	 13.0540 	  0.0480
		    SkyMapper_g     	 12.4320 	  0.0030
		    GaiaDR2v2_BP    	 12.5227 	  0.0017
		  GROUND_JOHNSON_V  	 12.2480 	  0.0690
		    SkyMapper_r     	 12.0630 	  0.0050
		    GaiaDR2v2_G     	 12.1088 	  0.0005
		    SkyMapper_i     	 11.8740 	  0.0050
		    GaiaDR2v2_RP    	 11.5532 	  0.0014
		    SkyMapper_z     	 11.8610 	  0.0080
		      2MASS_J       	 10.9110 	  0.0260
		      2MASS_H       	 10.6020 	  0.0220
		      2MASS_Ks      	 10.4810 	  0.0230
		    WISE_RSR_W1     	 10.4360 	  0.0230
		    WISE_RSR_W2     	 10.4940 	  0.0200
```

A list of allowed filters can be found
[here](https://github.com/jvines/astroARIADNE/blob/v2/filters.md)

<!-- DOC:DATASOURCES-START -->
### Data sources & spectro-photometry

When a `Star` is created, **ARIADNE**'s librarian assembles the broadband SED
automatically. It pulls catalog photometry (Gaia, 2MASS, AllWISE, Pan-STARRS,
SkyMapper, SDSS, APASS, Strömgren, Tycho, GALEX, and the far-IR catalogs) and
then *gap-fills* with synthetic photometry derived from public spectroscopic
and spectro-photometric surveys. Every magnitude is added **first-write-wins**:
a real catalog measurement is never overwritten by a synthetic one, and the
synthetic sources only ever fill wavelengths the catalogs leave uncovered.

| Source | Range | Role | Opt out with |
|--------|-------|------|--------------|
| Catalog photometry | UV–far-IR | the SED backbone | per-catalog `ignore=[...]` |
| **Gaia XP** synthetic | 0.4–1.05 µm | fills optical gaps (Strömgren / JKC / SDSS / PS1 standardized systems) for `G ≤ 16` | `ignore=['XP']` |
| **DESI / BOSS** | 0.36–1.0 µm | the faint-end (`G > 16`) continuation of the optical fill | `ignore=['DESI']`, `ignore=['BOSS']` |
| **SPHEREx** QR2 | 1.0–5.0 µm | super-band spectro-photometry bridging the J/H/Ks/W1/W2 gaps (incl. the 2MASS–WISE void) | `ignore=['SPHEREX']` |
| **CASSIS / Spitzer-IRS** | 5–38 µm | IR-excess super-bands (10/18 µm silicate features) — never fit, only quantified as excess | `ignore=['CASSIS']` |
| **IUE / INES** | 0.12–0.33 µm | GALEX FUV/NUV for bright stars where GALEX saturates (replaces the saturated catalog value) | `ignore=['IUE']` |

Each synthetic band is convolved through the same filter curves the fit uses,
carries a measured calibration floor on its uncertainty, and is announced in
the log as it is added, e.g.

```
*** Gaia XP synthetic photometry filled 9 uncovered band(s): STROMGREN_v,
    STROMGREN_b, GROUND_JOHNSON_B, GROUND_COUSINS_R, ... ***
*** SPHEREx QR2 spectro-photometry added 10 super-band(s): SPHEREX_1130(3),
    SPHEREX_1400(4), ... ***
```

The XP/SPHEREx fills also act as a **quality control** on the catalog: an
observed magnitude that disagrees with its own synthetic counterpart by more
than the tolerance (tightened for high-proper-motion stars, where multi-epoch
surveys split the source) is flagged as a wrong/split crossmatch and dropped.

**Turning the spectro-photometry off.** The whole synthetic layer is opt-out at
the fit level — `Fitter.spectra = False` drops every XP/DESI/BOSS fill and the
SPHEREx super-bands, leaving a catalog-only SED:

```python
f = Fitter()
f.star = s
f.spectra = False        # catalog photometry only (no XP / SPHEREx / spectra)
```

Why use the fills? On a sample of stars with directly-measured (interferometric)
temperatures, the XP optical fills tighten the recovered `Teff` posterior by
~35 % and can make otherwise unfittable, sparsely-observed stars fit at all —
see [the validation below](#validating-the-fills).
<!-- DOC:DATASOURCES-END -->

### Interstellar extinction

**ARIADNE** has an incorporated prior for the interstellar extinction in the
Visual band, $A_{\rm V}$ which consists of a uniform prior between 0 and the
maximum line-of-sight value provided by the
[SFD dust maps](https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S/abstract).
This, however, can be changed either by a custom prior (see Fitter setup) or by
changing the dustmap used. We provide following dustmaps:

- [SFD (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S/abstract)
- [Planck Collaboration (2013)](http://adsabs.harvard.edu/abs/2014A%26A...571A..11P)
- [Planck Collaboration (2016; GNILC)](https://ui.adsabs.harvard.edu/abs/2016A%26A...596A.109P/abstract)
- [Lenz, Hensley & Doré (2017)](https://arxiv.org/abs/1706.00011)
- [Bayestar (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G)

**These maps are all implemented through the
[dustmaps](https://dustmaps.readthedocs.io/en/latest/index.html) package and
need to be downloaded. Instructions to download the dustmaps can be found in
its documentation.**

To change the dustmap you need to provide the `dustmap` parameter to the `Star` constructor, for example:

```python
ra = 75.795
dec = -30.399
starname = 'NGTS-6'
gaia_id = 4875693023844840448

s = Star(starname, ra, dec, g_id=gaia_id, dustmap='Bayestar')
```

This concludes the stellar setup and now we're ready to set up the parameters
for the fitting routine.

## Fitter setup

In this section we'll detail how to set up the fitter for the Bayesian Model
Averaging (BMA) mode of **ARIADNE**. For single models the procedure is very
similar.

First, import the fitter from **ARIADNE**

```python
from astroARIADNE.fitter import Fitter
```

There are several configuration parameters we have to setup, the first one is
the output folder where we want **ARIADNE** to output the fitting files and
results, next we have to select the fitting engine (`'dynesty'`), number of
live points to use, evidence tolerance threshold, bounding method, sampling
method, threads, and whether to use the dynamic nested sampler. After selecting
all of those, we need to select the models we want to use and finally, we feed
them all to the fitter:

> **Note:** `'dynesty'` is the engine. The legacy `'multinest'` engine was
> **removed in 2.0** — it only ever supported single-grid fits (not BMA), and
> dynesty is faster besides. Always use `'dynesty'`.

```python
out_folder = 'your folder here'

engine = 'dynesty'
nlive = 500
dlogz = 0.5
bound = 'multi'
sample = 'rwalk'
threads = 4
dynamic = False

setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

# This is also ARIADNE's default ensemble, so setting `f.models` is optional —
# a BMA run uses exactly this list if you don't override it. Comment out any
# grid you don't want.
models = [
	'phoenix',
	'btsettl',
	'newera',    # PHOENIX-NewEra modern opacities, 2300-12000 K
	'btnextgen', # gas-phase BT; only below 2600 K (identical to BT-Settl above)
	'btcond',    # full-settling BT; only below 2600 K (same reason)
	'btdusty',   # max-dust BT limit; only below 2600 K (above it, no dust
	             # condenses and it is just an older BT opacity vintage)
	'kurucz',
	'ck04',
	'bosz',      # MARCS+ATLAS9/Synspec, 2800-16000 K
	'sphinx',    # M dwarfs; only contributes at Teff < 4000 K
	'stagger',   # 3D RHD; only contributes at 4000-7000 K
	'coelho'     # independent 1D LTE physics (Coelho 2014)
]

f = Fitter()
f.star = s
f.setup = setup
f.av_law = 'fitzpatrick'
f.out_folder = out_folder
f.bma = True
f.models = models
f.n_samples = 100000
```

**Note:** While you can always select all 11 models, **ARIADNE** has an
internal filter put in place in order to avoid having the user unintentionally
bias the results. BT-NextGen and BT-Cond are identical to BT-Settl (and to
each other) **by construction above ~2600 K** — verified on the source data:
18–20% apart at 2000 K, exactly converged at 2600 K — so both are used only
at/below 2600 K, the late-M/L regime where their dust treatments genuinely
diverge (they also serve the IR-excess **companion** hypothesis, where their
800 K reach brackets the L/T-transition physics). Kurucz and Castelli &
Kurucz are known to work poorly on stars with Teff < 4000 K, thus they
aren't used in that regime. BT-Dusty is likewise gated to <= 2600 K: above
the dust-condensation regime its differences from BT-Settl are an older
opacity vintage, not dust physics. SPHINX only matters for M dwarfs, so it
is used only at/below 4000 K and silently dropped above it;
STAGGER's 3D models cover FGK stars, so it contributes only between 4000 and
7000 K; Coelho is dropped below 3000 K. This is why listing every grid is
safe — each one contributes only where it is valid. The filter is **not**
user-overridable — it is what keeps one model
family from being counted several times in the average. If the star has no
catalogue Teff at all (e.g. no Gaia GSP-Phot entry), **ARIADNE** estimates one
by fitting a blackbody to the observed photometry — calibrated on the fly
against BT-Settl synthetic photometry in the star's own bands, since raw
blackbody fits run several hundred K cool on real SEDs — and gates on that
(a notice tells you the value; pass `temp` to the `Star` constructor to
override it).

**The full grid roster.** Everything **ARIADNE** ships or supports beyond the
classic PHOENIX/BT/Kurucz families — some are default-ensemble members,
some standalone:

| grid | regime | Teff (K) | use |
| --- | --- | --- | --- |
| `bosz` | FGK + warm | 2800–16000 | BMA member (independent MARCS+ATLAS9/Synspec physics) |
| `sphinx` | M dwarfs | 2000–4000 | BMA member (Teff-gated < 4000 K) — in the default ensemble |
| `tlusty` | hot O/B (NLTE) | 15000–55000 | standalone |
| `coelho` | FGK + warm | 3000–26000 | BMA member (independent 1D LTE physics) — in the default ensemble |
| `koester` | DA white dwarfs | 5000–80000 | standalone (no [Fe/H] axis; WD priors) |
| `newera` | FGKM (modern opacities) | 2300–12000 | **default BMA member** |

`bosz`, `sphinx` and `coelho` are part of the default BMA ensemble and are
temperature-gated to their validity ranges (SPHINX contributes only below
4000 K, Coelho is dropped below 3000 K), so you can leave them in the `models`
list for any target. Coelho's library spans 0.13–10 µm, which covers every
band the photosphere fit uses (the reddest fit bands are WISE W2 and IRAC
4.5 µm; everything redder belongs to the IR-excess set), and a cross-grid
audit shows its synthetic photometry agrees with BT-Settl/PHOENIX at the
1–3% level across the shared bands. `tlusty` is meant to be run **standalone**
(see below): it uses a wide uniform Teff prior because the FGK population
prior cannot reach the hot-star regime.

`koester` is a **DA white-dwarf** grid (Koester 2010): a perfectly rectilinear
82 × 13 (Teff, log g) cube with **no metallicity axis** (a single pinned
[Fe/H]=0 node). It runs standalone with white-dwarf-appropriate priors — uniform
Teff over 5000–80000 K and uniform log g over 6.5–9.5 — and its ~3 µm red cutoff
drops WISE/Spitzer (a featureless Rayleigh-Jeans tail there). It lets ARIADNE
fit isolated white dwarfs and WD components, a regime no other grid reaches.

`newera` is **PHOENIX-NewEra** (Husser et al. 2025), the modern-opacity
successor to the bundled ACES PHOENIX grid, with rebuilt line lists and [M/H]
down to −4.0. The grid is too large (tens of GB of spectra) to ship, so it is
**not pre-built**: stage the spectra and build the DataFrame with
`notebooks/download_newera.py` + `notebooks/build_model_grids.py newera`, then
add `'newera'` to your `models` list. It is BMA-capable across the full FGKM
range (it shares ACES's axes); take the α=0 slice, which the builder does
automatically.

### Running a single model (standalone)

To fit a single grid instead of doing BMA, use the dedicated `fit_single`
endpoint — it's one call, no flags to juggle, and there's no ambiguity about
what's running:

```python
f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = out_folder
f.prior_setup = {
	'teff': ('default'), 'logg': ('default'), 'z': ('default'),
	'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.fit_single('phoenix')   # or 'bosz', 'sphinx', 'tlusty', 'coelho', ...
```

This is the right mode for the standalone-only grids `tlusty` (hot stars) and
`koester` (white dwarfs); `sphinx` and `coelho` can also be run this way for a
single-physics fit, though both now participate in the default BMA ensemble.
The output goes to the same files as a BMA run minus the model-averaging
products.

Every case below ships as a ready-to-run script in
[`examples/`](https://github.com/jvines/astroARIADNE/tree/master/examples) —
run it and you should reproduce the figure shown!

#### Coelho (single grid): HD 209458

Coelho 2014 spans 0.13–10 µm, so a standalone Coelho fit uses everything
from the blue through 2MASS/WISE/IRAC (its historical ~0.9 µm reliability
cap was lifted after a cross-grid audit showed 1–3% agreement with
BT-Settl/PHOENIX in the NIR). Two more things this example shows: we skip
GALEX (a G dwarf's UV is chromospheric, not photospheric — and Coelho's
grid starts at 0.13 µm besides); and at 48 pc the `Av` prior auto-pins
to 0. The fit recovers Teff = 5989 K and R = 1.19 R☉ against the literature
6065 K / 1.16 R☉ — within 2%! Reproduce with
[`examples/single_coelho.py`](https://github.com/jvines/astroARIADNE/blob/v2/examples/single_coelho.py):

![HD 209458 Coelho fit](https://raw.githubusercontent.com/jvines/astroARIADNE/v2/docs/img/HD209458_coelho_SED.png)

#### TLUSTY (hot stars): HD 65041

For OB stars the LTE grids are out of their depth — `tlusty` is the NLTE
grid for the job. HD 65041 is a B dwarf with a spectroscopic benchmark
(Teff = 17605 ± 200 K; Huang, Gies & McSwain 2010) at galactic latitude
+30°, and the broadband fit recovers ~16900 K — within 4% from photometry
alone, because at this temperature the Balmer jump still lives in the
optical. For hotter stars the optical turns pure Rayleigh–Jeans and you
*need* UV photometry (most bright OB stars saturate GALEX — add UV with
`Star.add_mag` when you have it). Reproduce with
[`examples/single_tlusty.py`](https://github.com/jvines/astroARIADNE/blob/v2/examples/single_tlusty.py):

![HD 65041 TLUSTY fit](https://raw.githubusercontent.com/jvines/astroARIADNE/v2/docs/img/HD65041_tlusty_SED.png)

#### Koester (white dwarfs): WDJ031149.21+190055.62

For `koester` **ARIADNE** switches to white-dwarf priors automatically, so
the fit needs nothing beyond picking the grid. This DA comes out at a
SED-consistent ~15900 K with every band from GALEX FUV to 2MASS Ks agreeing
within 10%. One honest warning: the Gaia-photometry catalogue (Gentile
Fusillo et al. 2021) quotes 18185 K for this star from three optical bands —
broadband WD temperatures carry ~10% systematics between methods, and they
lean heavily on the UV (mind the caveats below). Reproduce with
[`examples/single_koester.py`](https://github.com/jvines/astroARIADNE/blob/v2/examples/single_koester.py):

![WDJ0311+1900 Koester fit](https://raw.githubusercontent.com/jvines/astroARIADNE/v2/docs/img/WD0311_koester_SED.png)

Under the hood, `koester` mode means uniform Teff over 5000–80000 K, uniform
log g over 6.5–9.5, a WD-appropriate radius range, and `[Fe/H]` pinned to 0
(the grid has no metallicity axis).

Two caveats specific to white dwarfs, both inherent to broadband SED fitting
rather than to the grid:

- **Constrain `Av` for nearby WDs.** ARIADNE's default `Av` prior runs up to the
  *full* line-of-sight dust maximum, which for a nearby WD (tens of pc) is far
  more extinction than the star actually sees. The Teff–`Av` degeneracy then
  biases Teff hot. Pin it — `f.prior_setup = {..., 'Av': ('fixed', 0.0)}` (or a
  tight prior from a 3-D dust map) — for stars within ~100 pc.
- **Hot WDs need UV.** Above ~25000 K the optical SED is nearly Rayleigh–Jeans,
  so Teff and radius trade off unless GALEX (or other UV) photometry pins the
  far-UV slope. Add it with `Star.add_mag` when the catalogue crossmatch misses
  it (bright standards are often GALEX-saturated).

**Grid wavelength coverage.** Some grids do not span the full set of filters
**ARIADNE** supports. The Coelho 2014 low-resolution SED library spans
0.13–10 µm, and a cross-grid audit shows its NIR/IRAC photometry matches
BT-Settl/PHOENIX to ~1–3% — so its historical 0.9 µm reliability cap was
lifted, Coelho fits use 2MASS/WISE/IRAC, and the grid now sits in the
**default BMA ensemble** (10 µm comfortably covers every photosphere fit
band). For grids that genuinely stop short — Koester's ~3 µm red cutoff,
the ATLAS9-based `ck04`/`kurucz` whose sparse IR sampling caps them at
8.5 µm — **ARIADNE** automatically excludes any photometry redder than the
grid's coverage from a **standalone** fit (a warning lists the dropped
bands). Coverage limits are declared in `grid_wave_coverage` in `config.py`.
The exclusion is **not** applied in BMA mode, where every grid must be fit to
the same bands for the evidences to be comparable.

**Performance & tuning knobs.** A few optional attributes control runtime:

- `f.walks` (default `25`): number of `rwalk` MCMC steps per proposal. Can
  also be passed as an 8th element of `setup`. Lower is faster but mixes less.
- `f.n_grid_jobs` (default `1`): number of BMA model grids to fit
  concurrently, one core each. Set to `len(models)` to run them all at once.
  **CAUTION:** this spawns that many processes — pick a value your machine's
  cores and memory can handle.
- `f.isochrone_dlogz` (default `0.01`): evidence tolerance for the MIST
  isochrone age/mass fit (the dominant cost of a BMA run, parallelised across
  `threads`). Raising it trades age/mass precision for speed.

We allow the use of five different extinction laws, set via `f.av_law`:

- fitzpatrick (Fitzpatrick 1999; the default)
- cardelli (Cardelli, Clayton & Mathis 1989)
- odonnell (O'Donnell 1994)
- calzetti (Calzetti et al. 2000)
- fm07 (Fitzpatrick & Massa 2007)

The next step is setting up the priors to use:

```python
f.prior_setup = {
	'teff': ('default'),
	'logg': ('default'),
	'z': ('default'),
	'dist': ('default'),
	'rad': ('default'),
	'Av': ('default')
}
```

A quick explanation on the priors:

The default prior for Teff is an empirical prior drawn from the RAVE survey
temperatures distribution, the distance prior is drawn from the
[Bailer-Jones](https://ui.adsabs.harvard.edu/abs/2021AJ....161..147B/abstract)
distance estimate from Gaia EDR3, and the radius has a flat prior ranging from
0.5 to 20 R$_\odot$. The default prior for the metallicity `z` is a population
normal `N(−0.125, 0.234)` dex, and the default log g prior is uniform on
3.5–6.0 dex (replaced by an isochrone `N(logg, σ)` estimate when MIST provides
one). The default prior for Av is a flat prior that ranges from 0 to the maximum
of line-of-sight as per the SFD map, finally the excess noise parameters all
have gaussian priors centered around their respective uncertainties.

In all cases a star-specific spectroscopic prior takes precedence over these
defaults: Teff/log g are drawn from the survey chain
PASTEL → APOGEE DR17 → GALAH DR3 → RAVE DR6 → LAMOST DR11 (first match wins),
and [Fe/H] defaults to the Hypatia Catalog (Hinkel et al. 2014), falling back to
that survey chain on a Hypatia miss.

**Cool dwarfs (automatic M-dwarf priors).** The population Teff/log g priors
above are tuned to FGK stars and mishandle M dwarfs — the Teff prior pulls them
several hundred K hot, and the broad log g prior admits a spurious giant
solution. **ARIADNE** therefore handles cool stars automatically, in two steps:

- **Cool stars** — flagged by Gaia colour BP−RP > 1.8 (falling back to a Gaia
  Teff < 4000 K when colour is unavailable) — get a cool-dwarf **uniform Teff
  prior** instead of the FGK population prior. Its lower edge is not a fixed
  M-dwarf value: it tracks the coolest temperature the in-play grids actually
  reach, so an L/T primary (e.g. an L0 near 1900 K) is not pinned at the ~2300 K
  (M9) floor — with the cloudy/brown-dwarf grids in the ensemble the floor drops
  to ~400–1000 K. Warm-only ensembles keep the historical 2300 K floor; the
  upper edge is 4200 K.
- **Cool dwarfs** — cool stars that are *also* on the main sequence, flagged by
  absolute magnitude M_G > 8 (falling back to luminosity < 0.5 L⊙) —
  *additionally* get a tight main-sequence **`N(5.0, 0.25)` log g prior**, which
  is what keeps the broadband fit from drifting into the giant solution.

This needs no configuration. A spectroscopic Teff/log g — the kind the librarian
attaches to the `Star` from survey catalogues, not a value you set by hand in
`prior_setup` — always takes precedence over both automatic choices.

We offer customization on the priors as well, those are listed in the following
table.

| Prior | Hyperparameters |
| :------: | :----------: |
| Fixed | value |
| Normal | mean, std |
| TruncNorm | mean, std, lower\_lim, uppern\_lim |
| Uniform | ini, end |
| RAVE (log g only) | --- |
| Default | --- | 

So if you knew (from a spectroscopic analysis, for example) that the effective
temperature is 5600 +/- 100 and the metallicity is [Fe/H] = 0.09 +/- 0.05 and
you wanted to use them as priors, and the star is nearby (< 70 pc), so you
wanted to fix Av to 0, your prior dictionary should look like this:

```python
f.prior_setup = {
	'teff': ('normal', 5600, 100),
	'logg': ('default'),
	'z': ('normal', 0.09, 0.05),
	'dist': ('default'),
	'rad': ('default'),
	'Av': ('fixed', 0)
}
```
Though leaving everything at default usually works well enough.

```python
f.prior_setup = {
    'teff': ('default'),
    'logg': ('default'),
    'z': ('default'),
    'dist': ('default'),
    'rad': ('default'),
    'Av': ('default')
}

```

After having set up everything we can finally initialize the fitter and start
fitting

```python
f.initialize()
f.fit_bma()
```

Now we wait for our results!

## Unresolved binaries (SB2 mode)

If a target is a known (or suspected) unresolved binary, **ARIADNE** can fit the
SED as the **sum of two stellar components** sharing distance, extinction and
metallicity. Turn it on with `f.binary = True`. The mode **requires at least one
flux-ratio (contrast) constraint** — a magnitude difference Δm = |m₂ − m₁|
measured in a band (e.g. from speckle/AO imaging, or Gaia) — because the
secondary is otherwise degenerate. It also runs only in the explicit
radius/distance parametrisation, so it needs a **real parallax-based distance**
(it refuses `use_norm` mode). Supply each contrast with `add_contrast`:

```python
f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.5, 'multi', 'rwalk', 4, False]
f.bma = False
f.grid = 'phoenix'                      # single grid, or f.bma=True + f.models
f.binary = True
f.add_contrast('SDSS_g', 5.64, 0.05)    # Δmag and its 1σ in a blue band
f.add_contrast('2MASS_Ks', 2.73, 0.05)  # ...and a red one
f.prior_setup = {
    'teff': ('default'), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.initialize()
f.fit()                                 # or f.fit_bma() for a BMA-binary fit
```

The model adds the secondary's `teff2`, `logg2` and `rad2` (with `teff1 ≥ teff2`
enforced); the primary keeps the usual names. Things to know:

- **One contrast band pins the *dilution* but not the secondary Teff.** With a
  single Δm the secondary lies on a `teff2`/`rad2` ridge; **ARIADNE** flags a
  prior-dominated secondary in `out['binary_secondary_unconstrained']` (set when
  the Teff₂ posterior scatter exceeds 15 % of its median). Two contrast bands of
  *different colour* constrain the secondary properly (recovered to ~1 % on
  synthetic two-band tests).
- **The main deliverable is the per-band host-flux fraction**, the received-flux
  ratio (R₁²f₁)/(R₁²f₁ + R₂²f₂) — the Furlan & Howell planet-radius dilution /
  third-light correction — in `out['binary_dilution']` (and as `dilution_<band>`
  posterior traces). With BMA those traces are averaged across grids, so the
  dilution carries model-systematic uncertainty.
- **A fair model comparison runs automatically.** After the fit, **ARIADNE** fits
  *both* the single-star and the binary model on the *same photometry* (no
  contrast term) and reports the **SED-only** Bayesian evidence in
  `out['binary_vs_single']`: does the photometry alone favour two temperatures
  (`delta_lnZ > 0`), independent of the contrast detection. This adds two extra
  fits, so it roughly triples the run time.
- `plot_SED()` overlays both components and their sum, with the usual gray
  posterior-draw band.

## Visualization

After the fitting has finished, we need to visualize our results. **ARIADNE**
includes a plotter object to do just that! We first star by importing the
plotter:

```python
from astroARIADNE.plotter import SEDPlotter
```

The setup for the plotter is already made for you, but if you really want to
change them, instructions on how to change it can be found
[here](https://github.com/jvines/astroARIADNE/blob/v2/customization.md)

If you want SED model plots, make sure you have either the spectra cache
(recommended) or the full model grids installed — see
[Model Spectra for SED Plotting](#model-spectra-for-sed-plotting) above.
The spectra cache is used automatically if present; the full model grids
are used via the `ARIADNE_MODELS` environment variable as a fallback.

Now we only need to specify the results file location and the output folder
for the plots!

```python
in_file = out_folder + 'BMA.pkl'
plots_out_folder = 'your plots folder here'
```

Now we instantiate the plotter and call the desired plotting methods!
We offer 5 different plots: 

- A RAW SED plot
- A SED plot with the model and synthetic photometry
- A corner plot
- An HR diagram taken from MIST isochrones
- Histograms showing the parameter distributions for each model.

```python
artist = SEDPlotter(in_file, plots_out_folder)
artist.plot_SED_no_model()
artist.plot_SED()
artist.plot_bma_hist()
artist.plot_bma_HR(10)
artist.plot_corner()
```

The number given to `plot_bma_HR` is the number of extra tracks you want to
plot, drawn randomly from the posterior distribution.

If you're iterating through lots of stars you can call the SEDPlotter `clean`
method to clear opened figures with `artist.clean()`

If you don't have either the spectra cache or the full model grids, then the
`plot_SED` method will be skipped.

An example usage file is provided in the repository called `test_bma.py` demonstrating
the recommended BMA (Bayesian Model Averaging) approach.

<!-- DOC:USAGE-END -->

## OUTPUT FILES

<!-- DOC:OUTPUT-START -->

After **ARIADNE** has finished running, it will output a series of files and
plots showing the results of the fit and other information.

The most important file holds the best-fitting parameters with their 1σ error
bars and 3σ confidence intervals. A single-grid run writes it as `best_fit.dat`;
a BMA run instead writes two — `best_fit_sample.dat` (from the evidence-weighted
resampled posterior) and `best_fit_average.dat` (from the supersampled weighted
average). Then there are pickle files for each of the used models
(`{grid}_out.pkl`) plus a last one for the BMA (`BMA.pkl`); these contain raw
information about the results. There is a `prior.dat` file that shows the priors
used and a `mags.dat` file with the used magnitudes and filters.

Another important output are the plots. Inside the plots folder you can find
`CORNER.png/pdf` with the cornerplot (the plot showing the distribution of the
parameters agains eachother), `HR_diagram.png/pdf` only for the BMA, with the HR
diagram showing the position of the star, `SED_no_model.png/pdf` with the RAW
SED showing each photometry point color coded to their respective filter
(coloured by wavelength, overlaid with the blackbody fit from the photometry
quality-control step, and any QC-flagged bands circled in red — see below), and
`SED.png/pdf` with the SED with the catalog photometry plus synthetic
photometry. If BMA was done, there's also a `histograms` folder inside the plot
folder with various histograms of the fitted parameters and their distribution
per model, highlighting the benefits of BMA.

Examples of those figures:

![SED plot](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/SED.png)
![HR Diagram](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/HR_diagram.png)
![Corner plot](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/CORNER.png)
![Histogram example](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/rad.png)

### Photometry quality control

The raw SED plot doubles as a quality-control diagnostic. ARIADNE fits a
blackbody to the photometry (anchored on the Gaia bands) and overlays it on
`SED_no_model.png`. Points are coloured by effective wavelength (violet/blue at
short wavelengths through to orange/red at long ones), and bands that sit more
than 5σ off the fit are flagged and **circled in red** so you can eyeball
whether a catalogue point genuinely belongs to the stellar continuum — useful
for spotting blends, bad cross-matches or real infrared excesses. By default
flagged bands are only warned about, not removed, so the decision stays with
you.

In the example below the star is a clean ~5500 K blackbody and the WISE W2
point (circled in red) shows a clear infrared excess above the fit:

![Blackbody QC](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/SED_blackbody.png)

### Fit-health diagnostics

Every fit now records a purely diagnostic health summary (it never alters the
fit) and prints it after the run. Each grid's pickle carries an `out['diagnostics']`
dict with:

- `logz` / `logzerr` — the grid's log-evidence and its error.
- `eff_samples` — the Kish effective sample size of the importance weights; a
  low value flags an under-sampled, unreliable posterior.
- `dlogz_final` — the last log-evidence increment, a convergence proxy.
- `edge_flags` — any of `teff`/`logg`/`z` whose posterior median sits within 2 %
  of a grid boundary, i.e. the fit is pegged against the edge of the model grid
  and its (tight) error bar should not be trusted.

A BMA run aggregates these into `out['per_model_diagnostics']` (keyed by grid)
plus `out['evidences_err']`, and prints a **PER-GRID FIT HEALTH** table so you
can see at a glance which grids converged cleanly and which railed against an
edge.

### Binary-fit outputs

When run in [SB2 mode](#unresolved-binaries-sb2-mode), each **per-grid pickle**
(`{grid}_out.pkl`) carries extra keys on top of the usual single-star parameters
(which now include `teff2`, `logg2`, `rad2` for the secondary):

- `binary` — `True`, so downstream tools and the plotter know to treat the fit
  as two-component.
- `binary_dilution` — per-band received-flux fraction (R₁²f₁)/(R₁²f₁ + R₂²f₂),
  the Furlan & Howell dilution / third-light correction. The same quantity is
  also stored band-by-band as `dilution_<band>` posterior traces, which (unlike
  the dict) propagate into `BMA.pkl` and are therefore grid-averaged under BMA.
- `binary_secondary_unconstrained` — `True` when the secondary Teff posterior is
  prior-dominated, i.e. its relative scatter `std(Teff₂)/median(Teff₂) > 0.15`
  (the usual outcome of supplying only one contrast band).
- `binary_vs_single` — the automatic SED-only model comparison: a dict (one per
  grid) with `lnZ_binary_sed`, `lnZ_single` and `delta_lnZ` (binary minus
  single). `delta_lnZ > 0` means the photometry alone favours two temperatures.

These binary keys live in the per-grid pickles; under BMA only the grid-averaged
`dilution_<band>` traces are carried into `BMA.pkl`. The SED plot overlays both
stellar components and their sum.


## NetCDF Export (Ecosystem Interop)

As of version 1.4.0 (extended in 1.4.1), ARIADNE can export its results as an
[arviz](https://python.arviz.org/) DataTree in netCDF4 format. This is the
canonical inter-tool format for passing full posterior distributions to
downstream tools (LACHESIS, PROTEUS, etc.) without reducing to point estimates.

### Exporting

After a BMA fit completes, call `to_netcdf` on the `Fitter` object:

```python
f.fit()             # run the fit as usual
f.to_netcdf('ariadne_result.nc')
```

Or get the raw dictionary (useful for programmatic access without writing a
file):

```python
result = f.to_dict()
teff_samples = result['posterior']['Teff']      # shape (1, n_draws)
model_weights = result['sample_stats']['model_weights']
```

### Reading the output

```python
import arviz as az

dt = az.from_netcdf('ariadne_result.nc')

# BMA-weighted posterior (combined across all models)
teff = dt['posterior'].ds.Teff          # shape (chain=1, draw=100000)
print(f"Teff = {float(teff.median()):.0f} K")

# Observed photometry used in the fit
obs = dt['observed_data'].ds
print(obs.wavelength.values)            # micron
print(obs.flux.values)                  # erg/s/cm^2/micron

# Model evidence and BMA weights
cd = dt['constant_data'].ds
print(cd.model_names.values)            # e.g. ['phoenix', 'kurucz', 'btsettl']
print(cd.model_weights.values)          # e.g. [0.08, 0.26, 0.66]
print(float(cd.log_evidence))           # BMA-weighted log evidence

# Per-model posteriors (individual model fits before averaging)
phoenix = dt['posterior_phoenix'].ds
print(f"Phoenix Teff = {float(phoenix.Teff.median()):.0f} K")
print(f"Phoenix log-likelihood median = {float(phoenix.log_likelihood.median()):.2f}")

# Construct a KDE prior for downstream tools
from scipy.stats import gaussian_kde
kde = gaussian_kde(teff.values.flatten())
```

Since 1.4.1 you can attach the best-fit model SED before exporting, so the
`.nc` file carries everything needed to re-plot it:

```python
# Assuming `artist` is an SEDPlotter that has been initialized
f.out['model_sed'] = {
    'model_flux': artist.model,    # f_λ at filter wavelengths
    'wavelengths': artist.wave,
}
f.to_netcdf('ariadne_result.nc')
```

### File structure

```
/
├── posterior/                          BMA-weighted combined posterior
│   ├── Teff          (chain, draw)    K
│   ├── logg          (chain, draw)    dex
│   ├── feh           (chain, draw)    dex
│   ├── radius        (chain, draw)    R_sun
│   ├── luminosity    (chain, draw)    L_sun
│   ├── distance      (chain, draw)    pc
│   ├── Av            (chain, draw)    mag
│   ├── age           (chain, draw)    Gyr          [1.4.1+, from MIST]
│   ├── iso_mass      (chain, draw)    M_sun        [1.4.1+, from MIST]
│   └── eep           (chain, draw)                 [1.4.1+, from MIST]
│
├── posterior_{model}/                  One group per stellar model used
│   ├── Teff          (chain, draw)    K
│   ├── ...
│   └── log_likelihood (chain, draw)   Per-sample log-likelihood
│
├── observed_data/                     Photometry used in the fit
│   ├── wavelength    (band,)          micron
│   ├── flux          (band,)          erg/s/cm^2/micron
│   ├── flux_err      (band,)          erg/s/cm^2/micron
│   ├── filter_names  (band,)          str           [1.4.1+]
│   ├── bandwidths    (band,)          micron         [1.4.1+]
│   └── model_flux    (band,)          erg/s/cm^2/micron  [1.4.1+, if injected]
│
├── constant_data/                     Scalar metadata
│   ├── log_evidence                   BMA-weighted log evidence
│   ├── model_weights (n_models,)      BMA posterior model probabilities
│   ├── model_names   (n_models,)      Model grid names
│   ├── best_fit_averaged__{param}     Summary stats  [1.4.1+]
│   ├── uncertainties_averaged__{param}               [1.4.1+]
│   └── confidence_interval_averaged__{param}         [1.4.1+]
│
```

Each model's posterior group (e.g. `posterior_phoenix`, `posterior_kurucz`) has
its own draw count reflecting the nested sampling output for that grid. The
top-level `posterior/` group contains the BMA-weighted resampled draws across
all models, including MIST isochrone-derived parameters (age, mass, EEP) when
available.


## Infrared Excess (IRX)

**ARIADNE** quantifies and characterizes infrared excess — debris disks and
unresolved cool companions — as a fully Bayesian *post-fit* analysis: bands
redward of ~5 µm never bias the photosphere fit; they are compared against
the posterior-predictive photosphere afterwards and classified by Bayesian
evidence among four hypotheses (no excess, single- and two-temperature
modified-blackbody disks, and an unresolved cool companion evaluated as a
BMA over brown-dwarf-capable model grids). The librarian fetches the
mid/far-IR photometry automatically (WISE, AKARI, IRAS, Spitzer, Herschel),
proper-motion-propagated to each catalog's epoch, and a dedicated figure
renders the SED together with the winning excess model — the classic
`plot_SED()` figure is unchanged.

IR excess is characterized only by the dedicated endpoint — the default
`fit_bma()` never runs it. ARIADNE's four explicit fitting endpoints:

```Python
f.fit_bma()      # default: BMA photosphere fit, no IR-excess
f.fit_single()   # one explicit grid (koester, tlusty, coelho, ...)
f.fit_binary()   # the SB2 two-component fit
f.fit_irx()      # BMA + IR-excess; deep-IR ensemble auto-selected

artist = SEDPlotter(in_file, plots_out_folder, ir_excess=True)
artist.plot_SED()   # the classic SED figure (unchanged)
artist.IRX_SED()    # SED + IR-excess characterization
```

As a taste of what it does: `examples/irx_disk.py` runs eps Eridani through
`fit_irx()`, quantifies twelve excess bands out to SPIRE 350 µm, and
classifies it as a **disk** at T_dust = 62 ± 3 K (literature: 60–65 K), with
the companion hypothesis rejected at lnBF ≈ 430:

![eps Eri IRX](https://raw.githubusercontent.com/jvines/astroARIADNE/v2/docs/img/epsEri_IRX_SED.png)

The full documentation — band routing, ingested catalogs, the evidence
framework, output keys, validation set and caveats — lives in the
[IRX documentation page](https://github.com/jvines/astroARIADNE/blob/v2/docs/irx.md)
(also on Read the Docs).

<!-- DOC:OUTPUT-END -->

## Citing ARIADNE

For a more in depth look on the inner workings of **ARIADNE** consider
[reading the paper!](https://ui.adsabs.harvard.edu/abs/2022MNRAS.tmp..920V/abstract)

Additionally, you can find how to cite **ARIADNE** and its dependencies
[here](https://github.com/jvines/astroARIADNE/blob/v2/citations.md)