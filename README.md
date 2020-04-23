# ARIADNE (spectrAl eneRgy dIstribution bAyesian moDel averagiNg fittEr)
## Characterize stellar atmospheres easily!
**ARIADNE** Is a code written in python 3 (sorry python 2 users!) designed to fit broadband photometry to different stellar atmosphere models automatically using Nested sampling algorithms.

# Installation
To install **ARIADNE** you can clone this repository with

```
git clone https://github.com/jvines/astroARIADNE.git
```

And then run

```
python setupy.py install
```

Or try ```pip install astroARIADNE``` (Soon to be available!)

But for the code to work, first you must install the necessary dependencies:

## Dependencies:
- Numpy (<https://numpy.org/>)
- Scipy (<https://www.scipy.org/>)
- Pandas (<https://pandas.pydata.org/>)
- numba (<http://numba.pydata.org/>)
- astropy (<https://astropy.readthedocs.io/en/stable/>)
- astroquery (<https://astroquery.readthedocs.io/en/latest/>)
- regions (<https://astropy-regions.readthedocs.io/en/latest/index.html>)
- PyAstronomy (<http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/index.html>)
- corner (<https://corner.readthedocs.io/en/latest/>)
- tqdm (<https://tqdm.github.io/>)
- matplotlib (<https://matplotlib.org/>)
- termcolor (<https://pypi.org/project/termcolor/>)
- extinction (<https://extinction.readthedocs.io/en/latest/>)
- pyphot (<http://mfouesneau.github.io/docs/pyphot/>)
- dustmaps (<https://dustmaps.readthedocs.io/en/latest/>)
- PyMultinest (<https://johannesbuchner.github.io/PyMultiNest/>)
- dynesty (<https://dynesty.readthedocs.io/en/latest/>)
- isochrones (<https://isochrones.readthedocs.io/en/latest/>)

Most can be easily installed with pip or conda but some might have special instructions (like PyMultinest!!)

## In order to plot the models, you have to download them first:
But note that plotting the SED model is optional. You can run the code withouth them!

| Model        | Link           |
| ------------- |:-------------:|
| Phoenix v2      | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/> |
| BT-Models      | <http://osubdd.ens-lyon.fr/phoenix/>  |
| Castelli & Kurucz | <http://ssb.stsci.edu/cdbs/tarfiles/synphot3.tar.gz>      |
| Kurucz 1993 | <http://ssb.stsci.edu/cdbs/tarfiles/synphot4.tar.gz>  |

For the code to find these models, you have to place them somewhere in your computer as follows:

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

### Notes:
- The Phoenix v2 models with alpha enhancements are unused
- BT-models are BT-Settl, BT-Cond, and BT-NextGen

# How to use?

## Stellar information setup

To use **ARIADNE** start by setting up the stellar information, this is done by importing the Star module.

```python
from astroARIADNE.star import Star
```

After importing, a star has to be defined.

Stars are defined in **ARIADNE** by their RA and DEC in degrees, a name, and optionally the Gaia DR2 source id, for example:

```python
ra = 75.795
dec = -30.399
starname = 'NGTS-6'
gaia_id = 4875693023844840448

s = Star(starname, ra, dec, g_id=gaia_id)
```
The starname is used purely for user identification later on, and the gaia\_id is provided to make sure the automatic photometry retrieval collects the correct magnitudes, otherwise **ARIADNE** will try and get the gaia\_id by itself using a cone search centered around the RA and DEC.

Executing the previous block will start the photometry and stellar parameter retrieval routine. **ARIADNE** will query Gaia DR2 for an estimate on the temperature, radius and the parallax, which can be used as priors for the fitting routine, and luminosity for completeness, as it's not used during the fit, and prints them along with its TIC, KIC IDs if any of those exist, its Gaia DR2 ID, and maximum line-of-sight extinction Av:

```
			Gaia DR2 ID : 4875693023844840448
			TIC : 1528696
			Effective temperature : 4975.000 +/- 104.390
			Stellar radius : 0.656 +/- 0.141
			Stellar Luminosity : 0.238 +/- 0.003
			Parallax : 3.297 +/- 0.036
			Maximum Av : 0.030
```

If you already know any of those values, you can override the search for them by providing them in the Star constructor with their respective uncerainties. Likewise if you already have the magnitudes and wish to override the on-line search, you can provide a dictionary where the keys are the filters and values are the mag, mag_err tuples.

If you want to check the retrieved magnitudes you can call the `print_mags` method from Star:

```python
s.print_mags()
```

This will print the filters used, magnitudes and uncertainties. For NGTS-6 this would look like this:

```
Filter              Magnitude    Uncertainty
----------------  -----------  -------------
2MASS_H               11.767          0.038
2MASS_J               12.222          0.033
2MASS_Ks              11.65           0.032
GROUND_JOHNSON_V      14.087          0.021
GROUND_JOHNSON_B      15.171          0.014
GaiaDR2v2_G           13.8175         0.0006
GaiaDR2v2_RP          13.1127         0.0015
GaiaDR2v2_BP          14.4012         0.0027
SDSS_g                14.639          0.058
SDSS_i                13.378          0.057
SDSS_r                13.703          0.032
WISE_RSR_W1           11.555          0.027
WISE_RSR_W2           11.636          0.027
GALEX_NUV             21.952          0.409
TESS                  13.1686         0.0062
```
The way the photometry retrieval works is that Gaia DR2 crossmatch catalogs are queried for the Gaia ID, these crossmatch catalogs exist for ALL-WISE, APASS, Pan-STARRS1, SDSS, 2MASS and Tycho-2, so finding photometry relies on these crossmatches. In the case of NGTS-6, there are also Pan-STARRS1 photometry which **ARIADNE** couldn't find due to the Pan-STARRS1 source not being identified in the Gaia DR2 crossmatch, in this case if you wanted to add that photometry manually, you can do so by using the `add_mag` method from Star, for example, if we wanted to add the PS1_r mag to our Star object we would do:

```python
s.add_mag(13.751, 0.032, 'PS1_r')
```

A list of allowed filters can be found [here](https://github.com/jvines/astroARIADNE/blob/master/filters.md)

After the photometry + stellar parameter retrieval has finished, we can estimate the star's log g to use as prior later with the `estimate_logg` method:

```python
s.estimate_logg()
```

This concludes the stellar setup and now we're ready to set up the parameters for the fitting routine.

## Fitter setup

In this section we'll detail how to set up the fitter for the Bayesian Model Averaging (BMA) mode of **ARIADNE**. For single models the procedure is very similar.

First, import the fitter from **ARIADNE**

```python
from astroARIADNE.fitter import Fitter
```

There are several configuration parameters we have to setup, the first one is the output folder where we want **ARIADNE** to output the fitting files and results, mext we have to select the fitting engine (for BMA we can only use dynesty for now), number of live points to use, evidence tolerance threshold, and the following only apply for dynesty: bounding method, sampling method, threads, dynamic nested sampler. After selecting all of those, we need to select the models we want to use and finally, we feed them all to the fitter:

```python
out_folder = 'your folder here'

engine = 'dynesty'
nlive = 500
dlogz = 0.5
bound = 'multi'
sample = 'rwalk'
threads = 4
dynamic = True

setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

# Feel free to uncomment any unneeded/unwanted models
models = [
	'phoenix',
	'btsettl',
	'btnextgen',
	'btcond',
	'kurucz',
	'ck04'
]

f = Fitter()
f.star = s
f.setup = setup
f.av_law = 'fitzpatrick
f.out_folder = out_folder
f.bma = True
f.models = models
f.n_samples = 100000
```

We allow the use of four different extinction laws:

- fitzpatrick
- cardelli
- odonnell
- calzetti

The next step is setting up the priors to use:

```python
f.prior_setup = {
	'teff': ('rave'),
	'logg': ('default'),
	'z': ('default'),
	'dist': ('default'),
	'rad': ('default'),
	'Av': ('default')
}
```

A quick explanation on the priors:

The default priors for Teff, distance, and radius are the values found in Gaia DR2. the RAVE prior applies only to Teff and consists on the Teff distribution from the RAVE survey, the default prior for the metallicity `z` and log g are also their respective distributions from the RAVE survey, the default prior for Av is a flat prior that ranges from 0 to the maximum of line-of-sight as per the SFD map, finally the excess noise parameters all have gaussian priors centered around their respective uncertainties.

We offer customization on the priors as well, those are listed in the following table.

| Prior | Hyperparameters |
| :------: | :----------: |
| Fixed | value |
| Normal | mean, std |
| TruncNorm | mean, std, lower\_lim, uppern\_lim |
| Uniform | ini, end |
| RAVE (Teff only) | --- |
| Default | --- | 

After having set up everything we can finally initialize the fitter and start fitting

```python
f.initialize()
f.fit_bma()
```

Now we wait for our results!

## Visualization

After the fitting has finished, we need to visualize our results. **ARIADNE** includes a plotter object to do just that! We first star by importing the plotter:

```python
from astroARIADNE.plotter import SEDPlotter
```

The setup for the plotter is already made for you, but instructions on how to change it can be found here **_insert link here_**

We only need to specify the results file location and the output folder for the plots!

```python
in_file = out_folder + 'BMA_out.pkl'
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

The number given to `plot_bma_HR` is the number of extra tracks you want to plot, drawn randomly from the posterior distribution.

If you don't have the models in your computer, then the `plot_SED` method will fail, as it needs the complete model grid.


An example usage file is provided in the repository called test_bma.py for the BMA approach and test.py for single model fitting.
## TODO

- customizing plots