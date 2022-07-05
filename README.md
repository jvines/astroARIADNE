# ARIADNE (spectrAl eneRgy dIstribution bAyesian moDel averagiNg fittEr)
## Characterize stellar atmospheres easily!
**ARIADNE** Is a code written in python 3.7+ designed to fit broadband
photometry to different stellar atmosphere models automatically using Nested
Sampling algorithms.

# Installation

You can install **ARIADNE** with `pip install astroARIADNE`

Otherwise you can clone this repository with

```
git clone https://github.com/jvines/astroARIADNE.git
cd astroARIADNE
```

And then run

```
python setupy.py install
```

But for the code to work, first you must install the necessary dependencies.

## Dependencies:
- Numpy (<https://numpy.org/>)
- Scipy (<https://www.scipy.org/>)
- Pandas (<https://pandas.pydata.org/>)
- numba (<http://numba.pydata.org/>)
- astropy (<https://astropy.readthedocs.io/en/stable/>)
- astroquery (<https://astroquery.readthedocs.io/en/latest/>)
- regions (<https://astropy-regions.readthedocs.io/en/latest/index.html>)
- PyAstronomy (<https://pyastronomy.readthedocs.io/en/v_0-15-2/>)
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

Most can be easily installed with pip or conda but some might have special
instructions (like PyMultinest!!)

**ARIADNE** has been tested on OS X up to Catalina and Linux. It does **NOT**
run on Windows because healpy, a dependency of dustmaps isn't available for
Windows (see [https://github.com/healpy/healpy/issues/25#issue-2987102](https://github.com/healpy/healpy/issues/25#issue-2987102))

## In order to plot the models, you have to download them first:
But note that plotting the SED model is optional. You can run the code without
them!

| Model        | Link           |
| ------------- |:-------------:|
| Phoenix v2      | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/> |
| Phoenix v2   wavelength file   | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits> |
| BT-Models      | <http://osubdd.ens-lyon.fr/phoenix/>  |
| Castelli & Kurucz | <http://ssb.stsci.edu/cdbs/tarfiles/synphot3.tar.gz>      |
| Kurucz 1993 | <http://ssb.stsci.edu/cdbs/tarfiles/synphot4.tar.gz>  |

The wavelength file for the Phoenix model has to be placed in the root folder
of the PHOENIXv2 models.

For the code to find these models, you have to place them somewhere in your
computer as follows:

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

### Notes:
- The Phoenix v2 models with alpha enhancements are unused
- BT-models are BT-Settl, BT-Cond, and BT-NextGen

# How to use?

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
ra = 75.795
dec = -30.399
starname = 'NGTS-6'
gaia_id = 4875693023844840448

s = Star(starname, ra, dec, g_id=gaia_id)
```
The starname is used purely for user identification later on, and the 
`gaia_id` is provided to make sure the automatic photometry retrieval collects
the correct magnitudes, otherwise **ARIADNE** will try and get the `gaia_id` by
itself using a cone search centered around the RA and DEC.

Executing the previous block will start the photometry and stellar parameter
retrieval routine. **ARIADNE** will query Gaia DR2 for an estimate on the
temperature, radius, parallax and luminosity for display as preliminar
information, as it's not used during the fit, and prints them along with its
TIC, KIC IDs if any of those exist, its Gaia DR3 ID, and maximum line-of-sight
extinction Av:

```
			Gaia DR2 ID : 4875693023844840448
			TIC : 1528696
			Effective temperature : 4975.000 +/- 104.390
			Stellar radius : 0.656 +/- 0.141
			Stellar Luminosity : 0.238 +/- 0.003
			Parallax : 3.297 +/- 0.036
			Maximum Av : 0.030
```

If you already know any of those values, you can override the search for them by
providing them in the Star constructor with their respective uncerainties.
Likewise if you already have the magnitudes and wish to override the on-line
search, you can provide a dictionary where the keys are the filters and values
are the mag, mag_err tuples.

If you want to check the retrieved magnitudes you can call the `print_mags`
method from Star:

```python
s.print_mags()
```

This will print the filters used, magnitudes and uncertainties. For NGTS-6 this
would look like this:

```
		     Filter     	Magnitude	Uncertainty
		----------------	---------	-----------
		    2MASS_H     	 11.7670 	  0.0380
		    2MASS_J     	 12.2220 	  0.0330
		    2MASS_Ks    	 11.6500 	  0.0320
		GROUND_JOHNSON_V	 14.0870 	  0.0210
		GROUND_JOHNSON_B	 15.1710 	  0.0140
		  GaiaDR2v2_G   	 13.8175 	  0.0006
		  GaiaDR2v2_RP  	 13.1127 	  0.0015
		  GaiaDR2v2_BP  	 14.4012 	  0.0027
		     SDSS_g     	 14.6390 	  0.0580
		     SDSS_i     	 13.3780 	  0.0570
		     SDSS_r     	 13.7030 	  0.0320
		  WISE_RSR_W1   	 11.5550 	  0.0270
		  WISE_RSR_W2   	 11.6360 	  0.0270
		   GALEX_NUV    	 21.9520 	  0.4090
		      TESS      	 13.1686 	  0.0062
```
**Note:**  **ARIADNE** automatically prints and saves the used magnitudes and
filters to a file.

The way the photometry retrieval works is that Gaia DR2 crossmatch catalogs are
queried for the Gaia ID, these crossmatch catalogs exist for ALL-WISE, APASS,
Pan-STARRS1, SDSS, 2MASS and Tycho-2, so finding photometry relies on these
crossmatches. In the case of NGTS-6, there are also Pan-STARRS1 photometry which
**ARIADNE** couldn't find due to the Pan-STARRS1 source not being identified in
the Gaia DR2 crossmatch, in this case if you wanted to add that photometry
manually, you can do so by using the `add_mag` method from Star, for example, if
you wanted to add the PS1_r mag to our `Star` object you would do:

```python
s.add_mag(13.751, 0.032, 'PS1_r')
```

If for whatever reason **ARIADNE** found a bad photometry point and you needed
to remove it, you can invoke the `remove_mag` method. For example you wanted
to remove the TESS magnitude due to it being from a blended source, you can just
run

```python
s.remove_mag('NGTS')
```

A list of allowed filters can be found
[here](https://github.com/jvines/astroARIADNE/blob/master/filters.md)

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

These maps are all implemented through the
[dustmaps](https://dustmaps.readthedocs.io/en/latest/index.html) package and
need to be downloaded. Instructions to download the dustmaps can be found in
its documentation.

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
results, next we have to select the fitting engine (for BMA only dynesty is
supported), number of live points to use, evidence tolerance threshold, and the
following only apply for dynesty: bounding method, sampling method, threads,
dynamic nested sampler. After selecting all of those, we need to select the
models we want to use and finally, we feed them all to the fitter:

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
f.av_law = 'fitzpatrick'
f.out_folder = out_folder
f.bma = True
f.models = models
f.n_samples = 100000
```

**Note:** While you can always select all 6 models, **ARIADNE** has an internal
filter put in place in order to avoid having the user unintentionally bias the
results. For stars with Teff > 4000 K BT-Settl, BT-NextGen and BT-Cond are
identical and thus only BT-Settl is used, even if the three are selected. On the
other hand, Kurucz and Castelli & Kurucz are known to work poorly on stars with
Teff < 4000 K, thus they aren't used in that regime.

We allow the use of four different extinction laws:

- fitzpatrick
- cardelli
- odonnell
- calzetti

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
0.5 to 20 R$_\odot$. The default prior for the metallicity `z` and log g are
also their respective distributions from the RAVE survey, the default prior for
Av is a flat prior that ranges from 0 to the maximum of line-of-sight as per the
SFD map, finally the excess noise parameters all have gaussian priors centered
around their respective uncertainties.

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

After having set up everything we can finally initialize the fitter and start
fitting

```python
f.initialize()
f.fit_bma()
```

Now we wait for our results!

## Visualization

After the fitting has finished, we need to visualize our results. **ARIADNE**
includes a plotter object to do just that! We first star by importing the
plotter:

```python
from astroARIADNE.plotter import SEDPlotter
```

The setup for the plotter is already made for you, but if you really want to
change them, instructions on how to change it can be found
[here](https://github.com/jvines/astroARIADNE/blob/master/customization.md)

Before we plot the SEDs we need to tell **ARIADNE** where to find our models.
This step isn't necessary if you don't want or need SED plots and are happy with
the HR diagram, histograms, cornerplot and RAW SED. This is done with an
environmental variable called ARIADNE_MODELS, to set it up you just need to run
`export ARIADNE_MODELS='/path/to/Models_Dir/'` in your terminal. You can also
add that instruction to your `.bash_profile` or `.bashrc` and the run
`source ~/.bash_profile` so you don't have to export everytime.

Now that **ARIADNE** knows where to find the models we only need to specify
the results file location and the output folder for the plots!

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

The number given to `plot_bma_HR` is the number of extra tracks you want to
plot, drawn randomly from the posterior distribution.

If you're iterating through lots of stars you can call the SEDPlotter `clean`
method to clear opened figures with `artist.clean()`

If you don't have the models in your computer, then the `plot_SED` method will
fail, as it needs the complete model grid.

An example usage file is provided in the repository called `test_bma.py` for the
BMA approach and test.py for single model fitting.

## OUTPUT FILES
After **ARIADNE** has finished running, it will output a series of files and
plots showing the results of the fit and other information.

The most important file is the `best_fit.dat` which contains the best fiting
parameters with the 1 sigma error bars and the 3 sigma confidence interval.
Then there are pickle files for each of the used models plus a last one for the
BMA, these contain raw information about the results. There is a `prior.dat`
file that shows the priors used and a `mags.dat` file with the used magnitudes
and filters.

Another important output are the plots. Inside the plots folder you can find
`CORNER.png/pdf` with the cornerplot (the plot showing the distribution of the
parameters agains eachother), `HR_diagram.png/pdf` only for the BMA, with the HR
diagram showing the position of the star, `SED_no_model.png/pdf` with the RAW
SED showing each photometry point color coded to their respective filter, and
`SED.png/pdf` with the SED with the catalog photometry plus synthetic
photometry. If BMA was done, there's also a `histograms` folder inside the plot
folder with various histograms of the fitted parameters and their distribution
per model, highlighting the benefits of BMA.

Examples of those figures:

![SED plot](https://github.com/jvines/astroARIADNE/blob/master/img/SED.png)
![HR Diagram](https://github.com/jvines/astroARIADNE/blob/master/img/HR_diagram.png)
![Corner plot](https://github.com/jvines/astroARIADNE/blob/master/img/CORNER.png)
![Histogram example](https://github.com/jvines/astroARIADNE/blob/master/img/rad.png)


## Infrared Excess

As of version 1.0, **ARIADNE** now allows for Infrared Excess visualization!

To visualize infrared excess you just need to add the relevant photometric
observations to the `Star` object with the `add_mag()` method. After the fitting
is done, you then need to initiate the `Plotter` object with the `ir_excess`
parameter set to `True`:

```Python
artist = SEDPlotter(in_file, plots_out_folder, pdf=True, ir_excess=True)
```
Finally after plotting, you should get an SED figure with your manually added
photometry!

Allowed filters for infrared excess plots are **WISE W3, WISE W4, HERSCHEL PACS
BLUE, GREEN and RED**, names for these filters can be found in the
[filters page.](https://github.com/jvines/astroARIADNE/blob/master/filters.md)


## Citing ARIADNE

For a more in depth look on the inner workings of **ARIADNE** consider
[reading the paper!](https://ui.adsabs.harvard.edu/abs/2022MNRAS.tmp..920V/abstract)

Additionally, you can find how to cite **ARIADNE** and its dependencies
[here](https://github.com/jvines/astroARIADNE/blob/master/citations.md)