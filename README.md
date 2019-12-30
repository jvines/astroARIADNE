# ARIADNE (spectrAl eneRgy dIstribution bAyesian moDel averagiNg fittEr)
## Characterize stellar atmospheres easily!
ARIADNE Is a code written in python 3 (sorry python 2 users!) designed to fit broadband photometry to different stellar atmosphere models automatically using Nested sampling algorithms.

# Installation
To install ARIADNE you can clone this repository with

```
git clone https://github.com/jvines/astroARIADNE.git
```

And the run

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
- tabulate (<https://pypi.org/project/tabulate/>)

Most can be easily installed with pip or conda but some might have special instructions (like PyMultinest!!)

After installing, you can run ... to test the installation

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
## TODO
