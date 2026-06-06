"""Some configs for the code to run."""
import inspect
import os
import numpy as np

from importlib.resources import files

__ROOT__ = '/'.join(os.path.abspath(inspect.getfile(inspect.currentframe())
                                    ).split('/')[:-1])
_datafiles = files('astroARIADNE').joinpath('Datafiles')
gridsdir = str(_datafiles.joinpath('model_grids'))
priorsdir = str(_datafiles.joinpath('prior'))
filesdir = str(_datafiles)

# Grid name -> on-disk basename. Grids are stored as ``{basename}.h5`` (a
# version-stable HDF5 cube: axes + flux + filter names), not pandas pickles, so
# they survive numpy/pandas upgrades. ``GridInterpolator`` (sed_library) loads
# them; ``load_grid`` resolves a name to the file.
GRID_BASENAMES = {
    'phoenix': 'Phoenixv2', 'btsettl': 'BTSettl', 'btnextgen': 'BTNextGen',
    'btcond': 'BTCond', 'ck04': 'CK04', 'kurucz': 'Kurucz', 'coelho': 'Coelho',
    'bosz': 'BOSZ', 'sphinx': 'SPHINX', 'tlusty': 'TLUSTY', 'koester': 'Koester',
    'newera': 'NewEra',
}

try:
    modelsdir = os.environ['ARIADNE_MODELS']
except KeyError:
    modelsdir = None

# Spectra cache for the continuous model SED in plots. Resolution order:
#   1) $ARIADNE_SPECTRA_CACHE  -- a local/NAS cache (e.g. one extended with the
#      BOSZ/SPHINX/TLUSTY grids) that needn't live under the package,
#   2) the package Datafiles copy (fetch with
#      `python -c "from astroARIADNE.fetch import fetch_spectra_cache; fetch_spectra_cache()"`).
_env_spectra_cache = os.environ.get('ARIADNE_SPECTRA_CACHE')
_spectra_cache_path = os.path.join(filesdir, 'spectra_cache.h5')
if _env_spectra_cache and os.path.isfile(_env_spectra_cache):
    spectra_cache = _env_spectra_cache
elif os.path.isfile(_spectra_cache_path):
    spectra_cache = _spectra_cache_path
else:
    spectra_cache = None

# Custom-filter library (mid/far-IR bands pyphot's bundled library lacks:
# Spitzer MIPS, AKARI/IRC, IRAS, Herschel SPIRE PLW). Built once from SVO by
# notebooks/build_custom_filters.py; loaded by phot_utils alongside pyphot.
custom_filters_path = os.path.join(filesdir, 'custom_filters.h5')

# pyphot filter names.
filter_names = np.array([
       'GALEX_FUV', 'GALEX_NUV', 'STROMGREN_u', 'SkyMapper_u', 'SDSS_u',
       'GROUND_JOHNSON_U', 'SkyMapper_v', 'STROMGREN_v', 'TYCHO_B_MvB',
       'GROUND_JOHNSON_B', 'STROMGREN_b', 'SDSS_g', 'PS1_g',
       'SkyMapper_g', 'GaiaDR2v2_BP', 'TYCHO_V_MvB', 'STROMGREN_y',
       'GROUND_JOHNSON_V', 'SkyMapper_r', 'SDSS_r', 'PS1_r', 'PS1_w',
       'KEPLER_Kp', 'GaiaDR2v2_G', 'GROUND_COUSINS_R', 'NGTS_I', 'SDSS_i',
       'PS1_i', 'SkyMapper_i', 'GaiaDR2v2_RP', 'GROUND_COUSINS_I', 'TESS',
       'PS1_z', 'SDSS_z', 'SkyMapper_z', 'PS1_y', '2MASS_J', '2MASS_H',
       '2MASS_Ks', 'WISE_RSR_W1', 'SPITZER_IRAC_36', 'SPITZER_IRAC_45',
       'WISE_RSR_W2', 'WISE_RSR_W3', 'WISE_RSR_W4', 'HERSCHEL_PACS_BLUE',
       'HERSCHEL_PACS_GREEN', 'HERSCHEL_PACS_RED',
       # Mid/far-IR extension (>=5 um), all treated as IR-excess bands (see
       # irx_bands below). Appended rather than wavelength-interleaved -- the
       # SED plot sorts by effective wavelength, and excess membership is by the
       # explicit irx_bands set, so position here is immaterial. SPIRE 250-500
       # is TLUSTY-only at the grid level (reach-gated per fit).
       'SPITZER_IRAC_58', 'SPITZER_IRAC_80', 'AKARI_IRC_S9W', 'IRAS_12',
       'AKARI_IRC_L18W', 'IRAS_25', 'SPITZER_MIPS_24', 'IRAS_60',
       'SPITZER_MIPS_70', 'IRAS_100', 'SPITZER_MIPS_160',
       'HERSCHEL_SPIRE_PSW', 'HERSCHEL_SPIRE_PMW', 'HERSCHEL_SPIRE_PLW'
])

# IR-excess bands: redward of ~5 um, where warm-dust / debris-disk excess
# appears. These are deliberately NOT fit (they would bias the photosphere on
# exactly the stars of interest); add_mag routes them to the star's irx set and
# they are quantified post-fit as excess -- gated to the bands every in-play
# grid actually reaches (the rest are shown plot-only). Mirrors the >=5 um block
# convolved into the grids (build_model_grids.py _IR_EXTENSION + TLUSTY SPIRE).
irx_bands = frozenset([
    'SPITZER_IRAC_58', 'SPITZER_IRAC_80',
    'AKARI_IRC_S9W', 'AKARI_IRC_L18W',
    'IRAS_12', 'IRAS_25', 'IRAS_60', 'IRAS_100',
    'WISE_RSR_W3', 'WISE_RSR_W4',
    'SPITZER_MIPS_24', 'SPITZER_MIPS_70', 'SPITZER_MIPS_160',
    'HERSCHEL_PACS_BLUE', 'HERSCHEL_PACS_GREEN', 'HERSCHEL_PACS_RED',
    'HERSCHEL_SPIRE_PSW', 'HERSCHEL_SPIRE_PMW', 'HERSCHEL_SPIRE_PLW',
])

# Native wavelength coverage (upper limit, in micron) of model grids whose
# spectra do not span the full ARIADNE filter set. Photometry redder than the
# limit is excluded from a standalone fit with that grid, because the grid's
# flux there was extrapolated when the grid was built and is unreliable.
# Grids absent from this dict are assumed to cover all filters.
# Coelho 2014 spectra span ~0.25-0.9 um, so its IR photometry (2MASS, WISE,
# Spitzer, Herschel) is not trustworthy.
# Koester DA white-dwarf spectra span ~0.09-3.0 um; WISE W1/W2 and Spitzer IRAC
# fall on the featureless Rayleigh-Jeans tail and are dropped (lossless).
# Castelli-Kurucz / Kurucz (ATLAS9) extend to 160 um but are densely sampled
# only to ~10 um (~9 wavelength points beyond, one in 10-15 um), so the mid-IR
# band convolutions (W3 onward) are sparse-sampling artefacts -- their photo-
# spheric W3/W4 come out ~2x the value the densely-sampled grids (BT-Settl, BOSZ)
# and a blackbody agree on. Capped at 8.5 um: IRAC 5.8/8.0 (which match BT-Settl
# to 1%) survive; AKARI 9 (already 18% off) and the unreliable W3+ are dropped.
grid_wave_coverage = {
    'coelho': 0.9,
    'koester': 3.0,
    'ck04': 8.5,
    'kurucz': 8.5,
}

# termcolor colors.
colors = [
        'red', 'green', 'blue', 'yellow',
        'grey', 'magenta', 'cyan', 'white'
    ]

# Isochrone mask array. Positionally aligned to filter_names; the 14 mid/far-IR
# extension bands (the tail) are not isochrone bands, hence the trailing zeros.
iso_mask = np.array([
    0, 0, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
    0, 1, 0, 0,
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 1,
    0, 0, 0, 0, 1, 1,
    1, 1, 0, 0,
    1, 0, 0, 0,
    0, 0,
    0, 0, 0, 0, 0, 0, 0,        # IR extension (IRAC 58/80, AKARI, IRAS, MIPS)
    0, 0, 0, 0, 0, 0, 0,        # ... and Herschel SPIRE
])

# Filter bands for isochrones.
iso_bands = [
    'U', 'B', 'BP', 'V', 'G', 'RP', 'TESS',
    'J', 'H', 'K', 'W1', 'W2'
]


__bibtex__ = '''
@article{ARIADNE,
    author = {Vines, Jose I and Jenkins, James S},
    title = "{ARIADNE: Measuring accurate and precise stellar parameters through SED fitting}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2022},
    month = {04},
    issn = {0035-8711},
    doi = {10.1093/mnras/stac956},
    url = {https://doi.org/10.1093/mnras/stac956},
    note = {stac956},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stac956/43296921/stac956.pdf},
}

@ARTICLE{Dynesty,
       author = {{Speagle}, Joshua S.},
        title = "{DYNESTY: a dynamic nested sampling package for estimating Bayesian posteriors and evidences}",
      journal = {\mnras},
     keywords = {methods: data analysis, methods: statistical, Astrophysics - Instrumentation and Methods for Astrophysics, Statistics - Computation},
         year = 2020,
        month = apr,
       volume = {493},
       number = {3},
        pages = {3132-3158},
          doi = {10.1093/mnras/staa278},
archivePrefix = {arXiv},
       eprint = {1904.02180},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{Skilling2004,
       author = {{Skilling}, John},
        title = "{Nested Sampling}",
     keywords = {02.50.Tt, Inference methods},
    booktitle = {American Institute of Physics Conference Series},
         year = "2004",
       editor = {{Fischer}, Rainer and {Preuss}, Roland and {Toussaint}, Udo Von},
       series = {American Institute of Physics Conference Series},
       volume = {735},
        month = "Nov",
        pages = {395-405},
          doi = {10.1063/1.1835238},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@article{Skilling2006,
       author = "Skilling, John",
          doi = "10.1214/06-BA127",
     fjournal = "Bayesian Analysis",
      journal = "Bayesian Anal.",
        month = "12",
       number = "4",
        pages = "833--859",
    publisher = "International Society for Bayesian Analysis",
        title = "Nested sampling for general Bayesian computation",
          url = "https://doi.org/10.1214/06-BA127",
       volume = "1",
         year = "2006"
}

@MISC{isochrones,
       author = {{Morton}, Timothy D.},
        title = "{isochrones: Stellar model grid package}",
     keywords = {Software},
         year = "2015",
        month = "Mar",
          eid = {ascl:1503.010},
        pages = {ascl:1503.010},
archivePrefix = {ascl},
       eprint = {1503.010},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2015ascl.soft03010M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{MIST,
       author = {{Dotter}, Aaron},
        title = "{MESA Isochrones and Stellar Tracks (MIST) 0: Methods for the Construction of Stellar Isochrones}",
      journal = {\apjs},
     keywords = {methods: numerical, stars: evolution, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = "2016",
        month = "Jan",
       volume = {222},
       number = {1},
          eid = {8},
        pages = {8},
          doi = {10.3847/0067-0049/222/1/8},
archivePrefix = {arXiv},
       eprint = {1601.05144},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2016ApJS..222....8D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@article{Husser2013,
        abstract = {We present a new library of high-resolution synthetic spectra based on the stellar atmosphere code PHOENIX that can be used for a wide range of applications of spectral analysis and stellar parameter synthesis. The spherical mode of PHOENIX was used to create model atmospheres and to derive detailed synthetic stellar spectra from them. We present a new self-consistent way of describing micro-turbulence for our model atmospheres. The synthetic spectra cover the wavelength range from 500AA to 50.000AA with resolutions of R=500.000 in the optical and near IR, R=100.000 in the IR and a step size of 0.1AA in the UV. The parameter space covers 2.300K{\textless}=Teff{\textless}=12.000K, 0.0{\textless}=log(g){\textless}=+6.0, -4.0{\textless}=[Fe/H]{\textless}=+1.0, and -0.2{\textless}=[alpha/Fe]{\textless}=+1.2. The library is a work in progress and we expect to extend it up to Teff=25.000 K.},
   archivePrefix = {arXiv},
         arxivId = {1303.5632},
          author = {Husser, Tim-Oliver and von Berg, Sebastian Wende - and Dreizler, Stefan and Homeier, Derek and Reiners, Ansgar and Barman, Travis and Hauschildt, Peter H},
             doi = {10.1051/0004-6361/201219058},
          eprint = {1303.5632},
            issn = {0004-6361},
         journal = {Astronomy {\&} Astrophysics},
        keywords = {atmospheres,convection,late-type,stars},
           pages = {A6},
           title = {{A new extensive library of PHOENIX stellar atmospheres and synthetic spectra}},
             url = {http://arxiv.org/abs/1303.5632{\%}0Ahttp://dx.doi.org/10.1051/0004-6361/201219058},
          volume = {553},
            year = {2013}
}

@article{Allard2012,
    abstract = {Within the next few years, GAIA and several instruments aiming at imag- ing extrasolar planets will see first light. In parallel, low mass planets are being searched around red dwarfs which offer more favourable conditions, both for radial velocity de- tection and transit studies, than solar-type stars. Authors of the model atmosphere code which has allowed the detection of water vapour in the atmosphere of Hot Jupiters re- view recent advancement in modelling the stellar to substellar transition. The revised solar oxygen abundances and cloud model allow for the first time to reproduce the pho- tometric and spectroscopic properties of this transition. Also presented are highlight results of a model atmosphere grid for stars, brown dwarfs and extrasolar planets.},
      author = {Allard, F. and Homeier, D. and Freytag, B.},
         doi = {10.1098/rsta.2011.0269},
        issn = {1364503X},
     journal = {Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences},
    keywords = {Brown dwarfs,CO5BOLD,Opacities,Stars,Very-low-mass stars},
      number = {1968},
       pages = {2765--2777},
       title = {{Models of very-low-mass stars, brown dwarfs and exoplanets}},
      volume = {370},
        year = {2012}
}

@article{Hauschildt1999,
    abstract = {We present our NextGen Model Atmosphere grid for low-mass stars for effective temperatures larger than 3000 K. These LTE models are calculated with the same basic model assumptions and input physics as the VLMS part of the NextGen grid so that the complete grid can be used, e.g., for consistent stellar evolution calculations and for internally consistent analysis of cool star spectra. This grid is also the starting point for a large grid of detailed NLTE model atmospheres for dwarfs and giants. The models were calculated from 3000 to 10,000 K (in steps of 200 K) for 3.5{\textless}=logg{\textless}=5.5 (in steps of 0.5) and metallicities of -4.0{\textless}=[M/H]{\textless}=0.0. We discuss the results of the model calculations and compare our results to the Kurucz grid. Some comparisons to standard stars like Vega and the Sun are presented and compared with detailed NLTE calculations.},
      author = {Hauschildt, Peter H and Allard, France and Baron, E},
         doi = {10.1086/430754},
        issn = {0004-637X},
     journal = {The Astrophysical Journal},
      number = {2},
       pages = {865--872},
       title = {{THE NEXTGEN MODEL ATMOSPHERE GRID FOR 3000 {\textless}= Teff {\textless}= 10,000 K}},
      volume = {629},
        year = {1999}
}

@ARTICLE{1993KurCD..13.....K,
       author = {{Kurucz}, Robert},
        title = "{ATLAS9 Stellar Atmosphere Programs and 2 km/s grid.}",
      journal = {ATLAS9 Stellar Atmosphere Programs and 2 km/s grid. Kurucz CD-ROM No. 13. Cambridge},
         year = "1993",
        month = "Jan",
       volume = {13},
       adsurl = {https://ui.adsabs.harvard.edu/abs/1993KurCD..13.....K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{Castelli2004,
       author = {{Castelli}, F. and {Kurucz}, R.~L.},
        title = "{New Grids of ATLAS9 Model Atmospheres}",
     keywords = {Astrophysics},
    booktitle = {Modelling of Stellar Atmospheres},
         year = {2003},
       editor = {{Piskunov}, N. and {Weiss}, W.~W. and {Gray}, D.~F.},
       volume = {210},
        month = {jan},
        pages = {A20},
        series = {},
archivePrefix = {arXiv},
       eprint = {astro-ph/0405087},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2003IAUS..210P.A20C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
'''