"""IR-excess characterization of a debris disk: eps Eridani.

fit_irx() is the only endpoint that quantifies and classifies IR excess. It
auto-selects the deep-IR ensemble (the grids that clip the common reach as
little as possible), fits the photosphere by BMA, then runs the Bayesian
evidence comparison: no excess / single-T disk / two-T disk / cool companion.

The librarian fetches WISE W3/W4, AKARI, IRAS, Spitzer and Herschel
photometry automatically (proper-motion-propagated to each catalog's epoch).
eps Eri should classify as a cold disk at T_dust ~ 60-65 K.

Run from anywhere; outputs land in ./epsEri_irx/
"""
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter

s = Star('epsEri', 53.22833, -9.45826, g_id=5164707970261890560)

f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = 'epsEri_irx'
f.prior_setup = {
    'teff': ('default'), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.fit_irx()        # one call: auto-ensemble + BMA + IRX

artist = SEDPlotter('epsEri_irx/BMA.pkl', 'epsEri_irx/plots', ir_excess=True)
artist.plot_SED()   # the classic SED (unchanged by IRX)
artist.IRX_SED()    # SED + the winning excess model
