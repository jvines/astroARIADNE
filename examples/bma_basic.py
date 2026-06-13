"""The bread-and-butter ARIADNE run: BMA fit of WASP-19.

This is the default operating mode -- a Bayesian Model Averaging fit over the
full Teff-gated ensemble (now including the STAGGER 3D grid and BT-Dusty).
No IR-excess analysis happens here; that lives in its own endpoint (see
irx_disk.py / irx_companion.py).

Run from anywhere; outputs land in ./WASP19_bma/
"""
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter

s = Star('WASP-19', 148.41676, -45.65911, g_id=5411736896952029568)

f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = 'WASP19_bma'
f.prior_setup = {
    'teff': ('default'), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.bma = True  # must be set BEFORE initialize() so the full ensemble loads
f.initialize()
f.fit_bma()

artist = SEDPlotter('WASP19_bma/BMA.pkl', 'WASP19_bma/plots')
artist.plot_SED()
