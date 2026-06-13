"""Standalone TLUSTY fit: the B2/3 dwarf HD 65041.

TLUSTY is the hot-star (NLTE) grid, standalone-only. HD 65041 is a
spectroscopic benchmark (Teff = 17605 +/- 200 K, Huang, Gies & McSwain 2010)
at galactic latitude +30, so extinction is nearly zero and the optical
Balmer jump pins the temperature: ARIADNE recovers ~16900 K from broadband
photometry alone. For hotter O/early-B stars the optical turns pure
Rayleigh-Jeans and you NEED UV photometry -- see the caveats below.

Run from anywhere; outputs land in ./HD65041_tlusty/
"""
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter

s = Star('HD65041', 119.64529, 43.50389, g_id=923063449155383936)

f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = 'HD65041_tlusty'
f.prior_setup = {
    'teff': ('default'), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.fit_single('tlusty')

artist = SEDPlotter('HD65041_tlusty/tlusty_out.pkl', 'HD65041_tlusty/plots')
artist.plot_SED()
