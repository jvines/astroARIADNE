"""Standalone Coelho fit: HD 209458.

Coelho 2014 is optical-only, so ARIADNE automatically drops every band redward
of ~0.9 um from the fit (you'll see the warning listing them). HD 209458 also
sits at ~48 pc, so the default Av prior auto-pins to 0 -- watch for the notice!

Run from anywhere; outputs land in ./HD209458_coelho/
"""
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter

# An optical-only grid deserves optical-only data: a G dwarf's GALEX UV is
# chromospheric, not photospheric, so we tell the librarian to skip it.
s = Star('HD209458', 330.79503, 18.88424, g_id=1779546757669063552,
         ignore=['GALEX'])

f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = 'HD209458_coelho'
f.prior_setup = {
    'teff': ('default'), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.fit_single('coelho')

artist = SEDPlotter('HD209458_coelho/coelho_out.pkl', 'HD209458_coelho/plots')
artist.plot_SED()
