"""Standalone Koester fit: the DA white dwarf WDJ031149.21+190055.62.

For `koester` ARIADNE switches to white-dwarf priors automatically (uniform
Teff 5000-80000 K, uniform log g 6.5-9.5, a WD radius range) and pins
[Fe/H]=0, so the fit needs nothing beyond picking the grid. This DA has
clean PS1/SDSS photometry plus GALEX FUV (hot WDs lean heavily on the UV --
see the README caveats), and ARIADNE recovers a SED-consistent ~15900 K with
every band from the FUV to Ks agreeing within 10%. Note the Gaia-photometry
catalogue (Gentile Fusillo et al. 2021) quotes 18185 K from three optical
bands: broadband WD temperatures carry ~10% systematics between methods, so
compare like with like.

Run from anywhere; outputs land in ./WD0311_koester/
"""
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter

s = Star('WDJ0311+1900', 47.95520, 19.01496, g_id=59077760488621184)

f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = 'WD0311_koester'
f.prior_setup = {
    'teff': ('default'), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.fit_single('koester')

artist = SEDPlotter('WD0311_koester/koester_out.pkl', 'WD0311_koester/plots')
artist.plot_SED()
