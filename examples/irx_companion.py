"""IR-excess characterization of an unresolved companion: SCR 1845-6357.

The same fit_irx() endpoint, but here the excess is NOT dust: SCR 1845 is an
M8.5 dwarf with an unresolved T6 brown-dwarf companion (~950 K) that shows up
as a W3/W4 excess. The evidence comparison should favor the companion
hypothesis, with the companion's Teff recovered by a BMA over the
brown-dwarf-capable grids (cool BT-Settl, ATMO 2020, Sonora Bobcat).

Run from anywhere; outputs land in ./SCR1845_irx/
"""
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter

s = Star('SCR1845', 281.29804, -63.96057, g_id=6439125097427143808)

f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = 'SCR1845_irx'
f.prior_setup = {
    'teff': ('default'), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.fit_irx()

artist = SEDPlotter('SCR1845_irx/BMA.pkl', 'SCR1845_irx/plots',
                    ir_excess=True)
artist.plot_SED()
artist.IRX_SED()
