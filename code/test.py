from fitter import *
from Plotter import *
from Star import *

if __name__ == '__main__':
    ra = 164.856
    dec = -56.623
    starname = 'HD95338'
    gaia_id = 5340648488081462528

    out_folder = '../outs/single_dynesty_' + starname  # + '_norm'

    s = Star(starname, ra, dec, g_id=gaia_id)

    # For debugging purposes. Are the mags right?
    s.print_mags()
    # SEDPlotter('raw', '../outs/').plot_SED_no_model(s)

    # Comment to use a custom log g prior or the default prior
    # s.estimate_logg()

    # Setup parameters
    engine = 'dynesty'  # Either multinest or dynesty
    nlive = 500  # number of live points to use
    dlogz = 0.5  # evidence tolerance
    bound = 'multi'  # dynesty only: unit cube bounds.
    sample = 'rwalk'  # dynesty only: sampling method.
    threads = 8  # dynesty only: number of threads to use.
    dynamic = False  # dynesty only: use dynamic nested sampling?
    setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

    f = Fitter()
    f.star = s
    f.setup = setup
    f.norm = False  # fit normalization constant instead of radius + distance
    f.grid = 'phoenix'
    f.av_law = 'fitzpatrick'
    f.verbose = True
    f.out_folder = out_folder
    f.prior_setup = {
        'teff': ('default',),
        'logg': ('normal', 4.533, 0.298),
        'z': ('default',),
        'dist': ('default',),
        'rad': ('default',),
        'Av': ('default',)
    }

    f.initialize()
    f.fit()

    in_file = out_folder + '/' + engine + '_out.pkl'
    out_folder = out_folder + '/plots'
    artist = SEDPlotter(in_file, out_folder)
    # artist.plot_SED()
    artist.plot_chains()
    artist.plot_like()
    artist.plot_post()
    artist.plot_corner()
