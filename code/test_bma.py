from fitter import *
from Plotter import *
from Star import *

if __name__ == '__main__':
    # ra = 338.399
    # dec = -43.436
    # starname = 'TOI-132'

    # mags = {
    #     'GROUND_JOHNSON_B': (11.50599957, 0.082),
    #     'GROUND_JOHNSON_V': (10.98600006, 0.086),
    #     '2MASS_J': (4.34, 0.26),
    #     '2MASS_H': (3.61, 0.23),
    #     '2MASS_Ks': (3.46, 0.2),
    #     'WISE_RSR_W1': (8.89, 0.021),
    #     'WISE_RSR_W2': (2.863, 0.491),
    #     'GaiaDR2v2_G': (6.5220, 0.0009),
    #     'GaiaDR2v2_BP': (7.6155, 0.0052),
    #     'GaiaDR2v2_RP': (5.5111, 0.0031),
    # }

    # ra = 144.581535
    # dec = -67.505097
    # starname = 'TOI-1047'

    # ra = 164.8563
    # dec = -56.6236
    # starname = 'HD95338'
    # gaia_id = 5340648488081462528

    # ra = 25.005615
    # dec = -54.522772
    # starname = 'TIC231005575'
    # gaia_id = 4912474299133826560

    # ra = 75.795
    # dec = -30.399
    # starname = 'NGTS-6'
    # gaia_id = 4875693023844840448

    # --------------------------------------- #
    #        PAPER BENCHMARK STARS            #
    # --------------------------------------- #

    # ra = 330.795
    # dec = 18.884
    # starname = 'HD209458'  # done
    # gaia_id = 1779546757669063552

    ra = 300.182
    dec = 22.710
    starname = 'HD189733'  # done
    gaia_id = 1827242816201846144

    # ra = 195.4438495819606
    # dec = 63.61032798802685
    # starname = 'HD113337'  # done
    # gaia_id = 1676282377934772608

    # ra = 348.3372026315853
    # dec = 57.169625529622316
    # starname = 'HD219134'  # done
    # gaia_id = 2009481748875806976

    # ra = 155.86848664075347
    # dec = -0.9023998277443014
    # starname = 'HD90043'  # Used log g prior done
    # gaia_id = 3830897080395058048

    # ra = 97.69603109164852
    # dec = 58.16117534213966
    # starname = 'HD45410'  # done
    # gaia_id = 1004358968092652544

    # ra = 124.60100914896653
    # dec = -12.636424623475861
    # starname = 'HD69830'  # done
    # gaia_id = 5726982995343100928

    # ra = 1.6551385404819694
    # dec = 29.020738353472627
    # starname = 'HD166'
    # gaia_id = 2860924621205256704

    # ra = 236.00739448041563
    # dec = 2.5145776349240343
    # starname = 'HD140538'  # done
    # gaia_id = 4423865487962417024

    # ra = 332.55110325841594
    # dec = 6.197966395587226
    # starname = 'HD210418'  # done
    # gaia_id = 2720428303852169216

    # ra = 337.8238475644936
    # dec = 50.282567015441394
    # starname = 'HD213558'
    # gaia_id = 1988193348339562880

    # ra = 109.52303862754744
    # dec = 16.540230717750458
    # starname = 'HD56537'  # done
    # gaia_id = 3168265196643176832

    # ra = 346.50275731118126
    # dec = -35.84734894375071
    # starname = 'GJ887'  # done
    # gaia_id = 6553614253923452800

    # ra = 264.10417311195084
    # dec = 68.33367405672519
    # starname = 'GJ687'  # done
    # gaia_id = 1637645127018395776

    # ra = 237.80800119826972
    # dec = 35.65588176597139
    # starname = 'HD142091'
    # gaia_id = 1372702716380418688

    # ra = 272.6326267925868
    # dec = 54.28761840926928
    # starname = 'HD167042'
    # gaia_id = 2149564787289902848

    # --------------------------------------- #
    #               TOIS TO RUN               #
    # --------------------------------------- #

    # ra = 0.362099
    # dec = 39.383795
    # starname = 'TOI-1476'
    # gaia_id = 2881784280929695360

    # ra = 120.593607
    # dec = 3.337163
    # starname = 'TOI-488'
    # gaia_id = 3094290054327367168

    # ra = 83.26925
    # dec = -26.72387
    # starname = 'TOI-431'
    # gaia_id = 2908664557091200768

    # ra = 64.190221
    # dec = -12.084864
    # starname = 'TOI-442'
    # gaia_id = 3189306030970782208

    # ra = 61.941059
    # dec = -25.208803
    # starname = 'TOI-954'
    # gaia_id = 4890874702443960832

    # ra = 306.741134
    # dec = 33.744388
    # starname = 'TIC-199376584'
    # gaia_id = 2056007995732413312

    # ra = 348.337203
    # dec = 57.169626
    # starname = 'TOI-1469'
    # gaia_id = 2009481748875806976

    # ra = 76.649617
    # dec = -20.245613
    # starname = 'TOI-942'
    # gaia_id = 2974906868489280768

    # ra = 353.062066
    # dec = -37.255863
    # starname = 'TOI-251'
    # gaia_id = 6539037542941988736

    # --------------------------------------- #
    #            ZAIRAS M DWARFS              #
    # --------------------------------------- #

    # ra = 346.50275731118126
    # dec = -35.84734894375071
    # starname = 'GJ887'
    # gaia_id = 6553614253923452800

    # ra = 264.10417311195084
    # dec = 68.33367405672519
    # starname = 'GJ687'
    # gaia_id = 1637645127018395776

    # ra = 153.07295861081224
    # dec = -3.7467137970794804
    # starname = 'GJ382'
    # gaia_id = 3828238392559860992

    # ra = 233.04714494316104
    # dec = -41.28003088172311
    # starname = 'GJ588'
    # gaia_id = 6002807341299977216

    # ra = 44.262014
    # dec = -56.191869
    # starname = 'TOI-179'
    # gaia_id = 4728513943538448512

    # --------------------------------------- #
    #            EDS POSSIBLE BD              #
    # --------------------------------------- #

    # ra = 84.985182
    # dec = -4.337143
    # starname = 'NOI-105872'
    # gaia_id = 3023459583984210304
    #
    # mags = {
    #     'SDSS_u':  (19.870,  0.091),
    #     'SDSS_g':  (17.361,  0.006),
    #     'SDSS_r':   (15.912,  0.003),
    #     'SDSS_i':   (14.832,  0.003),
    #     'SDSS_z':  (14.203,  0.003),
    #     'GaiaDR2v2_G':   (15.4811, 0.0044),
    #     'GaiaDR2v2_BP': (16.7387, 0.0205),
    #     'GaiaDR2v2_RP':  (14.3633, 0.0117),
    #     '2MASS_J':  (12.828,  0.028),
    #     '2MASS_H':  (12.042,  0.023),
    #     '2MASS_Ks': (11.864,  0.023),
    #     'WISE_RSR_W1':   (11.714,  0.024),
    #     'WISE_RSR_W2':   (11.595,  0.021),
    #     'GROUND_JOHNSON_V': (16.43, -1)
    # }

    # --------------------------------------- #
    #              OLIS PLANETS               #
    # --------------------------------------- #

    # ra = 217.68335340300
    # dec = -31.73241549430
    # starname = 'NOI-101302'
    # gaia_id = 6216644481523819776

    # ra = 206.02663100100
    # dec = -32.52298159570
    # starname = 'NOI-101195'
    # gaia_id = 6171451912217279872

    s = Star(starname, ra, dec, g_id=gaia_id)

    # For debugging purposes. Are the mags right?
    s.print_mags()
    # SEDPlotter('raw', '../outs/').plot_SED_no_model(s)

    # Comment to use a custom log g prior or the default prior
    s.estimate_logg()

    out_folder = '../outs/SED paper outputs/{}_short'.format(starname)
    # out_folder = '../outs/bma_{}'.format(starname)
    in_file = out_folder + '/BMA_out.pkl'
    plots_out_folder = out_folder + '/plots'
    # Setup parameters
    engine = 'dynesty'  # Only dynesty is available for BMA
    nlive = 500  # number of live points to use
    dlogz = 0.5  # evidence tolerance
    bound = 'multi'  # Unit cube bounds. Options are multi, single
    sample = 'rwalk'  # Sampling method. Options are rwalk, unif
    threads = 8  # Number of threads to use.
    dynamic = False  # Use dynamic nested sampling?
    setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]
    models = ['phoenix', 'btsettl', 'btnextgen', 'btcond', 'ck04', 'kurucz']
    f = Fitter()
    f.star = s
    f.setup = setup
    f.norm = False  # fit normalization constant instead of radius + distance
    f.av_law = 'fitzpatrick'
    f.verbose = True
    # f.priorfile = '../Datafiles/Template_prior.dat'
    f.out_folder = out_folder
    f.bma = True
    f.sequential = True  # Fit models sequentially instead of in parallel.
    f.models = models
    f.n_samples = 100000  # If set as None it will choose automatically.
    f.experimental = False  # If set to True, let threads be smallish.
    f.prior_setup = {
        'teff': ('default',),
        'logg': ('default',),
        'z': ('default',),
        'dist': ('default',),
        'rad': ('default',),
        'Av': ('default',)
    }

    f.initialize()
    f.fit_bma()
    # f = None

artist = SEDPlotter(in_file, plots_out_folder, pdf=True)
artist.plot_SED_no_model()
# artist.plot_SED()
artist.plot_bma_hist()
artist.plot_corner()
