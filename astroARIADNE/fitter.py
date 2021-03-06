"""Main driver of the fitting routine."""
import pickle
import random
import time
import warnings
from multiprocessing import Pool, Process

import astropy.units as u
import pandas as pd
import numpy as np
import scipy.stats as st
from astropy.constants import sigma_sb
from isochrones.interp import DFInterpolator
from termcolor import colored

from .config import filesdir, gridsdir, priorsdir
from .error import *
from .isochrone import estimate
from .phot_utils import *
from .sed_library import *
from .utils import *

try:
    import dynesty
    from dynesty.utils import resample_equal

    bma_flag = True
except ModuleNotFoundError:
    wrn = 'Dynesty package not found. BMA and log g estimation unavailable.'
    warnings.warn(wrn)
    bma_flag = False
try:
    import pymultinest
except ModuleNotFoundError:
    warnings.warn(
        '(py)MultiNest installation (or libmultinest.dylib) not detected.'
    )


class Fitter:
    """The Fitter class handles the fitting routines and parameter estimation.

    Examples
    --------
    The fitter isn't instantiaded with any arguments, rather you instantiate a
    Fitter object, then you set up the configurations and finally you
    initialize the object by running the initialize method.
    >>> f = Fitter()
    >>> f.star = s  # s must be a valid Star object.
    >>> f.initialize()

    Attributes
    ----------
    out_folder : type
        Description of attribute `out_folder`.
    verbose : type
        Description of attribute `verbose`.
    star : type
        Description of attribute `star`.
    setup : type
        Description of attribute `setup`.
    norm : type
        Description of attribute `norm`.
    grid : type
        Description of attribute `grid`.
    estimate_logg : type
        Description of attribute `estimate_logg`.
    priorfile : type
        Description of attribute `priorfile`.
    av_law : type
        Description of attribute `av_law`.
    n_samples : type
        Description of attribute `n_samples`.
    bma : type
        Description of attribute `bma`.
    prior_setup : type
        Description of attribute `prior_setup`.
    sequential : type
        Description of attribute `sequential`.

    """

    def __init__(self):

        # Default values for attributes
        self._interpolators = []
        self._grids = []
        self.out_folder = None
        self.verbose = True
        self.star = None
        self.setup = ['dynesty']
        self.norm = False
        self.grid = 'phoenix'
        self.estimate_logg = False
        self.priorfile = None
        self.av_law = 'fitzpatrick'
        self.n_samples = None
        self.bma = False
        self.prior_setup = None
        self.sequential = False

    @property
    def star(self):
        """Star to fit for."""
        return self._star

    @star.setter
    def star(self, star):
        # if not isinstance(star, Star) and star is not None:
        #     InstanceError(star, Star).__raise__()
        self._star = star

    @property
    def setup(self):
        """Set up options."""
        return self._setup

    @setup.setter
    def setup(self, setup):
        err_msg = 'The setup has to contain at least the fitting engine'
        err_msg += ', multinest or dynesty.'
        if len(setup) < 1:
            InputError(setup, err_msg).__raise__()
        self._setup = setup
        self._engine = setup[0]
        defaults = False
        if len(setup) == 1:
            defaults = True
        if self._engine == 'multinest':
            if defaults:
                self._nlive = 500
                self._dlogz = 0.5
            else:
                self._nlive = setup[1]
                self._dlogz = setup[2]
        if self._engine == 'dynesty':
            if defaults:
                self._nlive = 500
                self._dlogz = 0.5
                self._bound = 'multi'
                self._sample = 'rwalk'
                self._threads = 1
                self._dynamic = False
            else:
                self._nlive = setup[1]
                self._dlogz = setup[2]
                self._bound = setup[3]
                self._sample = setup[4]
                self._threads = setup[5]
                self._dynamic = setup[6]

    @property
    def norm(self):
        """Bool to decide if a normalization constant will be fitted.

        Set as True to not fit for radius and distance and fit for a
        normalization constant. After the fit a radius is calculated using
        the Gaia parallax.
        """
        return self._norm

    @norm.setter
    def norm(self, norm):
        if type(norm) is not bool:
            InputError(norm, 'norm must be True or False.').__raise__()
        self._norm = norm

    @property
    def grid(self):
        """Model grid selected."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        assert type(grid) == str
        self._grid = grid
        directory = '../Datafiles/model_grids/'
        # directory = './'
        if grid.lower() == 'phoenix':
            with open(gridsdir + '/Phoenixv2_DF.pkl', 'rb') as intp:
                self._interpolator = DFInterpolator(pd.read_pickle(intp))
        if grid.lower() == 'btsettl':
            with open(gridsdir + '/BTSettl_DF.pkl', 'rb') as intp:
                self._interpolator = DFInterpolator(pd.read_pickle(intp))
        if grid.lower() == 'btnextgen':
            with open(gridsdir + '/BTNextGen_DF.pkl', 'rb') as intp:
                self._interpolator = DFInterpolator(pd.read_pickle(intp))
        if grid.lower() == 'btcond':
            with open(gridsdir + '/BTCond_DF.pkl', 'rb') as intp:
                self._interpolator = DFInterpolator(pd.read_pickle(intp))
        if grid.lower() == 'ck04':
            with open(gridsdir + '/CK04_DF.pkl', 'rb') as intp:
                self._interpolator = DFInterpolator(pd.read_pickle(intp))
        if grid.lower() == 'kurucz':
            with open(gridsdir + '/Kurucz_DF.pkl', 'rb') as intp:
                self._interpolator = DFInterpolator(pd.read_pickle(intp))
        if grid.lower() == 'coelho':
            with open(gridsdir + '/Coelho_DF.pkl', 'rb') as intp:
                self._interpolator = DFInterpolator(pd.read_pickle(intp))

    @property
    def bma(self):
        """Bayesian Model Averaging (BMA).

        Set to True if BMA is wanted. This loads every model grid interpolator
        and fits an SED to all of them, so the runtime will be slower!
        """
        return self._bma

    @bma.setter
    def bma(self, bma):
        self._bma = bma if bma_flag else False

    @property
    def models(self):
        """Models to be used in BMA."""
        return self._bma_models

    @models.setter
    def models(self, mods):
        self._bma_models = mods

    @property
    def sequential(self):
        """Set to True to make BMA sequentially instead of parallel."""
        return self._sequential

    @sequential.setter
    def sequential(self, sequential):
        self._sequential = sequential

    @property
    def n_samples(self):
        """Set number of samples for BMA."""
        return self._nsamp

    @n_samples.setter
    def n_samples(self, nsamp):
        self._nsamp = nsamp

    @property
    def verbose(self):
        """Program verbosity. Default is True."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if type(verbose) is not bool:
            InputError(verbose, 'verbose must be True or False.').__raise__()
        self._verbose = verbose

    @property
    def priorfile(self):
        """Priorfile location."""
        return self._priorfile

    @priorfile.setter
    def priorfile(self, priorfile):
        if type(priorfile) is not str and priorfile is not None:
            err_msg = 'Priorfile must be an address or None.'
            InputError(priorfile, err_msg).__raise__()
        self._priorfile = priorfile

    @property
    def out_folder(self):
        """Output folder.

        If none is provided the default will be the starname.
        """
        return self._out_folder

    @out_folder.setter
    def out_folder(self, out_folder):
        if type(out_folder) is not str and out_folder is not None:
            err_msg = 'Output folder must be an address or None.'
            InputError(out_folder, err_msg).__raise__()
        self._out_folder = out_folder

    @property
    def av_law(self):
        """Select extinction law."""
        return self._av_law

    @av_law.setter
    def av_law(self, law):
        laws = [
            'cardelli',
            'odonnell',
            'calzetti',
            'fitzpatrick'
        ]
        law = law.lower()
        if law not in laws:
            err_msg = 'Extinction law {} not recognized. Available extinction'
            err_msg += ' laws are: `cardelli`, `odonnell`'
            err_msg += ', `calzetti`, and `fitzpatrick`'
            err_msg.format(law)
            InputError(law, err_msg).__raise__()
        import extinction
        law_f = None
        if law == laws[0]:
            law_f = extinction.ccm89
        if law == laws[1]:
            law_f = extinction.odonnell94
        if law == laws[2]:
            law_f = extinction.calzetti00
        if law == laws[3]:
            law_f = extinction.fitzpatrick99
        self._av_law = law_f

    def initialize(self):
        """Initialize the fitter.

        To be run only after every input is added.
        This function calculates the number of dimensions, runs the prior
        creation, creates output directory, initializes coordinators and sets
        up global variables.
        """
        global prior_dict, coordinator, fixed, order, star, use_norm, av_law
        self.start = time.time()
        err_msg = 'No star is detected. Please create an instance of Star.'
        if self.star is None:
            er = InputError(self.star, err_msg)
            er.log(self.out + '/output.log')
            er.__raise__()
        star = self.star
        if not self._bma:
            global interpolator
            interpolator = self._interpolator
        use_norm = self.norm
        # Extinction law
        av_law = self._av_law

        # Declare order of parameters.
        if not self.norm:
            order = np.array(
                [
                    'teff', 'logg', 'z',
                    'dist', 'rad', 'Av'
                ]
            )
        else:
            order = np.array(['teff', 'logg', 'z', 'norm', 'Av'])

        # Create output directory
        if self.out_folder is None:
            self.out_folder = self.star.starname + '/'
        create_dir(self.out_folder)

        self.star.save_mags(self.out_folder + '/')

        # Parameter coordination.
        # Order for the parameters are:
        # teff, logg, z, dist, rad, Av, noise
        # or
        # teff, logg, z, norm, Av, noise
        npars = 6 if not self.norm else 5
        npars += self.star.used_filters.sum()
        npars = int(npars)
        self.coordinator = np.zeros(npars)  # 1 for fixed params
        self.fixed = np.zeros(npars)
        coordinator = self.coordinator
        fixed = self.fixed

        # Setup priors.
        self.default_priors = self._default_priors()
        if self.prior_setup is None:
            self.create_priors(self.priorfile)
        else:
            self.create_priors_from_setup()
        prior_dict = self.priors

        # Get dimensions.
        self.ndim = self.get_ndim()

        # warnings
        if len(self._setup) == 1:
            print('USING DEFAULT SETUP VALUES.')

        # BMA settings
        # if BMA is used, load all interpolators requested.
        if self._bma:
            if self.n_samples is None:
                self.n_samples = 'max'

            if self.star.offline:
                self.star.temp = 4001
                off_msg = 'Offline mode assumes that the stellar'
                off_msg += ' temperature is greater than 4000 K'
                off_msg += '. If you believe this is not the case then please '
                off_msg += 'add a temperature to the Star constructor'
                print(off_msg)

            for mod in self._bma_models:
                # We'll assume that if ARIADNE is running in offline mode
                # Then the star will have > 4000 K
                if mod.lower() == 'phoenix':
                    with open(gridsdir + '/Phoenixv2_DF.pkl', 'rb') as intp:
                        df = DFInterpolator(pd.read_pickle(intp))
                if mod.lower() == 'btsettl':
                    with open(gridsdir + '/BTSettl_DF.pkl', 'rb') as intp:
                        df = DFInterpolator(pd.read_pickle(intp))
                if mod.lower() == 'btnextgen':
                    if self.star.temp > 4000:
                        continue
                    else:
                        with open(gridsdir + '/BTNextGen_DF.pkl', 'rb') as inp:
                            df = DFInterpolator(pd.read_pickle(inp))
                if mod.lower() == 'btcond':
                    if self.star.temp > 4000:
                        continue
                    else:
                        with open(gridsdir + '/BTCond_DF.pkl', 'rb') as intp:
                            df = DFInterpolator(pd.read_pickle(intp))
                if mod.lower() == 'ck04':
                    if self.star.temp > 4000:
                        with open(gridsdir + '/CK04_DF.pkl', 'rb') as intp:
                            df = DFInterpolator(pd.read_pickle(intp))
                    else:
                        # Warning temp too low for model
                        continue
                if mod.lower() == 'kurucz':
                    if self.star.temp > 4000:
                        with open(gridsdir + '/Kurucz_DF.pkl', 'rb') as intp:
                            df = DFInterpolator(pd.read_pickle(intp))
                    else:
                        # Warning temp too low for model.
                        continue
                # if mod.lower() == 'coelho':
                #     if self.star.temp > 3500:
                #         with open(gridsdir + '/Coelho_DF.pkl', 'rb') as intp:
                #             df = DFInterpolator(pd.read_pickle(intp))
                #     else:
                #         # Warning
                #         continue
                self._interpolators.append(df)
                self._grids.append(mod)
            thr = self._threads if self._sequential else len(
                self._interpolators)
        else:
            thr = self._threads
        en = 'Bayesian Model Averaging' if self._bma else self._engine
        display_routine(en, self._nlive, self._dlogz, self.ndim, self._bound,
                        self._sample, thr, self._dynamic)

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = 6 if not self.norm else 5
        ndim += self.star.used_filters.sum()
        ndim -= self.coordinator.sum()
        return int(ndim)

    def _default_priors(self):
        global order
        defaults = dict()
        # Logg prior setup.
        if self.star.get_logg:
            defaults['logg'] = st.norm(
                loc=self.star.logg, scale=self.star.logg_e)
        else:
            with open(priorsdir + '/logg_ppf.pkl', 'rb') as jar:
                defaults['logg'] = pickle.load(jar)
        # Teff prior from RAVE
        with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
            defaults['teff'] = pickle.load(jar)
        # [Fe/H] prior setup.
        defaults['z'] = st.norm(loc=-0.125, scale=0.234)
        # Distance prior setup.
        if not self._norm:
            if self.star.dist != -1:
                defaults['dist'] = st.norm(
                    loc=self.star.dist, scale=3 * self.star.dist_e)
            else:
                defaults['dist'] = st.uniform(loc=1, scale=3000)
            # Radius prior setup.
            defaults['rad'] = st.uniform(loc=0.05, scale=100)
        # Normalization prior setup.
        else:
            up = 1 / 1e-20
            defaults['norm'] = st.truncnorm(a=0, b=up, loc=0, scale=1e-15)
        # Extinction prior setup.
        if self.star.Av == 0.:
            av_idx = 4 if self._norm else 5
            self.coordinator[av_idx] = 1
            self.fixed[av_idx] = 0
            defaults['Av'] = None
        else:
            defaults['Av'] = st.uniform(loc=0, scale=self.star.Av)
        # Noise model prior setup.
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
            p_ = get_noise_name(filt) + '_noise'
            mu = 0
            sigma = flx_e * 10
            b = (1 - flx) / flx_e
            defaults[p_] = st.truncnorm(loc=mu, scale=sigma, a=0, b=b)
            # defaults[p_] = st.uniform(loc=0, scale=5)
            order = np.append(order, p_)
        return defaults

    def create_priors(self, priorfile):
        """Read the prior file.

        Returns a dictionary with each parameter's prior
        """
        param_list = [
            'teff', 'logg', 'z',
            'dist', 'rad', 'norm',
            'Av'
        ]

        noise = []
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
            p_ = get_noise_name(filt) + '_noise'
            noise.append(p_)

        if priorfile:
            param, prior, bounds = np.loadtxt(
                priorfile, usecols=[0, 1, 2], unpack=True, dtype=object)
            copy = np.vstack((param, prior, bounds)).T
            np.savetxt(self.out_folder + '/prior.dat', copy, fmt='%s')
            # Dict with priors.
            prior_dict = dict()
            for par, pri, bo in zip(param, prior, bounds):
                if par not in param_list:
                    er = PriorError(par, 0)
                    er.log(self.out_folder + '/output.log')
                    er.__raise__()
                if self.norm and (par == 'dist' or par == 'rad'):
                    er = PriorError(par, 1)
                    er.log(self.out_folder + '/output.log')
                    er.__raise__()
                if pri.lower() == 'uniform':
                    a, b = bo.split(',')
                    a, b = float(a), float(b)
                    prior_dict[par] = st.uniform(loc=a, scale=b - a)
                elif pri.lower() == 'normal':
                    mu, sig = bo.split(',')
                    mu, sig = float(mu), float(sig)
                    prior_dict[par] = st.norm(loc=mu, scale=sig)
                elif pri.lower() == 'truncatednormal':
                    mu, sig, up, low = bo.split(',')
                    mu, sig, up, low = float(mu), float(
                        sig), float(up), float(low)
                    b, a = (up - mu) / sig, (low - mu) / sig
                    priot_dict[par] = st.truncnorm(a=a, b=b, loc=mu, scale=sig)
                elif pri.lower() == 'default':
                    prior_dict[par] = self.default_priors[par]
                elif pri.lower() == 'fixed':
                    idx = np.where(par == order)[0]
                    self.coordinator[idx] = 1
                    self.fixed[idx] = float(bo)

                for par in noise:
                    prior_dict[par] = self.default_priors[par]
            self.priors = prior_dict
        else:
            warnings.warn('No priorfile detected. Using default priors.')
            self.priors = self.default_priors
        pass

    def create_priors_from_setup(self):
        """Create priors from the manual setup."""
        prior_dict = dict()
        keys = self.prior_setup.keys()
        noise = []
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
            p_ = get_noise_name(filt) + '_noise'
            noise.append(p_)
        prior_out = 'Parameter\tPrior\tValues\n'
        if 'norm' in keys and ('rad' in keys or 'dist' in keys):
            er = PriorError('rad or dist', 1)
            er.log(self.out_folder + '/output.log')
            er.__raise__()
        for k in keys:
            if type(self.prior_setup[k]) == str:
                if self.prior_setup[k] == 'default':
                    prior_dict[k] = self.default_priors[k]
                    prior_out += k + '\tdefault\n'
                if self.prior_setup[k].lower() == 'rave':
                    # RAVE prior only available for teff and logg. It's already
                    # the default for [Fe/H]
                    if k == 'teff' or k == 'logg':
                        with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
                            prior_dict[k] = pickle.load(jar)
                        prior_out += k + '\tRAVE\n'
            else:
                prior = self.prior_setup[k][0]
                if prior == 'fixed':
                    value = self.prior_setup[k][1]
                    idx = np.where(k == order)[0]
                    self.coordinator[idx] = 1
                    self.fixed[idx] = value
                    prior_out += k + '\tfixed\t{}\n'.format(value)
                if prior == 'normal':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    prior_dict[k] = st.norm(loc=mu, scale=sig)
                    prior_out += k + '\tnormal\t{}\t{}\n'.format(mu, sig)
                if prior == 'truncnorm':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    low = self.prior_setup[k][3]
                    up = self.prior_setup[k][4]
                    b, a = (up - mu) / sig, (low - mu) / sig
                    prior_dict[k] = st.truncnorm(a=a, b=b, loc=mu, scale=sig)
                    prior_out += k
                    prior_out += '\ttruncatednormal\t{}\t{}\t{}\t{}\n'.format(
                        mu, sig, low, up)
                if prior == 'uniform':
                    low = self.prior_setup[k][1]
                    up = self.prior_setup[k][2]
                    prior_dict[k] = st.uniform(loc=low, scale=up - low)
                    prior_out += k + '\tuniform\t{}\t{}\n'.format(low, up)
        for par in noise:
            prior_dict[par] = self.default_priors[par]
        ff = open(self.out_folder + '/prior.dat', 'w')
        ff.write(prior_out)
        ff.close()
        del ff
        self.priors = prior_dict
        pass

    def fit(self):
        """Run fitting routine."""
        if self._engine == 'multinest':
            self.fit_multinest()
        else:
            self.fit_dynesty()
        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time,
            self.out_folder, self._engine, self.norm)
        pass

    def fit_bma(self):
        """Perform the fit with different models and the average the output.

        Only works with dynesty.
        """
        if len(self.star.filter_names[self.star.filter_mask]) <= 5:
            print(colored('\t\t\tNOT ENOUGH POINTS TO MAKE THE FIT! !', 'red'))
            return
        thr = self._threads if self._sequential else len(self._interpolators)
        # display('Bayesian Model Averaging', self.star, self._nlive,
        #         self._dlogz, self.ndim, self._bound, self._sample,
        #         thr, self._dynamic)
        if not self._sequential:
            jobs = []
            n_threads = len(self._interpolators)
            for intp, gr in zip(self._interpolators, self._grids):
                p = Process(target=self._bma_dynesty, args=([intp, gr]))
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        else:
            global interpolator
            for intp, gr in zip(self._interpolators, self._grids):
                interpolator = intp
                self.grid = gr
                out_file = self.out_folder + '/' + gr + '_out.pkl'
                print('\t\t\tFITTING MODEL : ' + gr)
                try:
                    self.fit_dynesty(out_file=out_file)
                except ValueError as e:
                    dump_out = self.out_folder + '/' + gr + '_DUMP.pkl'
                    pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                    DynestyError(dump_out, gr, e).__raise__()
                    continue

        # Now that the fitting finished, read the outputs and average
        # the posteriors
        outs = []
        for g in self._grids:
            in_folder = self.out_folder + '/' + g + '_out.pkl'
            with open(in_folder, 'rb') as out:
                outs.append(pickle.load(out))

        avgd = self.bayesian_model_average(outs, self._grids)
        self.save_bma(avgd)

        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time,
            self.out_folder, 'Bayesian Model Averaging', self.norm)
        pass

    def bayesian_model_average(self, outputs, grids):
        """Perform Bayesian Model Averaging."""
        # CONSIDER MAKING STATIC
        choice = np.random.choice
        evidences = []
        post_samples = []
        for o in outputs:
            evidences.append(o['global_lnZ'])
            post_samples.append(o['posterior_samples'])
        evidences = np.array(evidences)
        weights = evidences - evidences.min()
        weights = [np.exp(e) / np.exp(weights).sum() for e in weights]
        weights = np.array(weights)
        ban = ['loglike', 'priors', 'posteriors', 'mass']
        if self._norm:
            ban.append('rad')
        out = dict()
        out['originals'] = dict()
        out['weights'] = dict()
        for i, o in enumerate(outputs):
            out['weights'][o['model_grid']] = weights[i]
        for o in outputs:
            out['originals'][o['model_grid']] = o['posterior_samples']
        out['averaged_samples'] = dict()
        if self.n_samples == 'max':
            lens = []
            for o in post_samples:
                lens.append(len(o['teff']))
            lens = np.array(lens)
            self.n_samples = lens.min()
        for k in post_samples[0].keys():
            if k in ban:
                continue
            out['averaged_samples'][k] = np.zeros(self.n_samples)
            for i, o in enumerate(post_samples):
                # Skip fixed params
                try:
                    len(o[k])
                except TypeError:
                    continue
                weighted_samples = choice(o[k], self.n_samples) * weights[i]
                out['averaged_samples'][k] += weighted_samples
        out['evidences'] = dict()
        for e, g in zip(evidences, grids):
            out['evidences'][g] = e
        return out

    def _bma_dynesty(self, intp, grid):
        # interpolator, grid = args
        global interpolator
        interpolator = intp

        # Parallel parallelized routine experiment
        if self.experimental:
            if self._dynamic:
                with Pool(self._threads) as executor:
                    sampler = dynesty.DynamicNestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz_init=self._dlogz,
                                       nlive_batch=self._nlive,
                                       wt_kwargs={'pfrac': .95})
            else:
                with Pool(self._threads) as executor:
                    sampler = dynesty.NestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz=self._dlogz)

        elif self._dynamic:
            sampler = dynesty.DynamicNestedSampler(
                dynesty_loglike_bma, pt_dynesty, self.ndim,
                bound=self._bound, sample=self._sample, logl_args=([intp])

            )
            sampler.run_nested(dlogz_init=self._dlogz,
                               nlive_init=self._nlive,
                               wt_kwargs={'pfrac': .95})
        else:
            try:
                self.sampler = dynesty.NestedSampler(
                    dynesty_loglike_bma, pt_dynesty, self.ndim,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample,
                    logl_args=([intp])
                )
                self.sampler.run_nested(dlogz=self._dlogz)
            except Error:
                dump_out = self.out_folder + '/' + grid + '_DUMP.pkl'
                pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                er = DynestyError(dump_out, grid)
                er.log(self.out + '/output.log')
                er.__raise__()

        results = self.sampler.results
        out_file = self.out_folder + '/' + grid + '_out.pkl'
        self.save(out_file, results=results)
        pass

    def fit_multinest(self):
        """Run MuiltiNest."""
        # Set up some globals
        global mask, flux, flux_er, filts, wave
        mask = star.filter_mask
        flux = star.flux[mask]
        flux_er = star.flux_er[mask]
        filts = star.filter_names[mask]
        wave = star.wave[mask]
        path = self.out_folder + '/mnest/'
        create_dir(path)  # Create multinest path.
        pymultinest.run(
            multinest_log_like, pt_multinest, self.ndim,
            n_params=self.ndim,
            sampling_efficiency=0.8,
            evidence_tolerance=self._dlogz,
            n_live_points=self._nlive,
            outputfiles_basename=path + 'chains',
            max_modes=100,
            verbose=self.verbose,
            resume=False
        )
        out_file = self.out_folder + '/' + self._engine + '_out.pkl'
        self.save(out_file=out_file)
        pass

    def fit_dynesty(self, out_file=None):
        """Run dynesty."""
        # Set up some globals
        global mask, flux, flux_er, filts, wave
        mask = star.filter_mask
        flux = star.flux[mask]
        flux_er = star.flux_er[mask]
        filts = star.filter_names[mask]
        wave = star.wave[mask]
        if self._dynamic:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.DynamicNestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample,
                        pool=executor, walks=25,
                        queue_size=self._threads - 1
                    )
                    self.sampler.run_nested(dlogz_init=self._dlogz,
                                            nlive_init=self._nlive,
                                            wt_kwargs={'pfrac': 1})
            else:
                self.sampler = dynesty.DynamicNestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=25,
                    bound=self._bound, sample=self._sample

                )
                self.sampler.run_nested(dlogz_init=self._dlogz,
                                        nlive_init=self._nlive,
                                        wt_kwargs={'pfrac': 1})
        else:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.NestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample,
                        pool=executor, walks=25,
                        queue_size=self._threads - 1,
                    )
                    self.sampler.run_nested(dlogz=self._dlogz)
            else:
                self.sampler = dynesty.NestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=25,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample
                )
                self.sampler.run_nested(dlogz=self._dlogz)
        results = self.sampler.results
        if out_file is None:
            out_file = self.out_folder + '/' + self._engine + '_out.pkl'
        self.save(out_file, results=results)
        pass

    def save(self, out_file, results=None):
        """Save multinest/dynesty output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ : The global evidence.
        lnZerr : The global evidence error.
        posterior_samples : A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed : An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit : The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star : The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)
        engine : The fitting engine used (i.e. MultiNest or Dynesty)

        Also creates a log file with the best fit parameters and 1 sigma
        error bars.

        """
        out = dict()
        logdat = '#Parameter\tmedian\tupper\tlower\t3sig_CI\n'
        log_out = self.out_folder + '/' + 'best_fit.dat'
        if self._engine == 'multinest':
            lnz, lnzer, posterior_samples = self.multinest_results()
        else:
            lnz, lnzer, posterior_samples = self.dynesty_results(results)

        n = int(self.star.used_filters.sum())
        mask = self.star.filter_mask

        # Save global evidence

        if self._engine == 'dynesty':
            out['dynesty'] = results
        out['global_lnZ'] = lnz
        out['global_lnZerr'] = lnzer

        # Create raw samples holder

        out['posterior_samples'] = dict()
        j = 0
        k = 0  # filter counter
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                samples = posterior_samples[:, j]
                if 'noise' in param:
                    filt = self.star.filter_names[mask][k]
                    flx = self.star.flux[mask][k]
                    _, samples = flux_to_mag(flx, samples, filt)
                    k += 1
                out['posterior_samples'][param] = samples
                j += 1
            else:
                out['posterior_samples'][param] = self.fixed[i]

        # Save loglike, priors and posteriors.

        out['posterior_samples']['loglike'] = np.zeros(
            posterior_samples.shape[0]
        )

        # If normalization constant was fitted, create a distribution of radii
        # only if there's a distance available.

        if use_norm and star.dist != -1:
            rad = self._get_rad(
                out['posterior_samples']['norm'], star.dist, star.dist_e
            )
            out['posterior_samples']['rad'] = rad

        # Create a distribution of masses.

        logg_samp = out['posterior_samples']['logg']
        rad_samp = out['posterior_samples']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        out['posterior_samples']['grav_mass'] = mass_samp

        # Create a distribution of luminosities.

        teff_samp = out['posterior_samples']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        out['posterior_samples']['lum'] = lum_samp

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['posterior_samples']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            out['posterior_samples']['AD'] = ad_samp

        for i in range(posterior_samples.shape[0]):
            theta = build_params(
                posterior_samples[i, :], flux, flux_er, filts, coordinator,
                fixed, self.norm)
            out['posterior_samples']['loglike'][i] = log_likelihood(
                theta, flux, flux_er, wave, filts, interpolator, self.norm,
                av_law)
        lnlike = out['posterior_samples']['loglike']

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.
        # do only if not bma

        if not self.bma:
            out['best_fit'] = dict()
            out['uncertainties'] = dict()
            out['confidence_interval'] = dict()
            best_theta = np.zeros(order.shape[0])

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if 'noise' in param:
                        continue
                    samp = out['posterior_samples'][param]

                    if param == 'z':
                        logdat = out_filler(samp, logdat, param, '[Fe/H]', out)
                    elif param == 'norm':
                        logdat = out_filler(samp, logdat, param, '(R/D)^2',
                                            out, fmt='e')
                        if star.dist != 1:
                            logdat = out_filler(
                                out['posterior_samples']['rad'], logdat, 'rad',
                                'R', out
                            )
                    else:
                        logdat = out_filler(samp, logdat, param, param, out)
                else:
                    logdat = out_filler(
                        0, logdat, param, out, fixed=self.fixed[i]
                    )
                best_theta[i] = out['best_fit'][param]

            # Add derived mass to best fit dictionary.

            samp = out['posterior_samples']['grav_mass']
            logdat = out_filler(
                samp, logdat, 'grav_mass', 'grav_mass', out
            )

            # Add derived luminosity to best fit dictionary.

            samp = out['posterior_samples']['lum']
            logdat = out_filler(samp, logdat, 'lum', 'lum', out)

            # Add derived angular diameter to best fit dictionary.

            if not use_norm:
                samp = out['posterior_samples']['AD']
                logdat = out_filler(samp, logdat, 'AD', 'AD', out)

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if 'noise' not in param:
                        continue
                    samp = out['posterior_samples'][param]
                    logdat = out_filler(samp, logdat, param,
                                        param, out, fmt='f')

            # Fill in best loglike, prior and posterior.

            out['best_fit']['loglike'] = log_likelihood(
                best_theta, flux, flux_er, wave,
                filts, interpolator, self.norm, av_law
            )
            lnlike = out['best_fit']['loglike']

            # Spectral type

            # Load Mamajek spt table
            mamajek_spt = np.loadtxt(
                filesdir + '/mamajek_spt.dat', dtype=str, usecols=[0])
            mamajek_temp = np.loadtxt(
                filesdir + '/mamajek_spt.dat', usecols=[1])

            # Find spt
            spt_idx = np.argmin(abs(mamajek_temp - out['best_fit']['teff']))
            spt = mamajek_spt[spt_idx]
            out['spectral_type'] = spt

        # Utilities for plotting.

        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['engine'] = self._engine
        out['norm'] = self.norm
        out['model_grid'] = self.grid
        out['av_law'] = av_law
        if not self.bma:
            with open(log_out, 'w') as logfile:
                logfile.write(logdat)
        pickle.dump(out, open(out_file, 'wb'))
        pass

    def save_bma(self, avgd):
        """Save BMA output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ : The global evidences.
        posterior_samples : A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed : An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit : The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star : The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)

        Also creates a log file with the best fit parameters and 1 sigma
        error bars.

        """
        out = dict()
        logdat = '#Parameter\tmedian\tupper\tlower\t3sig_low\t3sig_up\n'
        log_out = self.out_folder + '/best_fit.dat'
        prob_out = self.out_folder + '/model_probabilities.dat'

        n = int(self.star.used_filters.sum())
        mask = self.star.filter_mask

        # Save global evidence of each model.
        out['lnZ'] = avgd['evidences']

        # Save original samples.
        out['originals'] = avgd['originals']

        # Save weights.
        out['weights'] = avgd['weights']

        # Create raw samples holder
        out['posterior_samples'] = dict()
        j = 0
        for i, par in enumerate(order):
            if not self.coordinator[i]:
                out['posterior_samples'][par] = avgd['averaged_samples'][par]
                j += 1
            else:
                out['posterior_samples'][par] = self.fixed[i]

        # If normalization constant was fitted, create a distribution of radii.

        if use_norm and star.dist != -1:
            rad = self._get_rad(
                out['posterior_samples']['norm'], star.dist, star.dist_e
            )
            out['posterior_samples']['rad'] = rad

        # Create a distribution of masses

        logg_samp = out['posterior_samples']['logg']
        rad_samp = out['posterior_samples']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        out['posterior_samples']['grav_mass'] = mass_samp

        # Create a distribution of luminosities.

        teff_samp = out['posterior_samples']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        out['posterior_samples']['lum'] = lum_samp

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['posterior_samples']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            out['posterior_samples']['AD'] = ad_samp

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.

        out['best_fit'] = dict()
        out['uncertainties'] = dict()
        out['confidence_interval'] = dict()
        best_theta = np.zeros(order.shape[0])
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if 'noise' in param:
                    continue
                samp = out['posterior_samples'][param]

                if param == 'z':
                    logdat = out_filler(samp, logdat, param, '[Fe/H]', out)
                elif param == 'norm':
                    logdat = out_filler(samp, logdat, param, '(R/D)^2', out,
                                        fmt='e')
                    if star.dist != 1:
                        logdat = out_filler(
                            out['posterior_samples']['rad'], logdat, 'rad',
                            'R', out
                        )
                else:
                    logdat = out_filler(samp, logdat, param, param, out)
            else:
                logdat = out_filler(
                    0, logdat, param, param, out, fixed=self.fixed[i]
                )
            best_theta[i] = out['best_fit'][param]

        # Add derived mass to best fit dictionary.

        samp = out['posterior_samples']['grav_mass']
        logdat = out_filler(samp, logdat, 'grav_mass', 'grav_mass', out)

        # Add derived luminosity to best fit dictionary.

        samp = out['posterior_samples']['lum']
        logdat = out_filler(samp, logdat, 'lum', 'lum', out)

        # Add derived angular diameter to best fit dictionary.

        if not use_norm:
            samp = out['posterior_samples']['AD']
            logdat = out_filler(samp, logdat, 'AD', 'AD', out)

        # Add estimated age to best fit dictionary.

        age_samp, mass_samp, eep_samp = self.estimate_age(out['best_fit'],
                                                          out['uncertainties'])
        out['posterior_samples']['age'] = age_samp
        out['posterior_samples']['iso_mass'] = mass_samp
        out['posterior_samples']['eep'] = eep_samp
        logdat = out_filler(age_samp, logdat, 'age', 'age', out)
        logdat = out_filler(mass_samp, logdat, 'iso_mass', 'iso_mass', out)
        logdat = out_filler(eep_samp, logdat, 'eep', 'eep', out)
        probdat = ''

        for k in avgd['weights'].keys():
            probdat += '{}_probability\t{:.4f}\n'.format(k, avgd['weights'][k])

        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if 'noise' not in param:
                    continue
                samp = out['posterior_samples'][param]
                logdat = out_filler(samp, logdat, param, param, out, fmt='f')

        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['norm'] = self.norm
        out['engine'] = 'Bayesian Model Averaging'
        out['av_law'] = av_law

        # Spectral type

        # Load Mamajek spt table
        mamajek_spt = np.loadtxt(
            filesdir + '/mamajek_spt.dat', dtype=str, usecols=[0])
        mamajek_temp = np.loadtxt(filesdir + '/mamajek_spt.dat', usecols=[1])

        # Find spt
        spt_idx = np.argmin(abs(mamajek_temp - out['best_fit']['teff']))
        spt = mamajek_spt[spt_idx]
        out['spectral_type'] = spt
        out_file = self.out_folder + '/BMA_out.pkl'
        with open(log_out, 'w') as logfile:
            logfile.write(logdat)
        with open(prob_out, 'w') as logfile:
            logfile.write(probdat)
        pickle.dump(out, open(out_file, 'wb'))
        pass

    def multinest_results(self):
        """Extract posterior samples, global evidence and its error."""
        path = self.out_folder + '/mnest/'
        output = pymultinest.Analyzer(outputfiles_basename=path + 'chains',
                                      n_params=self.ndim)
        posterior_samples = output.get_equal_weighted_posterior()[:, :-1]
        lnz = output.get_stats()['global evidence']
        lnzer = output.get_stats()['global evidence error']
        return lnz, lnzer, posterior_samples

    def dynesty_results(self, results):
        """Extract posterior samples, global evidence and its error."""
        weights = np.exp(results['logwt'] - results['logz'][-1])
        posterior_samples = resample_equal(results.samples, weights)
        lnz = results.logz[-1]
        lnzer = results.logzerr[-1]
        return lnz, lnzer, posterior_samples

    def _get_mass(self, logg, rad):
        """Calculate mass from logg and radius."""
        # Solar logg = 4.437
        # g = g_Sol * M / R**2
        mass = logg + 2 * np.log10(rad) - 4.437
        mass = 10 ** mass
        return mass

    def _get_lum(self, teff, rad):
        sb = sigma_sb.to(u.solLum / u.K ** 4 / u.solRad ** 2).value
        L = 4 * np.pi * rad ** 2 * sb * teff ** 4
        return L

    def _get_rad(self, samples, dist, dist_e):
        """Calculate radius from the normalization constant and distance."""
        norm = samples
        # Create a synthetic distribution for distance.
        # N = (R / D) ** 2
        d = st.norm(loc=dist, scale=dist_e).rvs(size=norm.shape[0])
        n = np.sqrt(norm)
        r = n * d  # This is in pc
        r *= u.pc.to(u.solRad)  # Transform to Solar radii
        return r

    def _get_angular_diameter(self, rad, dist):
        diameter = 2 * rad
        ad = (diameter / (dist * u.pc.to(u.solRad))) * u.rad.to(u.marcsec)
        return ad

    def estimate_age(self, bf, unc):
        """Estimate age using MIST isochrones.

        Parameters
        ----------
        bf : dict
            A dictionary with the best fit parameters.

        unc : dict
            A dictionary with the uncertainties.

        """
        colors = [
            'red', 'green', 'blue', 'yellow',
            'grey', 'magenta', 'cyan', 'white'
        ]
        c = random.choice(colors)
        print(
            colored(
                '\t\t*** ESTIMATING AGE AND MASS USING MIST ISOCHRONES ***', c
            )
        )
        params = dict()  # params for isochrones.
        for i, k in enumerate(order):
            if k == 'logg' or 'noise' in k:
                continue
            if k == 'teff':
                par = 'Teff'
            if k == 'z':
                par = 'feh'
            if k == 'dist':
                par = 'distance'
            if k == 'rad':
                par = 'radius'
            if k == 'Av':
                par = 'AV'
            if not self.coordinator[i]:
                if k != 'norm':
                    params[par] = (bf[k], max(unc[k]))
                if k == 'norm' and star.dist != -1:
                    params['distance'] = (star.dist, star.dist_e)
                if par == 'distance':
                    err = max(unc[k])
                    params['parallax'] = (1000 / bf[k], 1000 * err / bf[k] ** 2)
            else:
                continue

        params['mass'] = (bf['grav_mass'], max(unc['grav_mass']))
        if star.lum != 0 and star.lum_e != 0:
            params['logL'] = (np.log10(bf['lum']),
                              abs(np.log10(max(unc['lum']))))
        mask = np.array([1, 1, 1,
                         0, 0,
                         1, 1, 1,
                         0, 0,
                         0, 0, 0, 0,
                         1, 1, 1,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         1, 1,
                         0, 0,
                         0, 0,
                         0, 1, 0])
        mags = self.star.mags[mask == 1]
        mags_e = self.star.mag_errs[mask == 1]
        bands = [
            'H', 'J', 'K',
            'U', 'V', 'B',
            'G', 'RP', 'BP',
            'W1', 'W2',
            'TESS'
        ]
        used_bands = []
        for m, e, b in zip(mags, mags_e, bands):
            if m != 0:
                params[b] = (m, e)
                used_bands.append(b)

        age_samp, mass_samp, eep_samp = estimate(used_bands, params, logg=False)

        return age_samp, mass_samp, eep_samp


#####################
# Dynesty and multinest wrappers


def dynesty_loglike_bma(cube, interpolator):
    """Dynesty log likelihood wrapper for BMA."""
    theta = build_params(cube, coordinator, fixed, use_norm)
    return log_likelihood(theta, star, interpolator, use_norm, av_law)


def dynesty_log_like(cube):
    """Dynesty log likelihood wrapper."""
    theta = build_params(
        cube, flux, flux_er, filts, coordinator, fixed, use_norm
    )
    return log_likelihood(theta, flux, flux_er, wave,
                          filts, interpolator, use_norm, av_law)


def pt_dynesty(cube):
    """Dynesty prior transform."""
    return prior_transform_dynesty(cube, flux, flux_er, filts, prior_dict,
                                   coordinator, use_norm)


def multinest_log_like(cube, ndim, nparams):
    """Multinest log likelihood wrapper."""
    theta = [cube[i] for i in range(ndim)]
    theta = build_params(
        theta, flux, flux_er, filts, coordinator, fixed, use_norm
    )
    return log_likelihood(theta, flux, flux_er, wave,
                          filts, interpolator, use_norm, av_law)


def pt_multinest(cube, ndim, nparams):
    """Multinest prior transform."""
    prior_transform_multinest(cube, flux, flux_er, filts, prior_dict,
                              coordinator, use_norm)
