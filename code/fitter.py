"""Main driver of the fitting routine."""
import os
import pickle
import random
import time
from contextlib import closing
from multiprocessing import Pool, cpu_count

import astropy.units as u
import scipy as sp
import scipy.stats as st
from isochrones import SingleStarModel, get_ichrone

import dynesty
import pymultinest
from dynesty.utils import resample_equal
from isochrone import estimate
from phot_utils import *
from sed_library import *
from utils import *

# TODO: Add a log file

# GLOBAL VARIABLES

order = sp.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation'])
with closing(open('interpolations.pkl', 'rb')) as intp:
    interpolators = pickle.load(intp)


class Fitter:

    def __init__(self, star, setup, priorfile=None, estimate_logg=False,
                 engine='multinest', out_folder=None,
                 verbose=True):
        global prior_dict, coordinator, fixed
        # Global settings
        self.start = time.time()
        self.star = star
        self.engine = engine
        self.verbose = verbose

        if out_folder is None:
            self.out_folder = self.star.starname + '/'
        else:
            self.out_folder = out_folder

        create_dir(self.out_folder)  # Create output folder.

        if engine == 'multinest':
            self.live_points = setup[0]
            self.dlogz = setup[1]
        if engine == 'dynesty':
            self.live_points = setup[0]
            self.dlogz = setup[1]
            self.bound = setup[2]
            self.sample = setup[3]
            self.nthreads = setup[4]
            self.dynamic = setup[5]

        # Parameter coordination.
        # Order for the parameters are:
        # tef, logg, z, dist, rad, Av
        self.coordinator = sp.zeros(7)  # 1 for fixed params
        self.fixed = sp.zeros(7)
        coordinator = self.coordinator
        fixed = self.fixed

        # Setup priors.
        self.default_priors = self._default_priors(estimate_logg)
        self.priorfile = priorfile
        self.create_priors(priorfile)
        prior_dict = self.priors

        # Get dimensions.
        self.ndim = self.get_ndim()

        if engine == 'multinest':
            display(self.engine, self.star, self.live_points,
                    self.dlogz, self.ndim)
        if engine == 'dynesty':
            display(self.engine, self.star, self.live_points,
                    self.dlogz, self.ndim, self.bound, self.sample,
                    self.nthreads, self.dynamic)

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = 7 - self.coordinator.sum()
        return int(ndim)

    def _default_priors(self, estimate_logg):
        defaults = dict()
        # Logg prior setup.
        if not estimate_logg:
            with closing(open('../Datafiles/logg_ppf.pkl', 'rb')) as jar:
                defaults['logg'] = pickle.load(jar)
        else:
            params = dict()  # params for isochrones.
            if self.star.get_temp:
                params['Teff'] = (self.star.temp, self.star.temp_e)
            if self.star.get_lum:
                params['LogL'] = (self.star.lum, self.star.lum_e)
            if self.star.get_rad:
                params['radius'] = (self.star.rad, self.star.rad_e)
            if self.star.get_plx:
                params['parallax'] = (self.star.plx, self.star.plx_e)
            mask = sp.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
            mags = self.star.mags[mask == 1]
            mags_e = self.star.mag_errs[mask == 1]
            bands = ['H', 'J', 'K', 'G', 'RP', 'BP', 'W1', 'W2']
            used_bands = []
            for m, e, b in zip(mags, mags_e, bands):
                if m != 0:
                    params[b] = (m, e)
                    used_bands.append(b)
            if self.verbose and estimate_logg:
                print('*** ESTIMATING LOGG USING MIST ISOCHRONES ***')
            logg_est = estimate(used_bands, params)
            if logg_est is not None:
                defaults['logg'] = st.norm(loc=logg_est[0], scale=logg_est[1])
        # Teff prior setup.
        if self.star.get_temp:
            defaults['teff'] = st.norm(
                loc=self.star.temp, scale=self.star.temp_e)
        else:
            with closing(open('../Datafiles/teff_ppf.pkl', 'rb')) as jar:
                defaults['teff'] = pickle.load(jar)
            defaults['teff'] = teff_prior['teff']
        defaults['z'] = st.norm(loc=-0.125, scale=0.234)
        defaults['dist'] = st.norm(
            loc=self.star.dist, scale=self.star.dist_e)
        defaults['rad'] = st.norm(
            loc=self.star.rad, scale=self.star.rad_e)
        defaults['Av'] = st.uniform(loc=0, scale=self.star.Av)
        up, low = (5 - 0.5) / 0.5, (0 - 0.5) / 0.5
        defaults['inflation'] = st.truncnorm(a=low, b=up, loc=0.5, scale=0.5)
        return defaults

    def create_priors(self, priorfile):
        """Read the prior file.

        Returns a dictionary with each parameter's prior
        """
        if priorfile:
            param, prior, bounds = sp.loadtxt(
                priorfile, usecols=[0, 1, 2], unpack=True, dtype=object)
            # Dict with priors.
            prior_dict = dict()
            for par, pri, bo in zip(param, prior, bounds):
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
                    up, low = (up - mu) / sig, (low - mu) / sig
                    priot_dict[par] = st.truncnorm(
                        a=low, b=up, loc=mu, scale=sig)
                elif pri.lower() == 'default':
                    prior_dict[par] = self.default_priors[par]
                elif pri.lower() == 'fixed':
                    idx = sp.where(par == order)[0]
                    self.coordinator[idx] = 1
                    self.fixed[idx] = float(bo)
            self.priors = prior_dict
        else:
            print('No priorfile detected. Using default priors.')
            self.priors = self.default_priors
        pass

    def fit(self):
        """Run fitting routine."""
        global star
        star = self.star
        if self.engine == 'multinest':
            self.fit_multinest()
        else:
            self.fit_dynesty()
        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time, self.out_folder)
        pass

    def fit_multinest(self):
        """Run MuiltiNest."""
        path = self.out_folder + 'multinest/'
        create_dir(path)  # Create multinest path.
        pymultinest.run(
            multinest_log_like, pt_multinest, self.ndim,
            n_params=self.ndim,
            sampling_efficiency=0.8,
            evidence_tolerance=self.dlogz,
            n_live_points=self.live_points,
            outputfiles_basename=path + 'chains',
            max_modes=100,
            verbose=self.verbose,
            resume=False
        )
        self.save()
        pass

    def fit_dynesty(self):
        """Run dynesty."""
        if self.dynamic:
            if self.nthreads > 1:
                with closing(Pool(self.nthreads)) as executor:
                    sampler = dynesty.DynamicNestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        bound=self.bound, sample=self.sample, pool=executor,
                        queue_size=self.nthreads - 1
                    )
                    sampler.run_nested(dlogz_init=self.dlogz,
                                       nlive_init=self.live_points)
            else:
                sampler = dynesty.DynamicNestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim,
                    bound=self.bound, sample=self.sample

                )
                sampler.run_nested(dlogz_init=self.dlogz,
                                   nlive_init=self.live_points)
        else:
            if self.nthreads > 1:
                with closing(Pool(self.nthreads)) as executor:
                    sampler = dynesty.NestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        nlive=self.live_points, bound=self.bound,
                        sample=self.sample, pool=executor,
                        queue_size=self.nthreads - 1
                    )
                    sampler.run_nested(dlogz=self.dlogz)
            else:
                sampler = dynesty.NestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim,
                    nlive=self.live_points, bound=self.bound,
                    sample=self.sample
                )
                sampler.run_nested(dlogz=self.dlogz)
        results = sampler.results
        self.save(results=results)
        pass

    def save(self, results=None):
        """Analyze and save multinest output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ : The global evidence
        lnZerr : The global evidence error
        posterior_samples : A dictionary containing the samples of each
                            parameter (even if it's fixed) and the
                            log likelihood for each set of sampled parameters.
        fixed : An array with the fixed parameter values
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit : The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star : The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)
        engine : The fitting engine used (i.e. MultiNest or Dynesty)

        """
        out = dict()
        out_file = self.out_folder + '/' + self.engine + '_out.pkl'
        if self.engine == 'multinest':
            lnz, lnzer, posterior_samples = self.multinest_results()
        else:
            lnz, lnzer, posterior_samples = self.dynesty_results(results)

        out['lnZ'] = lnz
        out['lnZerr'] = lnzer

        out['posterior_samples'] = dict()
        j = 0
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                out['posterior_samples'][param] = posterior_samples[:, j]
                j += 1
            else:
                out['posterior_samples'][param] = self.fixed[i]
        out['posterior_samples']['loglike'] = sp.zeros(
            posterior_samples.shape[0])
        for i in range(posterior_samples.shape[0]):
            theta = build_params(posterior_samples[i, :], coordinator, fixed)
            out['posterior_samples']['loglike'][i] = log_likelihood(
                theta, self.star, interpolators)
        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['best_fit'] = dict()
        best_theta = sp.zeros(order.shape[0])
        j = 0
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                out['best_fit'][param] = sp.median(posterior_samples[:, j])
                j += 1
            else:
                out['best_fit'][param] = self.fixed[i]
            best_theta[i] = out['best_fit'][param]
        out['best_fit']['likelihood'] = log_likelihood(
            best_theta, self.star, interpolators)
        out['star'] = self.star
        out['engine'] = self.engine
        pickle.dump(out, open(out_file, 'wb'))
        pass

    def multinest_results(self):
        """Extract posterior samples, global evidence and its error."""
        path = self.out_folder + 'multinest/'
        output = pymultinest.Analyzer(outputfiles_basename=path + 'chains',
                                      n_params=self.ndim)
        posterior_samples = output.get_equal_weighted_posterior()[:, :-1]
        lnz = output.get_stats()['global evidence']
        lnzer = output.get_stats()['global evidence error']
        return lnz, lnzer, posterior_samples

    def dynesty_results(self, results):
        """Extract posterior samples, global evidence and its error."""
        weights = sp.exp(results['logwt'] - results['logz'][-1])
        posterior_samples = resample_equal(results.samples, weights)
        lnz = results.logz[-1]
        lnzer = results.logzerr[-1]
        return lnz, lnzer, posterior_samples


#####################
def dynesty_log_like(cube):
    """Dynesty log likelihood wrapper."""
    theta = build_params(cube, coordinator, fixed)
    return log_likelihood(theta, star, interpolators)


def pt_dynesty(cube):
    """Dynesty prior transform."""
    return prior_transform_dynesty(cube, star, prior_dict, coordinator)


def multinest_log_like(cube, ndim, nparams):
    """Multinest log likelihood wrapper."""
    theta = [cube[i] for i in range(ndim)]
    theta = build_params(theta, coordinator, fixed)
    return log_likelihood(theta, star, interpolators)


def pt_multinest(cube, ndim, nparams):
    """Multinest prior transform."""
    prior_transform_multinest(cube, star, prior_dict, coordinator)


def log_prob(theta, star):
    """Wrap for sed_library.log_probability."""
    # DEPRECATED
    return log_probability(theta, star, prior_dict, coordinator, interpolators,
                           fixed)
