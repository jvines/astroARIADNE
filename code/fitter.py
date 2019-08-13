"""Main driver of the fitting routine."""
from __future__ import division, print_function

import pickle
import time
from multiprocessing import Pool, cpu_count

import astropy.units as u
import dill
import scipy as sp
import scipy.stats as st
from extinction import apply, fitzpatrick99

import emcee
import pymultinest
from phot_utils import *
from sed_library import *

# GLOBAL VARIABLES

order = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
with open('interpolations.pkl', 'rb') as intp:
    interpolators = pickle.load(intp)


class Fitter:

    def __init__(self, star, setup, priorfile=None, engine='emcee'):
        global prior_dict
        self.star = star
        # Interpolations
        # with open('interpolations.pkl', 'rb') as intp:
        #     self.interpolators = pickle.load(intp)
        # Parameter coordination.
        # Order for the parameters are:
        # tef, logg, z, dist, rad, Av
        self.coordinator = sp.zeros(6)  # 1 for fixed params

        # Setup priors.
        self.default_priors = self._default_priors()
        self.priorfile = priorfile
        self.create_priors(priorfile)
        prior_dict = self.priors

        # Global settings
        self.ndim = self.get_ndim()

        if engine == 'emcee':
            # emcee settings
            self.nwalkers = setup[0]
            self.nsteps = setup[1]
            if len(setup) == 3:
                self.burnout = setup[2]
            else:
                self.burnout = nsteps // 2
            self.cores = cpu_count()
        if engine == 'multinest' or engine == 'dynesty':
            # Nested sampling settings
            self.live_points = setup[0]
            with open('logg_ppf.pkl', 'rb') as jar:
                prior_dict['logg'] = pickle.load(jar)
            if not self.star.get_temp or not self.star.temp:
                with open('teff_ppf.pkl', 'rb') as jar:
                    prior_dict['teff'] = pickle.load(jar)

    def create_artist(self):
        """Instantiate the plotter here."""
        pass

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = 6 - self.coordinator.sum()
        return int(ndim)

    def _default_priors(self):
        defaults = dict()
        with open('logg_kde.pkl', 'rb') as pkl:
            logg_prior = pickle.load(pkl)
        if self.star.get_temp or self.star.temp:
            defaults['teff'] = st.norm(
                loc=self.star.temp, scale=self.star.temp_e)
        else:
            with open('teff_kde.pkl', 'rb') as pkl:
                teff_prior = pickle.load(pkl)
            defaults['teff'] = teff_prior['teff']
        defaults['logg'] = logg_prior['logg']
        defaults['z'] = st.norm(loc=-0.125, scale=0.234)
        defaults['dist'] = st.norm(
            loc=self.star.dist, scale=self.star.dist_e)
        defaults['rad'] = st.norm(
            loc=self.star.rad, scale=self.star.rad_e)
        defaults['Av'] = st.uniform(loc=0, scale=.032)
        # defaults['inflation'] = st.norm(loc=0, scale=.05)
        return defaults

    def create_priors(self, priorfile):
        """Read the prior file.

        Returns a dictionary with the pdfs of each parameter/prior
        """
        if priorfile:
            param, prior, bounds = sp.loadtxt(
                priorfile, usecols=[0, 1, 2], unpack=True)
        # Dict with priors.
        prior_dict = dict()
        if priorfile:
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
                    self.coordinator[par] = bo
            self.priors = prior_dict
        else:
            print('No priorfile detected. Using default priors.')
            self.priors = self.default_priors

    def pos(self):
        """Set initial position of the walkers."""
        p0 = sp.empty(self.nwalkers)

        for i, k in enumerate(order):
            flag = True if self.coordinator[i] else False

            if flag:
                continue
            if k == 'logg' and not self.priorfile:
                p0 = sp.vstack((p0, self.priors[k].resample(self.nwalkers)))
            elif k == 'teff' and not self.priorfile and \
                    not self.star.get_temp and self.star.temp is None:
                p0 = sp.vstack((p0, self.priors[k].resample(self.nwalkers)))
            else:
                p0 = sp.vstack((p0, self.priors[k].rvs(size=self.nwalkers)))
        self.p0 = p0.T[:, 1:]

    def save_self(self):
        with open('test.pkl', 'wb') as jar:
            dill.dump(self, jar)
        pass

    def fit_emcee(self):
        """Run emcee."""
        # Randomize starting position for the walkers.
        self.pos()

        with Pool(cpu_count()) as pool:
            self.sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, log_prob,
                args=[
                    self.star,
                    self.coordinator,
                ],
                pool=pool
            )
            self.sampler.run_mcmc(self.p0, self.nsteps + self.burnout,
                                  progress=True)
        flat_samples = self.sampler.get_chain(
            discard=self.burnout, thin=1, flat=True)
        self.chain = flat_samples
        self.save_self()

    def fit_multinest(self):
        """Run MuiltiNest."""
        global star
        out = dict()
        star = self.star
        pymultinest.run(
            log_like, pt_multinest, self.ndim,
            n_params=self.ndim,
            sampling_efficiency=0.8,
            evidence_tolerance=0.01,
            n_live_points=self.live_points,
            outputfiles_basename='test/chains',
            verbose=True,
            resume=False
        )
        output = pymultinest.Analyzer(outputfiles_basename='test/',
                                      n_params=self.ndim)
        posterior_samples = output.get_equal_weighted_posterior()[:, :-1]
        out['lnZ'] = output.get_stats()['global evidence']
        out['lnZerr'] = output.get_stats()['global evidence error']
        out['posterior_samples'] = dict()
        for i, param in enumerate(order):
            out['posterior_samples'][param] = posterior_samples[:, i]
        out['posterior_samples']['loglike'] = sp.zeros(
            posterior_samples.shape[0])
        for i in range(posterior_samples.shape[0]):
            out['posterior_samples']['log_like'][i] = log_likelihood(
                posterior_samples[i, :], self.star, interpolators)

        pickle.dump(out, open('multinest_out.pkl', 'wb'))
#####################


def log_like(cube, ndim, nparams):
    """Multinest log likelihood wrapper."""
    # import pdb; pdb.set_trace()
    theta = [cube[i] for i in range(ndim)]
    return log_likelihood(theta, star, interpolators)


def pt_multinest(cube, ndim, nparams):
    """Multinest prior transform."""
    prior_transform_multinest(cube, star, prior_dict)


def log_prob(theta, star, coordinator):
    """Wrap for sed_library.log_probability."""
    return log_probability(theta, star, prior_dict, coordinator, interpolators)
