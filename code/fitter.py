"""Main driver of the fitting routine."""
from __future__ import division, print_function

from multiprocessing import Pool, cpu_count

import scipy as sp
import scipy.stats as st
from scipy.interpolate import griddata

import emcee
from phot_utils import *


class Fitter:

    def __init__(self, star, nwalkers, nsteps, priorfile=None, burnout=False):
        self.star = star

        # emcee settings
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnout = nsteps // 2 if not burnout else burnout
        self.cores = cpu_count()

        # Setup priors.
        self.create_priors(priorfile)

    def create_artist(self):
        """Instantiate the plotter here."""
        pass

    def create_priors(self, priorfile):
        """Read the prior file.

        Returns a dictionary with the pdfs of each parameter/prior
        """
        param, prior, bounds = sp.loadtxt(
            priorfile, usecols=[0, 1, 2], unpack=True)
        # Dict with priors.
        prior_dict = dict()
        for par, pri, bo in zip(param, prior, bounds):
            if pri.lower() == 'uniform':
                a, b = bounds.split(',')
                a, b = float(a), float(b)
                prior_dict[par] = st.uniform(loc=a, scale=b - a)
            elif pri.lower() == 'normal':
                mu, sig = bounds.split(',')
                mu, sig = float(mu), float(sig)
                prior_dict[par] = st.norm(loc=mu, scale=sig)
            elif pri.lower() == 'truncatednormal':
                mu, sig, up, low = bounds.split(',')
                mu, sig, up, low = float(mu), float(sig), float(up), float(low)
                up, low = (up - mu) / sig, (low - mu) / sig
                priot_dict[par] = st.truncnorm(
                    a=low, b=up, loc=mu, scale=sig)
            elif pri.lower() == 'fixed':
                continue
        self.priors = prior_dict

        def pos(self, ndim, prior_dict):
            p0 = sp.empty(self.nwalkers)

            for k in prior_dict.keys():
                p0 = sp.vstack((p0, prior_dict[k].rvs(size=self.nwalkers)))
            self.p0 = p0.T[:, 1:]

        def fit(self):
            """Run emcee here."""
            # The parameters are teff, logg, z, rad, dist, unless z is fixed.
            ndim = 5 if not self.star.fixed_z else 4
            # Randomize starting position for the walkers.
            self.pos(ndim, prior_dict)
            sampler = emcee.EnsembleSampler(
                self.nwalkers, ndim, log_probability,
                args=(self.star, self.prior_dict))
            sampler.run_mcmc(self.p0, self.steps, progress=True)
            flat_samples = sampler.get_chain(
                discard=self.burnout, thin=1, flat=True)
            for i in range(ndim):
                print(flat_samples[:, i].mean())
            pass
